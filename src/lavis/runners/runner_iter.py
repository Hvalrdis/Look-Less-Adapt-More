"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import logging
import os
import time, math, random

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import download_cached_file, is_main_process, main_process
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.runners.runner_base import RunnerBase
from torch.utils.data.dataset import ChainDataset
import contextlib

@registry.register_runner("runner_iter")
class RunnerIter(RunnerBase):
    """
    Run training based on the number of iterations. This is common when
    the training dataset size is large. Underhood logic is similar to
    epoch-based training by considering every #iters_per_inner_epoch as an
    inner epoch.

    In iter-based runner, after every #iters_per_inner_epoch steps, we

        1) do a validation epoch;
        2) schedule the learning rate;
        3) save the checkpoint.

    We refer every #iters_per_inner_epoch steps as an inner epoch.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        super().__init__(cfg, task, model, datasets, job_id)

        self.start_iters = 0

        self.max_iters = int(self.config.run_cfg.get("max_iters", -1))
        assert self.max_iters > 0, "max_iters must be greater than 0."

        self.iters_per_inner_epoch = int(
            self.config.run_cfg.get("iters_per_inner_epoch", -1)
        )
        assert (
            self.iters_per_inner_epoch > 0
        ), "iters_per_inner_epoch must be greater than 0."

    @property
    def max_epoch(self):
        return int(self.max_iters / self.iters_per_inner_epoch)

    @property
    def cur_epoch(self):
        try:
            return self.train_loader.epoch
        except AttributeError:
            # pipeline data (e.g. LAION) is streaming, have no concept of epoch
            return 0

    def _progress(self, cur_iters):
        return "{}_iters={}".format(self.cur_epoch, cur_iters)

    def _barrier_if_dist(self):
        try:
            import torch.distributed as dist
            if getattr(self, "distributed", False) and dist.is_available() and dist.is_initialized():
                dist.barrier()
        except Exception:
            pass

    # === Fallback: bump batch size after prune ===
    def _maybe_bump_batch_size_after_prune(self, factor: float = 1.0):
        """
        """
        try:
            return super()._maybe_bump_batch_size_after_prune(factor)
        except AttributeError:
            logging.warning("[Recovery] bump batch size skipped: RunnerBase has no method; fallback no-op.")
            return False

    def _maybe_diag_act_stats(self, split_name: str):
        """
        """
        from lavis.common.dist_utils import is_main_process

        if not is_main_process():
            return

        run = getattr(self.config, "run_cfg", {}) or {}
        if not bool(run.get("diag_act_stats", False)):
            return

        if getattr(self, "_diag_act_done", False):
            return

        model = self.unwrap_dist_model(self.model)

        if not hasattr(model, "_toggle_ffn_act_stats") or not hasattr(model, "_collect_ffn_act_stats"):
            logging.warning("[Diag-Act] model has no FFN act stat hooks; skip.")
            self._diag_act_done = True
            return

        #  split  dataloader
        dl_map = getattr(self, "dataloaders", None)
        dl = None
        if isinstance(dl_map, dict) and split_name in dl_map:
            dl = dl_map[split_name]
        if dl is None:
            logging.warning(f"[Diag-Act] no dataloader for split={split_name}; skip.")
            self._diag_act_done = True
            return

        subset = int(run.get("diag_act_subset", run.get("diag_attn_subset", 128)))
        device = self.device

        #  FFN 
        processed = 0

        vision_norm_sum = 0.0
        vision_norm_sq_sum = 0.0
        vision_tok_count = 0         # token 

        used_norm_sum = 0.0
        used_norm_sq_sum = 0.0
        used_tok_count = 0

        model.eval()

        try:
            model._toggle_ffn_act_stats(True)
        except Exception as e:
            logging.warning(f"[Diag-Act] enable hooks failed: {e}")
            self._diag_act_done = True
            return

        with torch.no_grad():
            for _, samples in enumerate(dl):
                # 
                image = samples.get("image", None)
                if image is None:
                    for k in ["images", "image_tensor", "pixel_values"]:
                        if k in samples:
                            image = samples[k]
                            break
                if image is None:
                    continue

                image = image.to(device, non_blocking=True)

                if hasattr(model, "ln_vision") and hasattr(model, "visual_encoder"):
                    img = model.ln_vision(model.visual_encoder(image))
                else:
                    img = model.visual_encoder(image)

                with torch.no_grad():
                    tok_norm = img.norm(dim=-1)          # [B, T_v]
                    vision_norm_sum += tok_norm.sum().item()
                    vision_norm_sq_sum += (tok_norm ** 2).sum().item()
                    vision_tok_count += tok_norm.numel()

                atts = torch.ones(img.size()[:-1], dtype=torch.long, device=image.device)
                q = model.query_tokens.expand(img.shape[0], -1, -1)
                _ = model.Qformer.bert(
                    query_embeds=q,
                    encoder_hidden_states=img,
                    encoder_attention_mask=atts,
                    return_dict=True,
                )

                try:
                    keep = float(getattr(model, "_vision_token_keep_ratio", 1.0))
                    if keep < 1.0 and hasattr(model, "_select_vision_tokens"):
                        mode = getattr(model, "_vision_token_score_mode", "l2norm")
                        tail = getattr(model, "_vision_token_tail_strategy", "drop")

                        attn_for_score = None
                        mode_l = str(mode).lower()
                        if mode_l in ("attn_entropy", "attn", "entropy"):
                            #  cross-attn hook  forward 
                            if hasattr(model, "_collect_cross_attn_maps"):
                                xs = model._collect_cross_attn_maps(with_layer_idx=False)  # List[Tensor(B,H,Q,N)]
                                if xs:
                                    base = xs[0].shape
                                    xs_same = [t for t in xs if tuple(t.shape) == tuple(base)]
                                    if xs_same:
                                        attn_for_score = torch.stack(xs_same, dim=0).mean(dim=0)


                        used_img, _ = model._select_vision_tokens(
                            img, atts,
                            attn_bhqk=attn_for_score,
                            keep_ratio=keep,
                            mode=mode,
                            tail_strategy=tail,
                        )

                        with torch.no_grad():
                            u = used_img.norm(dim=-1)  # [B, used_len]
                            used_norm_sum += u.sum().item()
                            used_norm_sq_sum += (u ** 2).sum().item()
                            used_tok_count += u.numel()
                except Exception:
                    pass


                processed += image.size(0)
                if processed >= subset:
                    break

        try:
            model._toggle_ffn_act_stats(False)
        except Exception:
            pass

        #  mean/std
        try:
            stats = model._collect_ffn_act_stats()
        except Exception as e:
            logging.warning(f"[Diag-Act] collect stats failed: {e}")
            self._diag_act_done = True
            return

        lines = []
        for st in stats:
            if st is None:
                lines.append("None")
            else:
                lines.append(
                    f"L{st['layer']}: mean={st['mean']:.4f}, std={st['std']:.4f}, n={st['count']}"
                )

        logging.info(
            f"[Diag-Act] split={split_name} subset={processed}, layer-wise FFN output stats:\n  "
            + "\n  ".join(lines)
        )
        
        if vision_tok_count > 0:
            v_mean = vision_norm_sum / max(vision_tok_count, 1)
            v_var = vision_norm_sq_sum / max(vision_tok_count, 1) - v_mean * v_mean
            if v_var < 0.0:
                v_var = 0.0
            v_std = v_var ** 0.5

            logging.info(
                "[Diag-VisionAct] split=%s subset=%d, "
                "token_L2_mean=%.6f, token_L2_std=%.6f, num_tokens=%d",
                split_name,
                processed,
                v_mean,
                v_std,
                vision_tok_count,
            )

            if used_tok_count > 0:
                u_mean = used_norm_sum / float(used_tok_count)
                u_var = used_norm_sq_sum / float(used_tok_count) - u_mean * u_mean
                if u_var < 0.0:
                    u_var = 0.0
                u_std = u_var ** 0.5
                logging.info(
                    "[Diag-VisionAct-Used] split=%s subset=%d, used_token_L2_mean=%.6f, used_token_L2_std=%.6f, used_num_tokens=%d",
                    split_name, processed, u_mean, u_std, used_tok_count
                )


        self._diag_act_done = True





    def _maybe_prune_before_train(self):
        """
        2)  token 
        3)  KV 
        8) 

        """
        import os, logging, random, torch
        from torch.utils.data import Subset

        run_cfg = self.config.run_cfg
        cfg_get = run_cfg.get

        if bool(cfg_get("use_sdpa_attention", False)):
            try:
                self._enable_sdpa()
            except Exception as e:
                logging.warning(f"[SDPA] enable failed in runner_iter: {e}")


        run = getattr(self.config, "run_cfg", {}) or {}
        if getattr(self, "_train_prune_done", False):
            return

        model = self.unwrap_dist_model(self.model)

        try:
            enabled = bool(run.get("diag_vtoken_usage_enabled", False))
            setattr(model, "_diag_vtoken_usage_enabled", enabled)
            if not enabled and hasattr(model, "_vision_token_usage_stats"):
                # 
                model._vision_token_usage_stats = None
        except Exception as e:
            logging.warning(f"[VToken-Usage@train] toggle diag flag failed: {e}")

        # Optional token-usage evidence logging is controlled by run_cfg.
        try:
            evidence = bool(run.get("vision_token_evidence_log", False))
            setattr(model, "_vision_token_debug_log", evidence)
            setattr(model, "_vision_token_evidence_log", evidence)
            logging.info(f"[VToken-Evidence@train] enabled={evidence}")
        except Exception as e:
            logging.warning(f"[VToken-Evidence@train] toggle evidence flag failed: {e}")


        def cfg_get(key, default=None):
            v = run.get(key, default)
            if isinstance(v, str) and v.lower() in ("none", "null"):
                return default
            return v

        want_split = str(cfg_get("prune_split", "train"))
        subset_n   = int(cfg_get("prune_subset_size", 0))
        dataset, collate_fn = None, None
        try:
            if hasattr(self, "dataloaders") and want_split in self.dataloaders:
                base_ds = getattr(self.dataloaders[want_split], "dataset", None)
                dataset = base_ds
                collate_fn = getattr(base_ds, "collater", None)
            elif hasattr(self, "datasets") and want_split in self.datasets:
                dataset = self.datasets[want_split]
                collate_fn = getattr(dataset, "collater", None)
            if dataset is not None and subset_n > 0 and hasattr(dataset, "__len__"):
                L = len(dataset)
                n = min(subset_n, L)
                idx = list(range(L))
                random.Random(int(cfg_get("seed", 42))).shuffle(idx)
                dataset = Subset(dataset, idx[:n])
        except Exception:
            pass
        logging.info(f"[Pruning-before-train] split={want_split}, subset={subset_n if subset_n>0 else 'FULL'}")

        gate_enabled = bool(cfg_get("ffn_gate_enabled", False))
        if gate_enabled and hasattr(model, "attach_qformer_ffn_gates"):
            gsize   = int(cfg_get("ffn_gate_group_size", 16))
            lam     = float(cfg_get("ffn_gate_lambda", 1e-4))
            pen_p   = float(cfg_get("ffn_gate_penalty_p", 1.0))
            warmup  = float(cfg_get("ffn_gate_warmup", cfg_get("ffn_gate_warmup_steps", 0.0)))
            steps   = int(cfg_get("ffn_gate_steps", 0))          # 
            lr_gate = float(cfg_get("ffn_gate_lr", 1e-4))        # 
            try:
                model.attach_qformer_ffn_gates(group_size=gsize)
                logging.info(f"[FFN-Gate@train] attached: group_size={gsize}, steps={steps}, lr={lr_gate}, lambda={lam}, p={pen_p}, warmup={warmup}")
            except Exception as e:
                logging.warning(f"[FFN-Gate@train] attach failed: {e}")

            if hasattr(model, "set_ffn_gate_loss_cfg"):
                try:
                    model.set_ffn_gate_loss_cfg(lmbda=lam, group_size=gsize, p=pen_p, warmup=warmup)
                    logging.info("[FFN-Gate@train] gate loss cfg set.")
                except Exception as e:
                    logging.warning(f"[FFN-Gate@train] set loss cfg failed: {e}")

            if bool(cfg_get("ffn_finalize_at_train", False)) and hasattr(model, "finalize_qformer_ffn_gates"):
                thr   = cfg_get("ffn_gate_threshold",       None)
                spars = cfg_get("ffn_gate_target_sparsity", None)
                try: thr   = float(thr)   if thr   is not None else None
                except Exception: thr = None
                try: spars = float(spars) if spars is not None else None
                except Exception: spars = None
                try:
                    model.finalize_qformer_ffn_gates(threshold=thr, target_sparsity=spars)
                    logging.info(f"[FFN-Gate@train] finalized: threshold={thr}, target_sparsity={spars}")
                except Exception as e:
                    logging.warning(f"[FFN-Gate@train] finalize failed: {e}")


        if bool(cfg_get("vision_token_prune_enabled", False)) and hasattr(model, "set_runtime_token_pruning"):
            score = str(cfg_get("vision_token_score_mode", "l2norm"))  #  "entropy"/"learned"
            tail  = str(cfg_get("vision_token_tail_strategy", "drop"))  # "drop"/"merge"

            #   vision_token_prune_mode = "ratio"
            #   vision_token_prune_mode = "loss_tolerance"
            mode = str(cfg_get("vision_token_prune_mode", "ratio")).lower()
            if mode in ("loss_tolerance", "loss", "tolerance", "loss-based"):
                self._vtoken_prune_mode = "loss_tolerance"
                start_keep = cfg_get("vision_token_start_ratio", 1.0)
                try:
                    start_keep = float(start_keep)
                except Exception:
                    start_keep = 1.0
                self._vtoken_keep_ratio_cur = float(start_keep)
            else:
                self._vtoken_prune_mode = "ratio"
                base_keep = cfg_get("vision_token_keep_ratio", 1.0)
                try:
                    base_keep = float(base_keep)
                except Exception:
                    base_keep = 1.0
                self._vtoken_keep_ratio_cur = float(base_keep)

            self._vtoken_dynamic_scores = []
            self._vtoken_best_avg_score = None

            use_head_cka = bool(cfg_get("use_vision_head_cka", True))
            setattr(model, "_use_vision_head_cka", use_head_cka)

            two_pass = bool(cfg_get("vision_token_lagged_two_pass", True))
            setattr(model, "_vision_token_lagged_two_pass", two_pass)
            logging.info(f"[Vision-Token@train] lagged_two_pass={two_pass}")

            try:
                beta = float(cfg_get("vision_head_cka_momentum", 0.9))
            except Exception:
                beta = 0.9
            setattr(model, "_vision_head_cka_momentum", beta)
            logging.info(f"[Vision-Token@train] use_head_cka={use_head_cka}, head_cka_momentum={beta}")

            # ===  teacher-based head CKA  recovery  teacher ===
            tpath = cfg_get("kd_teacher_pretrained", None)
            setattr(model, "_kd_teacher_pretrained", tpath)

            try:
                n_eval = int(cfg_get("vision_head_cka_eval_batches", 2))
            except Exception:
                n_eval = 2
            setattr(model, "_vision_head_cka_eval_batches", n_eval)
            setattr(model, "_vision_head_cka_update_on_eval", True)

            logging.info(f"[Vision-HeadCKA@train] teacher_ckpt={tpath} eval_batches={n_eval}")


            try:
                if str(score).lower() in ("learned", "mlp", "score_net") and hasattr(model, "enable_learned_vision_token_scorer"):
                    try:
                        if getattr(model, "_vision_token_scorer", None) is None:
                            init_zero = bool(cfg_get("vision_token_learned_init_zero", True))
                            model.enable_learned_vision_token_scorer(init_zero=init_zero)
                            logging.info(f"[Vision-Token@train] learned scorer created (init_zero={init_zero}).")
                        else:
                            logging.info("[Vision-Token@train] learned scorer already exists; skip re-init.")
                    except Exception as e:
                        logging.warning(f"[Vision-Token@train] learned scorer failed: {e}")

                extra = {}
                if str(score).lower() in ("learned", "mlp", "score_net"):
                    tmode = cfg_get("vision_token_learned_teacher_mode", cfg_get("vision_token_learned_teacher", None))
                    lam   = cfg_get("vision_token_learned_lambda", None)
                    temp  = cfg_get("vision_token_learned_temp", None)
                    if tmode is None:
                        tmode = "entropy"
                    if lam is None:
                        lam = 0.1
                    extra = dict(
                        learned_teacher_mode=str(tmode),
                        learned_lambda=float(lam),
                        learned_temp=(None if temp is None else float(temp)),
                    )

                model.set_runtime_token_pruning(
                    keep_ratio=float(self._vtoken_keep_ratio_cur),
                    score_mode=str(score),
                    tail_strategy=str(tail),
                    **extra,
                )

                logging.info(
                    f"[Token-Prune@train] enabled: mode={self._vtoken_prune_mode}, "
                    f"keep={self._vtoken_keep_ratio_cur}, score={score}, tail={tail}"
                )
            except Exception as e:
                logging.warning(f"[Token-Prune@train] set failed: {e}")



        # ========== 3)  KV  ==========
        if bool(cfg_get("text_kv_prune_enabled", False)) and hasattr(model, "set_text_kv_pruning"):
            kv_mode        = cfg_get("text_kv_mode", "pool")  # "window"/"pool"
            kv_window      = int(cfg_get("text_kv_window_size", 256))
            kv_keep_ratio  = cfg_get("text_kv_keep_ratio", None)
            kv_keep_ratio  = float(kv_keep_ratio) if kv_keep_ratio is not None else None
            kv_pool_bins   = int(cfg_get("text_kv_pool_bins", 64))
            kv_keep_recent = int(cfg_get("text_kv_keep_recent", 64))
            try:
                model.set_text_kv_pruning(
                    mode=kv_mode,
                    window_size=kv_window,
                    keep_ratio=kv_keep_ratio,
                    pool_bins=kv_pool_bins,
                    keep_recent=kv_keep_recent,
                )
                logging.info(f"[Text-KV@train] enabled: mode={kv_mode}, window={kv_window}, keep_ratio={kv_keep_ratio}, "
                            f"pool_bins={kv_pool_bins}, keep_recent={kv_keep_recent}")
            except Exception as e:
                logging.warning(f"[Text-KV@train] set failed: {e}")

        if bool(cfg_get("head_merge_enabled", False)) and hasattr(model, "merge_similar_qformer_heads") and (dataset is not None):
            try:
                hb     = int(cfg_get("head_merge_batches", 8))
                bsz    = int(cfg_get("head_merge_batch_size", 4))
                thr    = float(cfg_get("head_merge_threshold", 0.97))
                method = str(cfg_get("head_merge_method", "cosine"))  # "cosine"/"cka"
                mmax   = int(cfg_get("head_merge_max_per_layer", 1))
                dft_text = cfg_get("head_merge_default_text",
                                cfg_get("importance_default_text", getattr(model, "prompt", "a photo of")))
                logging.info(f"[Head-Merge@train] batches={hb}, bs={bsz}, thr={thr}, method={method}, max/layer={mmax}, text='{dft_text}'")
                model.merge_similar_qformer_heads(
                    dataset=dataset, device=self.device,
                    num_batches=hb, batch_size=bsz,
                    sim_threshold=thr, method=method, max_merge_per_layer=mmax,
                    collate_fn=collate_fn, default_text=dft_text
                )
            except Exception as e:
                logging.warning(f"[Head-Merge@train] skipped: {e}")

        p = cfg_get("head_dropout_prob", cfg_get("headdrop_prob", 0.0))
        try:
            p = float(p)
        except Exception:
            p = 0.0
        if p > 0 and hasattr(model, "enable_random_head_dropout"):
            try:
                model.enable_random_head_dropout(drop_prob=p)
                logging.info(f"[HeadDrop@train] enabled: p={p}")
            except Exception as e:
                logging.warning(f"[HeadDrop@train] failed: {e}")


        head_prune_when = str(cfg_get("head_prune_when", "before_eval")).lower()
        do_hard_prune_now = (head_prune_when != "never") and (head_prune_now := head_prune_when == "before_train")
        head_enabled_cfg = bool(cfg_get("enable_qformer_pruning", False))
        env_flag = os.getenv("ENABLE_QFORMER_PRUNING")
        if env_flag is not None:
            head_enabled_cfg = env_flag.lower() not in ("0", "false", "no")

        if do_hard_prune_now and head_enabled_cfg and hasattr(model, "compute_and_prune_qformer_cross_attn") and (dataset is not None):
            try:
                num_batches     = int(cfg_get("importance_batches", 50))
                bs_importance   = int(cfg_get("importance_batch_size", 4))
                heads_per_layer = int(cfg_get("heads_per_layer_to_prune", 2))
                default_text    = cfg_get("importance_default_text", "a photo of")
                score_mode      = str(cfg_get("score_mode", "taylor"))  # "taylor"/"ablation"/"grad*act"
                k_folds         = int(cfg_get("k_folds", 5))
                save_ckpt       = cfg_get("save_pruned_ckpt", None)
                logging.info(f"[Head-Prune@train] batches={num_batches}, bs={bs_importance}, "
                            f"heads/layer={heads_per_layer}, score={score_mode}, k_folds={k_folds}")
                model.compute_and_prune_qformer_cross_attn(
                    dataset=dataset, device=self.device,
                    num_batches=num_batches, batch_size=bs_importance,
                    heads_per_layer=heads_per_layer, default_text=default_text,
                    num_workers=0, collate_fn=collate_fn,
                    score_mode=score_mode, k_folds=k_folds,
                    save_pruned_ckpt=save_ckpt
                )
            except Exception as e:
                logging.warning(f"[Head-Prune@train] skipped: {e}")
            finally:
                try:
                    it = getattr(model, "_iter_cross_attn_modules", None)
                    if callable(it):
                        for _, _, sa in it():
                            if hasattr(sa, "save_attention"):
                                sa.save_attention = False
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            logging.info("[Head-Prune@train] deferred (head_prune_when != 'before_train' or disabled).")

        # ========== 7)  ==========
        if bool(cfg_get("lowrank_enabled", False)) and hasattr(model, "apply_lowrank_to_qformer"):
            try:
                r = int(cfg_get("lowrank_rank", 64))
                scope = cfg_get("lowrank_scope", ["qkv", "ffn"])
                if isinstance(scope, str): scope = [scope]
                model.apply_lowrank_to_qformer(rank=r, scope=scope)
                logging.info(f"[LowRank@train] applied: rank={r}, scope={scope}")
            except Exception as e:
                logging.warning(f"[LowRank@train] skipped: {e}")

        # ========== 8)  ==========
        try:
            if hasattr(model, "log_pruning_summary"):
                model.log_pruning_summary(stage="train")
        except Exception as e:
            logging.warning(f"[Summary] log_pruning_summary failed: {e}")

        # 
        setattr(self, "_train_prune_done", True)


    def _maybe_prune_before_eval(self):
        """
        2)  KV 
        8) / FLOPs 
        """
        from torch.utils.data import Subset

        run = getattr(self.config, "run_cfg", {}) or {}

        try:
            enabled = bool(run.get("diag_vtoken_usage_enabled", False))
            model_flag = self.unwrap_dist_model(self.model)
            setattr(model_flag, "_diag_vtoken_usage_enabled", enabled)

            if enabled:
                if hasattr(model_flag, "_vision_token_usage_stats"):
                    model_flag._vision_token_usage_stats = None
            else:
                if hasattr(model_flag, "_vision_token_usage_stats"):
                    model_flag._vision_token_usage_stats = None

        except Exception as e:
            logging.warning(f"[VToken-Usage@eval] toggle diag flag failed: {e}")

        try:
            evidence = bool(run.get("vision_token_evidence_log", False))
            model_flag = self.unwrap_dist_model(self.model)
            setattr(model_flag, "_vision_token_debug_log", evidence)
            setattr(model_flag, "_vision_token_evidence_log", evidence)
            logging.info(f"[VToken-Evidence@eval] enabled={evidence}")
        except Exception as e:
            logging.warning(f"[VToken-Evidence@eval] toggle evidence flag failed: {e}")


        #  eval 
        if getattr(self, "_eval_prune_done", False):
            return

        def cfg_get(key, default=None):
            v = run.get(key, default)
            if isinstance(v, str) and v.lower() in ("none", "null"):
                return default
            return v

        if not bool(run.get("prune_before_eval", False)):
            model = self.unwrap_dist_model(self.model)

            if bool(cfg_get("text_kv_prune_enabled", False)) and hasattr(model, "set_text_kv_pruning"):
                kv_mode        = cfg_get("text_kv_mode", "pool")     # "window" / "pool"
                kv_window      = int(cfg_get("text_kv_window_size", 256))
                kv_keep_ratio  = cfg_get("text_kv_keep_ratio", None)
                kv_keep_ratio  = float(kv_keep_ratio) if kv_keep_ratio is not None else None
                kv_pool_bins   = int(cfg_get("text_kv_pool_bins", 64))
                kv_keep_recent = int(cfg_get("text_kv_keep_recent", 64))
                try:
                    model.set_text_kv_pruning(
                        mode=kv_mode, window_size=kv_window, keep_ratio=kv_keep_ratio,
                        pool_bins=kv_pool_bins, keep_recent=kv_keep_recent
                    )
                    logging.info(
                        f"[Text-KV@eval-runtime] enabled: mode={kv_mode}, window={kv_window}, "
                        f"keep_ratio={kv_keep_ratio}, pool_bins={kv_pool_bins}, keep_recent={kv_keep_recent}"
                    )
                except Exception as e:
                    logging.warning(f"[Text-KV@eval-runtime] set failed: {e}")

            if bool(cfg_get("vision_token_prune_enabled", False)) and hasattr(model, "set_runtime_token_pruning"):
                mode = str(cfg_get("vision_token_prune_mode", "ratio")).lower()

                cur_keep = getattr(self, "_vtoken_keep_ratio_cur", None)
                if cur_keep is None:
                    if mode in ("loss_tolerance", "loss", "tolerance", "loss-based"):
                        # loss_tolerance  ->  vision_token_start_ratio
                        cur_keep = cfg_get("vision_token_start_ratio", 1.0)
                    else:
                        # ratio /  ->  vision_token_keep_ratio
                        cur_keep = cfg_get("vision_token_keep_ratio", 1.0)

                try:
                    keep = float(cur_keep)
                except Exception:
                    keep = 1.0

                score = str(cfg_get("vision_token_score_mode", "entropy"))
                tail  = str(cfg_get("vision_token_tail_strategy", "merge"))
                try:
                    extra = {}
                    if str(score).lower() in ("learned", "mlp", "score_net"):
                        tmode = cfg_get("vision_token_learned_teacher_mode", cfg_get("vision_token_learned_teacher", None))
                        lam   = cfg_get("vision_token_learned_lambda", None)
                        temp  = cfg_get("vision_token_learned_temp", None)
                        if tmode is None:
                            tmode = "entropy"
                        if lam is None:
                            lam = 0.1
                        extra = dict(
                            learned_teacher_mode=str(tmode),
                            learned_lambda=float(lam),
                            learned_temp=(None if temp is None else float(temp)),
                        )

                        if hasattr(model, "enable_learned_vision_token_scorer"):
                            try:
                                if getattr(model, "_vision_token_scorer", None) is None:
                                    init_zero = bool(cfg_get("vision_token_learned_init_zero", True))
                                    model.enable_learned_vision_token_scorer(init_zero=init_zero)
                                    logging.info(f"[Vision-Token@eval-runtime] learned scorer created (init_zero={init_zero}).")
                                else:
                                    logging.info("[Vision-Token@eval-runtime] learned scorer already exists; skip re-init.")
                            except Exception as e:
                                logging.warning(f"[Vision-Token@eval-runtime] learned scorer failed: {e}")

                    model.set_runtime_token_pruning(
                        keep_ratio=keep, score_mode=score, tail_strategy=tail, **extra
                    )

                    logging.info(
                        "[Vision-Token@eval-runtime] enabled: keep=%.4f, score=%s, tail=%s",
                        keep, score, tail
                    )
                except Exception as e:
                    logging.warning(f"[Vision-Token@eval-runtime] set failed: {e}")




        if bool(run.get("prune_before_eval", False)):
            model = self.unwrap_dist_model(self.model)

            mids_before = None

            split = str(cfg_get("prune_split", "val"))
            subset_n = int(cfg_get("prune_subset_size", 0))
            ds_for_prune, collate_fn_for_prune = None, None
            try:
                if hasattr(self, "dataloaders") and split in self.dataloaders:
                    base_ds = getattr(self.dataloaders[split], "dataset", None)
                    ds_for_prune = base_ds
                    collate_fn_for_prune = getattr(base_ds, "collater", None)
                elif hasattr(self, "datasets") and split in self.datasets:
                    base_ds = self.datasets[split]
                    ds_for_prune = base_ds
                    collate_fn_for_prune = getattr(base_ds, "collater", None)
                if subset_n > 0 and hasattr(ds_for_prune, "__len__"):
                    L = len(ds_for_prune)
                    n = min(subset_n, L)
                    idx = list(range(L))
                    random.Random(int(cfg_get("seed", 42))).shuffle(idx)
                    ds_for_prune = Subset(ds_for_prune, idx[:n])
            except Exception:
                pass
            logging.info(f"[Pruning-before-eval] split={split}, subset={subset_n if subset_n>0 else 'FULL'}")

            finalized_by_mid = False

            gate_enabled = bool(cfg_get("ffn_gate_enabled", False))
            if gate_enabled and hasattr(model, "attach_qformer_ffn_gates"):
                gsize = int(cfg_get("ffn_gate_group_size", 16))
                try:
                    model.attach_qformer_ffn_gates(group_size=gsize)
                    logging.info(f"[FFN-Gate@eval] attached: group_size={gsize}")
                    mids_before = [d for _, _, _, _, _, d in model._iter_qformer_ffn_modules()]
                except Exception as e:
                    logging.warning(f"[FFN-Gate@eval] attach failed: {e}")

                global_keep = cfg_get("ffn_global_keep_ratio", None)
                global_prune = cfg_get("ffn_global_prune_ratio", None)
                rest_keep_cfg = cfg_get("ffn_rest_keep_ratio", None)
                rest_prune_cfg = cfg_get("ffn_rest_prune_ratio", None)

                keep_ratio = None
                if global_keep is not None:
                    try:
                        keep_ratio = float(global_keep)
                    except Exception:
                        logging.warning(f"[FFN-GlobalPrune] invalid ffn_global_keep_ratio={global_keep}")
                elif global_prune is not None:
                    try:
                        keep_ratio = 1.0 - float(global_prune)
                    except Exception:
                        logging.warning(f"[FFN-GlobalPrune] invalid ffn_global_prune_ratio={global_prune}")
                else:
                    #  rest_*  keep_ratio
                    if rest_keep_cfg is not None:
                        try:
                            keep_ratio = float(rest_keep_cfg)
                        except Exception:
                            logging.warning(f"[FFN-GlobalPrune] invalid ffn_rest_keep_ratio={rest_keep_cfg}")
                    elif rest_prune_cfg is not None:
                        try:
                            keep_ratio = 1.0 - float(rest_prune_cfg)
                        except Exception:
                            logging.warning(f"[FFN-GlobalPrune] invalid ffn_rest_prune_ratio={rest_prune_cfg}")

                if keep_ratio is not None and hasattr(model, "global_prune_qformer_ffn_by_ratio"):
                    try:
                        kr = max(min(float(keep_ratio), 1.0), 0.0)
                        if kr <= 0.0:
                            logging.warning(f"[FFN-GlobalPrune] keep_ratio={kr} <= 0, skip global ratio prune.")
                        else:
                            min_keep_groups = int(cfg_get("ffn_global_min_keep_groups", 1))
                            shallow_layers = cfg_get("ffn_shallow_layers", None)
                            shallow_min_ratio = cfg_get("ffn_shallow_min_ratio", None)

                            # rest_*  cfg 
                            rest_keep = rest_keep_cfg
                            rest_prune = rest_prune_cfg
                            if rest_keep is None and rest_prune is not None:
                                try:
                                    rest_keep = 1.0 - float(rest_prune)
                                except Exception:
                                    rest_keep = None

                            model.global_prune_qformer_ffn_by_ratio(
                                keep_ratio=kr if rest_keep is None else None,
                                group_size=gsize,
                                min_keep_groups_per_layer=min_keep_groups,
                                shallow_layers=shallow_layers,
                                shallow_min_ratio=float(shallow_min_ratio) if shallow_min_ratio is not None else None,
                                rest_keep_ratio=float(rest_keep) if rest_keep is not None else None,
                            )
                            finalized_by_mid = True

                    except Exception as e:
                        logging.warning(f"[FFN-GlobalPrune] failed, fallback to other FFN prune mode: {e}")


                if keep_ratio is not None and hasattr(model, "global_prune_qformer_ffn_by_ratio"):
                    try:
                        kr = max(min(float(keep_ratio), 1.0), 0.0)
                        if kr <= 0.0:
                            logging.warning(f"[FFN-GlobalPrune] keep_ratio={kr} <= 0, skip global ratio prune.")
                        else:
                            min_keep_groups = int(cfg_get("ffn_global_min_keep_groups", 1))
                            shallow_layers = cfg_get("ffn_shallow_layers", None)
                            shallow_min_ratio = cfg_get("ffn_shallow_min_ratio", None)
                            rest_keep = rest_keep_cfg
                            rest_prune = rest_prune_cfg
                            if rest_keep is None and rest_prune is not None:
                                try:
                                    rest_keep = 1.0 - float(rest_prune)
                                except Exception:
                                    rest_keep = None

                            model.global_prune_qformer_ffn_by_ratio(
                                keep_ratio=kr if rest_keep is None else None,
                                group_size=gsize,
                                min_keep_groups_per_layer=min_keep_groups,
                                shallow_layers=shallow_layers,
                                shallow_min_ratio=float(shallow_min_ratio) if shallow_min_ratio is not None else None,
                                rest_keep_ratio=float(rest_keep) if rest_keep is not None else None,
                            )
                            finalized_by_mid = True

                            logging.info(
                                f"[FFN-GlobalPrune] done. keep_ratio={kr:.4f}, "
                                f"min_keep_groups_per_layer={min_keep_groups}"
                            )
                    except Exception as e:
                        logging.warning(f"[FFN-GlobalPrune] failed, fallback to other FFN prune mode: {e}")

                if (not finalized_by_mid) and bool(cfg_get("ffn_gate_scan_min_stable", False)) and hasattr(model, "scan_qformer_ffn_min_stable_mids"):
                    try:
                        tol = float(cfg_get("ffn_scan_loss_tolerance", 0.02))
                        nb  = int(cfg_get("ffn_scan_num_batches", 6))
                        bsz = int(cfg_get("ffn_scan_batch_size", 4))
                        bins = cfg_get("ffn_fixed_mid_bins", None)  # /None => 

                        mids_min = model.scan_qformer_ffn_min_stable_mids(
                            dataset=ds_for_prune, device=self.device,
                            group_size=gsize, num_batches=nb, batch_size=bsz,
                            loss_tol=tol, collate_fn=collate_fn_for_prune
                        )
                        ...


                        orig_mids = [dmid for _, _, _, _, _, dmid in model._iter_qformer_ffn_modules()]

                        def _round_up_group(x, g):
                            x = int(x); g = int(g)
                            return int(((x + g - 1) // g) * g)

                        use_fixed_bins = isinstance(bins, (list, tuple)) and len(bins) > 0

                        min_keep_ratio = float(cfg_get("ffn_scan_min_keep_ratio", 0.60))  #  60%
                        min_mid_abs    = int(cfg_get("ffn_scan_min_mid_abs",   1024))     #  1024


                        mapped = []
                        if use_fixed_bins:
                            bins = [int(b) for b in bins]
                            for m, d_orig in zip(mids_min, orig_mids):
                                # 
                                m = _round_up_group(m, gsize)
                                # 
                                floor_ratio = _round_up_group(math.ceil(int(d_orig) * min_keep_ratio), gsize)
                                floor_abs   = _round_up_group(min_mid_abs, gsize)
                                m = max(m, min(floor_ratio, int(d_orig)), min(floor_abs, int(d_orig)))
                                valid = [b for b in bins if (b <= int(d_orig)) and (b >= m)]
                                choose = min(valid) if len(valid) > 0 else min(m, int(d_orig))
                                mapped.append(choose)
                            bins_mode = "fixed"
                        else:
                            for m, d_orig in zip(mids_min, orig_mids):
                                m = _round_up_group(m, gsize)
                                floor_ratio = _round_up_group(math.ceil(int(d_orig) * min_keep_ratio), gsize)
                                floor_abs   = _round_up_group(min_mid_abs, gsize)
                                m = max(m, min(floor_ratio, int(d_orig)), min(floor_abs, int(d_orig)))
                                m = min(m, int(d_orig))  # 
                                mapped.append(m)
                            bins_mode = "auto"

                        logging.info(f"[FFN-Scan] safety floor applied: ratio={min_keep_ratio}, abs={min_mid_abs}; mapped={mapped}")

                        global_tol = float(cfg_get("ffn_global_loss_tolerance", tol))
                        if (global_tol > 0.0
                            and ds_for_prune is not None
                            and hasattr(model, "adjust_qformer_ffn_mids_global_loss")):
                            try:
                                mapped_global = model.adjust_qformer_ffn_mids_global_loss(
                                    dataset=ds_for_prune,
                                    device=self.device,
                                    init_mids=mapped,
                                    group_size=gsize,
                                    num_batches=nb,
                                    batch_size=bsz,
                                    num_workers=int(cfg_get("ffn_scan_num_workers", 0)),
                                    loss_tol=global_tol,
                                    collate_fn=collate_fn_for_prune,
                                )
                                logging.info(
                                    f"[FFN-Scan][Global] adjusted mids by global loss<= {global_tol:.4f}: {mapped_global}"
                                )
                                mapped = [int(m) for m in mapped_global]
                            except Exception as e:
                                logging.warning(f"[FFN-Scan][Global] adjust failed, fallback to per-layer mapped: {e}")


                        # 4)  mapped 
                        if hasattr(model, "finalize_qformer_ffn_to_mid"):
                            model.finalize_qformer_ffn_to_mid(mapped, group_size=gsize)
                            logging.info(f"[FFN-Scan] finalized mids: {mapped} (group={gsize}, bins={bins_mode})")
                            finalized_by_mid = True

                    except Exception as e:
                        logging.warning(f"[FFN-Scan] failed; fallback to target_mid/sparsity finalize: {e}")


                target_mid = cfg_get("ffn_gate_eval_target_mid", None)
                if (not finalized_by_mid) and target_mid is not None and hasattr(model, "finalize_qformer_ffn_to_mid"):
                    try:
                        model.finalize_qformer_ffn_to_mid(target_mid, group_size=gsize)  #  int  list[int]
                        finalized_by_mid = True
                        logging.info(f"[FFN-Finalize@mid] done: target_mid={target_mid}, group={gsize}")
                    except Exception as e:
                        logging.warning(f"[FFN-Finalize@mid] failed, fallback to sparsity/threshold: {e}")

                if (not finalized_by_mid) and hasattr(model, "finalize_qformer_ffn_gates"):
                    thr     = cfg_get("ffn_gate_threshold",       None)
                    sp      = cfg_get("ffn_gate_target_sparsity", None)
                    eval_thr   = cfg_get("ffn_gate_eval_target_thr",   None)
                    eval_ratio = cfg_get("ffn_gate_eval_target_ratio", None)
                    thr = float(eval_thr) if eval_thr is not None else (float(thr) if thr is not None else None)
                    sp  = float(sp) if sp is not None else None
                    ratio = float(eval_ratio) if eval_ratio is not None else None
                    try:
                        model.finalize_qformer_ffn_gates(threshold=thr, target_sparsity=sp, target_ratio=ratio)
                        logging.info(f"[FFN-Gate@eval] finalized: thr={thr}, sparsity={sp}, ratio={ratio}")
                    except TypeError:
                        model.finalize_qformer_ffn_gates(threshold=thr, target_sparsity=sp)
                        logging.info(f"[FFN-Gate@eval] finalized(legacy): thr={thr}, sparsity={sp}")

            try:
                mids_after = [d for _, _, _, _, _, d in model._iter_qformer_ffn_modules()]
            except Exception:
                mids_after = None

            if (mids_before is not None) and (mids_after is not None):
                diffs = []
                for i, (b, a) in enumerate(zip(mids_before, mids_after)):
                    if int(b) != int(a):
                        diffs.append(f"L{i}:{int(b)}->{int(a)}")
                if len(diffs) == 0:
                    logging.info("[FFN-Finalize] MIDs unchanged.")
                else:
                    logging.info("[FFN-Finalize] MIDs before->after: " + ", ".join(diffs))

            if hasattr(model, "set_ffn_gate_loss_cfg"):
                try:
                    model.set_ffn_gate_loss_cfg(lmbda=0.0)
                    logging.info("[FFN-Gate@eval] gate sparsity loss disabled for evaluation.")
                except TypeError:
                    try:
                        model.set_ffn_gate_loss_cfg(lambda_l1=0.0, lambda_group=0.0)
                        logging.info("[FFN-Gate@eval] gate sparsity loss disabled (legacy keys).")
                    except Exception:
                        pass
            try:
                export_rel = cfg_get("export_slim_path", None)
                if export_rel:
                    from lavis.common.registry import registry
                    out_dir = registry.get_path("output_dir")
                    export_abs = export_rel if os.path.isabs(export_rel) else os.path.join(out_dir, export_rel)
                    os.makedirs(os.path.dirname(export_abs), exist_ok=True)

                    model_no_ddp = self.unwrap_dist_model(self.model)
                    if hasattr(model_no_ddp, "export_slim_checkpoint"):
                        model_no_ddp.export_slim_checkpoint(export_abs)
                        logging.info(f"[Export@eval] slim checkpoint saved to {export_abs}")
            except Exception as e:
                logging.warning(f"[Export@eval] skipped: {e}")



            if bool(cfg_get("text_kv_prune_enabled", False)) and hasattr(model, "set_text_kv_pruning"):
                kv_mode        = cfg_get("text_kv_mode", "pool")     # "window" / "pool"
                kv_window      = int(cfg_get("text_kv_window_size", 256))
                kv_keep_ratio  = cfg_get("text_kv_keep_ratio", None)
                kv_keep_ratio  = float(kv_keep_ratio) if kv_keep_ratio is not None else None
                kv_pool_bins   = int(cfg_get("text_kv_pool_bins", 64))
                kv_keep_recent = int(cfg_get("text_kv_keep_recent", 64))
                try:
                    model.set_text_kv_pruning(
                        mode=kv_mode, window_size=kv_window, keep_ratio=kv_keep_ratio,
                        pool_bins=kv_pool_bins, keep_recent=kv_keep_recent
                    )
                    logging.info(f"[Text-KV] enabled: mode={kv_mode}, window={kv_window}, keep_ratio={kv_keep_ratio}, "
                                f"pool_bins={kv_pool_bins}, keep_recent={kv_keep_recent}")
                except Exception as e:
                    logging.warning(f"[Text-KV] set failed: {e}")

            if bool(cfg_get("vision_token_prune_enabled", False)) and hasattr(model, "set_runtime_token_pruning"):
                keep  = float(cfg_get("vision_token_keep_ratio", 1.0))
                score = str(cfg_get("vision_token_score_mode", "entropy"))
                tail  = str(cfg_get("vision_token_tail_strategy", "merge"))
                try:
                    if score == "learned" and hasattr(model, "enable_learned_vision_token_scorer"):
                        try:
                            if getattr(model, "_vision_token_scorer", None) is None:
                                model.enable_learned_vision_token_scorer()
                                logging.info("[Vision-Token@train] learned scorer enabled.")
                            else:
                                logging.info("[Vision-Token@train] learned scorer already exists; skip enable.")
                        except Exception as e:
                            logging.warning(f"[Vision-Token@train] learned scorer failed: {e}")

                    model.set_runtime_token_pruning(keep_ratio=keep, score_mode=score, tail_strategy=tail)
                    logging.info(f"[Vision-Token] enabled: keep={keep}, score={score}, tail={tail}")
                except Exception as e:
                    logging.warning(f"[Vision-Token] set failed: {e}")

            if bool(cfg_get("head_merge_enabled", False)) and hasattr(model, "merge_similar_qformer_heads") and ds_for_prune is not None:
                try:
                    hb     = int(cfg_get("head_merge_batches", 8))
                    bsz    = int(cfg_get("head_merge_batch_size", 4))
                    thr    = float(cfg_get("head_merge_threshold", 0.97))
                    method = str(cfg_get("head_merge_method", "cosine"))  # "cosine" / "cka"
                    mmax   = int(cfg_get("head_merge_max_per_layer", 1))
                    dft_text = cfg_get("head_merge_default_text", cfg_get("importance_default_text", getattr(model, "prompt", "a photo of")))
                    logging.info(f"[Head-Merge@eval] batches={hb}, bs={bsz}, thr={thr}, method={method}, max/layer={mmax}, text='{dft_text}'")
                    model.merge_similar_qformer_heads(
                        dataset=ds_for_prune, device=self.device,
                        num_batches=hb, batch_size=bsz,
                        sim_threshold=thr, method=method, max_merge_per_layer=mmax,
                        collate_fn=collate_fn_for_prune, default_text=dft_text
                    )
                except Exception as e:
                    logging.warning(f"[Head-Merge@eval] skipped: {e}")

            head_prune_when = str(cfg_get("head_prune_when", "before_eval")).lower()
            head_enabled = bool(cfg_get("enable_qformer_pruning", False))
            env_flag = os.getenv("ENABLE_QFORMER_PRUNING")
            if env_flag is not None:
                head_enabled = env_flag.lower() not in ("0", "false", "no")

            do_soft_mask_now  = head_enabled and (head_prune_when in ("after_kd", "before_eval_soft", "before_eval"))
            do_hard_prune_now = head_enabled and (head_prune_when == "before_eval")

            if hasattr(model, "compute_and_prune_qformer_cross_attn") and ds_for_prune is not None and do_soft_mask_now:
                try:
                    num_batches     = int(cfg_get("importance_batches", 50))
                    bs_importance   = int(cfg_get("importance_batch_size", 4))
                    heads_per_layer = int(cfg_get("heads_per_layer_to_prune", 2))
                    default_text    = cfg_get("importance_default_text", "a photo of")
                    score_mode      = str(cfg_get("score_mode", "taylor"))   # "taylor" / "ablation" / "grad*act"
                    k_folds         = int(cfg_get("k_folds", 5))
                    logging.info(f"[Head-Select@eval] batches={num_batches}, bs={bs_importance}, "
                                f"heads/layer={heads_per_layer}, score={score_mode}, k_folds={k_folds}, "
                                f"hard={do_hard_prune_now}")

                    try:
                        model.compute_and_prune_qformer_cross_attn(
                            dataset=ds_for_prune, device=self.device,
                            num_batches=num_batches, batch_size=bs_importance,
                            heads_per_layer=heads_per_layer, default_text=default_text,
                            num_workers=0, collate_fn=collate_fn_for_prune,
                            score_mode=score_mode, k_folds=k_folds,
                            save_pruned_ckpt=False,
                            hard=bool(do_hard_prune_now)
                        )
                    except TypeError:
                        if not do_hard_prune_now and hasattr(model, "ensure_head_mask_only"):
                            model.ensure_head_mask_only(dataset=ds_for_prune, device=self.device)
                except Exception as e:
                    logging.warning(f"[Head-Select@eval] skipped: {e}")
                finally:
                    try:
                        it = getattr(model, "_iter_cross_attn_modules", None)
                        if callable(it):
                            for _, _, sa in it():
                                if hasattr(sa, "save_attention"):
                                    sa.save_attention = False
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                logging.info("[Head-Select@eval] skipped or deferred (head_prune_when=%s, enabled=%s).",
                            head_prune_when, str(head_enabled))


            # ========== 6)  ==========
            if bool(cfg_get("lowrank_enabled", False)) and hasattr(model, "apply_lowrank_to_qformer"):
                try:
                    r = int(cfg_get("lowrank_rank", 64))
                    scope = cfg_get("lowrank_scope", ["qkv", "ffn"])
                    if isinstance(scope, str):
                        scope = [scope]
                    model.apply_lowrank_to_qformer(rank=r, scope=scope)
                    logging.info(f"[LowRank@eval] applied: rank={r}, scope={scope}")
                except Exception as e:
                    logging.warning(f"[LowRank@eval] skipped: {e}")

            # ========== 7)  ==========
            if bool(cfg_get("verify_after_prune", True)):
                try:
                    default_text = str(cfg_get("importance_default_text", "a photo of"))
                    if hasattr(model, "quick_random_ablate_check") and (ds_for_prune is not None):
                        model.quick_random_ablate_check(
                            dataset=ds_for_prune, device=self.device,
                            max_batches=16, group_size=int(cfg_get("ffn_gate_group_size", 16)),
                            collate_fn=collate_fn_for_prune, default_text=default_text
                        )
                    elif hasattr(self, "_run_prune_verification") and (ds_for_prune is not None):
                        self._run_prune_verification(dataset=ds_for_prune, num_batches=8, batch_size=4, collate_fn=collate_fn_for_prune)
                except Exception as e:
                    logging.warning(f"[Verify] skipped: {e}")

            # ========== 8)  ==========
            try:
                if hasattr(model, "log_pruning_summary"):
                    model.log_pruning_summary(stage="eval")
            except Exception as e:
                logging.warning(f"[Summary] log_pruning_summary failed: {e}")
            try:
                if hasattr(model, "summarize_pruned_structure_and_cost"):
                    model.summarize_pruned_structure_and_cost()
            except Exception as e:
                logging.warning(f"[Summary] summarize_pruned_structure_and_cost failed: {e}")

            try:
                model.freeze_sparsity_targets()
            except Exception:
                pass

            try:
                if hasattr(model, "opt_model") and hasattr(model.opt_model, "generate"):
                    model.eval()
                    with torch.no_grad():
                        dft_text = str(cfg_get("importance_default_text", "a photo of"))
                        samples = [{"image": None, "text_input": dft_text}]  #  minimal collate
                        out = self.task.evaluation(model, samples=samples, use_generate=True)
                        logging.info("[Text-KV] warm generate done; stats=%s", getattr(model, "_text_kv_stats", {}))
            except Exception as e:
                logging.warning(f"[Text-KV] warm generate skipped: {e}")


        try:
            self._maybe_bump_batch_size_after_prune(
                factor=float(cfg_get("bump_batch_size_factor", 1.0))
            )
        except Exception as e:
            logging.warning(f"[Recovery] bump batch size skipped: {e}")

        # ===  ===
        rec_epochs = int(cfg_get("recovery_epochs", 0))
        if rec_epochs > 0:
            try:
                if bool(cfg_get("recovery_kd_enabled", False)):
                    import copy
                    teacher = getattr(self, "_kd_teacher_model", None)
                    if teacher is None:
                        tpath = cfg_get("kd_teacher_pretrained", None)
                        if tpath:
                            import copy
                            base_cfg = copy.deepcopy(self.config)

                            inner_cfg = getattr(base_cfg, "config", base_cfg)

                            model_cfg = None
                            try:
                                # OmegaConf: inner_cfg.model.xxx
                                model_cfg = inner_cfg.model
                            except Exception:
                                try:
                                    # dict: inner_cfg["model"]["xxx"]
                                    model_cfg = inner_cfg["model"]
                                except Exception:
                                    model_cfg = None

                            # 4)  model  ckpt 
                            if model_cfg is not None:
                                try:
                                    #  OmegaConf / dataclass 
                                    model_cfg.pretrained = tpath
                                except Exception:
                                    try:
                                        #  dict 
                                        model_cfg["pretrained"] = tpath
                                    except Exception:
                                        pass

                            # 5)  base_cfg 
                            teacher = self.task.build_model(base_cfg).cpu().eval()
                        else:
                            tfrom = str(cfg_get("kd_teacher_from", "original")).lower()
                            if tfrom in ("original", "orig"):
                                teacher = self.task.build_model(self.config).cpu().eval()


                    if teacher is not None:
                        for p in teacher.parameters():
                            p.requires_grad_(False)
                        self._kd_teacher_model = teacher
                        logging.info("[Recovery-Setup@iter] KD teacher ready on CPU.")
            except Exception as e:
                logging.warning(f"[Recovery-Setup@iter] KD teacher build failed: {e}")

            try:
                self._run_recovery_phase(
                    epochs=rec_epochs,
                    lr_scale=float(cfg_get("recovery_lr_scale", 0.1)),
                    kd_temp=float(cfg_get("kd_temp", 2.0)),
                    kd_lambda_logit=float(cfg_get("kd_lambda_logit", 0.5)),
                    kd_lambda_attn=float(cfg_get("kd_lambda_attn", 0.1)),
                    kd_attn_topk=float(cfg_get("kd_attn_topk", 0.2)),
                    kd_attn_l9_boost=float(cfg_get("kd_attn_l9_boost", 1.0)),
                    kd_attn_l9_var_target=float(cfg_get("kd_attn_l9_var_target", 0.7)),
                    kd_attn_l9_var_lambda=float(cfg_get("kd_attn_l9_var_lambda", 0.0)),
                    kd_attn_l8_boost=float(cfg_get("kd_attn_l8_boost", 1.0)),
                    kd_attn_l10_boost=float(cfg_get("kd_attn_l10_boost", 1.0)),
                    penalty_latency=float(cfg_get("penalty_latency_lambda", 0.0)),
                    penalty_mem=float(cfg_get("penalty_memory_lambda", 0.0)),
                )

            except Exception as e:
                logging.warning(f"[Recovery@eval] skipped due to error: {e}")



        # 
        setattr(self, "_eval_prune_done", True)

    def _maybe_diag_attn_similarity(self, split_name: str):
        """
        """
        try:
            if not bool(self.config.run_cfg.get("diag_attn_similarity", False)):
                return
            if getattr(self, "_diag_attn_done", False):
                return

            model = self.unwrap_dist_model(self.model)
            run = getattr(self.config, "run_cfg", {}) or {}

            # === 1.  KD teacher ===
            teacher = getattr(model, "_kd_teacher", None)

            if teacher is None:
                teacher = getattr(self, "_kd_teacher_model", None)

                if teacher is None:
                    try:
                        import copy

                        def cfg_get(k, d=None):
                            return run.get(k, d)

                        tpath = cfg_get("kd_teacher_pretrained", None)
                        tfrom = str(cfg_get("kd_teacher_from", "original")).lower()
                        teacher_cfg = None

                        if tpath:
                            base_cfg = copy.deepcopy(self.config)
                            inner_cfg = getattr(base_cfg, "config", base_cfg)

                            model_cfg = None
                            try:
                                model_cfg = inner_cfg.model
                            except Exception:
                                try:
                                    model_cfg = inner_cfg["model"]
                                except Exception:
                                    model_cfg = None

                            #  model  ckpt 
                            if model_cfg is not None:
                                try:
                                    model_cfg.pretrained = tpath
                                except Exception:
                                    try:
                                        model_cfg["pretrained"] = tpath
                                    except Exception:
                                        pass

                            teacher_cfg = base_cfg
                        elif tfrom in ("original", "orig"):
                            teacher_cfg = copy.deepcopy(self.config)

                        if teacher_cfg is not None:
                            teacher = self.task.build_model(teacher_cfg).cpu().eval()
                            for p in teacher.parameters():
                                p.requires_grad_(False)
                            self._kd_teacher_model = teacher
                            logging.info("[Diag-AttnKL] KD teacher built on CPU for diag.")
                    except Exception as e:
                        logging.warning(f"[Diag-AttnKL] build KD teacher failed: {e}")
                        teacher = None

                if teacher is not None and hasattr(model, "set_kd_teacher"):
                    try:
                        #  device  orig_dev / device 
                        model.set_kd_teacher(
                            teacher,
                            temp=float(run.get("kd_temp", 2.0)),
                            lambda_logit=0.0,   #  logit KD loss
                            lambda_attn=0.0,
                            attn_topk=float(run.get("kd_attn_topk", 0.3)),
                        )
                    except Exception as e:
                        logging.warning(f"[Diag-AttnKL] set_kd_teacher failed: {e}")
                    # set_kd_teacher  teacher  model._kd_teacher 
                    teacher = getattr(model, "_kd_teacher", None)


            if teacher is None:
                logging.warning("[Diag-AttnKL] no KD teacher available; skip.")
                setattr(self, "_diag_attn_done", True)
                return

            data_loader = self.dataloaders.get(split_name, None)
            if data_loader is None:
                logging.warning("[Diag-AttnKL] data_loader is None; skip.")
                setattr(self, "_diag_attn_done", True)
                return

            subset = int(self.config.run_cfg.get("diag_attn_subset", 128))
            topk = float(self.config.run_cfg.get(
                "diag_attn_topk",
                self.config.run_cfg.get("kd_attn_topk", 0.3),
            ))

            #  acc_overlap 
            processed, n_batches, acc_kl, acc_cos, acc_overlap = 0, 0, 0.0, 0.0, 0.0
            n_batches_p, accp_kl, accp_cos, accp_overlap = 0, 0.0, 0.0, 0.0
            accp_t_keep_mass, accp_s_tail_mass = 0.0, 0.0
            device = self.device

            orig_dev = next(iter(teacher.parameters())).device
            move_back = False
            if orig_dev != device:
                try:
                    teacher.to(device)
                    move_back = True
                except Exception:
                    logging.warning(f"[Diag-AttnKL] move teacher to {device} failed; skip.")
                    setattr(self, "_diag_attn_done", True)
                    return

            model.eval()
            try:
                model._toggle_cross_attn_save(True)
                if hasattr(teacher, "_toggle_cross_attn_save"):
                    teacher._toggle_cross_attn_save(True)
            except Exception:
                pass

            with torch.no_grad():
                for _, samples in enumerate(data_loader):
                    image = samples.get("image", None)
                    if image is None:
                        for k in ["images", "image_tensor", "pixel_values"]:
                            if k in samples:
                                image = samples[k]
                                break
                    if image is None:
                        continue
                    image = image.to(device, non_blocking=True)

                    s_img_full = model.ln_vision(model.visual_encoder(image))
                    s_atts_full = torch.ones(s_img_full.size()[:-1], dtype=torch.long, device=image.device)
                    sq_full = model.query_tokens.expand(s_img_full.shape[0], -1, -1)
                    _ = model.Qformer.bert(
                        query_embeds=sq_full,
                        encoder_hidden_states=s_img_full,
                        encoder_attention_mask=s_atts_full,
                        return_dict=True,
                    )
                    stud_maps_full = model._collect_cross_attn_maps()

                    stud_maps_pruned, keep_idx = None, None
                    v_keep_now = float(getattr(model, "_vision_token_keep_ratio", 1.0) or 1.0)
                    score_mode_now = str(getattr(model, "_vision_token_score_mode", "l2norm")).lower()
                    tail_strategy_now = str(getattr(model, "_vision_token_tail_strategy", "drop")).lower()

                    if v_keep_now < 1.0 and hasattr(model, "_select_vision_tokens"):
                        try:
                            mode_for_attn = score_mode_now
                            if score_mode_now in ("learned", "mlp", "score_net"):
                                mode_for_attn = str(getattr(model, "_vision_token_learned_teacher_mode", "entropy")).lower()

                            need_attn = mode_for_attn in (
                                "attn_entropy", "attn", "entropy", "attn_mean",
                                "attn_grad", "grad", "taylor_attn",
                            )

                            attn_for_prune, grad_for_prune = None, None
                            if need_attn and hasattr(model, "_get_cross_attn_for_prune_lagged"):
                                fwd_id = int(getattr(model, "_vision_token_fwd_id", 0) or 0) + 1
                                setattr(model, "_vision_token_fwd_id", fwd_id)
                                (
                                    attn_for_prune,
                                    grad_for_prune,
                                    _attn_refresh,
                                    _grad_refresh,
                                    _want_grads,
                                    _attn_src,
                                    _did_refresh,
                                    _refresh_reason,
                                ) = model._get_cross_attn_for_prune_lagged(
                                    image_embeds=s_img_full,
                                    image_atts=s_atts_full,
                                    v_keep=v_keep_now,
                                    mode_for_attn=mode_for_attn,
                                    fwd_id=fwd_id,
                                )

                            s_img_pruned, s_atts_pruned = model._select_vision_tokens(
                                s_img_full,
                                s_atts_full,
                                attn_bhqk=attn_for_prune,
                                attn_grad_bhqk=grad_for_prune,
                                keep_ratio=v_keep_now,
                                mode=score_mode_now,
                                tail_strategy=tail_strategy_now,
                            )
                            keep_idx = getattr(model, "_last_vtoken_keep_idx", None)

                            # [FIX] pruned diag: _get_cross_attn_for_prune_lagged() may disable cross-attn saving; re-enable & clear cache
                            try:
                                if hasattr(model, "_toggle_cross_attn_save"):
                                    model._toggle_cross_attn_save(True)
                                if hasattr(model, "_clear_saved_attn_cache"):
                                    model._clear_saved_attn_cache()
                            except Exception:
                                pass

                            sq_p = model.query_tokens.expand(s_img_pruned.shape[0], -1, -1)
                            _ = model.Qformer.bert(
                                query_embeds=sq_p,
                                encoder_hidden_states=s_img_pruned,
                                encoder_attention_mask=s_atts_pruned,
                                return_dict=True,
                            )
                            stud_maps_pruned = model._collect_cross_attn_maps()
                        except Exception:
                            stud_maps_pruned, keep_idx = None, None

                    stud_maps = stud_maps_full

                    #  cross-attn maps
                    t_img = teacher.ln_vision(teacher.visual_encoder(image))
                    t_atts = torch.ones(t_img.size()[:-1], dtype=torch.long, device=image.device)
                    tq = teacher.query_tokens.expand(t_img.shape[0], -1, -1)
                    _ = teacher.Qformer.bert(
                        query_embeds=tq,
                        encoder_hidden_states=t_img,
                        encoder_attention_mask=t_atts,
                        return_dict=True,
                    )
                    teach_maps = teacher._collect_cross_attn_maps()

                    kd_attn_all, cos_all, overlap_all, n_used = 0.0, 0.0, 0.0, 0
                    for S, Tm in zip(stud_maps, teach_maps):
                        if S is None or Tm is None:
                            continue

                        S = (S + 1e-12) / (S + 1e-12).sum(dim=-1, keepdim=True)
                        Tm = (Tm + 1e-12) / (Tm + 1e-12).sum(dim=-1, keepdim=True)

                        #  top-k 
                        if 0.0 < topk < 1.0:
                            k = max(1, int(round(Tm.shape[-1] * topk)))
                            topv, topi = torch.topk(Tm, k=k, dim=-1)
                            weight = torch.zeros_like(Tm)
                            weight.scatter_(-1, topi, topv)
                        else:
                            weight = Tm

                        # ===== 1) KL(student || teacher) =====
                        kl = torch.sum(
                            S * (torch.log(S + 1e-12) - torch.log(Tm + 1e-12)),
                            dim=-1,
                        )  # [B,H,Q]
                        w = weight.sum(dim=-1).clamp_min(1e-6)  # [B,H,Q]
                        kd_attn_all = kd_attn_all + (kl * w).sum() / w.sum()

                        # ===== 2)  =====
                        S_w = S * weight
                        T_w = Tm * weight
                        B = S_w.shape[0]
                        S_vec = S_w.reshape(B, -1)
                        T_vec = T_w.reshape(B, -1)
                        num = (S_vec * T_vec).sum(dim=-1)
                        denom = (S_vec.norm(dim=-1) * T_vec.norm(dim=-1)).clamp_min(1e-12)
                        cos_batch = (num / denom).mean()
                        cos_all += float(cos_batch)

                        k_idx = max(1, int(round(Tm.shape[-1] * topk)))
                        _, idx_t = torch.topk(Tm, k=k_idx, dim=-1)
                        _, idx_s = torch.topk(S,  k=k_idx, dim=-1)

                        mask_t = torch.zeros_like(Tm, dtype=torch.bool).scatter_(-1, idx_t, True)
                        mask_s = torch.zeros_like(S,  dtype=torch.bool).scatter_(-1, idx_s, True)

                        inter = (mask_t & mask_s).float().sum(dim=-1)
                        union = (mask_t | mask_s).float().sum(dim=-1).clamp_min(1e-6)
                        jacc = (inter / union).mean()

                        overlap_all += float(jacc)

                        n_used += 1

                    if n_used > 0:
                        acc_kl      += (kd_attn_all   / n_used).item()
                        acc_cos     += (cos_all       / n_used)
                        acc_overlap += (overlap_all   / n_used)
                        n_batches   += 1


                    if stud_maps_pruned is not None and keep_idx is not None and v_keep_now < 1.0:
                        try:
                            if torch.is_tensor(keep_idx) and keep_idx.dim() == 2:
                                keep_idx_use = keep_idx.to(device=device, dtype=torch.long)
                                K_keep = int(keep_idx_use.shape[1])
                                eps = 1e-12

                                kd_p, cos_p, overlap_p = 0.0, 0.0, 0.0
                                t_keep_mass_p, s_tail_mass_p = 0.0, 0.0
                                n_used_p = 0

                                for Sp, Tm in zip(stud_maps_pruned, teach_maps):
                                    if Sp is None or Tm is None:
                                        continue

                                    # token 
                                    Sp = (Sp + eps) / (Sp + eps).sum(dim=-1, keepdim=True)
                                    Tm = (Tm + eps) / (Tm + eps).sum(dim=-1, keepdim=True)

                                    # 
                                    if keep_idx_use.numel() == 0:
                                        continue
                                    if int(keep_idx_use.max().item()) >= int(Tm.shape[-1]):
                                        continue
                                    if int(Sp.shape[-1]) < int(K_keep):
                                        continue

                                    # expand keep_idx  [B,H,Q,K]
                                    B0, H0, Q0 = int(Sp.shape[0]), int(Sp.shape[1]), int(Sp.shape[2])
                                    idx_exp = keep_idx_use.view(B0, 1, 1, K_keep).expand(B0, H0, Q0, K_keep)

                                    # teacher: gather  keep token
                                    T_keep_raw = torch.gather(Tm, dim=-1, index=idx_exp)  # [B,H,Q,K]
                                    keep_mass = T_keep_raw.sum(dim=-1)
                                    t_keep_mass_p += float(keep_mass.mean().item())

                                    S_keep_raw = Sp[..., :K_keep]                         # [B,H,Q,K]
                                    s_keep_mass = S_keep_raw.sum(dim=-1)                  # [B,H,Q]
                                    s_tail_mass = (1.0 - s_keep_mass).clamp(min=0.0, max=1.0)
                                    s_tail_mass_p += float(s_tail_mass.mean().item())

                                    T_keep = T_keep_raw / T_keep_raw.sum(dim=-1, keepdim=True).clamp_min(eps)
                                    S_keep = S_keep_raw / S_keep_raw.sum(dim=-1, keepdim=True).clamp_min(eps)

                                    if 0.0 < topk < 1.0:
                                        kk = max(1, int(round(K_keep * topk)))
                                        topv, topi = torch.topk(T_keep, k=kk, dim=-1)
                                        weight = torch.zeros_like(T_keep)
                                        weight.scatter_(-1, topi, topv)
                                    else:
                                        weight = T_keep

                                    # 1) KL(student||teacher) in keep-space
                                    kl = torch.sum(
                                        S_keep * (torch.log(S_keep + eps) - torch.log(T_keep + eps)),
                                        dim=-1,
                                    )  # [B,H,Q]
                                    w = weight.sum(dim=-1).clamp_min(1e-6)
                                    kd_p = kd_p + (kl * w).sum() / w.sum()

                                    S_w = S_keep * weight
                                    T_w = T_keep * weight
                                    S_vec = S_w.reshape(B0, -1)
                                    T_vec = T_w.reshape(B0, -1)
                                    num = (S_vec * T_vec).sum(dim=-1)
                                    denom = (S_vec.norm(dim=-1) * T_vec.norm(dim=-1)).clamp_min(1e-12)
                                    cos_batch = (num / denom).mean()
                                    cos_p += float(cos_batch)

                                    k_idx = max(1, int(round(K_keep * topk)))
                                    _, idx_t = torch.topk(T_keep, k=k_idx, dim=-1)
                                    _, idx_s = torch.topk(S_keep, k=k_idx, dim=-1)
                                    mask_t = torch.zeros_like(T_keep, dtype=torch.bool).scatter_(-1, idx_t, True)
                                    mask_s = torch.zeros_like(S_keep, dtype=torch.bool).scatter_(-1, idx_s, True)
                                    inter = (mask_t & mask_s).float().sum(dim=-1)
                                    union = (mask_t | mask_s).float().sum(dim=-1).clamp_min(1e-6)
                                    jacc = (inter / union).mean()
                                    overlap_p += float(jacc)

                                    n_used_p += 1

                                if n_used_p > 0:
                                    accp_kl += (kd_p / n_used_p).item()
                                    accp_cos += (cos_p / n_used_p)
                                    accp_overlap += (overlap_p / n_used_p)
                                    accp_t_keep_mass += (t_keep_mass_p / n_used_p)
                                    accp_s_tail_mass += (s_tail_mass_p / n_used_p)
                                    n_batches_p += 1
                        except Exception:
                            pass


                    processed += image.size(0)
                    if processed >= subset:
                        break

            #  & 
            try:
                model._toggle_cross_attn_save(False)
                if hasattr(teacher, "_toggle_cross_attn_save"):
                    teacher._toggle_cross_attn_save(False)
            except Exception:
                pass
            if move_back:
                try:
                    teacher.to(orig_dev)
                except Exception:
                    pass

            avg_kl = (acc_kl / max(1, n_batches))
            avg_cos = (acc_cos / max(1, n_batches))
            avg_overlap = acc_overlap / max(1, n_batches)

            logging.info(
                f"[Diag-AttnKL] split={split_name} subset={processed} topk={topk} "
                f"avg_KL={avg_kl:.6f} avg_cos={avg_cos:.6f} "
                f"avg_topk_overlap={avg_overlap:.6f} (~{avg_overlap*100:.2f}% overlap, over {n_batches} batches)"
            )

            if n_batches_p == 0:
                try:
                    v_keep_cfg = float(getattr(model, "_vision_token_keep_ratio", 1.0) or 1.0)
                    tail_cfg = str(getattr(model, "_vision_token_tail_strategy", "drop"))
                    logging.info(f"[Diag-AttnKL-Pruned] skipped (no valid pruned maps). keep={v_keep_cfg:.4f} tail={tail_cfg}. "
                                 f"Common causes: pruning disabled (keep=1.0), keep_idx missing, or cross-attn saving was off during pruned forward.")
                except Exception:
                    pass

            if n_batches_p > 0:
                avgp_kl = (accp_kl / max(1, n_batches_p))
                avgp_cos = (accp_cos / max(1, n_batches_p))
                avgp_overlap = (accp_overlap / max(1, n_batches_p))
                avgp_t_keep_mass = (accp_t_keep_mass / max(1, n_batches_p))
                avgp_s_tail_mass = (accp_s_tail_mass / max(1, n_batches_p))

                v_keep_cfg = float(getattr(model, "_vision_token_keep_ratio", 1.0) or 1.0)
                tail_cfg = str(getattr(model, "_vision_token_tail_strategy", "drop"))

                logging.info(
                    f"[Diag-AttnKL-Pruned] split={split_name} subset={processed} keep={v_keep_cfg:.4f} tail={tail_cfg} topk={topk} "
                    f"t_keep_mass={avgp_t_keep_mass:.4f} s_tail_mass={avgp_s_tail_mass:.4f} "
                    f"avg_KL={avgp_kl:.6f} avg_cos={avgp_cos:.6f} "
                    f"avg_topk_overlap={avgp_overlap:.6f} (~{avgp_overlap*100:.2f}% overlap, over {n_batches_p} batches)"
                )

            setattr(self, "_diag_attn_done", True)

        except Exception as e:
            logging.warning(f"[Diag-AttnKL] failed: {e}")
            setattr(self, "_diag_attn_done", True)


    def _maybe_diag_tail_grad(self, split_name: str):
        """

        """
        import torch

        run = getattr(self.config, "run_cfg", {}) or {}

        enabled = bool(run.get("diag_tail_grad_enabled",
                               run.get("diag_tail_merge_grad", False)))
        if not enabled:
            return

        if getattr(self, "_diag_tail_grad_done", False):
            return

        if not is_main_process():
            return

        model = self.unwrap_dist_model(self.model)

        if not hasattr(model, "_iter_cross_attn_modules"):
            logging.warning("[Diag-TailGrad] model has no cross-attn modules; skip.")
            self._diag_tail_grad_done = True
            return

        #  split  dataloader
        dl = self.dataloaders.get(split_name, None)
        if dl is None:
            logging.warning(f"[Diag-TailGrad] no dataloader for split={split_name}; skip.")
            self._diag_tail_grad_done = True
            return

        subset = int(run.get("diag_tail_grad_subset",
                             run.get("diag_attn_subset", 64)))
        default_text = run.get(
            "diag_tail_default_text",
            run.get("importance_default_text", "a photo of"),
        )
        device = self.device

        modules = list(getattr(model, "_iter_cross_attn_modules")())
        if not modules:
            logging.warning("[Diag-TailGrad] no cross-attn modules returned; skip.")
            self._diag_tail_grad_done = True
            return

        stats = {}
        for li, _, sa in modules:
            stats[int(li)] = {
                "sink_sum": 0.0,
                "sink_cnt": 0,
                "non_sum": 0.0,
                "non_cnt": 0,
                "entropy_sum": 0.0,
                "entropy_cnt": 0,
            }

        processed = 0
        model.eval()

        #  cross-attn attention map / grad 
        try:
            if hasattr(model, "_toggle_cross_attn_save"):
                model._toggle_cross_attn_save(True)
        except Exception as e:
            logging.warning(f"[Diag-TailGrad] enable cross-attn save failed: {e}")
            self._diag_tail_grad_done = True
            return

        for _, samples in enumerate(dl):
            if processed >= subset:
                break

            # 
            image = samples.get("image", None)
            if image is None:
                for k in ["images", "image_tensor", "pixel_values"]:
                    if k in samples:
                        image = samples[k]
                        break
            if image is None:
                continue

            image = image.to(device, non_blocking=True)
            bs = image.size(0)

            samples = dict(samples)  #  batch
            #  Tensor  device
            for k, v in list(samples.items()):
                if hasattr(v, "to"):
                    samples[k] = v.to(device, non_blocking=True)

            if "text_input" not in samples:
                samples["text_input"] = [default_text] * bs

            with torch.enable_grad():
                try:
                    setattr(model, "_diag_tail_grad_force_attn_save", True)
                    out = model.forward(samples)
                finally:
                    try:
                        setattr(model, "_diag_tail_grad_force_attn_save", False)
                    except Exception:
                        pass
                loss = out.get("loss", None)
                if loss is None:
                    continue

                if hasattr(model, "Qformer"):
                    model.Qformer.zero_grad(set_to_none=True)
                if hasattr(model, "opt_model"):
                    model.opt_model.zero_grad(set_to_none=True)
                loss.backward()

            #  attn_gradients
            for li, _, sa in modules:
                g = None
                get_grad = getattr(sa, "get_attn_gradients", None)
                if callable(get_grad):
                    g = get_grad()
                else:
                    g = getattr(sa, "attn_gradients", None)
                if g is None:
                    continue
                if not torch.is_tensor(g) or g.dim() != 4:
                    continue

                B, H, Q, N = g.shape
                if N <= 1:
                    continue

                g_abs = g.detach().abs()         # [B,H,Q,N]
                sink = g_abs[..., -1]
                non  = g_abs[..., :-1]           #  token

                st = stats[int(li)]
                st["sink_sum"] += float(sink.sum().item())
                st["sink_cnt"] += sink.numel()
                st["non_sum"]  += float(non.sum().item())
                st["non_cnt"]  += non.numel()

                flat = g_abs.view(-1, N)
                S = flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                p = flat / S
                ent = (-(p * p.clamp_min(1e-12).log()).sum(dim=-1)).mean().item()
                st["entropy_sum"] += float(ent)
                st["entropy_cnt"] += 1

            processed += bs

        # 
        try:
            if hasattr(model, "_toggle_cross_attn_save"):
                model._toggle_cross_attn_save(False)
        except Exception:
            pass

        # 
        lines = []
        for li in sorted(stats.keys()):
            st = stats[li]
            if st["sink_cnt"] == 0 or st["non_cnt"] == 0:
                continue
            sink_mean = st["sink_sum"] / st["sink_cnt"]
            non_mean  = st["non_sum"] / max(1, st["non_cnt"])
            ratio = sink_mean / max(non_mean, 1e-12)
            if st["entropy_cnt"] > 0:
                entropy = st["entropy_sum"] / st["entropy_cnt"]
            else:
                entropy = float("nan")
            lines.append(
                f"L{li}: grad_sink={sink_mean:.4e}, grad_others={non_mean:.4e}, "
                f"ratio={ratio:.2f}, grad_token_entropy={entropy:.3f}"
            )

        if lines:
            logging.info(
                f"[Diag-TailGrad] split={split_name} subset={processed}, "
                "per-layer sink-vs-others grad stats:\n  " + "\n  ".join(lines)
            )
        else:
            logging.info(
                f"[Diag-TailGrad] split={split_name} subset={processed}, no valid gradients collected."
            )

        self._diag_tail_grad_done = True


    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_iters = 0

        try:
            from lavis.common.dist_utils import is_main_process
        except Exception:
            def is_main_process():
                return True

        self.log_config()

        run = getattr(self.config, "run_cfg", {}) or {}
        eval_prune_only = bool(run.get("vision_eval_prune_only", False))

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)

        if not self.evaluate_only:
            self._maybe_prune_before_train()

        for start_iters in range(
            self.start_iters, self.max_iters, self.iters_per_inner_epoch
        ):
            end_iters = start_iters + self.iters_per_inner_epoch

            # training phase
            if not self.evaluate_only:
                logging.info(
                    "Start training, max_iters={}, in total {} inner epochs.".format(
                        self.max_iters, int(self.max_iters / self.iters_per_inner_epoch)
                    )
                )
                if start_iters == self.start_iters:
                    self.task.before_training(
                        model=self.unwrap_dist_model(self.model),
                        dataset=self.datasets,
                    )
                train_stats = self.train_iters(self.cur_epoch, start_iters)
                self.log_stats(split_name="train", stats=train_stats)

            # evaluation phase
            if len(self.valid_splits) > 0 and (self.evaluate_only or (end_iters//self.iters_per_inner_epoch)%self.val_freq == 0):
                for split_name in self.valid_splits:
                    logging.info("Evaluating on {}.".format(split_name))

                    # --- / eval-time  ---
                    if not getattr(self, "_eval_prune_done", False):
                        try:
                            self._maybe_prune_before_eval()  #  prune_before_eval 
                            self._eval_prune_done = True
                            model_no_ddp = self.unwrap_dist_model(self.model)
                            if hasattr(model_no_ddp, "log_pruning_summary"):
                                model_no_ddp.log_pruning_summary(stage="eval")
                        except Exception as e:
                            logging.warning(f"[Eval-Prune@train] skipped: {e}")


                    do_diag = False

                    if getattr(self, "evaluate_only", False):
                        do_diag = True
                    else:
                        try:
                            if end_iters >= self.max_iters:
                                do_diag = True
                        except Exception:
                            do_diag = True

                    if do_diag:
                        # 1) FFN  / 
                        try:
                            self._maybe_diag_act_stats(split_name)
                        except Exception as e:
                            logging.warning(f"[Diag-Act] skipped before eval: {e}")

                        if bool(self.config.run_cfg.get("diag_attn_similarity", False)) and is_main_process():
                            try:
                                self._maybe_diag_attn_similarity(split_name)
                            except Exception as e:
                                logging.warning(f"[Diag-AttnKL] skipped before eval: {e}")

                        # 3)  token tail-merge 
                        if bool(self.config.run_cfg.get("diag_tail_grad_enabled",
                                                        self.config.run_cfg.get("diag_tail_merge_grad", False))) and is_main_process():
                            try:
                                self._maybe_diag_tail_grad(split_name)
                            except Exception as e:
                                logging.warning(f"[Diag-TailGrad] skipped before eval: {e}")

                        try:
                            model_no_ddp = self.unwrap_dist_model(self.model)
                            if hasattr(model_no_ddp, "log_pruning_summary"):
                                model_no_ddp.log_pruning_summary(stage="eval-post-diag")
                        except Exception as e:
                            logging.warning(f"[Vision-Token@usage] post-diag print skipped: {e}")


                    val_log = self.eval_epoch(
                        split_name=split_name, cur_epoch=self._progress(end_iters)
                    )

                    try:
                        model_no_ddp = self.unwrap_dist_model(self.model)
                        if hasattr(model_no_ddp, 'log_pruning_summary'):
                            model_no_ddp.log_pruning_summary(stage='eval-end')
                    except Exception as e:
                        logging.warning(f"[Summary] eval-end print skipped: {e}")

                    if (val_log is not None) and (split_name == "val"):
                        try:
                            self._update_val_loss_window(val_log)
                        except Exception as e:
                            logging.warning(f"[Val-Loss@window] update skipped due to error: {e}")

                                        
                    allow_dynamic = (not self.evaluate_only) or eval_prune_only
                    if (val_log is not None) and allow_dynamic and (split_name == "val"):
                        try:
                            self._maybe_update_vision_prune_after_eval(val_log)
                        except Exception as e:
                            logging.warning(f"[Vision-Token@dynamic] update skipped due to error: {e}")

                    if val_log is not None:
                        if is_main_process():
                            assert (
                                "agg_metrics" in val_log
                            ), "No agg_metrics found in validation log."

                            agg_metrics = val_log["agg_metrics"]
                            if agg_metrics > best_agg_metric and split_name == "val":
                                best_iters, best_agg_metric = end_iters, agg_metrics

                                self._save_checkpoint(end_iters, is_best=True)

                            val_log.update({"best_iters": best_iters})
                            self.log_stats(val_log, split_name)
            '''
            else:
                # if no validation split is provided, we just save the checkpoint at the end of each inner epoch.
                if not self.evaluate_only:
                    self._save_checkpoint(end_iters, is_best=False)
            '''

            if self.evaluate_only and not eval_prune_only:
                break

            # save checkpoint according to save freq
            # if self.save_freq>0 and (end_iters//self.iters_per_inner_epoch)%self.save_freq == 0:
            '''
            self._save_checkpoint(end_iters, is_best=False)
            '''

            self._barrier_if_dist() 

        # save last checkpoint
        if self.save_last and not self.evaluate_only:
            '''
            self._save_checkpoint(end_iters, is_best=False)
            '''
            self._save_checkpoint(end_iters, is_best=False, is_last=True)

        # testing phase
        self.evaluate(cur_epoch=self.cur_epoch)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _apply_vision_keep_ratio(self, keep_ratio: float):
        import logging
        model = self.unwrap_dist_model(self.model)
        run = getattr(self.config, "run_cfg", {}) or {}

        if not hasattr(model, "set_runtime_token_pruning"):
            return

        score = str(run.get("vision_token_score_mode", "l2norm"))
        tail  = str(run.get("vision_token_tail_strategy", "drop"))
        try:
            kr = float(keep_ratio)
        except Exception:
            kr = 1.0

        try:
            extra = {}
            if str(score).lower() in ("learned", "mlp", "score_net"):
                tmode = run.get("vision_token_learned_teacher_mode", run.get("vision_token_learned_teacher", None))
                lam   = run.get("vision_token_learned_lambda", None)
                temp  = run.get("vision_token_learned_temp", None)
                if tmode is None:
                    tmode = "entropy"
                if lam is None:
                    lam = 0.1
                extra = dict(
                    learned_teacher_mode=str(tmode),
                    learned_lambda=float(lam),
                    learned_temp=(None if temp is None else float(temp)),
                )

            if str(score).lower() == "learned" and hasattr(model, "enable_learned_vision_token_scorer"):
                if getattr(model, "_vision_token_scorer", None) is None:
                    try:
                        model.enable_learned_vision_token_scorer()
                    except Exception as e:
                        logging.warning(f"[Vision-Token@dynamic] learned scorer enable failed: {e}")

            if str(score).lower() == "learned" and getattr(model, "_vision_token_scorer", None) is not None:
                if not getattr(model, "_vision_token_scorer_ckpt_loaded", False):
                    try:
                        import os, torch
                        ckpt_path = None
                        ckpt_path = run.get("vision_token_scorer_ckpt_path", None)
                        if ckpt_path is None:
                            ckpt_path = getattr(getattr(self.config, "model_cfg", None), "pretrained", None)

                        if ckpt_path and os.path.isfile(ckpt_path):
                            ckpt = torch.load(ckpt_path, map_location="cpu")
                            sd = ckpt.get("model", ckpt)

                            prefix = "_vision_token_scorer."
                            sub = {k[len(prefix):]: v for k, v in sd.items() if isinstance(k, str) and k.startswith(prefix)}

                            if len(sub) > 0:
                                keys = model._vision_token_scorer.load_state_dict(sub, strict=False)
                                model._vision_token_scorer_ckpt_loaded = True
                                logging.info(f"[Vision-Token@dynamic] loaded learned-scorer from ckpt: {ckpt_path} "
                                            f"(missing={len(keys.missing_keys)}, unexpected={len(keys.unexpected_keys)})")
                    except Exception as e:
                        logging.warning(f"[Vision-Token@dynamic] load learned-scorer from ckpt failed: {e}")


            model.set_runtime_token_pruning(
                keep_ratio=kr,
                score_mode=str(score),
                tail_strategy=str(tail),
                **extra,
            )

            logging.info(
                "[Vision-Token@dynamic] updated keep_ratio=%.4f, score=%s, tail=%s",
                kr,
                score,
                tail,
            )
        except Exception as e:
            logging.warning(f"[Vision-Token@dynamic] set_runtime_token_pruning failed: {e}")

    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _maybe_update_vision_prune_after_eval(self, val_log: dict):
        import logging, math

        run = getattr(self.config, "run_cfg", {}) or {}

        if not bool(run.get("vision_token_prune_enabled", False)):
            return

        mode = str(run.get("vision_token_prune_mode", "ratio")).lower()
        if mode not in ("loss_tolerance", "loss", "tolerance", "loss-based"):
            return

        if not isinstance(val_log, dict):
            return

        if "agg_metrics" in val_log:
            try:
                m = float(val_log["agg_metrics"])
            except Exception:
                return
            score = -m
        elif "loss" in val_log:
            try:
                score = float(val_log["loss"])
            except Exception:
                return
        else:
            return

        # 
        if not hasattr(self, "_vtoken_dynamic_scores") or self._vtoken_dynamic_scores is None:
            self._vtoken_dynamic_scores = []
        hist = self._vtoken_dynamic_scores
        hist.append(score)

        win = int(run.get("vision_token_loss_window", 3))
        if win <= 0 or len(hist) < win:
            return

        window_scores = hist[-win:]
        avg = sum(window_scores) / float(len(window_scores))

        best = getattr(self, "_vtoken_best_avg_score", None)
        if best is None:
            self._vtoken_best_avg_score = avg
            return

        delta = avg - best
        if abs(best) < 1e-8:
            rel = 0.0 if delta <= 0 else float("inf")
        else:
            rel = delta / abs(best)

        tol  = float(run.get("vision_token_loss_tolerance", 0.02))
        step = float(run.get("vision_token_keep_step", 0.05))
        kmin = float(run.get("vision_token_min_keep_ratio", 0.3))
        kmax = float(run.get("vision_token_max_keep_ratio", 1.0))

        cur = getattr(self, "_vtoken_keep_ratio_cur", None)
        if cur is None:
            try:
                cur = float(run.get("vision_token_start_ratio", 1.0))
            except Exception:
                cur = 1.0


        debounce = int(run.get("vision_token_loss_debounce", 1))
        if debounce <= 0:
            debounce = 1

        bad_streak  = int(getattr(self, "_vtoken_bad_streak", 0))
        good_streak = int(getattr(self, "_vtoken_good_streak", 0))

        new = cur

        if rel <= tol:
            if avg < best:
                self._vtoken_best_avg_score = avg

            good_streak += 1
            bad_streak = 0

            if good_streak >= debounce:
                new = max(kmin, cur - step)
                good_streak = 0

        else:
            bad_streak += 1
            good_streak = 0

            if bad_streak >= debounce:
                new = min(kmax, cur + step)
                bad_streak = 0

        self._vtoken_bad_streak = bad_streak
        self._vtoken_good_streak = good_streak

        if abs(new - cur) < 1e-6:
            #  log / 
            return

        self._vtoken_keep_ratio_cur = new
        logging.info(
            "[Vision-Token@dynamic] win_avg=%.6f, best_avg=%.6f, rel_delta=%.4f, tol=%.4f, "
            "keep_ratio: %.4f -> %.4f (debounce=%d, good_streak=%d, bad_streak=%d)",
            avg,
            best,
            rel,
            tol,
            cur,
            new,
            debounce,
            good_streak,
            bad_streak,
        )
        self._apply_vision_keep_ratio(new)


    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    def _update_val_loss_window(self, val_log: dict):
        import logging

        if not isinstance(val_log, dict):
            return

        #  "loss" 
        if "loss" not in val_log:
            return

        try:
            loss = float(val_log["loss"])
        except Exception:
            return

        if not hasattr(self, "_val_loss_history") or self._val_loss_history is None:
            self._val_loss_history = []

        hist = self._val_loss_history
        hist.append(loss)

        run = getattr(self.config, "run_cfg", {}) or {}
        win = int(run.get("val_loss_window", 3))

        if win <= 0:
            # win<=0 
            return

        if len(hist) < win:
            logging.info(
                "[Val-Loss@window] step=%d, loss=%.6f (window not full, win=%d)",
                len(hist),
                loss,
                win,
            )
            return

        window_vals = hist[-win:]
        avg = sum(window_vals) / float(len(window_vals))

        best = getattr(self, "_best_val_loss_window", None)
        if (best is None) or (avg < best):
            best = avg
            self._best_val_loss_window = best

        logging.info(
            "[Val-Loss@window] step=%d, loss=%.6f, window(%d)_avg=%.6f, best_window_avg=%.6f",
            len(hist),
            loss,
            win,
            avg,
            best,
        )


    def train_iters(self, epoch, start_iters):
        # train by iterations
        self.model.train()


        return self.task.train_iters(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_inner_epoch=self.iters_per_inner_epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
        )

    @main_process
    def _save_checkpoint(self, cur_iters, is_best=False, is_last=False):
        import os, logging, torch
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()}

        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                del state_dict[k]

        save_obj = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "iters": cur_iters,
        }

        #  YAML  ckpt 
        run_cfg = getattr(self.config, "run_cfg", {})
        if is_best:
            fname = run_cfg.get("save_best_name", "checkpoint_best.pth")
        else:
            fname = run_cfg.get("save_last_name", "checkpoint_last.pth")

        save_to = os.path.join(self.output_dir, fname)
        logging.info("Saving checkpoint at iters {} to {}.".format(cur_iters, save_to))


        tmp_path = save_to + ".tmp"
        try:
            torch.save(save_obj, tmp_path)
            os.replace(tmp_path, save_to)

            if is_best:
                export_rel = self.config.run_cfg.get("export_slim_path", None)
                if export_rel:
                    from lavis.common.registry import registry
                    out_dir = registry.get_path("output_dir")
                    export_abs = export_rel if os.path.isabs(export_rel) else os.path.join(out_dir, export_rel)
                    os.makedirs(os.path.dirname(export_abs), exist_ok=True)

                    model_no_ddp = self.unwrap_dist_model(self.model)
                    if hasattr(model_no_ddp, "export_slim_checkpoint"):
                        model_no_ddp.export_slim_checkpoint(export_abs)
                        logging.info(f"[Export] slim checkpoint saved to {export_abs}")

        except Exception as e1:
            logging.warning(f"[CKPT] primary save failed: {e1}")
            try:
                if os.path.exists(tmp_path):
                    try: os.remove(tmp_path)
                    except: pass
                torch.save(save_obj, tmp_path, _use_new_zipfile_serialization=False)
                os.replace(tmp_path, save_to)
            except Exception as e2:
                logging.error(f"[CKPT] fallback save failed: {e2}")
                if self.config.run_cfg.get("save_weights_only", False):
                    try:
                        weights_only = {"model": self.unwrap_dist_model(self.model).state_dict()}
                        torch.save(weights_only, tmp_path)
                        os.replace(tmp_path, save_to + ".weights_only.pth")
                        logging.info(f"[CKPT] saved weights-only to {save_to+'.weights_only.pth'}")
                    except Exception as e3:
                        logging.error(f"[CKPT] weights-only save failed: {e3}")
                raise



    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_iters = checkpoint["iters"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            if dataset_ratios is None:
                # concatenate map-style datasets and chain wds.DataPipe datasets separately
                # training set becomes a tuple (ConcatDataset, ChainDataset), both are
                # optional but at least one of them is required. The resultant ConcatDataset
                # and ChainDataset will be sampled evenly.
                logging.info(
                    "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
                )

                datasets = reorg_datasets_by_split(self.datasets)
                self.datasets = concat_datasets(datasets)
            else:
                # create multi-loader with the provided ratios, without concatenating or chaining
                missing_keys = [k for k in dataset_ratios if k not in self.datasets]
                if len(missing_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not found: {}".format(
                            missing_keys
                        )
                    )

                unexpected_keys = [k for k in self.datasets if k not in dataset_ratios]
                if len(unexpected_keys) > 0:
                    raise ValueError(
                        "Datasets with the following split names are not expected: {}".format(
                            unexpected_keys
                        )
                    )

                dataset_ratios = [float(dataset_ratios[k]) for k in self.datasets]
                self.datasets = reorg_datasets_by_split(self.datasets)
                # to keep the same structure as return value of concat_datasets
                self.datasets = {
                    k: v[0] if len(v) == 1 else v for k, v in self.datasets.items()
                }

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [wds.DataPipeline, ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    try:
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    except TypeError:
                        # a single wds.DataPipeline or ChainDataset
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @torch.no_grad()
    def _run_prune_verification(self, dataset, *, num_batches=8, batch_size=4, collate_fn=None):
        """
        """
        from torch.utils.data import DataLoader, Subset

        if dataset is None:
            logging.warning("[Verify] dataset is None; skip.")
            return

        model = self.unwrap_dist_model(self.model)
        model.eval()

        default_text = str(
            self.config.run_cfg.get("importance_default_text",
                self.config.run_cfg.get("head_merge_default_text", "a photo of"))
        )

        #  +  DataLoader
        try:
            n = min(len(dataset), 128)
            idx = list(range(n))
        except Exception:
            idx = list(range(64))
        dl = DataLoader(Subset(dataset, idx), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, drop_last=False)

        base_losses, ablated_losses = [], []
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if self.cuda_enabled else contextlib.nullcontext()

        steps = 0
        for batch in dl:
            if steps >= int(num_batches):
                break

            # to(device)
            for k, v in list(batch.items()):
                if hasattr(v, "to"):
                    batch[k] = v.to(self.device, non_blocking=True)

            if "text_input" not in batch:
                bs = batch["image"].size(0)
                batch["text_input"] = [default_text] * bs

            # baseline
            with amp_ctx:
                out = model.forward(batch)
                base_loss = float(out["loss"])
            base_losses.append(base_loss)

            if hasattr(model, "_iter_qformer_ffn_modules"):
                for li, layer, inter_dense, out_dense, act_fn, dmid in model._iter_qformer_ffn_modules():
                    g = max(1, int(getattr(model, "_ffn_gate_group_size", 16)))
                    G = (dmid + g - 1) // g
                    gi = random.randint(0, max(G - 1, 0))
                    beg, end = gi * g, min((gi + 1) * g, dmid)

                    mask = torch.ones(dmid, device=self.device, dtype=torch.float32)
                    mask[beg:end] = 0.0

                    def _mask_fwd(_m, _inp, _out):
                        return _out * mask.view(1, 1, -1)

                    hk = act_fn.register_forward_hook(_mask_fwd)
                    try:
                        with amp_ctx:
                            out2 = model.forward(batch)
                            loss2 = float(out2["loss"])
                    finally:
                        hk.remove()

                    ablated_losses.append(loss2)

            steps += 1

        if base_losses and ablated_losses:
            b = sum(base_losses) / len(base_losses)
            a = sum(ablated_losses) / len(ablated_losses)
            logging.info(f"[Verify] base_loss={b:.4f}, ablated_loss={a:.4f}, delta={a-b:.4f}")
        else:
            logging.warning("[Verify] not enough samples; skip.")
