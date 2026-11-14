"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
from packaging import version
import os

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
#from transformers import AutoTokenizer, OPTForCausalLM, OPTConfig
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
import transformers
import math
from torch.utils.data import DataLoader, Subset
from typing import Optional, Union, Tuple, Dict, List, Set
import bitsandbytes as bnb
import json
from pathlib import Path

from contextlib import nullcontext

import math


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True, dtype=None, device=None):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False, dtype=dtype, device=device)
        self.B = nn.Linear(rank, out_features, bias=bias, dtype=dtype, device=device)

    @torch.no_grad()
    def from_full_(self, full: nn.Linear, rank: int):
        W = full.weight.detach().to("cpu")
        b = full.bias.detach().to("cpu") if full.bias is not None else None
        r = int(min(rank, min(W.shape)))
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.A.weight.copy_(Vh[:r, :])
        self.B.weight.copy_(U[:, :r] * S[:r])
        if b is not None and self.B.bias is not None:
            self.B.bias.copy_(b)
        dev, dt = full.weight.device, full.weight.dtype
        self.to(device=dev, dtype=dt)
        return self

    def forward(self, x):  # [*, in] -> [*, out]
        return self.B(self.A(x))

    @property
    def in_features(self) -> int:
        # A: [r, in]; B: [out, r]
        return int(self.A.in_features if hasattr(self.A, "in_features") else self.A.weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.B.out_features if hasattr(self.B, "out_features") else self.B.weight.shape[0])

    @property
    def weight(self) -> torch.Tensor:
        return self.B.weight @ self.A.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.B.bias


def _linear_to_lowrank(module: nn.Linear, rank: int) -> nn.Module:
    lr = LowRankLinear(module.in_features, module.out_features, rank,
                       bias=(module.bias is not None),
                       dtype=module.weight.dtype, device=module.weight.device)
    return lr.from_full_(module, rank)

# ---- head similarity helpers ----
def _cosine_sim(x: torch.Tensor, y: torch.Tensor, eps=1e-12):
    x = x / (x.norm(p=2, dim=-1, keepdim=True) + eps)
    y = y / (y.norm(p=2, dim=-1, keepdim=True) + eps)
    return (x * y).sum(dim=-1)

def _center(x): return x - x.mean(dim=-1, keepdim=True)

def _cka_similarity(x: torch.Tensor, y: torch.Tensor, eps=1e-12):
    x = _center(x); y = _center(y)
    num = (x * y).sum()
    den = (x.norm() * y.norm()) + eps
    return (num / den).clamp(min=-1.0, max=1.0) ** 2

def _avg_head_pattern(attn_bhqk: torch.Tensor):
    # input [B,H,Q,N] -> per-head mean pattern [H,N]
    return attn_bhqk.mean(dim=(0, 2))


def _pairwise_head_similarity(A: torch.Tensor, method: str = "cosine"):
    """A: [H, N] -> pairwise similarity matrix [H, H]."""
    A = A.float()
    H = int(A.shape[0])
    method = str(method).lower()
    if H <= 0:
        return torch.empty(0, 0, device=A.device, dtype=A.dtype)
    if method == "cosine":
        A = A / (A.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        return (A @ A.transpose(0, 1)).clamp(min=-1.0, max=1.0)

    sims = torch.empty(H, H, device=A.device, dtype=A.dtype)
    for i in range(H):
        sims[i, i] = 1.0
        for j in range(i + 1, H):
            s = _cka_similarity(A[i], A[j]) if method == "cka" else _cosine_sim(A[i], A[j])
            sims[i, j] = sims[j, i] = float(s)
    return sims


def _save_head_similarity_artifacts(
    sim_matrix: torch.Tensor,
    *,
    save_dir: str,
    prefix: str,
    method: str,
    head_meta: Optional[List[Dict[str, int]]] = None,
    layer_ranges: Optional[List[Dict[str, int]]] = None,
    save_png: bool = True,
    save_dtype: str = "float16",
):
    os.makedirs(save_dir, exist_ok=True)

    sim_cpu = sim_matrix.detach().float().cpu()
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    target_dtype = dtype_map.get(str(save_dtype).lower(), torch.float16)
    sim_to_save = sim_cpu.to(dtype=target_dtype)

    pt_path = os.path.join(save_dir, f"{prefix}.pt")
    json_path = os.path.join(save_dir, f"{prefix}.json")
    png_path = os.path.join(save_dir, f"{prefix}.png")

    payload = {
        "similarity": sim_to_save,
        "method": str(method),
        "shape": list(sim_to_save.shape),
        "dtype": str(sim_to_save.dtype),
        "head_meta": head_meta or [],
        "layer_ranges": layer_ranges or [],
    }
    torch.save(payload, pt_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": str(method),
                "shape": list(sim_to_save.shape),
                "dtype": str(sim_to_save.dtype),
                "num_heads_total": int(sim_to_save.shape[0]),
                "head_meta": head_meta or [],
                "layer_ranges": layer_ranges or [],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if save_png:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        n = int(sim_cpu.shape[0])
        fig_side = min(24.0, max(8.0, 0.22 * max(n, 1)))
        fig, ax = plt.subplots(figsize=(fig_side, fig_side))
        im = ax.imshow(sim_cpu.numpy(), vmin=-1.0 if str(method).lower() == "cosine" else 0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(f"Cross-Attn Head Similarity ({method})")
        ax.set_xlabel("Head index")
        ax.set_ylabel("Head index")

        if head_meta and n <= 96:
            labels = [f"L{int(x['layer_idx'])}H{int(x['head_idx'])}" for x in head_meta]
            font = max(4, min(7, int(480 / max(n, 1))))
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(labels, rotation=90, fontsize=font)
            ax.set_yticklabels(labels, fontsize=font)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        for item in (layer_ranges or []):
            end = int(item.get("end", 0))
            if 0 < end < n:
                ax.axhline(end - 0.5, linewidth=0.7)
                ax.axvline(end - 0.5, linewidth=0.7)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return {
        "pt": pt_path,
        "json": json_path,
        "png": png_path if save_png else None,
    }

def _load_head_similarity_payload(similarity_pt_path: str):
    """

      - similarity: [N, N]
      - method
      - head_meta
      - layer_ranges
    """
    if not similarity_pt_path:
        raise ValueError("similarity_pt_path is empty")

    payload = torch.load(str(similarity_pt_path), map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError(f"similarity payload must be dict, got {type(payload)}")

    if "similarity" not in payload:
        raise KeyError(f"similarity payload missing 'similarity': {similarity_pt_path}")

    sim = payload["similarity"]
    if not torch.is_tensor(sim):
        raise TypeError(f"payload['similarity'] must be Tensor, got {type(sim)}")

    if sim.dim() != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError(f"payload['similarity'] must be square [N,N], got {tuple(sim.shape)}")

    layer_ranges = payload.get("layer_ranges", None) or []
    head_meta = payload.get("head_meta", None) or []
    method = str(payload.get("method", "cosine"))

    if (not layer_ranges) and head_meta:
        ranges = []
        start = 0
        cur_layer = None
        for idx, item in enumerate(head_meta):
            li = int(item.get("layer_idx", -1))
            if cur_layer is None:
                cur_layer = li
                start = idx
            elif li != cur_layer:
                ranges.append({"layer_idx": int(cur_layer), "start": int(start), "end": int(idx)})
                cur_layer = li
                start = idx
        if cur_layer is not None:
            ranges.append({"layer_idx": int(cur_layer), "start": int(start), "end": int(len(head_meta))})
        layer_ranges = ranges

    # 
    if not layer_ranges:
        n = int(sim.shape[0])
        layer_ranges = [{"layer_idx": 0, "start": 0, "end": n}]
        if not head_meta:
            head_meta = [{"global_index": int(i), "layer_idx": 0, "head_idx": int(i)} for i in range(n)]

    return {
        "similarity": sim.detach().float().cpu(),
        "method": method,
        "head_meta": head_meta,
        "layer_ranges": layer_ranges,
        "path": str(similarity_pt_path),
    }

@registry.register_model("blip2_opt")
class Blip2OPT(Blip2Base):
    """
    BLIP2 OPT model.
    Supported model types:
        - pretrained_opt2.7b: pretrained model with OPT2.7b
        - pretrained_opt6.7b: pretrained model with OPT6.7b
        - caption_coco_opt2.7b: fintuned image captioning model with OPT2.7b
        - caption_coco_opt6.7b: fintuned image captioning model with OPT6.7b
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2_opt", "caption_coco_opt2.7b")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_opt2.7b": "configs/models/blip2/blip2_pretrain_opt2.7b.yaml",
        "pretrain_opt6.7b": "configs/models/blip2/blip2_pretrain_opt6.7b.yaml",
        "caption_coco_opt2.7b": "configs/models/blip2/blip2_caption_opt2.7b.yaml",
        "caption_coco_opt6.7b": "configs/models/blip2/blip2_caption_opt6.7b.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        opt_model="facebook/opt-2.7b",
        prompt="",
        max_txt_len=32,
        apply_lemmatizer=False,
        qformer_sdpa_enabled=False,
        opt_sdpa_enabled=False,
        opt_quant_mode="",
    ):
        """
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas.
        """
        super().__init__()
        transformers_version = version.parse(transformers.__version__)
        assert transformers_version >= version.parse("4.27"), "BLIP-2 OPT requires transformers>=4.27"

        self._qformer_sdpa_enabled = bool(qformer_sdpa_enabled)
        self._opt_sdpa_enabled = bool(opt_sdpa_enabled)
        self._opt_quant_mode = str(opt_quant_mode or "").lower()
        self._opt_quantized_on_init = False
        
        self.tokenizer = self.init_tokenizer()

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None

        #  query token  intermediate_query / output_query
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)


        load_kwargs = {}
        quant_mode = self._opt_quant_mode

        if quant_mode == "bnb_8bit":
            load_kwargs["load_in_8bit"] = True
            if torch.cuda.is_available():
                load_kwargs["device_map"] = {"": torch.cuda.current_device()}
            self._opt_quantized_on_init = True

        elif quant_mode in ("bnb_4bit", "4bit", "nf4"):
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            if torch.cuda.is_available():
                load_kwargs["device_map"] = {"": torch.cuda.current_device()}
            self._opt_quantized_on_init = True

        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model,
            **load_kwargs,
        )

        self.opt_model.can_generate_from_inputs_embeds = True
        self.opt_model.config.use_cache = True
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False

        self.pad_token_id = self.opt_tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.opt_tokenizer.eos_token_id if self.opt_tokenizer.eos_token_id is not None else 0
        self.opt_model.config.pad_token_id = int(self.pad_token_id)
        try:
            self.opt_model.generation_config.pad_token_id = int(self.pad_token_id)
        except Exception:
            pass

        self.eos_token_id = self.opt_tokenizer("\n", add_special_tokens=False).input_ids[0]
        if self.eos_token_id is None and self.opt_tokenizer.eos_token_id is not None:
            self.eos_token_id = int(self.opt_tokenizer.eos_token_id)

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )

        self._set_qformer_sdpa_enabled(self._qformer_sdpa_enabled)
        self._set_opt_sdpa_enabled(self._opt_sdpa_enabled)

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)
        
        self._apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None       

        if self._opt_quantized_on_init:
            logging.info(
                f"[Quant] OPT loaded in quantized mode={self._opt_quant_mode}; "
                f"checkpoint loading will skip opt_model.* and only restore non-OPT modules."
            )


    def forward(self, samples):
        image = samples["image"]
        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        image_embeds_full = image_embeds
        image_atts_full = image_atts

        v_keep = getattr(self, "_vision_token_keep_ratio", 1.0)
        v_keep = 1.0 if v_keep is None else float(v_keep)

        score_mode = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
        tail_strategy = str(getattr(self, "_vision_token_tail_strategy", "drop")).lower()

        fwd_id = int(getattr(self, "_vision_token_fwd_id", 0) or 0) + 1
        setattr(self, "_vision_token_fwd_id", fwd_id)
        N_full = int(image_embeds.shape[1])
        setattr(self, "_vision_token_last_n_full", int(N_full))

        evidence_flag = bool(getattr(self, "_vision_token_evidence_log", False)) or (os.environ.get("VTOKEN_EVIDENCE", "0") == "1") or (os.environ.get("VTOKEN_DEBUG", "0") == "1")
        do_evi = False
        if evidence_flag:
            is_rank0 = True
            try:
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    is_rank0 = (torch.distributed.get_rank() == 0)
            except Exception:
                is_rank0 = True
            if is_rank0:
                ecnt = int(getattr(self, "_vision_token_evidence_cnt", 0)) + 1
                setattr(self, "_vision_token_evidence_cnt", ecnt)
                do_evi = (ecnt <= 5) or (ecnt % 200 == 0)

        attn_for_prune = None
        grad_for_prune = None
        want_grads = False

        mode_for_attn = score_mode
        if score_mode in ("learned", "mlp", "score_net"):
            tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
            mode_for_attn = tmode

        need_attn_for_scoring = mode_for_attn in (
            "attn_entropy", "attn", "entropy", "attn_mean",
            "attn_grad", "grad", "taylor_attn",
        )

        
        attn_refresh = None
        grad_refresh = None
        attn_src = "none"
        did_refresh = False
        refresh_reason = ""

        if v_keep < 1.0 and need_attn_for_scoring:
            (
                attn_for_prune,
                grad_for_prune,
                attn_refresh,
                grad_refresh,
                want_grads,
                attn_src,
                did_refresh,
                refresh_reason,
            ) = self._get_cross_attn_for_prune_lagged(
                image_embeds=image_embeds,
                image_atts=image_atts,
                v_keep=v_keep,
                mode_for_attn=mode_for_attn,
                fwd_id=fwd_id,
            )
        else:
            try:
                self._toggle_cross_attn_save(False)
            except Exception:
                pass


        if do_evi and (v_keep < 1.0):
            try:
                a_shape = tuple(attn_for_prune.shape) if (attn_for_prune is not None and torch.is_tensor(attn_for_prune)) else None
                g_shape = tuple(grad_for_prune.shape) if (grad_for_prune is not None and torch.is_tensor(grad_for_prune)) else None
                r_shape = tuple(attn_refresh.shape) if (attn_refresh is not None and torch.is_tensor(attn_refresh)) else None
                logging.info(
                    f"[Vision-Token@evidence] fwd={fwd_id} attn_ready "
                    f"mode_for_attn={mode_for_attn} want_grads={want_grads} "
                    f"N_full={N_full} attn_src={attn_src} did_refresh={did_refresh} refresh={refresh_reason} "
                    f"attn_shape={a_shape} grad_shape={g_shape} refresh_shape={r_shape}"
                )
            except Exception as _e:
                logging.info(f"[Vision-Token@evidence] attn_ready log failed: {_e}")

        try:
            if (v_keep < 1.0) and bool(getattr(self, "_use_vision_head_cka", True)) and (attn_for_prune is not None):
                self._maybe_update_head_weights_from_teacher(
                    attn_student_bhqk=(attn_refresh if (attn_refresh is not None) else attn_for_prune),
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    fwd_id=fwd_id,
                )
        except Exception as _e:
            if do_evi:
                logging.info(f"[Vision-HeadCKA@evidence] fwd={fwd_id} update failed: {_e}")

        if v_keep < 1.0:
            image_embeds, image_atts = self._select_vision_tokens(
                image_embeds,
                image_atts,
                attn_bhqk=attn_for_prune,
                attn_grad_bhqk=grad_for_prune,
                keep_ratio=v_keep,
                mode=score_mode,
                tail_strategy=tail_strategy,
            )
            setattr(self, "_vision_token_last_n_after", int(image_embeds.shape[1]))
        else:
            setattr(self, "_vision_token_last_n_after", int(image_embeds.shape[1]))
        
        if attn_refresh is not None:
            try:
                if torch.is_tensor(attn_refresh) and attn_refresh.dim() == 4:
                    self._last_xattn = attn_refresh.detach()
                if grad_refresh is not None and torch.is_tensor(grad_refresh) and grad_refresh.dim() == 4:
                    self._last_xattn_grad = grad_refresh.detach()
                self._vision_token_cached_fwd = fwd_id
            except Exception:
                pass

        setattr(self, "_vision_token_last_n_after", int(image_embeds.shape[1]))
        if do_evi and (v_keep < 1.0):
            try:
                N_after = int(image_embeds.shape[1])
                used_ratio = (float(N_after) / float(N_full)) if N_full > 0 else 1.0
                _grad_src = (
                    "refresh" if (grad_refresh is not None)
                    else ("cache" if (grad_for_prune is not None) else "none")
                )
                logging.info(
                    f"[Vision-Token@evidence] fwd={fwd_id} prune_done "
                    f"N_full={N_full} N_after={N_after} keep={v_keep:.4f} used_ratio={used_ratio:.4f} "
                    f"score_mode={score_mode} tail={tail_strategy} attn_src={attn_src} grad_src={_grad_src}"
                )

                fb = self._get_qformer_flops_breakdown()
                if (fb.get("current_total") is not None) and (fb.get("full_total") is not None):
                    logging.info(
                        f"[Vision-Token@flops] fwd={fwd_id} "
                        f"current={fb['current_total']/1e9:.4f}G "
                        f"full={fb['full_total']/1e9:.4f}G "
                        f"saved={fb['saved']/1e9:.4f}G ({fb['saved_pct']:.2f}%) "
                        f"q={fb['q_len']} v_full={fb['v_full']} v_used={fb['v_current']} "
                        f"heads={fb['heads']}"
                    )
            except Exception as _e:
                logging.info(f"[Vision-Token@evidence] prune_done log failed: {_e}")


        # Q-Former  & dtype 
        assert image_embeds.dim() == 3 and image_atts.dim() == 2, "image_embeds [B,N,D], image_atts [B,N]"
        assert image_embeds.size(0) == image_atts.size(0) and image_embeds.size(1) == image_atts.size(1), "B/N mismatch after pruning"
        image_atts = image_atts.to(device=image_embeds.device, dtype=torch.long)

        if self.training:
            _fn = getattr(self, "_headdrop_step_fn", None)
            if callable(_fn):
                _fn()

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

        self.opt_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        opt_tokens = self.opt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(image.device)

        targets = opt_tokens.input_ids.masked_fill(
            opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

        # Text-KV learned scorer  teacher(attn) + student(past_kv)
        need_textkv_kd = False
        if self.training:
            try:
                lam_tkv = float(getattr(self, "_text_kv_learned_lambda", 0.0) or 0.0)
                need_textkv_kd = (lam_tkv > 0.0) and (getattr(self, "_text_kv_scorer", None) is not None)
            except Exception:
                need_textkv_kd = False

        labels_for_opt = targets
        if bool(getattr(self, "_text_kv_scorer_only", False)):
            labels_for_opt = None

        with self.maybe_autocast():
            outputs = self.opt_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=labels_for_opt,
                use_cache=need_textkv_kd,
                output_attentions=need_textkv_kd,
            )

        loss = outputs.loss
        if loss is None:
            loss = inputs_embeds.new_tensor(0.0)


        if self.training:
            lam = float(getattr(self, "_ffn_gate_loss_lambda", 0.0) or 0.0)
            if lam > 0.0 and hasattr(self, "_ffn_gates"):
                if hasattr(self, "ffn_gate_sparsity_loss"):
                    loss = loss + self.ffn_gate_sparsity_loss(
                        lambda_sparse=lam,
                        mode=str(getattr(self, "_ffn_gate_loss_mode", "l1"))
                    )
                elif hasattr(self, "_compute_ffn_gate_sparsity_loss"):
                    loss = loss + self._compute_ffn_gate_sparsity_loss()

        if self.training:
            mode_v = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
            if mode_v in ("learned", "mlp", "score_net"):
                lam_v = float(getattr(self, "_vision_token_learned_lambda", 0.0) or 0.0)
                scorer = getattr(self, "_vision_token_scorer", None)
                if lam_v > 0.0 and scorer is not None:
                    tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
                    temp = getattr(self, "_vision_token_learned_temp", None)
                    temp = 1.0 if temp is None else float(temp)

                    with torch.cuda.amp.autocast(enabled=False):
                        feats_f = image_embeds_full.detach().float()  # teacher 
                        if tmode in ("l2norm", "l2"):
                            teacher = feats_f.norm(p=2, dim=-1)  # [B,N]
                        else:
                            attn = getattr(self, "_last_xattn", None)
                            grad = getattr(self, "_last_xattn_grad", None)

                            teacher = None
                            if (tmode in ("attn_grad", "grad", "taylor_attn")) and (attn is not None) and (grad is not None):
                                if attn.dim() == 4 and grad.dim() == 4 and attn.shape == grad.shape:
                                    # [B,H,Q,N] -> [B,N]
                                    teacher = (attn * grad).abs().sum(dim=(1, 2)).float()

                            if teacher is None and (attn is not None) and attn.dim() == 4:
                                eps = 1e-12
                                p = attn.detach().float().clamp_min(eps)
                                p = p / (p.sum(dim=-1, keepdim=True) + eps)      # [B,H,Q,N]
                                p_bqk = p.mean(dim=1)                             # [B,Q,N]
                                H_bq = -(p_bqk * (p_bqk.clamp_min(eps).log())).sum(dim=-1)  # [B,Q]
                                wq = torch.softmax(-H_bq, dim=-1)                 # [B,Q]
                                Q = wq.size(1)
                                if Q > 0:
                                    base = 1.0 / float(Q)
                                    wq = wq.clamp(min=0.5 * base, max=1.5 * base)
                                    wq = wq / wq.sum(dim=-1, keepdim=True).clamp_min(eps)
                                teacher = (wq.unsqueeze(-1) * p_bqk).sum(dim=1)   # [B,N]

                            if teacher is None:
                                teacher = feats_f.norm(p=2, dim=-1)

                        student = scorer(image_embeds_full.float()).squeeze(-1)  # [B,N]

                        t = torch.softmax(teacher / temp, dim=-1).detach()
                        s = torch.log_softmax(student / temp, dim=-1)
                        vtoken_kd = torch.nn.functional.kl_div(s, t, reduction="batchmean") * (temp * temp)

                    loss = loss + lam_v * vtoken_kd

        if self.training:
            lam_tkv = float(getattr(self, "_text_kv_learned_lambda", 0.0) or 0.0)
            scorer = getattr(self, "_text_kv_scorer", None)
            if lam_tkv > 0.0 and scorer is not None:
                try:
                    temp = getattr(self, "_text_kv_learned_temp", None)
                    temp = 1.0 if temp is None else float(temp)
                    eps = 1e-12

                    teacher = None
                    try:
                        dec = getattr(getattr(self.opt_model, "model", None), "decoder", None)
                        layers = getattr(dec, "layers", None) if dec is not None else None
                        if layers is not None and len(layers) > 0:
                            teacher = getattr(layers[-1].self_attn, "_last_attn_importance", None)
                    except Exception:
                        teacher = None

                    if teacher is not None and hasattr(teacher, "dim") and teacher.dim() == 2:
                        pkv = getattr(outputs, "past_key_values", None)
                        if pkv is not None:
                            k_last = pkv[-1][0]  # [B, H, L, D] or [B, L, D]
                            if k_last.dim() == 4:
                                B = int(k_last.size(0)); H = int(k_last.size(1)); L = int(k_last.size(2)); D = int(k_last.size(3))
                                rep = k_last.transpose(1, 2).contiguous().view(B, L, H * D)  # [B, L, hidden]
                            else:
                                rep = k_last  # [B, L, hidden]

                            student_logits = scorer(rep.float()).squeeze(-1)  # [B, L]

                            attn_mask = attention_mask
                            if attn_mask is None:
                                valid = torch.ones_like(student_logits, dtype=torch.bool)
                            else:
                                valid = attn_mask.to(dtype=torch.bool)

                            q_len = int(getattr(self, "_q_len_for_kv", 0) or 0)
                            if q_len > 0 and q_len < valid.size(1):
                                valid[:, :q_len] = False

                            t = teacher.float()
                            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

                            if t.size(1) != student_logits.size(1):
                                Lm = min(int(t.size(1)), int(student_logits.size(1)))
                                t = t[:, :Lm]
                                student_logits = student_logits[:, :Lm]
                                valid = valid[:, :Lm]

                            num_valid = valid.sum(dim=-1)
                            keep = num_valid > 0
                            if not bool(keep.any()):
                                return loss
                            t = t[keep]
                            student_logits = student_logits[keep]
                            valid = valid[keep]

                            t = t * valid.to(dtype=t.dtype)
                            denom = t.sum(dim=-1, keepdim=True).clamp_min(eps)
                            t = (t / denom).detach()

                            s = student_logits.masked_fill(~valid, float("-inf"))
                            s_log = torch.log_softmax(s / temp, dim=-1)
                            tkv_kd = torch.nn.functional.kl_div(s_log, t, reduction="batchmean") * (temp * temp)
                            loss = loss + lam_tkv * tkv_kd

                except Exception:
                    #  loss
                    pass

        if self.training and getattr(self, "_kd_on", False) and getattr(self, "_kd_teacher", None) is not None:
            T = float(getattr(self, "_kd_temp", 2.0))
            lam_log = float(getattr(self, "_kd_l_logit", 0.5))
            lam_att = float(getattr(self, "_kd_l_attn", 0.1))

            try:
                with torch.no_grad():
                    t_outputs = self._kd_teacher.opt_model(
                        inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                        return_dict=True
                    )
                # mask  targets != -100 
                kd_mask = (targets != -100).unsqueeze(-1).to(outputs.logits.dtype)
                s = torch.log_softmax(outputs.logits / T, dim=-1)
                t = torch.softmax(t_outputs.logits / T, dim=-1).detach()
                kl = torch.nn.functional.kl_div(s, t, reduction="none") * (T * T)
                kd_logit = (kl * kd_mask).sum() / kd_mask.sum().clamp_min(1.0)
                loss = loss + lam_log * kd_logit
            except Exception:
                pass  # 

            try:
                self._toggle_cross_attn_save(True)
                if hasattr(self._kd_teacher, "_toggle_cross_attn_save"):
                    self._kd_teacher._toggle_cross_attn_save(True)
                if hasattr(self, "_clear_saved_attn_cache"):
                    self._clear_saved_attn_cache()

                stud_maps = self._collect_cross_attn_maps(with_layer_idx=True, raw=True)   # List[(layer_idx, Tensor  None)]

                with torch.no_grad():
                    t_img = self._kd_teacher.ln_vision(self._kd_teacher.visual_encoder(image))
                    t_atts = torch.ones(t_img.size()[:-1], dtype=torch.long, device=image.device)
                    tq = self._kd_teacher.query_tokens.expand(t_img.shape[0], -1, -1)
                    _ = self._kd_teacher.Qformer.bert(
                        query_embeds=tq,
                        encoder_hidden_states=t_img,
                        encoder_attention_mask=t_atts,
                        return_dict=True,
                    )
                teach_maps = self._kd_teacher._collect_cross_attn_maps(with_layer_idx=True, raw=True)

                topk_ratio = float(getattr(self, "_kd_attn_topk", 0.4))
                kd_attn_all, n_used = 0.0, 0

                cka_accum = None
                cka_count = 0

                l9_weight  = float(getattr(self, "_kd_attn_l9_boost", lam_att))
                l8_weight  = float(getattr(self, "_kd_attn_l8_boost", lam_att))
                l10_weight = float(getattr(self, "_kd_attn_l10_boost", lam_att))
                l9_var_target = float(getattr(self, "_kd_attn_l9_var_target", 0.7))
                l9_var_lambda = float(getattr(self, "_kd_attn_l9_var_lambda", 0.0))

                for (layer_idx, S), (layer_idx_t, Tm) in zip(stud_maps, teach_maps):
                    #  layer_idx  layer_idx_t 
                    if S is None or Tm is None:
                        continue

                    Tm = Tm.detach()

                    # 
                    S = (S + 1e-12) / (S + 1e-12).sum(dim=-1, keepdim=True)
                    Tm = (Tm + 1e-12) / (Tm + 1e-12).sum(dim=-1, keepdim=True)

                    if getattr(self, "_use_vision_head_cka", True):
                        with torch.no_grad():
                            try:
                                Bv, Hv, Qv, Nv = S.shape
                                S_flat = S.permute(1, 0, 2, 3).reshape(Hv, -1)   # [H, B*Q*N]
                                T_flat = Tm.permute(1, 0, 2, 3).reshape(Hv, -1)

                                sims = []
                                for h in range(Hv):
                                    sims.append(_cka_similarity(S_flat[h], T_flat[h]))
                                sims = torch.stack(sims, dim=0)  # [H]

                                if cka_accum is None:
                                    cka_accum = sims
                                else:
                                    cka_accum = cka_accum + sims
                                cka_count += 1
                            except Exception:
                                pass


                    if 0.0 < topk_ratio < 1.0:
                        k = max(1, int(round(Tm.shape[-1] * topk_ratio)))
                        topk_val, topk_idx = torch.topk(Tm, k=k, dim=-1)
                        weight = torch.zeros_like(Tm)
                        weight.scatter_(-1, topk_idx, topk_val)
                    else:
                        weight = Tm

                    kl = torch.sum(
                        S * (torch.log(S + 1e-12) - torch.log(Tm + 1e-12)),
                        dim=-1,
                    )  # [B,H,Q]

                    w = weight.sum(dim=-1).clamp_min(1e-6)  # [B,H,Q]
                    layer_loss = (kl * w).sum() / w.sum()

                    if layer_idx == 9 and l9_var_lambda > 0.0:
                        var_map = S.var(dim=-1, unbiased=False)  # [B,H,Q]
                        var_mean = (var_map * w).sum() / w.sum()
                        var_loss = (var_mean - l9_var_target) ** 2
                        layer_loss = layer_loss + l9_var_lambda * var_loss

                    if layer_idx == 9:
                        layer_weight = l9_weight
                    elif layer_idx == 8:
                        layer_weight = l8_weight
                    elif layer_idx == 10:
                        layer_weight = l10_weight
                    else:
                        layer_weight = lam_att

                    layer_loss = layer_weight * layer_loss

                    kd_attn_all = kd_attn_all + layer_loss
                    n_used += 1

                if getattr(self, "_use_vision_head_cka", True) and (cka_accum is not None) and (cka_count > 0):
                    with torch.no_grad():
                        sims = cka_accum / float(cka_count)          # [H]
                        sims = torch.relu(sims)
                        mean_val = sims.mean().clamp_min(1e-6)
                        sims = sims / mean_val

                        beta = float(getattr(self, "_vision_head_cka_momentum", 0.9))
                        old = getattr(self, "_vision_head_weights", None)
                        if old is None:
                            self._vision_head_weights = sims.detach()
                        else:
                            self._vision_head_weights = (
                                beta * old.to(sims.device, dtype=sims.dtype) +
                                (1.0 - beta) * sims.detach()
                            )
                        if (os.environ.get("VTOKEN_EVIDENCE", "0") == "1") or (os.environ.get("VTOKEN_DEBUG", "0") == "1") or bool(getattr(self, "_vision_token_evidence_log", False)):
                            try:
                                from lavis.common.dist_utils import get_rank
                                if get_rank() == 0:
                                    cc = int(getattr(self, "_vision_head_cka_log_cnt", 0)) + 1
                                    setattr(self, "_vision_head_cka_log_cnt", cc)
                                    if (cc <= 5) or (cc % 200 == 0):
                                        _s = sims.detach().float()
                                        _w = getattr(self, "_vision_head_weights", None)
                                        if _w is not None:
                                            _w = _w.detach().float()
                                            w_min = float(_w.min().item())
                                            w_mean = float(_w.mean().item())
                                            w_max = float(_w.max().item())
                                            w_str = f"{w_min:.4g}/{w_mean:.4g}/{w_max:.4g}"
                                        else:
                                            w_str = "na/na/na"
                                        logging.info(
                                            f"[Vision-HeadCKA@evidence] fwd={getattr(self, '_vision_token_fwd_id', None)} "
                                            f"cka_count={cka_count} beta={beta:.3f} "
                                            f"sims(min/mean/max)={float(_s.min().item()):.4g}/{float(_s.mean().item()):.4g}/{float(_s.max().item()):.4g} "
                                            f"head_w(min/mean/max)={w_str}"
                                        )
                            except Exception as _e:
                                logging.info(f"[Vision-HeadCKA@evidence] log failed: {_e}")


                if n_used > 0:
                    loss = loss + (kd_attn_all / n_used)


            except Exception:
                pass
            finally:
                try:
                    v_keep = float(getattr(self, "_vision_token_keep_ratio", 1.0) or 1.0)
                    mode_v = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
                    tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
                    mode_for_attn = tmode if mode_v in ("learned", "mlp", "score_net") else mode_v
                    need_attn = (v_keep < 1.0) and (mode_for_attn in ("attn", "entropy", "attn_entropy", "attn_mean", "attn_grad", "grad", "taylor_attn"))

                    self._toggle_cross_attn_save(need_attn)
                except Exception:
                    # 
                    pass

                try:
                    if hasattr(self, "_kd_teacher") and hasattr(self._kd_teacher, "_toggle_cross_attn_save"):
                        self._kd_teacher._toggle_cross_attn_save(False)
                except Exception:
                    pass


        if self.training and (
            float(getattr(self, "_recovery_cost_lat", 0.0)) > 0.0
            or float(getattr(self, "_recovery_cost_mem", 0.0)) > 0.0
        ):
            try:
                loss = loss + self._recovery_cost_loss()
            except Exception:
                pass

        try:
            v_keep = float(getattr(self, "_vision_token_keep_ratio", 1.0) or 1.0)
            lag_onepass_post = bool(getattr(self, "_vision_token_lag_onepass", False))
            if bool(getattr(self, "_vision_token_lagged_two_pass", True)) is False:
                lag_onepass_post = True

            if self.training and lag_onepass_post:
                pass
            elif self.training and v_keep < 1.0 and (getattr(self, "_vision_token_cached_fwd", None) != getattr(self, "_vision_token_fwd_id", None)):
                mode_v = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
                tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
                mode_for_attn = tmode if mode_v in ("learned", "mlp", "score_net") else mode_v

                if mode_for_attn in ("attn", "entropy", "attn_entropy", "attn_mean", "attn_grad", "grad", "taylor_attn"):
                    xs = []
                    grads = [] if mode_for_attn in ("attn_grad", "grad", "taylor_attn") else None

                    Bv = image_embeds.shape[0]
                    Nv = image_embeds.shape[1]

                    for _, _, sa in self._iter_cross_attn_modules():
                        get = getattr(sa, "get_attention_map", None)
                        attn = get() if callable(get) else getattr(sa, "_saved_attn_map", None)
                        if attn is None or attn.dim() != 4:
                            continue
                        if attn.shape[0] != Bv or attn.shape[-1] != Nv:
                            continue

                        xs.append(attn.detach())
                        if grads is not None:
                            get_grad = getattr(sa, "get_attn_gradients", None)
                            grad = get_grad() if callable(get_grad) else getattr(sa, "attn_gradients", None)
                            if grad is not None and grad.dim() == 4 and tuple(grad.shape) == tuple(attn.shape):
                                grads.append(grad.detach())

                    if xs:
                        try:
                            stacked = torch.stack(xs, dim=0)  # [L,B,H,Q,N]
                            self._last_xattn = stacked.mean(dim=0)  # [B,H,Q,N]
                        except Exception:
                            base = xs[0].shape
                            xs_same = [t for t in xs if tuple(t.shape) == tuple(base)]
                            self._last_xattn = torch.stack(xs_same, dim=0).mean(dim=0) if xs_same else None
                    else:
                        self._last_xattn = None

                    if grads is not None:
                        if grads:
                            try:
                                g_stacked = torch.stack(grads, dim=0)  # [L,B,H,Q,N]
                                self._last_xattn_grad = g_stacked.mean(dim=0)
                            except Exception:
                                base = grads[0].shape
                                g_same = [t for t in grads if tuple(t.shape) == tuple(base)]
                                self._last_xattn_grad = torch.stack(g_same, dim=0).mean(dim=0) if g_same else None
                        else:
                            self._last_xattn_grad = None
                    else:
                        self._last_xattn_grad = None

                    if hasattr(self, "_clear_saved_attn_cache"):
                        self._clear_saved_attn_cache()
        except Exception:
            # 
            pass



        return {"loss": loss}


    def reset_textkv_gen_stats(self):
        self._text_kv_gen_stats = {
            "n_seq": 0,
            "sum_new": 0,
            "sum_new_sq": 0,
            "n_has_eos": 0,
            "n_hit_max": 0,
            "sum_decode_time": 0.0,
            "last_max_new_tokens": None,
            "lens_tail": [],  # 
        }


    def reset_generate_stage_stats(self):
        self._generate_stage_stats = {
            "n_calls": 0,
            "n_seq": 0,

            # 
            "t_visual": 0.0,
            "t_vtoken": 0.0,          # runtime vtoken 

            "t_qformer": 0.0,
            "t_qformer_cross": 0.0,
            "t_qformer_ffn": 0.0,
            "t_qformer_other": 0.0,

            "t_opt_prep": 0.0,
            "t_generate": 0.0,
            "t_post_decode": 0.0,

            "t_vtoken_attn": 0.0,
            "t_vtoken_select": 0.0,   # _select_vision_tokens 
            "t_vtoken_cache": 0.0,    # lag cache  / refresh 

            "f_vtoken_attn": 0.0,
            "f_vtoken_select": 0.0,
            "f_vtoken_cache": 0.0,
            "f_vtoken_total": 0.0,
        }

    def set_generate_stage_profile(self, enabled: bool = False):
        self._profile_generate_stages = bool(enabled)
        if (not hasattr(self, "_generate_stage_stats")) or (not isinstance(getattr(self, "_generate_stage_stats", None), dict)):
            self.reset_generate_stage_stats()

    def set_qformer_inner_profile(self, enabled: bool = False):
        self._profile_qformer_inner = bool(enabled)
        try:
            enc = getattr(getattr(self.Qformer, "bert", None), "encoder", None)
            if enc is not None and hasattr(enc, "set_inner_profile"):
                enc.set_inner_profile(bool(enabled))
        except Exception:
            pass

    def _sync_cuda_for_timing(self):
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

    def _update_generate_stage_stats(self, name: str, dt: float):
        try:
            st = getattr(self, "_generate_stage_stats", None)
            if not isinstance(st, dict):
                self.reset_generate_stage_stats()
                st = self._generate_stage_stats
            key = f"t_{name}"
            st[key] = float(st.get(key, 0.0) or 0.0) + float(dt)
        except Exception:
            pass

    def _get_generate_stage_profile_dict(self):
        st = getattr(self, "_generate_stage_stats", None)
        if not isinstance(st, dict):
            return None

        n_calls = int(st.get("n_calls", 0) or 0)
        n_seq = int(st.get("n_seq", 0) or 0)
        if n_calls <= 0 or n_seq <= 0:
            return None

        main_keys = [
            "t_visual",
            "t_vtoken",
            "t_qformer",
            "t_qformer_cross",
            "t_qformer_ffn",
            "t_qformer_other",
            "t_opt_prep",
            "t_generate",
            "t_post_decode",
        ]
        vtoken_sub_keys = [
            "t_vtoken_attn",
            "t_vtoken_select",
            "t_vtoken_cache",
        ]

        total_sec = {k: float(st.get(k, 0.0) or 0.0) for k in main_keys}
        vtoken_total_sec = {k: float(st.get(k, 0.0) or 0.0) for k in vtoken_sub_keys}

        tracked_total_sec = sum(
            float(total_sec.get(k, 0.0) or 0.0)
            for k in [
                "t_visual",
                "t_vtoken",
                "t_qformer",
                "t_opt_prep",
                "t_generate",
                "t_post_decode",
            ]
        )

        avg_call_sec = {k: (v / float(n_calls)) for k, v in total_sec.items()}
        ms_seq = {k: (1000.0 * v / float(n_seq)) for k, v in total_sec.items()}

        vtoken_avg_call_sec = {k: (v / float(n_calls)) for k, v in vtoken_total_sec.items()}
        vtoken_ms_seq = {k: (1000.0 * v / float(n_seq)) for k, v in vtoken_total_sec.items()}

        f_vtoken_attn = float(st.get("f_vtoken_attn", 0.0) or 0.0)
        f_vtoken_select = float(st.get("f_vtoken_select", 0.0) or 0.0)
        f_vtoken_cache = float(st.get("f_vtoken_cache", 0.0) or 0.0)
        f_vtoken_total = float(st.get("f_vtoken_total", 0.0) or 0.0)

        vtoken_flops_total = {
            "f_vtoken_attn": f_vtoken_attn,
            "f_vtoken_select": f_vtoken_select,
            "f_vtoken_cache": f_vtoken_cache,
            "f_vtoken_total": f_vtoken_total,
        }
        vtoken_flops_avg_call = {
            "f_vtoken_attn": f_vtoken_attn / float(n_calls),
            "f_vtoken_select": f_vtoken_select / float(n_calls),
            "f_vtoken_cache": f_vtoken_cache / float(n_calls),
            "f_vtoken_total": f_vtoken_total / float(n_calls),
        }
        vtoken_flops_per_seq = {
            "f_vtoken_attn": f_vtoken_attn / float(n_seq),
            "f_vtoken_select": f_vtoken_select / float(n_seq),
            "f_vtoken_cache": f_vtoken_cache / float(n_seq),
            "f_vtoken_total": f_vtoken_total / float(n_seq),
        }

        stage4_total_sec = {
            "visual_encoder": float(total_sec.get("t_visual", 0.0) or 0.0),
            "runtime_vtoken": float(total_sec.get("t_vtoken", 0.0) or 0.0),
            "qformer": float(total_sec.get("t_qformer", 0.0) or 0.0),
            "llm_decode": float(
                (total_sec.get("t_opt_prep", 0.0) or 0.0)
                + (total_sec.get("t_generate", 0.0) or 0.0)
                + (total_sec.get("t_post_decode", 0.0) or 0.0)
            ),
        }

        stage4_avg_call_sec = {
            k: (v / float(n_calls)) for k, v in stage4_total_sec.items()
        }

        stage4_ms_seq = {
            k: (1000.0 * v / float(n_seq)) for k, v in stage4_total_sec.items()
        }

        if tracked_total_sec > 1e-12:
            stage4_share_tracked = {
                k: (v / tracked_total_sec) for k, v in stage4_total_sec.items()
            }
        else:
            stage4_share_tracked = {
                k: 0.0 for k in stage4_total_sec
            }

        vtoken_normal_total_sec = {
            "t_vtoken_normal_visual": float(total_sec.get("t_visual", 0.0) or 0.0),
            "t_vtoken_normal_qformer": float(total_sec.get("t_qformer", 0.0) or 0.0),
            "t_vtoken_normal_qformer_cross": float(total_sec.get("t_qformer_cross", 0.0) or 0.0),
            "t_vtoken_normal_qformer_ffn": float(total_sec.get("t_qformer_ffn", 0.0) or 0.0),
            "t_vtoken_normal_qformer_other": float(total_sec.get("t_qformer_other", 0.0) or 0.0),
            "t_vtoken_normal_total": float((total_sec.get("t_visual", 0.0) or 0.0) + (total_sec.get("t_qformer", 0.0) or 0.0)),
        }
        vtoken_normal_avg_call_sec = {k: (v / float(n_calls)) for k, v in vtoken_normal_total_sec.items()}
        vtoken_normal_ms_seq = {k: (1000.0 * v / float(n_seq)) for k, v in vtoken_normal_total_sec.items()}

        runtime_vtoken_total_sec = {
            "t_runtime_total": float(total_sec.get("t_vtoken", 0.0) or 0.0),
            "t_runtime_attn": float(vtoken_total_sec.get("t_vtoken_attn", 0.0) or 0.0),
            "t_runtime_select": float(vtoken_total_sec.get("t_vtoken_select", 0.0) or 0.0),
            "t_runtime_cache": float(vtoken_total_sec.get("t_vtoken_cache", 0.0) or 0.0),
        }
        runtime_vtoken_avg_call_sec = {k: (v / float(n_calls)) for k, v in runtime_vtoken_total_sec.items()}
        runtime_vtoken_ms_seq = {k: (1000.0 * v / float(n_seq)) for k, v in runtime_vtoken_total_sec.items()}

        return {
            "n_calls": n_calls,
            "n_seq": n_seq,
            "total_sec": total_sec,
            "avg_call_sec": avg_call_sec,
            "ms_seq": ms_seq,
            "vtoken_total_sec": vtoken_total_sec,
            "vtoken_avg_call_sec": vtoken_avg_call_sec,
            "vtoken_ms_seq": vtoken_ms_seq,
            "vtoken_normal_total_sec": vtoken_normal_total_sec,
            "vtoken_normal_avg_call_sec": vtoken_normal_avg_call_sec,
            "vtoken_normal_ms_seq": vtoken_normal_ms_seq,
            "runtime_vtoken_total_sec": runtime_vtoken_total_sec,
            "runtime_vtoken_avg_call_sec": runtime_vtoken_avg_call_sec,
            "runtime_vtoken_ms_seq": runtime_vtoken_ms_seq,

            "stage4_total_sec": stage4_total_sec,
            "stage4_avg_call_sec": stage4_avg_call_sec,
            "stage4_ms_seq": stage4_ms_seq,
            "stage4_share_tracked": stage4_share_tracked,

            "vtoken_flops_total": vtoken_flops_total,
            "vtoken_flops_avg_call": vtoken_flops_avg_call,
            "vtoken_flops_per_seq": vtoken_flops_per_seq,

            "tracked_total_sec": float(tracked_total_sec),
            "tracked_avg_call_sec": float(tracked_total_sec / float(n_calls)),
            "tracked_ms_seq": float(1000.0 * tracked_total_sec / float(n_seq)),
        }
            
    def set_text_kv_len_diag(self, enabled: bool = False, interval: int = 0):
        """

        interval:
            - ==1: 
            -  >1:  +  interval 
        """
        self._text_kv_len_diag_enabled = bool(enabled)
        try:
            self._text_kv_len_diag_interval = int(interval or 0)
        except Exception:
            self._text_kv_len_diag_interval = 0
        self._text_kv_len_diag_step = 0

    def _should_text_kv_len_diag(self) -> bool:
        if not bool(getattr(self, "_text_kv_len_diag_enabled", False)):
            return False

        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                if int(torch.distributed.get_rank()) != 0:
                    return False
        except Exception:
            pass

        try:
            interval = int(getattr(self, "_text_kv_len_diag_interval", 0) or 0)
        except Exception:
            interval = 0

        step = int(getattr(self, "_text_kv_len_diag_step", 0) or 0) + 1
        self._text_kv_len_diag_step = step

        if interval <= 0:
            return step == 1
        if interval == 1:
            return True
        return (step == 1) or (step % interval == 0)




    def _update_textkv_gen_stats(self, gen_ids, *, eos_token_id: int, pad_token_id: int, max_new_tokens: int, decode_time: float):
        """
        """
        try:
            if gen_ids is None or (not torch.is_tensor(gen_ids)) or gen_ids.dim() != 2:
                return
            B, L = int(gen_ids.size(0)), int(gen_ids.size(1))
            if B <= 0 or L <= 0:
                return

            eos_mask = (gen_ids == int(eos_token_id))
            has_eos = eos_mask.any(dim=1)  # [B]
            eos_pos = torch.argmax(eos_mask.to(dtype=torch.int32), dim=1)  # [B]
            nonpad = (gen_ids != int(pad_token_id)).sum(dim=1)
            gen_len = torch.where(has_eos, eos_pos + 1, nonpad).to(dtype=torch.int64)

            hit_max = (~has_eos) & (gen_len >= int(max_new_tokens))

            st = getattr(self, "_text_kv_gen_stats", None)
            if not isinstance(st, dict):
                self.reset_textkv_gen_stats()
                st = self._text_kv_gen_stats

            st["n_seq"] = int(st.get("n_seq", 0)) + B
            st["sum_new"] = int(st.get("sum_new", 0)) + int(gen_len.sum().item())
            st["sum_new_sq"] = int(st.get("sum_new_sq", 0)) + int((gen_len.to(dtype=torch.int64) ** 2).sum().item())

            st["n_has_eos"] = int(st.get("n_has_eos", 0)) + int(has_eos.sum().item())
            st["n_hit_max"] = int(st.get("n_hit_max", 0)) + int(hit_max.sum().item())
            st["sum_decode_time"] = float(st.get("sum_decode_time", 0.0)) + float(decode_time)
            st["last_max_new_tokens"] = int(max_new_tokens)

            try:
                tail = st.get("lens_tail", None)
                if not isinstance(tail, list):
                    tail = []
                tail.extend([int(x) for x in gen_len.detach().cpu().tolist()])
                if len(tail) > 2000:
                    tail[:] = tail[-2000:]
                st["lens_tail"] = tail
            except Exception:
                pass
        except Exception:
            return

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Text generation with OPT, supporting runtime visual-token pruning and text KV pruning.
        """
        image = samples["image"]

        do_len_diag = False
        try:
            do_len_diag = bool(self._should_text_kv_len_diag())
        except Exception:
            do_len_diag = False

        import time as _time

        profile_stage = bool(getattr(self, "_profile_generate_stages", False))
        batch_size = int(image.size(0))

        if profile_stage:
            try:
                if (not hasattr(self, "_generate_stage_stats")) or (not isinstance(getattr(self, "_generate_stage_stats", None), dict)):
                    self.reset_generate_stage_stats()
                self._generate_stage_stats["n_calls"] = int(self._generate_stage_stats.get("n_calls", 0) or 0) + 1
                self._generate_stage_stats["n_seq"] = int(self._generate_stage_stats.get("n_seq", 0) or 0) + batch_size
            except Exception:
                pass

        def _tic():
            if not profile_stage:
                return None
            self._sync_cuda_for_timing()
            return _time.perf_counter()

        def _toc(name: str, t0):
            if (not profile_stage) or (t0 is None):
                return
            self._sync_cuda_for_timing()
            self._update_generate_stage_stats(name, _time.perf_counter() - t0)

        with self.maybe_autocast():
            # Visual encode
            _t_visual = _tic()
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            _toc("visual", _t_visual)

            image_embeds_full = image_embeds

            N_full = int(image_embeds.shape[1])
            setattr(self, "_vision_token_last_n_full", int(N_full))
            setattr(self, "_vision_token_last_n_after", int(N_full))

            # (Optional) Vision token pruning at runtime
            v_keep = getattr(self, "_vision_token_keep_ratio", 1.0)
            v_keep = 1.0 if v_keep is None else float(v_keep)

            _t_vtoken_total = _tic()

            if v_keep < 1.0:
                score_mode = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
                tail_strategy = str(getattr(self, "_vision_token_tail_strategy", "drop")).lower()

                #  token  _select_vision_tokens(..., mode="attn_grad")
                select_mode = score_mode

                fwd_id = int(getattr(self, "_vision_token_fwd_id", 0)) + 1
                setattr(self, "_vision_token_fwd_id", fwd_id)

                attn_for_prune = None
                grad_for_prune = None
                attn_refresh = None
                grad_refresh = None
                want_grads = False
                attn_src = "none"
                did_refresh = False
                refresh_reason = "none"

                # 1) attn / grad 
                _t_vtoken_attn = _tic()
                if score_mode == "teacher_attn_grad_external":
                    attn_for_prune, grad_for_prune = self._get_teacher_attn_grad_for_prune(samples)
                    select_mode = "attn_grad"
                    attn_src = "teacher_external"
                    want_grads = True
                else:
                    mode_for_attn = score_mode
                    if score_mode in ("learned", "mlp", "score_net"):
                        tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
                        mode_for_attn = tmode

                    (
                        attn_for_prune,
                        grad_for_prune,
                        attn_refresh,
                        grad_refresh,
                        want_grads,
                        attn_src,
                        did_refresh,
                        refresh_reason,
                    ) = self._get_cross_attn_for_prune_lagged(
                        image_embeds=image_embeds,
                        image_atts=image_atts,
                        v_keep=v_keep,
                        mode_for_attn=mode_for_attn,
                        fwd_id=fwd_id,
                    )

                    #  student  attn  head CKA 
                    try:
                        if bool(getattr(self, "_use_vision_head_cka", True)) and (attn_for_prune is not None):
                            self._maybe_update_head_weights_from_teacher(
                                attn_student_bhqk=(attn_refresh if (attn_refresh is not None) else attn_for_prune),
                                image_embeds=image_embeds,
                                image_atts=image_atts,
                                fwd_id=fwd_id,
                            )
                    except Exception as _e:
                        if bool(getattr(self, "_vision_token_evidence_log", False)):
                            logging.info(f"[Vision-HeadCKA@evidence] generate update failed: {_e}")
                _toc("vtoken_attn", _t_vtoken_attn)

                # 2) token  / merge / drop 
                _t_vtoken_select = _tic()
                image_embeds, image_atts = self._select_vision_tokens(
                    image_embeds,
                    image_atts,
                    keep_ratio=v_keep,
                    mode=select_mode,
                    tail_strategy=tail_strategy,
                    attn_bhqk=attn_for_prune,
                    attn_grad_bhqk=grad_for_prune,
                )
                _toc("vtoken_select", _t_vtoken_select)

                # 3) lag cache  / refresh 
                _t_vtoken_cache = _tic()
                if (score_mode != "teacher_attn_grad_external") and (attn_refresh is not None):
                    try:
                        if torch.is_tensor(attn_refresh) and attn_refresh.dim() == 4:
                            self._last_xattn = attn_refresh.detach()
                        if grad_refresh is not None and torch.is_tensor(grad_refresh) and grad_refresh.dim() == 4:
                            self._last_xattn_grad = grad_refresh.detach()
                        self._vision_token_cached_fwd = fwd_id
                    except Exception:
                        pass
                _toc("vtoken_cache", _t_vtoken_cache)

            setattr(self, "_vision_token_last_n_after", int(image_embeds.shape[1]))

            _toc("vtoken", _t_vtoken_total)

            if profile_stage:
                try:
                    vt_fb = {
                        "attn": 0.0,
                        "select": 0.0,
                        "cache": 0.0,
                        "total": 0.0,
                    }
                    if v_keep < 1.0:
                        attn_for_flops = None
                        if attn_for_prune is not None and torch.is_tensor(attn_for_prune):
                            attn_for_flops = attn_for_prune
                        elif attn_refresh is not None and torch.is_tensor(attn_refresh):
                            attn_for_flops = attn_refresh

                        vt_fb = self._get_vtoken_flops_breakdown(
                            image_embeds_full=image_embeds_full,
                            image_embeds_used=image_embeds,
                            attn_bhqk=attn_for_flops,
                            score_mode=score_mode,
                            tail_strategy=tail_strategy,
                            did_refresh=bool(did_refresh),
                            used_grad=bool(
                                ((grad_for_prune is not None) and torch.is_tensor(grad_for_prune))
                                or ((grad_refresh is not None) and torch.is_tensor(grad_refresh))
                            ),
                        )

                    stg = getattr(self, "_generate_stage_stats", None)
                    if isinstance(stg, dict):
                        stg["f_vtoken_attn"] = float(stg.get("f_vtoken_attn", 0.0) or 0.0) + float(vt_fb["attn"])
                        stg["f_vtoken_select"] = float(stg.get("f_vtoken_select", 0.0) or 0.0) + float(vt_fb["select"])
                        stg["f_vtoken_cache"] = float(stg.get("f_vtoken_cache", 0.0) or 0.0) + float(vt_fb["cache"])
                        stg["f_vtoken_total"] = float(stg.get("f_vtoken_total", 0.0) or 0.0) + float(vt_fb["total"])
                except Exception:
                    pass

            # Q-Former  & dtype 
            assert image_embeds.dim() == 3 and image_atts.dim() == 2, "image_embeds [B,N,D], image_atts [B,N]"
            assert image_embeds.size(0) == image_atts.size(0) and image_embeds.size(1) == image_atts.size(1), "B/N mismatch after pruning"
            image_atts = image_atts.to(device=image_embeds.device, dtype=torch.long)

            # Q-Former queries
            score_mode = getattr(self, "_vision_token_score_mode", "")
            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

            need_grad_attn = (score_mode == "attn_grad")

            qf_inner_enabled = bool(profile_stage and getattr(self, "_profile_qformer_inner", False))
            qf_encoder = getattr(getattr(self.Qformer, "bert", None), "encoder", None)
            if qf_inner_enabled and qf_encoder is not None and hasattr(qf_encoder, "begin_inner_profile_call"):
                qf_encoder.begin_inner_profile_call()

            _t_qformer = _tic()
            with torch.set_grad_enabled(need_grad_attn):
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
            _toc("qformer", _t_qformer)

            if qf_inner_enabled and qf_encoder is not None and hasattr(qf_encoder, "end_inner_profile_call"):
                try:
                    qf_stats = qf_encoder.end_inner_profile_call()
                    self._update_generate_stage_stats("qformer_cross", float(qf_stats.get("cross", 0.0)))
                    self._update_generate_stage_stats("qformer_ffn", float(qf_stats.get("ffn", 0.0)))
                    self._update_generate_stage_stats("qformer_other", float(qf_stats.get("other", 0.0)))
                except Exception:
                    pass

            try:
                _to_store_attn = attn_refresh if (attn_refresh is not None) else attn_for_prune
                _to_store_grad = grad_refresh if (grad_refresh is not None) else grad_for_prune

                if (_to_store_attn is not None) and torch.is_tensor(_to_store_attn) and _to_store_attn.dim() == 4:
                    self._last_xattn = _to_store_attn.detach()          # [B,H,Q,N_full]
                if (_to_store_grad is not None) and torch.is_tensor(_to_store_grad) and _to_store_grad.dim() == 4:
                    self._last_xattn_grad = _to_store_grad.detach()
            except Exception:
                pass

            _t_opt_prep = _tic()

            inputs_opt = self.opt_proj(query_output.last_hidden_state)  # [B, Q, D]
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)  # [B, Q]

            # Prompt
            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt
            prompt = [prompt] * image.size(0)

            # Tokenize (left padding to align incremental decoding)
            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)
            if do_len_diag:
                try:
                    text_len = int(opt_tokens.input_ids.shape[1])  # padding="longest"
                    valid_tok = int(opt_tokens.attention_mask.sum(dim=1).float().mean().item())  # token
                    p0 = prompt[0] if (isinstance(prompt, list) and len(prompt) > 0) else str(prompt)
                    p0 = (p0[:160] + "...") if len(p0) > 160 else p0
                    logging.info(
                        f"[Len-Diag] max_txt_len={int(self.max_txt_len)} "
                        f"text_len(padded)={text_len} text_tok(valid_mean)~={valid_tok} "
                        f"prompt0={p0}"
                    )
                except Exception:
                    pass
            # <<< 

            # Attention mask = [query ones] + [text mask]
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            _toc("opt_prep", _t_opt_prep)

            prompt_length = opt_tokens.input_ids.shape[1]
            if profile_stage:
                self._sync_cuda_for_timing()
                _t0 = _time.perf_counter()
            else:
                _t0 = _time.time()

            if use_nucleus_sampling:
                outputs = self.opt_model.generate(
                    input_ids=opt_tokens.input_ids,
                    use_cache=True,
                    query_embeds=inputs_opt,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_captions,
                )
            else:
                outputs = self.opt_model.generate(
                    input_ids=opt_tokens.input_ids,
                    use_cache=True,
                    query_embeds=inputs_opt,
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    max_new_tokens=max_length,
                    min_new_tokens=min_length,
                    eos_token_id=self.eos_token_id,
                    pad_token_id=self.pad_token_id,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    num_return_sequences=num_captions,
                )


            if profile_stage:
                self._sync_cuda_for_timing()
                _decode_time = _time.perf_counter() - _t0
                self._update_generate_stage_stats("generate", _decode_time)
            else:
                _decode_time = _time.time() - _t0

            if do_len_diag:
                try:
                    prompt_length = int(opt_tokens.input_ids.size(1))
                    gen_ids = outputs[:, prompt_length:]  # [B, L_new_padded]
                    eos_id = int(self.eos_token_id)
                    pad_id = int(self.pad_token_id)

                    eos_mask = (gen_ids == eos_id)
                    has_eos = eos_mask.any(dim=1)
                    eos_pos = torch.argmax(eos_mask.to(dtype=torch.int32), dim=1)
                    nonpad = (gen_ids != pad_id).sum(dim=1)
                    gen_len = torch.where(has_eos, eos_pos + 1, nonpad).to(dtype=torch.int64)

                    gen_len_mean = float(gen_len.float().mean().item())
                    q_len = int(inputs_opt.size(1))
                    prompt_padded = int(opt_tokens.input_ids.size(1))
                    prompt_valid = opt_tokens.attention_mask.sum(dim=1).to(dtype=torch.int64)
                    prompt_valid_mean = float(prompt_valid.float().mean().item())

                    kv_total_mean_valid = float(q_len + prompt_valid_mean + gen_len_mean)
                    kv_total_mean_padded = float(q_len + float(prompt_padded) + gen_len_mean)
                    prunable_valid = float(prompt_valid_mean + gen_len_mean)
                    prunable_padded = float(float(prompt_padded) + gen_len_mean)

                    logging.info(
                        "[KV-Len-Diag] "
                        f"Q={q_len} "
                        f"prompt_padded={prompt_padded} prompt_valid_mean={prompt_valid_mean:.2f} "
                        f"gen_new_mean={gen_len_mean:.2f} "
                        f"KV_total_mean(valid)={kv_total_mean_valid:.2f} KV_total_mean(padded)={kv_total_mean_padded:.2f} "
                        f"prunable_mean(valid)={prunable_valid:.2f} prunable_mean(padded)={prunable_padded:.2f} "
                        f"decode_time={float(_decode_time):.4f}s"
                    )
                except Exception:
                    pass
            # <<< 

            try:
                self._update_textkv_gen_stats(
                    outputs[:, prompt_length:],
                    eos_token_id=int(self.eos_token_id),
                    pad_token_id=int(self.pad_token_id),
                    max_new_tokens=int(max_length),
                    decode_time=float(_decode_time),
                )
            except Exception:
                pass
            # Strip prompt and decode
            _t_post = _tic()

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [t.strip() for t in output_text]

            _toc("post_decode", _t_post)
            return output_text

        
        
    def predict_answers(
        self,
        samples,
        num_beams=5,
        inference_method="generate",
        max_len=10,
        min_len=1,
        num_ans_candidates=128,
        answer_list=None,
        prompt="",
        length_penalty=0,
        **kwargs
    ):
        image = samples["image"]

        do_len_diag = False
        try:
            do_len_diag = bool(self._should_text_kv_len_diag())
        except Exception:
            do_len_diag = False

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                image.device
            )

            v_keep = getattr(self, "_vision_token_keep_ratio", 1.0)
            v_keep = 1.0 if v_keep is None else float(v_keep)
            
            if v_keep < 1.0:
                score_mode = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
                tail_strategy = str(getattr(self, "_vision_token_tail_strategy", "drop")).lower()

                fwd_id = int(getattr(self, "_vision_token_fwd_id", 0)) + 1
                setattr(self, "_vision_token_fwd_id", fwd_id)

                mode_for_attn = score_mode
                if score_mode in ("learned", "mlp", "score_net"):
                    tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
                    mode_for_attn = tmode

                (
                    attn_for_prune,
                    grad_for_prune,
                    attn_refresh,
                    grad_refresh,
                    want_grads,
                    attn_src,
                    did_refresh,
                    refresh_reason,
                ) = self._get_cross_attn_for_prune_lagged(
                    image_embeds=image_embeds,
                    image_atts=image_atts,
                    v_keep=v_keep,
                    mode_for_attn=mode_for_attn,
                    fwd_id=fwd_id,
                )

                image_embeds, image_atts = self._select_vision_tokens(
                    image_embeds,
                    image_atts,
                    keep_ratio=v_keep,
                    mode=score_mode,
                    tail_strategy=tail_strategy,
                    attn_bhqk=attn_for_prune,
                    attn_grad_bhqk=grad_for_prune,
                )

                if attn_refresh is not None:
                    try:
                        if torch.is_tensor(attn_refresh) and attn_refresh.dim() == 4:
                            self._last_xattn = attn_refresh.detach()
                        if grad_refresh is not None and torch.is_tensor(grad_refresh) and grad_refresh.dim() == 4:
                            self._last_xattn_grad = grad_refresh.detach()
                        self._vision_token_cached_fwd = fwd_id
                    except Exception:
                        pass

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(
                image.device
            )

            if isinstance(samples["text_input"], str):
                samples["text_input"] = [samples["text_input"]]
            if prompt:
                text_input = [prompt.format(question) for question in samples["text_input"]]
            else:
                text_input = samples["text_input"]

            self.opt_tokenizer.padding_side = "left"
            opt_tokens = self.opt_tokenizer(
                text_input,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
            ).to(image.device)

            if do_len_diag:
                try:
                    q_len = int(inputs_opt.size(1))  # query tokens
                    prompt_len_padded = int(opt_tokens.input_ids.size(1))
                    prompt_valid = opt_tokens.attention_mask.sum(dim=1).to(dtype=torch.int64)
                    prompt_valid_mean = float(prompt_valid.float().mean().item())
                    logging.info(
                        "[Len-Diag] "
                        f"max_txt_len={int(self.max_txt_len)} "
                        f"Q={q_len} "
                        f"prompt_padded={prompt_len_padded} "
                        f"prompt_valid_mean={prompt_valid_mean:.2f}"
                    )
                except Exception:
                    pass
            # <<< 

            # Attention mask = [query ones] + [text mask]
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)
            
            if num_beams > 1:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)
            else:
                query_embeds = inputs_opt

            outputs = self.opt_model.generate(
                input_ids=opt_tokens.input_ids,
                query_embeds=query_embeds,
                attention_mask=attention_mask,
                use_cache=True, 
                do_sample=False,
                num_beams=num_beams,
                max_new_tokens=max_len,
                min_length=min_len,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                length_penalty=length_penalty,
            )

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.opt_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )

        return output_text
    
    def _lemmatize(self, answers):
        def apply(answer):
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy
                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer
        
    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        opt_model = cfg.get("opt_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        apply_lemmatizer = cfg.get("apply_lemmatizer", False)

        qformer_sdpa_enabled = bool(cfg.get("qformer_sdpa_enabled", False))
        opt_sdpa_enabled = bool(cfg.get("opt_sdpa_enabled", False))
        opt_quant_mode = str(cfg.get("opt_quant_mode", ""))

        enable_l9_aux_ln = bool(cfg.get("enable_qformer_l9_aux_ln", False))

        old_flag = getattr(cls, "_enable_l9_aux_ln_for_init", None)
        cls._enable_l9_aux_ln_for_init = enable_l9_aux_ln
        try:
            model = cls(
                vit_model=vit_model,
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                use_grad_checkpoint=use_grad_checkpoint,
                vit_precision=vit_precision,
                freeze_vit=freeze_vit,
                num_query_token=num_query_token,
                opt_model=opt_model,
                prompt=prompt,
                max_txt_len=max_txt_len,
                apply_lemmatizer=apply_lemmatizer,
                qformer_sdpa_enabled=qformer_sdpa_enabled,
                opt_sdpa_enabled=opt_sdpa_enabled,
                opt_quant_mode=opt_quant_mode,
            )
        finally:
            if old_flag is None:
                delattr(cls, "_enable_l9_aux_ln_for_init")
            else:
                cls._enable_l9_aux_ln_for_init = old_flag


        mid_override = None
        try:
            mid_override = int(cfg.get("qformer_ffn_mid_override", None))
        except Exception:
            mid_override = None

        rebuilt_by_mids = False
        if mid_override is None:
            pretrain_path = cfg.get("pretrained", None)
            if isinstance(pretrain_path, str) and os.path.isfile(pretrain_path):
                try:
                    try:
                        ckpt = torch.load(pretrain_path, map_location="cpu", mmap=True)
                        logging.info(f"[Slim-Autodetect] memory-mapped checkpoint inspection: {pretrain_path}")
                    except (RuntimeError, TypeError, ValueError) as mmap_error:
                        logging.warning(
                            f"[Slim-Autodetect] mmap unavailable for {pretrain_path}: {mmap_error}; "
                            "falling back to regular torch.load"
                        )
                        ckpt = torch.load(pretrain_path, map_location="cpu")
                    sd = ckpt.get("model", ckpt)  #  {"model": state_dict}  state_dict

                    # Materialized head-slim checkpoints need their Q-Former
                    # structure rebuilt before load_state_dict sees pruned tensors.
                    slim_meta = ckpt.get("slim_meta", {}) if isinstance(ckpt, dict) else {}
                    pruned_by_layer = slim_meta.get("heads_pruned_from_original", {}) or {}
                    if pruned_by_layer:
                        for raw_li, raw_heads in sorted(pruned_by_layer.items(), key=lambda item: int(item[0])):
                            li = int(raw_li)
                            heads = {int(h) for h in raw_heads}
                            if heads:
                                model.Qformer.bert.encoder.layer[li].crossattention.prune_heads(heads)
                        logging.info(
                            "[Slim-Autodetect] rebuilt Q-Former cross-attention heads from slim_meta: %s",
                            {int(k): [int(h) for h in v] for k, v in pruned_by_layer.items()},
                        )

                    try:
                        has_scorer = any(k.endswith("_vision_token_scorer.weight") for k in sd.keys())
                        if has_scorer and getattr(model, "_vision_token_scorer", None) is None and hasattr(model, "enable_learned_vision_token_scorer"):
                            model.enable_learned_vision_token_scorer(init_zero=False)
                            logging.info("[Learned-Scorer] detected in pretrained; module created before loading weights.")
                    except Exception as e:
                        logging.warning(f"[Learned-Scorer] pre-create failed: {e}")

                    mids_by_layer = {}
                    try:
                        nL = len(model.Qformer.bert.encoder.layer)
                    except Exception:
                        nL = 0

                    for li in range(nL):
                        k1 = f"Qformer.bert.encoder.layer.{li}.intermediate_query.dense.weight"
                        k2 = f"Qformer.bert.encoder.layer.{li}.intermediate.dense.weight"
                        key = k1 if k1 in sd else (k2 if k2 in sd else None)
                        if key is None:
                            continue
                        w = sd[key]
                        if hasattr(w, "dim") and w.dim() == 2:
                            mids_by_layer[li] = int(w.shape[0])

                    uniq_mids = sorted(set(mids_by_layer.values())) if mids_by_layer else []

                    if len(uniq_mids) == 1:
                        mid_override = int(uniq_mids[0])
                    elif len(uniq_mids) > 1:
                        model._rebuild_qformer_ffn_to_mids(mids_by_layer)
                        rebuilt_by_mids = True
                        mids_list = [mids_by_layer.get(li, None) for li in range(len(model.Qformer.bert.encoder.layer))]
                        logging.info(f"[Slim-Autodetect] Q-Former FFN per-layer mids = {mids_list}")
                
                    try:
                        del sd
                        del ckpt
                    except Exception:
                        pass
                    import gc as _gc
                    _gc.collect()
                
                except Exception as e:
                    logging.warning(f"[Slim-Autodetect] failed to inspect pretrained FFN mids: {e}")

        if (not rebuilt_by_mids) and isinstance(mid_override, int) and mid_override > 0:
            try:
                model._rebuild_qformer_ffn_to_mid(mid_override)
                logging.info(f"[Slim-Autodetect] Q-Former FFN intermediate_size = {mid_override}")
            except Exception as e:
                logging.warning(f"[Slim-Autodetect] failed to rebuild FFN mid: {e}")

        model.load_checkpoint_from_config(cfg)
        return model


    
        # ==== Q-Former cross-attention head pruning (per-layer) ====

    def _iter_cross_attn_modules(self, qformer=None):
        """Yield (layer_idx, bert_layer.crossattention, bert_layer.crossattention.self) for a given Q-Former.

        Args:
            qformer: optional Q-Former module (default: self.Qformer)
        """
        qformer = self.Qformer if qformer is None else qformer
        modules = []
        for li, layer in enumerate(qformer.bert.encoder.layer):
            if getattr(layer, "has_cross_attention", False):
                ca = layer.crossattention  # BertAttention
                modules.append((li, ca, ca.self))
        return modules

    def _toggle_cross_attn_save(self, flag: bool, qformer=None):
        for _, _, sa in self._iter_cross_attn_modules(qformer=qformer):
            # In Qformer.BertSelfAttention, when is_cross_attention and save_attention,
            # attention_probs are cached and a grad hook is registered.
            sa.save_attention = flag
            if flag:
                if hasattr(sa, "_saved_attn_map"):
                    sa._saved_attn_map = None
                if hasattr(sa, "attention_map"):
                    sa.attention_map = None
                if hasattr(sa, "_attention_map_raw"):
                    sa._attention_map_raw = None
                if hasattr(sa, "attn_gradients"):
                    sa.attn_gradients = None

    # ==== Teacher-based head CKA (runs on small eval/val batches; no recovery/KD required) ====
    def _init_head_cka_teacher(self):
        """Lazily build a frozen Q-Former teacher for head-CKA weighting.

        Teacher weights are loaded from (in priority order):
          1) self._vision_head_cka_teacher_pretrained
          2) self._kd_teacher_pretrained   (re-using existing config key)
        If none provided, will fall back to a frozen copy of current Q-Former.
        """
        if getattr(self, "_head_cka_teacher_ready", False):
            return

        tpath = getattr(self, "_vision_head_cka_teacher_pretrained", None)
        if not tpath:
            tpath = getattr(self, "_kd_teacher_pretrained", None)

        import copy
        teacher_qformer = copy.deepcopy(self.Qformer)
        teacher_qformer.eval()
        for p in teacher_qformer.parameters():
            p.requires_grad_(False)

        try:
            tdev = str(getattr(self, "_vision_head_cka_teacher_device", "same")).lower()
        except Exception:
            tdev = "same"
        try:
            home_device = next(self.Qformer.parameters()).device
        except Exception:
            home_device = torch.device("cpu")

        if tdev in ("cpu", "offload_cpu", "offload", "host"):
            teacher_qformer = teacher_qformer.to("cpu")
            try:
                teacher_qformer = teacher_qformer.float()
            except Exception:
                pass
            self._head_cka_teacher_home_device = torch.device("cpu")
        else:
            teacher_qformer = teacher_qformer.to(home_device)
            self._head_cka_teacher_home_device = home_device

        # Try to load teacher weights if a checkpoint path is provided.
        if tpath:
            try:
                ckpt = torch.load(tpath, map_location="cpu")
                state = ckpt.get("model", ckpt)
                # Load Q-Former submodule weights (strip 'Qformer.' prefix if present)
                qsd = {}
                for k, v in state.items():
                    if k.startswith("Qformer."):
                        qsd[k[len("Qformer."):]] = v
                if qsd:
                    teacher_qformer.load_state_dict(qsd, strict=False)
                else:
                    # Some checkpoints may save Q-Former keys without prefix
                    try:
                        teacher_qformer.load_state_dict({k: v for k, v in state.items() if k.startswith("bert.")}, strict=False)
                    except Exception:
                        pass
            except Exception as _e:
                # Teacher load failure should not break training/eval
                if bool(getattr(self, "_vision_token_evidence_log", False)) or bool(getattr(self, "_vision_token_debug_log", False)):
                    logging.info(f"[Vision-HeadCKA@evidence] failed to load teacher from {tpath}: {_e}")

        self._head_cka_teacher_qformer = teacher_qformer
        self._head_cka_teacher_ready = True

        if bool(getattr(self, "_vision_token_evidence_log", False)) or bool(getattr(self, "_vision_token_debug_log", False)):
            logging.info(f"[Vision-HeadCKA@evidence] teacher_qformer_ready=True from={tpath if tpath else 'frozen_copy'}")

    @torch.no_grad()
    def _compute_head_cka_sims(self, attn_student_bhqk: torch.Tensor, attn_teacher_bhqk: torch.Tensor, eps: float = 1e-6):
        """Compute per-head linear CKA similarity to a teacher distribution.

        Teacher distribution is defined as mean over teacher heads: T_ref = mean_h p_t.
        For each student head h, compute linear CKA between X_h and T_ref, where
        X_h and T_ref are matrices of shape [B*Q, N] (token-prob distributions).
        Returns:
            sims: [H] in [0, 1] (approximately)
        """
        # Normalize to token-probabilities per head
        ps = attn_student_bhqk.clamp_min(eps)
        ps = ps / (ps.sum(dim=-1, keepdim=True) + eps)  # [B,H,Q,N]
        pt = attn_teacher_bhqk.clamp_min(eps)
        pt = pt / (pt.sum(dim=-1, keepdim=True) + eps)  # [B,H,Q,N]

        B, H, Q, N = ps.shape
        M = B * Q

        # Teacher reference distribution: [B,Q,N] -> [M,N]
        y = pt.mean(dim=1).reshape(M, N).float()
        y = y - y.mean(dim=0, keepdim=True)
        yty = y.t().matmul(y)  # [N,N]
        den_y = torch.sqrt((yty * yty).sum().clamp_min(eps))

        sims = []
        for h in range(H):
            x = ps[:, h].reshape(M, N).float()
            x = x - x.mean(dim=0, keepdim=True)
            xty = x.t().matmul(y)
            xtx = x.t().matmul(x)
            num = (xty * xty).sum()
            den_x = torch.sqrt((xtx * xtx).sum().clamp_min(eps))
            sims.append((num / (den_x * den_y + eps)).clamp_min(0.0))
        sims = torch.stack(sims, dim=0)  # [H]
        return sims

    @torch.no_grad()
    def _maybe_update_head_weights_from_teacher(self, *, attn_student_bhqk: torch.Tensor, image_embeds: torch.Tensor, image_atts: torch.Tensor, fwd_id=None):
        """Update per-head weights using CKA similarity to a frozen teacher (small eval/val batch).

        This follows the advisor description: use a small validation batch, compute head similarity
        to teacher distribution via CKA, and update an EMA head-weight vector.
        """
        if not getattr(self, "_use_vision_head_cka", True):
            return
        if attn_student_bhqk is None or (not torch.is_tensor(attn_student_bhqk)):
            return

        cur_split = str(getattr(self, "_vision_head_cka_current_split", getattr(self, "_vision_runtime_split", "")) or "").lower()
        req_split = str(getattr(self, "_vision_head_cka_update_split", "") or "").lower()
        allow_update_flag = getattr(self, "_vision_head_cka_allow_update", None)
        if allow_update_flag is not None and (not bool(allow_update_flag)):
            return
        if req_split not in ("", "all", "any", "*") and cur_split != req_split:
            return

        update_on_train = bool(getattr(self, "_vision_head_cka_update_on_train", False))
        cur_train = bool(self.training)


        step_id = int(getattr(self, "_vision_token_fwd_id", 0) or 0)
        interval = int(getattr(self, "_vision_head_cka_update_interval", 200))
        if interval <= 0:
            interval = 200
        last_state = getattr(self, "_vision_head_cka_last_training", None)
        if last_state is None or bool(last_state) != cur_train:
            setattr(self, "_vision_head_cka_last_training", cur_train)
            if not cur_train:
                setattr(self, "_vision_head_cka_eval_used", 0)

        # 
        if cur_train and (not update_on_train):
            return
        if (not cur_train) and (not bool(getattr(self, "_vision_head_cka_update_on_eval", True))):
            return

        if not cur_train:
            budget = int(getattr(self, "_vision_head_cka_eval_batches", 2))
            used = int(getattr(self, "_vision_head_cka_eval_used", 0))
            if used >= budget:
                return
        else:
            interval = int(getattr(self, "_vision_head_cka_update_interval", 200))
            if interval <= 0:
                interval = 200
            step_id = int(getattr(self, "_vision_token_fwd_id", 0) or 0)
            if (step_id % interval) != 0 and (not bool(getattr(self, "_vision_head_cka_force_update", False))):
                return

        self._init_head_cka_teacher()
        teacher_qformer = getattr(self, "_head_cka_teacher_qformer", None)
        if teacher_qformer is None:
            return

        B, N = int(image_embeds.shape[0]), int(image_embeds.shape[1])
        # Use STUDENT query tokens for alignment across Q
        try:
            home = getattr(self, "_head_cka_teacher_home_device", None)
        except Exception:
            home = None

        try:
            run_device = image_embeds.device
        except Exception:
            try:
                run_device = next(self.Qformer.parameters()).device
            except Exception:
                run_device = torch.device("cpu")

        moved = False
        try:
            tcur = next(teacher_qformer.parameters()).device
        except Exception:
            tcur = run_device

        if tcur != run_device:
            teacher_qformer = teacher_qformer.to(run_device)
            moved = True

        query = self.query_tokens.to(run_device).expand(B, -1, -1)
        image_embeds_t = image_embeds.to(run_device)
        image_atts_t = image_atts.to(run_device)

        # Run teacher Q-Former to collect cross-attn maps
        self._toggle_cross_attn_save(True, qformer=teacher_qformer)
        _ = teacher_qformer.bert(
            query_embeds=query,
            encoder_hidden_states=image_embeds_t,
            encoder_attention_mask=image_atts_t,
            use_cache=False,
            return_dict=True,
        )
        attn_t, _ = self._collect_cross_attn(expect_b=B, expect_n=N, want_grads=False, qformer=teacher_qformer)
        self._toggle_cross_attn_save(False, qformer=teacher_qformer)

        try:
            if moved and (home is not None) and (str(home) == "cpu" or getattr(home, "type", None) == "cpu"):
                teacher_qformer = teacher_qformer.to(home)
        except Exception:
            pass

        self._head_cka_teacher_qformer = teacher_qformer

        if attn_t is None:
            if bool(getattr(self, "_vision_token_evidence_log", False)) or bool(getattr(self, "_vision_token_debug_log", False)):
                logging.info(f"[Vision-HeadCKA@evidence] fwd={fwd_id} teacher_attn=None (skip)")
            return

        sims = self._compute_head_cka_sims(attn_student_bhqk, attn_t)
        # Normalize: keep mean ~ 1 to avoid changing overall scale
        sims = torch.relu(sims)
        mean_val = sims.mean().clamp_min(1e-6)
        sims = sims / mean_val

        beta = float(getattr(self, "_vision_head_cka_momentum", 0.9))
        old = getattr(self, "_vision_head_weights", None)
        if old is None:
            self._vision_head_weights = sims.detach()
        else:
            self._vision_head_weights = beta * old.to(sims.device, dtype=sims.dtype) + (1.0 - beta) * sims.detach()

        if not cur_train:
            try:
                setattr(self, "_vision_head_cka_eval_used", int(getattr(self, "_vision_head_cka_eval_used", 0)) + 1)
            except Exception:
                pass

        if bool(getattr(self, "_vision_token_evidence_log", False)) or bool(getattr(self, "_vision_token_debug_log", False)):
            try:
                w = self._vision_head_weights.detach().float()
                logging.info(
                    f"[Vision-HeadCKA@evidence] fwd={fwd_id} step={step_id} beta={beta:.3f} interval={interval} "
                    f"sims(min/mean/max)={float(sims.min()):.4g}/{float(sims.mean()):.4g}/{float(sims.max()):.4g} "
                    f"head_w(min/mean/max)={float(w.min()):.4g}/{float(w.mean()):.4g}/{float(w.max()):.4g}"
                )
            except Exception as _e:
                logging.info(f"[Vision-HeadCKA@evidence] log failed: {_e}")

    def compute_and_prune_qformer_cross_attn(
        self,
        dataset,
        device,
        *,
        num_batches: int = 50,
        batch_size: int = 4,
        heads_per_layer: int = 2,
        default_text: str = "a photo of",
        num_workers: int = 0,
        collate_fn=None,
        score_mode: str = "taylor",      # "taylor" / "grad*act" / "ablation"
        k_folds: int = 1,
        save_pruned_ckpt: str = None,
        global_head_keep_ratio: float = None,
        min_heads_per_layer: int = 1,
        hard: bool = False,
    ):
        """

        - taylor   : |grad(attn) * attn|
        - grad*act : |grad(y_h) * y_h|, y_h  head 
        """
        _ = bool(hard)

        # 
        if getattr(self, "_qformer_pruned", False):
            return

        # ---------- / loader  tiny loader ----------
        tiny_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        old_use_cache = None
        ckpt_on = False
        old_qformer_gc = None
        try:
            self.eval()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            score_mode = str(score_mode).strip().lower()
            if score_mode == "grad_act":
                score_mode = "grad*act"

            if hasattr(self, "opt_model") and hasattr(self.opt_model, "config"):
                old_use_cache = getattr(self.opt_model.config, "use_cache", None)
                try:
                    self.opt_model.config.use_cache = False
                except Exception:
                    pass

            if hasattr(self, "Qformer") and hasattr(self.Qformer, "config"):
                try:
                    old_qformer_gc = getattr(self.Qformer.config, "gradient_checkpointing", None)
                    if score_mode == "taylor":
                        self.Qformer.config.gradient_checkpointing = False
                except Exception:
                    pass

            if score_mode != "taylor":
                if hasattr(self, "opt_model") and hasattr(self.opt_model, "gradient_checkpointing_enable"):
                    try:
                        self.opt_model.gradient_checkpointing_enable()
                        ckpt_on = True
                    except Exception:
                        pass
            else:
                if hasattr(self, "opt_model") and hasattr(self.opt_model, "gradient_checkpointing_disable"):
                    try:
                        self.opt_model.gradient_checkpointing_disable()
                    except Exception:
                        pass

            global_keep = None if global_head_keep_ratio is None else float(global_head_keep_ratio)
            use_global = global_keep is not None
            min_heads_floor = max(int(min_heads_per_layer), 1)

            #  taylor  attn map + grad
            self._toggle_cross_attn_save(True if score_mode == "taylor" else False)

            modules = self._iter_cross_attn_modules()  # [(li, BertAttention, BertSelfAttention)]
            device0 = device

            if score_mode in ("grad*act", "ablation"):
                self._ensure_runtime_head_mask_api()

            selection_cfg = self._build_qformer_head_selection_cfg(
                modules,
                heads_per_layer=heads_per_layer,
                global_head_keep_ratio=global_keep,
                min_heads_per_layer=min_heads_floor,
            )

            # ---------- K  ----------
            k_folds = max(1, int(k_folds))
            fold_len = max(1, math.ceil(num_batches / k_folds))

            fold_selected = [dict() for _ in range(k_folds)]
            fold_scores_sum = {li: torch.zeros(sa.num_attention_heads, device=device0) for li, _, sa in modules}

            amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()

            def _fmt_scores(x: torch.Tensor):
                try:
                    vals = x.detach().float().cpu().tolist()
                    return "[" + ", ".join(f"{float(v):.6g}" for v in vals) + "]"
                except Exception:
                    return str(x)

            def _rank_scores_for_log(s: torch.Tensor):
                s_det = s.detach()
                if s_det.numel() == 0:
                    return s_det
                if torch.any(s_det < 0):
                    return s_det
                return s_det / (s_det.sum() + 1e-12)

            logging.info(
                "[Q-Former] head-prune start: score_mode=%s, num_batches=%s, batch_size=%s, scope=%s, heads_per_layer=%s, global_keep_ratio=%s, min_heads_per_layer=%s, total_heads=%s, total_prunable=%s, target_total_prune=%s, k_folds=%s",
                str(score_mode), int(num_batches), int(batch_size), selection_cfg["scope"], int(selection_cfg["heads_per_layer"]),
                "None" if selection_cfg["global_head_keep_ratio"] is None else f"{float(selection_cfg['global_head_keep_ratio']):.6g}",
                int(selection_cfg["min_heads_per_layer"]), int(selection_cfg["total_heads"]), int(selection_cfg["total_prunable"]), int(selection_cfg["target_total_prune"]), int(k_folds),
            )

            it = iter(tiny_loader)
            with torch.enable_grad(), amp_ctx:
                for k in range(k_folds):
                    fold_scores = {li: torch.zeros(sa.num_attention_heads, device=device0) for li, _, sa in modules}
                    steps = 0
                    while steps < fold_len:
                        try:
                            samples = next(it)
                        except StopIteration:
                            break

                        for kk, vv in list(samples.items()):
                            if torch.is_tensor(vv):
                                samples[kk] = vv.to(device0, non_blocking=True)

                        if "text_input" not in samples:
                            bs = samples["image"].size(0)
                            samples["text_input"] = [default_text] * bs

                        if score_mode == "taylor":
                            out = self.forward(samples)
                            loss = out["loss"]
                            if hasattr(self, "Qformer"): self.Qformer.zero_grad(set_to_none=True)
                            if hasattr(self, "opt_model"): self.opt_model.zero_grad(set_to_none=True)
                            loss.backward()

                            for li, _, sa in modules:
                                attn = sa.get_attention_map() if hasattr(sa, "get_attention_map") else None
                                grads = sa.get_attn_gradients() if hasattr(sa, "get_attn_gradients") else None
                                if attn is None or grads is None:
                                    continue
                                contrib = (grads * attn).abs().sum(dim=(0, 2, 3))
                                fold_scores[li] += contrib.detach()
                                fold_scores_sum[li] += contrib.detach()

                            del out, loss
                        elif score_mode == "grad*act":
                            for _, _, sa in modules:
                                if hasattr(sa, "clear_head_output_cache"):
                                    sa.clear_head_output_cache()

                            out = self.forward(samples)
                            loss = out["loss"]
                            if hasattr(self, "Qformer"): self.Qformer.zero_grad(set_to_none=True)
                            if hasattr(self, "opt_model"): self.opt_model.zero_grad(set_to_none=True)
                            loss.backward()

                            for li, _, sa in modules:
                                ctx  = sa.get_head_output() if hasattr(sa, "get_head_output") else None
                                gctx = sa.get_head_output_grad() if hasattr(sa, "get_head_output_grad") else None
                                if ctx is None or gctx is None:
                                    logging.warning("[Q-Former][grad*act] layer=%s missing head output/grad; skip this batch.", int(li))
                                    continue
                                if ctx.shape != gctx.shape:
                                    logging.warning(
                                        "[Q-Former][grad*act] layer=%s shape mismatch: ctx=%s gctx=%s; skip this batch.",
                                        int(li), tuple(ctx.shape), tuple(gctx.shape),
                                    )
                                    continue
                                contrib = (gctx * ctx).abs().sum(dim=(0, 2, 3))
                                fold_scores[li]     += contrib.detach()
                                fold_scores_sum[li] += contrib.detach()

                            del out, loss
                        elif score_mode == "ablation":
                            base_out = self.forward(samples)
                            base_loss = base_out["loss"].detach()
                            del base_out

                            for li, _, sa in modules:
                                H = sa.num_attention_heads
                                for h in range(H):
                                    mask = torch.ones(H, device=device0, dtype=torch.float32)
                                    mask[h] = 0.0
                                    sa.set_runtime_head_mask(mask)
                                    masked_out = self.forward(samples)
                                    dloss = (masked_out["loss"] - base_loss).detach()
                                    fold_scores[li][h] += dloss
                                    fold_scores_sum[li][h] += dloss
                                    sa.set_runtime_head_mask(None)
                                    del masked_out
                                sa.set_runtime_head_mask(None)
                        else:
                            raise ValueError(f"Unknown score_mode={score_mode}")

                        steps += 1
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    sel, sel_meta = self._select_qformer_heads_from_scores(
                        modules,
                        fold_scores,
                        heads_per_layer=heads_per_layer,
                        global_head_keep_ratio=global_keep,
                        min_heads_per_layer=min_heads_floor,
                    )

                    for li, _, sa in modules:
                        raw_s = fold_scores[li]
                        rank_s = _rank_scores_for_log(raw_s)
                        target_k = len(sel.get(li, set()))
                        logging.info(
                            "[Q-Former][fold %s/%s] layer=%s H=%s target_k=%s raw_scores=%s rank_scores=%s selected=%s",
                            int(k + 1), int(k_folds), int(li), int(sa.num_attention_heads), int(target_k),
                            _fmt_scores(raw_s), _fmt_scores(rank_s), sorted(list(sel.get(li, set()))),
                        )

                    if selection_cfg["scope"] == "global":
                        flat_sel = sorted((int(li), int(h)) for li, heads in sel.items() for h in heads)
                        logging.info(
                            "[Q-Former][fold %s/%s][global] target_total=%s selected_total=%s min_heads_per_layer=%s selected=%s",
                            int(k + 1), int(k_folds), int(sel_meta["target_total_prune"]), int(sum(len(v) for v in sel.values())), int(sel_meta["min_heads_per_layer"]), flat_sel,
                        )
                    fold_selected[k] = sel

            final_to_prune = {}
            if selection_cfg["scope"] == "global":
                per_fold_global_sets = []
                for k in range(k_folds):
                    flat = set()
                    for li, heads in fold_selected[k].items():
                        for h in heads:
                            flat.add((int(li), int(h)))
                    per_fold_global_sets.append(flat)

                inter_global = None
                for S in per_fold_global_sets:
                    inter_global = S if inter_global is None else (inter_global & S)
                inter_global = inter_global or set()

                inter_dict = {}
                for li, h in sorted(inter_global):
                    inter_dict.setdefault(int(li), set()).add(int(h))

                final_to_prune, fusion_meta = self._select_qformer_heads_from_scores(
                    modules,
                    fold_scores_sum,
                    heads_per_layer=heads_per_layer,
                    global_head_keep_ratio=global_keep,
                    min_heads_per_layer=min_heads_floor,
                    preselected=inter_dict,
                )

                logging.info(
                    "[Q-Former][fusion][global] mode=intersection+fill per_fold_sizes=%s inter_size=%s target_total=%s final_total=%s final=%s",
                    [int(len(s)) for s in per_fold_global_sets], int(len(inter_global)), int(fusion_meta["target_total_prune"]), int(sum(len(v) for v in final_to_prune.values())),
                    sorted((int(li), int(h)) for li, heads in final_to_prune.items() for h in heads),
                )
            else:
                for li, _, sa in modules:
                    per_fold_sets = [fold_selected[k].get(li, set()) for k in range(k_folds)]
                    inter = None
                    for S in per_fold_sets:
                        inter = S if inter is None else (inter & S)

                    H = sa.num_attention_heads
                    k_pick = min(max(int(heads_per_layer), 0), max(H - 1, 0))
                    if inter and len(inter) > 0:
                        final_to_prune[li] = inter
                        fusion_rank_s = _rank_scores_for_log(fold_scores_sum[li])
                        logging.info(
                            "[Q-Former][fusion] layer=%s mode=intersection per_fold=%s final=%s fused_scores=%s",
                            int(li), [sorted(list(s)) for s in per_fold_sets], sorted(list(inter)), _fmt_scores(fusion_rank_s),
                        )
                    else:
                        sel_set, _ = self._select_qformer_heads_from_scores(
                            modules,
                            {li2: fold_scores_sum[li2] for li2, _, _ in modules},
                            heads_per_layer=heads_per_layer,
                            global_head_keep_ratio=None,
                            min_heads_per_layer=min_heads_floor,
                        )
                        final_to_prune[li] = sel_set.get(li, set())
                        fusion_rank_s = _rank_scores_for_log(fold_scores_sum[li])
                        logging.info(
                            "[Q-Former][fusion] layer=%s mode=fallback_topk per_fold=%s final=%s fused_scores=%s",
                            int(li), [sorted(list(s)) for s in per_fold_sets], sorted(list(final_to_prune[li])), _fmt_scores(fusion_rank_s),
                        )

            for li, _, sa in modules:
                chosen = sorted(list(final_to_prune.get(li, set())))
                logging.info(
                    "[Q-Former] Layer %s: H_now=%s, chosen_k=%s, chosen=%s, scope=%s",
                    int(li), int(sa.num_attention_heads), int(len(chosen)), chosen, selection_cfg["scope"],
                )

            self._heads_marked_for_hard_prune = getattr(self, "_heads_marked_for_hard_prune", dict())

            for li, heads in final_to_prune.items():
                if not heads:
                    continue
                layer = self.Qformer.bert.encoder.layer[li]
                sa = layer.crossattention.self
                H = sa.num_attention_heads

                m = torch.ones(H, 1, 1, device=device0, dtype=torch.float32)
                for h in sorted(list(heads)):
                    if 0 <= h < H:
                        m[h, 0, 0] = 0.0

                self._ensure_runtime_head_mask_api()

                prev = sa.get_permanent_head_mask() if hasattr(sa, "get_permanent_head_mask") else None
                if prev is not None:
                    if prev.dim() == 1 and prev.shape[0] == H:
                        prev = prev.view(H, 1, 1)
                    m = torch.minimum(prev.to(m.dtype), m)

                sa.set_permanent_head_mask(m)
                logging.info(
                    "[Q-Former][mask] layer=%s H=%s pruned=%s keep_mask=%s",
                    int(li), int(H), sorted(list(heads)), _fmt_scores(m.view(-1)),
                )

                S = set(self._heads_marked_for_hard_prune.get(li, []))
                S |= set(int(i) for i in heads)
                self._heads_marked_for_hard_prune[li] = sorted(list(S))

                logging.info(f"[Q-Former] Layer {li}: soft-masked {len(heads)} heads (deferred hard-prune at export) -> {sorted(list(heads))}")

            self._qformer_pruned = True

            if save_pruned_ckpt:
                os.makedirs(os.path.dirname(save_pruned_ckpt), exist_ok=True)
                torch.save({"model": self.state_dict()}, save_pruned_ckpt)

        finally:
            try:
                self._toggle_cross_attn_save(False)
            except Exception:
                pass
            if (old_use_cache is not None) and hasattr(self, "opt_model") and hasattr(self.opt_model, "config"):
                try:
                    self.opt_model.config.use_cache = old_use_cache
                except Exception:
                    pass
            if ckpt_on and hasattr(self, "opt_model") and hasattr(self.opt_model, "gradient_checkpointing_disable"):
                try:
                    self.opt_model.gradient_checkpointing_disable()
                except Exception:
                    pass
            try:
                if old_qformer_gc is not None and hasattr(self, "Qformer") and hasattr(self.Qformer, "config"):
                    self.Qformer.config.gradient_checkpointing = old_qformer_gc
            except Exception:
                pass
            self.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_active_qformer_head_indices(self, sa):
        H = int(getattr(sa, "num_attention_heads", 0) or 0)
        if H <= 0:
            return []
        get_perm = getattr(sa, "get_permanent_head_mask", None)
        perm = get_perm() if callable(get_perm) else getattr(sa, "_permanent_head_mask", None)
        if perm is None:
            return list(range(H))
        try:
            pm = perm.detach().float().view(-1)
        except Exception:
            return list(range(H))
        if int(pm.numel()) != H:
            return list(range(H))
        active = [int(i) for i in range(H) if float(pm[i].item()) > 0.5]
        return active if active else list(range(H))

    def _build_qformer_head_selection_cfg(
        self,
        modules,
        *,
        heads_per_layer: int = 2,
        global_head_keep_ratio: float = None,
        min_heads_per_layer: int = 1,
    ):
        total_heads = 0
        total_active = 0
        layer_heads = {}
        layer_active = {}
        for li, _, sa in modules:
            H = int(sa.num_attention_heads)
            active = self._get_active_qformer_head_indices(sa)
            total_heads += H
            total_active += len(active)
            layer_heads[int(li)] = H
            layer_active[int(li)] = [int(h) for h in active if 0 <= int(h) < H]

        already_pruned = max(int(total_heads - total_active), 0)
        cfg = {
            "scope": "per_layer",
            "heads_per_layer": int(heads_per_layer),
            "global_head_keep_ratio": None if global_head_keep_ratio is None else float(global_head_keep_ratio),
            "min_heads_per_layer": max(int(min_heads_per_layer), 1),
            "total_heads": int(total_heads),
            "total_active": int(total_active),
            "already_pruned": int(already_pruned),
            "target_total_prune": int(sum(min(max(int(heads_per_layer), 0), max(len(layer_active[li]) - 1, 0)) for li in layer_heads.keys())),
            "final_target_total_prune": int(sum(min(max(int(heads_per_layer), 0), max(len(layer_active[li]) - 1, 0)) for li in layer_heads.keys())) + int(already_pruned),
            "layer_heads": layer_heads,
            "layer_active": layer_active,
            "layer_capacity": {li: max(len(layer_active[li]) - 1, 0) for li in layer_heads.keys()},
        }

        if global_head_keep_ratio is not None:
            keep = float(global_head_keep_ratio)
            keep = min(max(keep, 0.0), 1.0)
            floor = max(int(min_heads_per_layer), 1)
            capacity = {li: max(len(layer_active[li]) - floor, 0) for li in layer_heads.keys()}
            total_prunable = int(sum(capacity.values()))
            final_target_total = int(round(total_heads * (1.0 - keep)))
            final_target_total = min(max(final_target_total, 0), int(sum(max(H - floor, 0) for H in layer_heads.values())))
            additional_target = max(final_target_total - already_pruned, 0)
            additional_target = min(additional_target, total_prunable)
            cfg.update({
                "scope": "global",
                "global_head_keep_ratio": keep,
                "min_heads_per_layer": floor,
                "total_prunable": total_prunable,
                "target_total_prune": int(additional_target),
                "final_target_total_prune": int(final_target_total),
                "layer_capacity": capacity,
            })
        return cfg

    def _select_qformer_heads_from_scores(
        self,
        modules,
        scores_by_layer,
        *,
        heads_per_layer: int = 2,
        global_head_keep_ratio: float = None,
        min_heads_per_layer: int = 1,
        preselected: dict = None,
    ):
        cfg = self._build_qformer_head_selection_cfg(
            modules,
            heads_per_layer=heads_per_layer,
            global_head_keep_ratio=global_head_keep_ratio,
            min_heads_per_layer=min_heads_per_layer,
        )

        if cfg["scope"] != "global":
            selected = {}
            for li, _, sa in modules:
                active_heads = list(cfg["layer_active"].get(int(li), []))
                k_pick = min(max(int(heads_per_layer), 0), max(len(active_heads) - 1, 0))
                s = scores_by_layer[li].detach()
                if s.numel() == 0 or k_pick <= 0 or not active_heads:
                    selected[int(li)] = set()
                    continue
                ranked = sorted(active_heads, key=lambda h: (float(s[h].detach().cpu()), int(h)))[:k_pick]
                selected[int(li)] = set(int(i) for i in ranked)
            return selected, cfg

        capacity = dict(cfg["layer_capacity"])
        target_total = int(cfg["target_total_prune"])
        selected = {int(li): set() for li, _, _ in modules}
        active_by_layer = {int(li): set(cfg["layer_active"].get(int(li), [])) for li, _, _ in modules}

        if preselected:
            for li, heads in preselected.items():
                li = int(li)
                valid = [int(h) for h in heads if int(h) in active_by_layer.get(li, set())]
                if not valid:
                    continue
                if len(valid) > capacity.get(li, 0):
                    s = scores_by_layer[li].detach()
                    valid = sorted(valid, key=lambda h: (float(s[h].detach().cpu()), int(h)))[:capacity.get(li, 0)]
                selected.setdefault(li, set()).update(valid)

        current_total = sum(len(v) for v in selected.values())
        if current_total >= target_total:
            if current_total > target_total:
                flat_selected = []
                for li, heads in selected.items():
                    s = scores_by_layer[li].detach()
                    for h in heads:
                        flat_selected.append((float(s[h].detach().cpu()), int(li), int(h)))
                flat_selected.sort(key=lambda x: (x[0], x[1], x[2]))
                keep_flat = flat_selected[:target_total]
                selected = {int(li): set() for li, _, _ in modules}
                for _, li, h in keep_flat:
                    selected[li].add(h)
            return selected, cfg

        ranking = []
        for li, _, sa in modules:
            s = scores_by_layer[li].detach()
            for h in cfg["layer_active"].get(int(li), []):
                if h in selected.get(int(li), set()):
                    continue
                ranking.append((float(s[h].detach().cpu()), int(li), int(h)))

        ranking.sort(key=lambda x: (x[0], x[1], x[2]))
        layer_count = {int(li): len(selected.get(int(li), set())) for li, _, _ in modules}

        for _, li, h in ranking:
            if current_total >= target_total:
                break
            if layer_count.get(li, 0) >= capacity.get(li, 0):
                continue
            selected.setdefault(li, set()).add(int(h))
            layer_count[li] = layer_count.get(li, 0) + 1
            current_total += 1

        return selected, cfg

    def _score_heads_on_batch(
self, modules, samples, mode="taylor", head_mask=None):
        """
         {layer_idx: tensor[H]} 
        mode:
        - "taylor": |grad(attn) * attn|  ( backward)
        - "grad*act": |grad(y_h) * y_h|  ( backward; y_h  head )
        """
        mode = str(mode).strip().lower()
        if mode == "grad_act":
            mode = "grad*act"

        scores = {li: torch.zeros(sa.num_attention_heads, device=samples["image"].device)
                for li, _, sa in modules}

        if mode in ("grad*act", "ablation"):
            self._ensure_runtime_head_mask_api()

        if mode == "taylor":
            out = self.forward(samples)
            loss = out["loss"]
            if hasattr(self, "Qformer"): self.Qformer.zero_grad(set_to_none=True)
            if hasattr(self, "opt_model"): self.opt_model.zero_grad(set_to_none=True)
            loss.backward()

            for li, _, sa in modules:
                attn = sa.get_attention_map() if hasattr(sa, "get_attention_map") else None
                grads = sa.get_attn_gradients() if hasattr(sa, "get_attn_gradients") else None
                if attn is None or grads is None: continue
                contrib = (grads * attn).abs().sum(dim=(0, 2, 3))
                scores[li] += contrib.detach()

        elif mode == "grad*act":
            for _, _, sa in modules:
                if hasattr(sa, "clear_head_output_cache"):
                    sa.clear_head_output_cache()

            out = self.forward(samples)
            loss = out["loss"]
            if hasattr(self, "Qformer"): self.Qformer.zero_grad(set_to_none=True)
            if hasattr(self, "opt_model"): self.opt_model.zero_grad(set_to_none=True)
            loss.backward()

            for li, _, sa in modules:
                ctx  = sa.get_head_output() if hasattr(sa, "get_head_output") else None
                gctx = sa.get_head_output_grad() if hasattr(sa, "get_head_output_grad") else None
                if ctx is None or gctx is None or ctx.shape != gctx.shape:
                    continue
                contrib = (gctx * ctx).abs().sum(dim=(0, 2, 3))
                scores[li] += contrib.detach()

        elif mode == "ablation":
            #  base loss
            base_out = self.forward(samples)
            base_loss = base_out["loss"].detach()
            del base_out

            for li, ca, sa in modules:
                H = sa.num_attention_heads
                for h in range(H):
                    mask = torch.ones(H, device=base_loss.device, dtype=torch.float32)
                    mask[h] = 0.0
                    if hasattr(sa, "set_runtime_head_mask"):
                        sa.set_runtime_head_mask(mask)
                    out = self.forward(samples)
                    dloss = (out["loss"] - base_loss).detach()
                    scores[li][h] += dloss
                    if hasattr(sa, "set_runtime_head_mask"):
                        sa.set_runtime_head_mask(None)
                    del out
                # 
                if hasattr(sa, "set_runtime_head_mask"):
                    sa.set_runtime_head_mask(None)
        else:
            raise ValueError(f"Unknown score mode: {mode}")

        return scores

    def _iter_qformer_ffn_modules(self):
        """
        Yield (li, layer, inter_dense, out_dense, act_fn, intermediate_dim) **FOR QUERY FLOW ONLY**.
        """
        qf = getattr(self, "Qformer", None)
        if qf is None or not hasattr(qf, "bert"):
            return
        for li, layer in enumerate(qf.bert.encoder.layer):
            if hasattr(layer, "intermediate_query") and hasattr(layer, "output_query"):
                inter_dense = layer.intermediate_query.dense
                out_dense = layer.output_query.dense
                act_fn = layer.intermediate_query.intermediate_act_fn  # GELU
                inter_dim = inter_dense.out_features
                yield li, layer, inter_dense, out_dense, act_fn, inter_dim

    def global_prune_qformer_ffn_by_ratio(
        self,
        keep_ratio: float = None,
        group_size: int = 16,
        min_keep_groups_per_layer: int = 1,
        *,
        shallow_layers=None,
        shallow_min_ratio: float = None,
        rest_keep_ratio: float = None
    ):
        """
        """
        from omegaconf import ListConfig

        gs = max(1, int(group_size))
        nL_total = len(self.Qformer.bert.encoder.layer)

        # 
        shallow_set = set()
        if shallow_layers is not None:
            if isinstance(shallow_layers, int):
                shallow_set = set(i for i in range(min(shallow_layers, nL_total)))
            elif isinstance(shallow_layers, (list, tuple, set, ListConfig)):
                shallow_set = set(int(i) for i in shallow_layers if int(i) < nL_total)
            else:
                try:
                    parts = [s.strip() for s in str(shallow_layers).split(',') if s.strip()!='']
                    shallow_set = set(int(x) for x in parts if int(x) < nL_total)
                except Exception:
                    shallow_set = set()

        if not hasattr(self, "_ffn_gates"):
            logging.warning("[FFN-GlobalPrune] no gates found; call attach_qformer_ffn_gates() first.")
            return
        if not hasattr(self, "_ffn_gate_scores") or self._ffn_gate_scores is None:
            self._ffn_gate_scores = [None] * nL_total

        #  group 
        layers_info = []     # (li, dmid, G)
        group_scores = {}    # li -> [G]
        total_groups, total_groups_rest = 0, 0

        for li, layer, inter, out, act, dmid in self._iter_qformer_ffn_modules():
            dmid = int(dmid)
            if dmid <= 0:
                continue
            G = (dmid + gs - 1) // gs
            total_groups += G
            if li not in shallow_set:
                total_groups_rest += G

            if self._ffn_gates is not None and li < len(self._ffn_gates) and self._ffn_gates[li] is not None:
                gparam = self._ffn_gates[li].detach()
                scores = torch.sigmoid(gparam).float()
                self._ffn_gate_scores[li] = scores.detach()
            elif self._ffn_gate_scores[li] is not None:
                scores = self._ffn_gate_scores[li].detach().float()
            else:
                with torch.no_grad():
                    w = inter.weight.detach()
                    energy = w.pow(2).sum(dim=1)
                    scores = energy.view(G, gs).mean(dim=1)

            if scores.numel() != G:
                scores = scores.view(-1)[:G]
                if scores.numel() < G:
                    pad = torch.zeros(G - scores.numel(), device=scores.device, dtype=scores.dtype)
                    scores = torch.cat([scores, pad], dim=0)

            group_scores[li] = scores
            layers_info.append((li, dmid, G))

        if total_groups == 0:
            logging.warning("[FFN-GlobalPrune] no FFN groups found; skip.")
            return

        min_keep_groups = max(1, int(min_keep_groups_per_layer))
        keep_groups = {li: set() for (li, _, _) in layers_info}

        # === A)  ===
        if shallow_min_ratio is not None and len(shallow_set) > 0:
            r = float(shallow_min_ratio)
            if not (0.0 < r <= 1.0):
                logging.warning(f"[FFN-GlobalPrune] shallow_min_ratio out of range: {r}, ignored.")
                r = None
            if r is not None:
                for (li, dmid, G) in layers_info:
                    if li in shallow_set:
                        k_floor = max(min_keep_groups, int(math.ceil(G * r)))
                        k_floor = min(k_floor, G)
                        scores = group_scores[li]
                        topk = torch.topk(scores, k=k_floor, largest=True, sorted=False).indices.tolist()
                        keep_groups[li].update(int(g) for g in topk)

        selected = sum(len(s) for s in keep_groups.values())

        def fill_by_global_ranking(target_total, candidate_layers):
            remaining = max(0, int(target_total) - selected)
            cand = []
            for (li, dmid, G) in layers_info:
                if li not in candidate_layers:
                    continue
                scores = group_scores[li]
                for gj in range(G):
                    if gj in keep_groups[li]:
                        continue
                    cand.append((float(scores[gj].item()), li, gj))
            cand.sort(key=lambda x: x[0], reverse=True)
            c = 0
            for score, li, gj in cand:
                if c >= remaining:
                    break
                keep_groups[li].add(gj)
                c += 1
            return c

        if rest_keep_ratio is not None:
            rk = float(rest_keep_ratio)
            rk = min(max(rk, 0.0), 1.0)
            keep_rest_total = int(math.floor(total_groups_rest * rk))
            for (li, dmid, G) in layers_info:
                if li in shallow_set:
                    continue
                k_local = min(min_keep_groups, G)
                if k_local > 0:
                    scores = group_scores[li]
                    topk = torch.topk(scores, k=k_local, largest=True, sorted=False).indices.tolist()
                    keep_groups[li].update(int(g) for g in topk)
            selected = sum(len(s) for s in keep_groups.values())
            _ = fill_by_global_ranking(
                keep_rest_total + sum(len(keep_groups[li]) for li in shallow_set),
                set(li for (li,_,_) in layers_info if li not in shallow_set)
            )
        else:
            try:
                kr = float(keep_ratio)
            except Exception:
                logging.warning(f"[FFN-GlobalPrune] invalid keep_ratio={keep_ratio}, skip.")
                return
            if not (0.0 < kr <= 1.0):
                logging.warning(f"[FFN-GlobalPrune] keep_ratio out of range: {kr}, skip.")
                return
            keep_total = int(math.floor(total_groups * kr))
            # 
            for (li, dmid, G) in layers_info:
                scores = group_scores[li]
                k_local = min(min_keep_groups, G)
                if k_local > 0:
                    topk = torch.topk(scores, k=k_local, largest=True, sorted=False).indices.tolist()
                    keep_groups[li].update(int(g) for g in topk)
            selected = sum(len(s) for s in keep_groups.values())
            # 
            _ = fill_by_global_ranking(keep_total, set(li for (li,_,_) in layers_info))

        # === C) keep_groups -> mids ===
        mids = []
        for li_idx in range(nL_total):
            tup = next(((li,d,G) for (li,d,G) in layers_info if li==li_idx), None)
            if tup is None:
                mids.append(0)
                continue
            li, dmid, G = tup
            kept = len(keep_groups.get(li, set()))
            kept = max(1, min(kept, G))
            mid = min(dmid, kept * gs)
            mids.append(int(mid))

        logging.info(f"[FFN-GlobalPrune] shallow_set={sorted(list(shallow_set)) if len(shallow_set)>0 else []}, "
                     f"shallow_min_ratio={shallow_min_ratio}, rest_keep_ratio={rest_keep_ratio}, "
                     f"mids={mids}")

        self.finalize_qformer_ffn_to_mid(mids, group_size=gs)
        logging.info("[FFN-GlobalPrune] finalize_qformer_ffn_to_mid done.")


    def compute_and_prune_qformer_ffn(
        self,
        dataset,
        device,
        *,
        mode: str = "taylor",         # "taylor" or "ablation"
        group_size: int = 16,
        target_sparsity: float = 0.2,
        num_batches: int = 8,
        batch_size: int = 1,
        default_text: str = "a photo of",
        num_workers: int = 0,
        collate_fn=None,
    ):
        """
        """

        assert 0.0 < target_sparsity < 1.0, "target_sparsity must be in (0,1)"
        self.eval()

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        ffn_scores = {}
        ffn_hooks = []
        saved_acts = {}

        def _make_hooks(li):
            def fwd_hook(module, inp, out):
                saved_acts[li] = out.detach()  # [B, L, D_mid]
            def bwd_hook(grad_out):
                # grad_out: [B, L, D_mid]
                act = saved_acts.pop(li, None)
                if act is not None:
                    contrib = (grad_out * act).abs().sum(dim=(0,1))  # [D_mid]
                    ffn_scores[li] = ffn_scores.get(li, 0) + contrib
            return fwd_hook, bwd_hook

        ablate_mask = {}

        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            ffn_scores[li] = 0
            if mode == "taylor":
                def _pre_out_fwd(module, inp, out):
                    # inp: tuple(tensor[B,L,D_mid]),  out_dense 
                    saved_acts[li] = inp[0].detach()
                def _pre_out_bwd(module, grad_in, grad_out):
                    act = saved_acts.pop(li, None)
                    if act is not None and grad_in[0] is not None:
                        contrib = (grad_in[0] * act).abs().sum(dim=(0,1))
                        ffn_scores[li] = ffn_scores.get(li, 0) + contrib.detach()
                ffn_hooks.append(out_dense.register_forward_hook(_pre_out_fwd))
                ffn_hooks.append(out_dense.register_full_backward_hook(_pre_out_bwd))

            elif mode == "ablation":
                ablate_mask[li] = None
                def _mask_fwd(module, inp, out):
                    mask = ablate_mask[li]
                    if mask is None:
                        return out
                    # out:[B,L,D_mid] * mask:[D_mid]
                    return out * mask.view(1,1,-1)
                ffn_hooks.append(act_fn.register_forward_hook(_mask_fwd))
            else:
                raise ValueError(f"Unknown mode: {mode}")

        # 
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
        steps = 0
        for step, samples in enumerate(dl):
            if step >= num_batches:
                break
            for k,v in list(samples.items()):
                if hasattr(torch, "is_tensor") and torch.is_tensor(v):
                    samples[k] = v.to(device, non_blocking=True)
            if "text_input" not in samples:
                bs = samples["image"].size(0)
                samples["text_input"] = [default_text] * bs

            # /
            if mode == "taylor":
                self.zero_grad(set_to_none=True)
                with amp_ctx:
                    out = self.forward(samples)   # {'loss': loss}
                    loss = out["loss"]
                loss.backward()
            else:  # ablation
                with torch.no_grad():
                    out_base = self.forward(samples)
                    base_loss = out_base["loss"].item()
                for li, _, _, _, _, dmid in self._iter_qformer_ffn_modules():
                    g = max(1, group_size)
                    G = (dmid + g - 1)//g
                    for gi in range(G):
                        mask = torch.ones(dmid, device=device, dtype=torch.float32)
                        beg = gi*g
                        end = min((gi+1)*g, dmid)
                        mask[beg:end] = 0.0
                        ablate_mask[li] = mask
                        with amp_ctx:
                            out2 = self.forward(samples)
                            loss2 = out2["loss"].item()
                        delta = max(0.0, loss2 - base_loss)
                        if dmid not in (0,):
                            if isinstance(ffn_scores[li], int):
                                ffn_scores[li] = torch.zeros(dmid, device=device)
                            ffn_scores[li][beg:end] += delta
                    ablate_mask[li] = None

            steps += 1
            # 
            if mode == "taylor":
                self.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # hook
        for h in ffn_hooks:
            try: h.remove()
            except: pass

        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            if isinstance(ffn_scores[li], int):
                continue
            scores = ffn_scores[li]
            # 
            scores = scores / (scores.sum() + 1e-12)

            #  + 
            g = max(1, group_size)
            G = (dmid + g - 1)//g
            group_scores = []
            for gi in range(G):
                beg = gi*g
                end = min((gi+1)*g, dmid)
                group_scores.append((gi, scores[beg:end].mean().item()))
            group_scores.sort(key=lambda x: x[1])  # 
            k_groups = max(0, int(math.floor(G * target_sparsity)))
            if k_groups == 0:
                continue
            pruned_groups = [gi for gi,_ in group_scores[:k_groups]]

            keep_mask = torch.ones(dmid, dtype=torch.bool, device=device)
            for gi in pruned_groups:
                beg = gi * g
                end = min((gi + 1) * g, dmid)
                keep_mask[beg:end] = False  # 
            keep = keep_mask.nonzero(as_tuple=False).view(-1).cpu()

            # 1) intermediate.dense: [out=D_mid, in=D_in] ->  out 
            W1 = inter_dense.weight.data
            b1 = inter_dense.bias.data if inter_dense.bias is not None else None
            W1_new = W1[keep, :]
            b1_new = b1[keep] if b1 is not None else None

            # 2) output.dense: [out=D_out, in=D_mid] ->  in 
            W2 = out_dense.weight.data
            b2 = out_dense.bias.data if out_dense.bias is not None else None
            W2_new = W2[:, keep]

            new_inter = nn.Linear(inter_dense.in_features, W1_new.size(0), bias=b1_new is not None).to(W1_new.device)
            new_inter.weight.data.copy_(W1_new)
            if b1_new is not None: new_inter.bias.data.copy_(b1_new)

            new_out = nn.Linear(W2_new.size(1), W2_new.size(0), bias=b2 is not None).to(W2_new.device)
            new_out.weight.data.copy_(W2_new)
            if b2 is not None: new_out.bias.data.copy_(b2)

            layer.intermediate_query.dense = new_inter
            layer.output_query.dense = new_out
            logging.info(f"[Q-Former][FFN] Layer {li}: pruned groups={len(pruned_groups)} "
                        f"({k_groups}/{G}), mid {dmid} -> {W1_new.size(0)}")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _rebuild_qformer_ffn_to_mid(self, target_mid: int):
        """
        """
        tgt = int(target_mid)
        assert tgt > 0, "target_mid must be positive integer"

        n_changed = 0
        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            if dmid == tgt:
                continue

            # 
            in1 = int(inter_dense.in_features)      #  768
            out2 = int(out_dense.out_features)      #  768

            new_inter = nn.Linear(in1, tgt, bias=(inter_dense.bias is not None),
                                device=inter_dense.weight.device, dtype=inter_dense.weight.dtype)
            new_out   = nn.Linear(tgt, out2, bias=(out_dense.bias is not None),
                                device=out_dense.weight.device, dtype=out_dense.weight.dtype)

            layer.intermediate_query.dense = new_inter
            layer.output_query.dense       = new_out
            n_changed += 1
            logging.info(f"[Q-Former][FFN] Layer {li}: rebuild mid {dmid} -> {tgt}")

        if n_changed == 0:
            logging.info("[Q-Former][FFN] all layers already match target mid; no rebuild performed.")

    def _rebuild_qformer_ffn_to_mids(self, mids_by_layer: dict, default_mid=None):
        """
        - mids_by_layer: {layer_idx: target_mid}
        """
        n_changed = 0
        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            tgt = mids_by_layer.get(li, default_mid)
            if tgt is None:
                continue
            tgt = int(tgt)
            if tgt <= 0 or tgt == dmid:
                continue

            in1 = int(inter_dense.in_features)
            out2 = int(out_dense.out_features)

            new_inter = nn.Linear(
                in1,
                tgt,
                bias=(inter_dense.bias is not None),
                device=inter_dense.weight.device,
                dtype=inter_dense.weight.dtype,
            )
            new_out = nn.Linear(
                tgt,
                out2,
                bias=(out_dense.bias is not None),
                device=out_dense.weight.device,
                dtype=out_dense.weight.dtype,
            )

            layer.intermediate_query.dense = new_inter
            layer.output_query.dense = new_out
            n_changed += 1
            logging.info(f"[Q-Former][FFN] Layer {li}: rebuild mid {dmid} -> {tgt}")

        if n_changed == 0:
            logging.info("[Q-Former][FFN] _rebuild_qformer_ffn_to_mids: no layer needs rebuild.")

    # Enable the optional learnable token scorer.
    def enable_learned_vision_token_scorer(self, init_zero=True):
        if getattr(self, "_vision_token_scorer", None) is not None:
            return True

        C = self.visual_encoder.num_features
        self._vision_token_scorer = nn.Linear(
            C, 1, bias=False,
            dtype=next(self.parameters()).dtype,
            device=next(self.parameters()).device
        )
        if init_zero:
            nn.init.zeros_(self._vision_token_scorer.weight)
        return True

    @torch.no_grad()
    # =========================
    # Text-KV learned scorer ()
    # =========================
    def enable_learned_text_kv_scorer(self, init_zero: bool = True):

        if getattr(self, "_text_kv_scorer", None) is not None:
            return True

        hidden_size = 0
        try:
            cfg = getattr(self.opt_model, "config", None)
            hidden_size = int(getattr(cfg, "hidden_size", 0) or 0)
            if hidden_size <= 0:
                hidden_size = int(getattr(cfg, "word_embed_proj_dim", 0) or 0)
        except Exception:
            hidden_size = 0

        if hidden_size <= 0:
            try:
                emb = self.opt_model.get_input_embeddings()
                if emb is not None and hasattr(emb, "weight") and torch.is_tensor(emb.weight):
                    hidden_size = int(emb.weight.shape[1])
            except Exception:
                hidden_size = 0

        if hidden_size <= 0:
            raise ValueError("[Text-KV] cannot infer OPT hidden_size for learned scorer.")

        self._text_kv_scorer = nn.Linear(hidden_size, 1, bias=False)
        setattr(self, "_text_kv_scorer_ckpt_loaded", False)

        if init_zero:
            try:
                nn.init.zeros_(self._text_kv_scorer.weight)
            except Exception:
                pass

        # / cfg.score_mode='learned' 
        setattr(self, "_text_kv_score_mode_default", "learned")
        return True

    def freeze_backbone_for_text_kv_scorer(self):

        # 1) 
        for _, p in self.named_parameters():
            p.requires_grad = False

        # 2)  scorer
        scorer = getattr(self, "_text_kv_scorer", None)
        if scorer is None:
            raise ValueError("[Text-KV] scorer not initialized. Call enable_learned_text_kv_scorer() first.")
        for p in scorer.parameters():
            p.requires_grad = True

        self._text_kv_scorer_only = True
        try:
            self.opt_model.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "visual_encoder") and self.visual_encoder is not None:
                self.visual_encoder.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "Qformer") and self.Qformer is not None:
                self.Qformer.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "opt_proj") and self.opt_proj is not None:
                self.opt_proj.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "query_tokens") and self.query_tokens is not None and torch.is_tensor(self.query_tokens):
                self.query_tokens.requires_grad = False
        except Exception:
            pass

        scorer.train()
        return True

    def freeze_backbone_for_query_opt_proj(self):

        # 1) 
        for _, p in self.named_parameters():
            p.requires_grad = False

        # 2)  query_tokens
        has_query_tokens = False
        try:
            if hasattr(self, "query_tokens") and self.query_tokens is not None and torch.is_tensor(self.query_tokens):
                self.query_tokens.requires_grad = True
                has_query_tokens = True
        except Exception:
            has_query_tokens = False

        # 3)  opt_proj
        has_opt_proj = False
        try:
            if hasattr(self, "opt_proj") and self.opt_proj is not None:
                for p in self.opt_proj.parameters():
                    p.requires_grad = True
                has_opt_proj = True
        except Exception:
            has_opt_proj = False

        if not has_query_tokens:
            raise ValueError("[Query-OPTProj] query_tokens not found.")
        if not has_opt_proj:
            raise ValueError("[Query-OPTProj] opt_proj not found.")

        # 4) 
        self._query_opt_proj_only = True
        #  only 
        self._text_kv_scorer_only = False

        try:
            self.opt_model.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "visual_encoder") and self.visual_encoder is not None:
                self.visual_encoder.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "ln_vision") and self.ln_vision is not None:
                self.ln_vision.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "Qformer") and self.Qformer is not None:
                self.Qformer.eval()
        except Exception:
            pass
        try:
            if hasattr(self, "opt_proj") and self.opt_proj is not None:
                self.opt_proj.train()
        except Exception:
            pass

        return True

    def train(self, mode: bool = True):
        super().train(mode)

        # Matched-recovery protocol: token filtering is a training-only
        # condition. Evaluation and checkpoint selection always use 677 tokens.
        if getattr(self, "_vision_token_train_only", False):
            keep = (
                getattr(self, "_vision_token_train_keep_ratio", 1.0)
                if mode
                else 1.0
            )
            self._vision_token_keep_ratio = float(keep)

        if getattr(self, "_text_kv_scorer_only", False):
            # backbone  eval
            try:
                self.opt_model.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "visual_encoder") and self.visual_encoder is not None:
                    self.visual_encoder.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "ln_vision") and self.ln_vision is not None:
                    self.ln_vision.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "Qformer") and self.Qformer is not None:
                    self.Qformer.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "opt_proj") and self.opt_proj is not None:
                    self.opt_proj.eval()
            except Exception:
                pass
            # scorer  train / eval  mode
            try:
                scorer = getattr(self, "_text_kv_scorer", None)
                if scorer is not None:
                    scorer.train(mode)
            except Exception:
                pass

        elif getattr(self, "_query_opt_proj_only", False):
            # backbone  eval
            try:
                self.opt_model.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "visual_encoder") and self.visual_encoder is not None:
                    self.visual_encoder.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "ln_vision") and self.ln_vision is not None:
                    self.ln_vision.eval()
            except Exception:
                pass
            try:
                if hasattr(self, "Qformer") and self.Qformer is not None:
                    self.Qformer.eval()
            except Exception:
                pass

            #  opt_proj  train / eval
            try:
                if hasattr(self, "opt_proj") and self.opt_proj is not None:
                    self.opt_proj.train(mode)
            except Exception:
                pass

            try:
                if hasattr(self, "query_tokens") and self.query_tokens is not None and torch.is_tensor(self.query_tokens):
                    self.query_tokens.requires_grad = True
            except Exception:
                pass

        return self

    def _select_vision_tokens(
        self,
        feats_bnc: torch.Tensor,          # [B, N, C]
        image_atts: torch.Tensor,         # [B, N]
        *,
        attn_bhqk: Optional[torch.Tensor] = None,   # [B, H, Q, N], optional
        attn_grad_bhqk: Optional[torch.Tensor] = None,
        mode: str = "l2norm",             # "l2norm" | "random" | "attn_entropy" | "attn_grad" | "learned"
        keep_ratio: float = 1.0,
        tail_strategy: str = "drop",      # "drop" | "merge" | "tail_token"
    ):

        B, N, C = feats_bnc.shape
        keep_ratio = float(max(0.0, min(1.0, keep_ratio)))
        K = max(1, int(round(N * keep_ratio)))

        diag_usage = getattr(self, "_diag_vtoken_usage_enabled", False)
        if diag_usage:
            st = getattr(self, "_vision_token_usage_stats", None)
            if not isinstance(st, dict):
                st = {"total_raw_tokens": 0, "total_used_tokens": 0, "calls": 0}
            st["total_raw_tokens"] += int(B * N)
            if K >= N:
                st["total_used_tokens"] += int(B * N)
                st["calls"] += 1
            self._vision_token_usage_stats = st

        if K >= N:
            return feats_bnc, image_atts  # no pruning


        # 1) get token scores
        scores = torch.zeros(B, N, device=feats_bnc.device, dtype=feats_bnc.dtype)

        attn_from_cache = False
        if attn_bhqk is None:
            attn_bhqk = getattr(self, "_last_xattn", None)  # [B,H,Q,N] or None
            attn_from_cache = True

        if attn_grad_bhqk is not None:
            attn_grad = attn_grad_bhqk
        elif attn_from_cache:
            attn_grad = getattr(self, "_last_xattn_grad", None)
        else:
            attn_grad = None

        head_w_eff = None
        head_w = getattr(self, "_vision_head_weights", None)
        if (attn_bhqk is not None) and (head_w is not None):
            hw = head_w
            if hw.dim() == 1:
                # [H] -> [1,H,1,1]
                hw = hw.view(1, -1, 1, 1)
            elif hw.dim() == 2:
                hw = hw.mean(dim=0, keepdim=True).view(1, -1, 1, 1)
            head_w_eff = hw.to(attn_bhqk.device, dtype=attn_bhqk.dtype)
        mode = str(mode).lower()
        if mode in ("attn_entropy", "attn", "entropy"):
            if attn_bhqk is not None:
                B2, H, Q, N2 = attn_bhqk.shape
                if (B2 != B) or (N2 != N):
                    attn_bhqk = None

            if attn_bhqk is not None:
                eps = 1e-12
                p = attn_bhqk.clamp_min(eps)
                #  head  token 
                p = p / (p.sum(dim=-1, keepdim=True) + eps)   # [B,H,Q,N]

                if head_w_eff is not None:
                    denom = head_w_eff.sum(dim=1, keepdim=False).clamp_min(eps)  # [1,1,1]
                    p_bqk = (p * head_w_eff).sum(dim=1) / denom                  # [B,Q,N]
                else:
                    p_bqk = p.mean(dim=1)       

                #  Query  H_q = -sum_k p log p
                H_bq = -(p_bqk * (p_bqk.clamp_min(eps).log())).sum(dim=-1)  # [B,Q]

                wq = torch.softmax(-H_bq, dim=-1)             # [B,Q]
                if Q > 0:
                    base = 1.0 / float(Q)
                    w_min = 0.5 * base
                    w_max = 1.5 * base
                    wq = wq.clamp(min=w_min, max=w_max)
                    wq = wq / wq.sum(dim=-1, keepdim=True).clamp_min(eps)

                #  token  [B,N]
                scores = (wq.unsqueeze(-1) * p_bqk).sum(dim=1)  # [B,N]

            else:
                scores = feats_bnc.norm(p=2, dim=-1)

        elif mode in ("attn_grad", "grad", "taylor_attn"):
            if attn_bhqk is not None:
                B2, H, Q, N2 = attn_bhqk.shape
                if (B2 != B) or (N2 != N):
                    attn_bhqk = None

            if (attn_bhqk is not None) and (attn_grad is not None):
                scores = (attn_bhqk * attn_grad).abs().sum(dim=(1, 2))  # [B,N]
            else:
                scores = feats_bnc.norm(p=2, dim=-1)

        # === 3) deterministic random-token control ===
        elif mode in ("random", "rand"):
            gen = getattr(self, "_vision_token_random_generator", None)
            gen_device = getattr(self, "_vision_token_random_generator_device", None)
            if gen is None or gen_device != feats_bnc.device:
                gen = torch.Generator(device=feats_bnc.device)
                gen.manual_seed(int(getattr(self, "_vision_token_random_seed", 42)))
                self._vision_token_random_generator = gen
                self._vision_token_random_generator_device = feats_bnc.device
            scores = torch.rand(B, N, device=feats_bnc.device, dtype=feats_bnc.dtype, generator=gen)

        elif mode in ("learned", "mlp", "score_net"):
            scorer = getattr(self, "_vision_token_scorer", None)
            if scorer is not None:
                scores = scorer(feats_bnc).squeeze(-1)  # [B,N]
            else:
                scores = feats_bnc.norm(p=2, dim=-1)

        else:  # "l2norm" and fallback
            scores = feats_bnc.norm(p=2, dim=-1)


        # 2) per-sample top-K 
        topk = torch.topk(scores, k=K, dim=1, largest=True, sorted=True)  # idx: [B,K]
        idx = topk.indices  # [B,K]
        try:
            self._last_vtoken_keep_idx = idx.detach()
            self._last_vtoken_full_N = int(N)
            self._last_vtoken_keep_K = int(K)
            self._last_vtoken_keep_ratio = float(keep_ratio)
            self._last_vtoken_tail_strategy = str(tail_strategy)
            self._last_vtoken_has_tail_token = (str(tail_strategy).lower() == "tail_token" and K < N)
        except Exception:
            pass

        try:
            #  tail  attn_grad  attn_bhqk 
            if str(tail_strategy).lower() in ("merge", "tail_token") and attn_bhqk is not None and attn_grad is not None:
                if attn_bhqk.shape == attn_grad.shape:
                    contrib = (attn_bhqk * attn_grad).abs().sum(dim=(1, 2))  # [B, N]
                    B_idx = torch.arange(B, device=idx.device).view(B, 1)
                    keep_mask = torch.zeros_like(contrib, dtype=torch.bool)
                    keep_mask[B_idx, idx] = True

                    keep_sum = contrib[keep_mask].sum().item()
                    tail_sum = contrib[~keep_mask].sum().item()
                    total = keep_sum + tail_sum

                    if total > 0.0:
                        keep_frac = keep_sum / total
                        tail_frac = tail_sum / total

                        st = getattr(self, "_tail_grad_profile", None)
                        if not isinstance(st, dict):
                            st = {"keep": 0.0, "tail": 0.0, "count": 0}
                        st["keep"] += float(keep_frac)
                        st["tail"] += float(tail_frac)
                        st["count"] += 1
                        self._tail_grad_profile = st
        except Exception:
            # 
            pass


        # gather feats & atts
        gather_idx = idx.unsqueeze(-1).expand(B, K, C)
        new_feats = torch.gather(feats_bnc, dim=1, index=gather_idx)      # [B,K,C]
        new_atts  = torch.gather(image_atts, dim=1, index=idx) 

        tail_strategy = str(tail_strategy).lower()
        if tail_strategy in ("merge", "tail_token") and K < N:
            mask_all = torch.ones(B, N, device=feats_bnc.device, dtype=torch.bool)
            mask_all.scatter_(1, idx, False)

            scores_detach = scores.detach()

            if tail_strategy == "tail_token":
                tail_agg = feats_bnc.new_zeros((B, C))
                tail_flag = torch.zeros(B, device=feats_bnc.device, dtype=torch.bool)

            for b in range(B):
                tail_mask = mask_all[b]
                if not tail_mask.any():
                    continue

                tail_feats = feats_bnc[b, tail_mask]      # [T,C]
                T = tail_feats.size(0)
                if T <= 0:
                    continue

                # (a)  + std 
                mu = tail_feats.mean(dim=0, keepdim=True)
                centered = tail_feats - mu
                var = centered.pow(2).mean(dim=0, keepdim=True)
                std = (var + 1e-6).sqrt()
                normed = centered / std                    # [T,C]

                tail_scores = scores_detach[b, tail_mask]  # [T]
                w = tail_scores.clamp_min(0)
                if float(w.sum()) <= 0.0:
                    w = torch.ones_like(w)
                w = w / w.sum()

                merged = (w.view(-1, 1) * normed).sum(dim=0, keepdim=True)  # [1,C]

                scale = torch.log1p(torch.tensor(float(T), device=merged.device, dtype=merged.dtype))
                merged = merged * scale

                if tail_strategy == "merge":
                    new_feats[b, -1, :] = 0.5 * new_feats[b, -1, :] + 0.5 * merged[0]
                    new_atts[b, -1] = 1
                else:
                    tail_agg[b] = merged[0]
                    tail_flag[b] = True

            if tail_strategy == "tail_token":
                for b in range(B):
                    if not tail_flag[b]:
                        tail_agg[b] = new_feats[b, -1, :]

                # [B, C] -> [B, 1, C]  token 
                tail_agg = tail_agg.view(B, 1, C)
                tail_att = new_atts.new_ones((B, 1))       #  att dtype/device 
                new_feats = torch.cat([new_feats, tail_agg], dim=1)   # [B, K+1, C]
                new_atts  = torch.cat([new_atts, tail_att], dim=1)    # [B, K+1]

        if getattr(self, "_diag_vtoken_usage_enabled", False) and K < N:
            st = getattr(self, "_vision_token_usage_stats", None)
            if isinstance(st, dict):
                used_len = int(new_feats.size(1))
                st["total_used_tokens"] += int(B * used_len)
                st["calls"] += 1
                try:
                    l2 = new_feats.detach().float().norm(p=2, dim=-1)  # [B, used_len]
                    s1 = l2.sum()
                    s2 = (l2 * l2).sum()
                    c  = torch.tensor(l2.numel(), device=l2.device, dtype=torch.long)

                    st["used_l2_sum"]    = st.get("used_l2_sum", 0.0) + s1
                    st["used_l2_sq_sum"] = st.get("used_l2_sq_sum", 0.0) + s2
                    st["used_l2_count"]  = st.get("used_l2_count", 0) + c
                except Exception:
                    pass

                self._vision_token_usage_stats = st

        return new_feats, new_atts




    def _collect_cross_attn_maps(self, with_layer_idx: bool = False, qformer=None, raw: bool = False):
        """

        """
        xs = []
        for li, _, sa in self._iter_cross_attn_modules(qformer=qformer):
            attn = getattr(sa, "_saved_attn_map", None)
            if raw:
                get = getattr(sa, "get_attention_map_raw", None)
                if attn is None and callable(get):
                    attn = get()
            else:
                get = getattr(sa, "get_attention_map", None)
                if attn is None and callable(get):
                    attn = get()

            attn_val = attn if attn is not None else None
            if with_layer_idx:
                xs.append((li, attn_val))
            else:
                xs.append(attn_val)
        return xs

    def _clear_saved_attn_cache(self, qformer=None):
        """
        """
        for _, _, sa in self._iter_cross_attn_modules(qformer=qformer):
            if hasattr(sa, "_saved_attn_map"):
                sa._saved_attn_map = None
            # Qformer.BertSelfAttention 
            if hasattr(sa, "attention_map"):
                sa.attention_map = None
            if hasattr(sa, "_attention_map_raw"):
                sa._attention_map_raw = None
            if hasattr(sa, "attn_gradients"):
                sa.attn_gradients = None

    
    def _reset_ffn_act_stats(self):
        """
        """
        hooks = getattr(self, "_ffn_act_hooks", None)
        if hooks is not None:
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
        self._ffn_act_hooks = []
        self._ffn_act_stats = None
        self._ffn_act_stats_enabled = False

    def _toggle_ffn_act_stats(self, flag: bool):
        """
        """

        # 
        if flag and getattr(self, "_ffn_act_stats_enabled", False):
            return

        if not flag:
            self._ffn_act_stats_enabled = False
            hooks = getattr(self, "_ffn_act_hooks", None)
            if hooks is not None:
                for h in hooks:
                    try:
                        h.remove()
                    except Exception:
                        pass
            self._ffn_act_hooks = []
            return

        self._reset_ffn_act_stats()
        self._ffn_act_stats_enabled = True

        try:
            nL = len(self.Qformer.bert.encoder.layer)
        except Exception:
            logging.warning("[FFN-ActStats] no Q-Former encoder; skip.")
            self._ffn_act_stats_enabled = False
            return

        self._ffn_act_stats = [
            {"sum": None, "sq_sum": None, "count": 0} for _ in range(nL)
        ]
        self._ffn_act_hooks = []

        for li, layer, inter, out, act_fn, dmid in self._iter_qformer_ffn_modules():

            def make_hook(layer_idx):
                def _hook(module, inputs, output):
                    if not getattr(self, "_ffn_act_stats_enabled", False):
                        return
                    if not isinstance(output, torch.Tensor):
                        return
                    with torch.no_grad():
                        x = output.detach()
                        # [B,Q,C] or [N,C]  [N,C]
                        if x.dim() >= 2:
                            x = x.view(-1, x.shape[-1])
                        else:
                            x = x.view(-1, 1)
                        s = x.sum(dim=0)              # [C]
                        sq = (x * x).sum(dim=0)       # [C]
                        n = x.shape[0]                # token 
                        st = self._ffn_act_stats[layer_idx]
                        if st["sum"] is None:
                            st["sum"] = s
                            st["sq_sum"] = sq
                            st["count"] = n
                        else:
                            st["sum"] = st["sum"] + s
                            st["sq_sum"] = st["sq_sum"] + sq
                            st["count"] += n
                return _hook

            h = out.register_forward_hook(make_hook(li))
            self._ffn_act_hooks.append(h)

        logging.info(f"[FFN-ActStats] hooks registered for {len(self._ffn_act_hooks)} FFN layers.")

    def _collect_ffn_act_stats(self):
        """
        List[{"layer": i, "count": n, "mean": float, "std": float}  None]
        """
        stats = []
        arr = getattr(self, "_ffn_act_stats", None)
        if arr is None:
            return []

        for li, st in enumerate(arr):
            if not isinstance(st, dict) or st.get("count", 0) <= 0 or st.get("sum") is None:
                stats.append(None)
                continue

            n = float(st["count"])
            s = st["sum"]
            sq = st["sq_sum"]

            mean_vec = (s / n).float()                             # [C]
            var_vec = (sq / n - mean_vec * mean_vec).clamp_min(1e-12)
            mean_scalar = float(mean_vec.mean().item())
            std_scalar = float(var_vec.sqrt().mean().item())

            stats.append(
                {
                    "layer": li,
                    "count": int(st["count"]),
                    "mean": mean_scalar,
                    "std": std_scalar,
                }
            )

        return stats
    
    """
    def set_kd_teacher(self, teacher, *, temp=2.0, lambda_logit=0.5, lambda_attn=0.1, attn_topk=0.2):
        import torch
        self._kd_teacher = teacher.eval()
        for p in self._kd_teacher.parameters():
            p.requires_grad = False
        self._kd_temp = float(temp)
        self._kd_l_logit = float(lambda_logit)
        self._kd_l_attn = float(lambda_attn)
        self._kd_attn_topk = float(attn_topk)
        self._kd_on = True
        # 
        self._toggle_cross_attn_save(True)
        if hasattr(self._kd_teacher, "_toggle_cross_attn_save"):
            self._kd_teacher._toggle_cross_attn_save(True)
    """

    def set_kd_teacher(
        self,
        teacher,
        *,
        temp=2.0,
        lambda_logit=0.5,
        lambda_attn=0.1,
        attn_topk=0.2,
        attn_l9_boost=1.0,        #  9  attention KD 
        attn_l9_var_target=0.7,   #  9  attention 
        attn_l9_var_lambda=0.0,
        attn_l8_boost=1.0,        #  8  attention KD 
        attn_l10_boost=1.0,       #  10  attention KD 
    ):
        if teacher is None:
            #  KD 
            self._kd_on = False

            #  cross-attn map 
            try:
                self._toggle_cross_attn_save(False)
            except Exception:
                pass

            #  teacher  CPU 
            old_t = getattr(self, "_kd_teacher", None)
            if old_t is not None:
                try:
                    old_t.to("cpu")
                except Exception:
                    pass
            self._kd_teacher = None
            return

        self._kd_teacher = teacher.eval()
        for p in self._kd_teacher.parameters():
            p.requires_grad_(False)

        try:
            dev = getattr(self, "device", None)
            if dev is not None:
                self._kd_teacher.to(dev)
        except Exception as e:
            logging.warning(f"[KD] move teacher to device failed: {e}")

        self._kd_temp = float(temp)
        self._kd_l_logit = float(lambda_logit)
        self._kd_l_attn = float(lambda_attn)
        self._kd_attn_topk = float(attn_topk)

        #  attention KD 
        self._kd_attn_l9_boost = float(attn_l9_boost)
        self._kd_attn_l9_var_target = float(attn_l9_var_target)
        self._kd_attn_l9_var_lambda = float(attn_l9_var_lambda)
        self._kd_attn_l8_boost = float(attn_l8_boost)
        self._kd_attn_l10_boost = float(attn_l10_boost)

        self._kd_on = True
        # 
        try:
            self._toggle_cross_attn_save(True)
            if hasattr(self._kd_teacher, "_toggle_cross_attn_save"):
                self._kd_teacher._toggle_cross_attn_save(True)
        except Exception as e:
            logging.warning(f"[KD] enable cross-attn save failed: {e}")


    def _compute_ffn_gate_sparsity_loss(self):
        """
        """
        lam = getattr(self, "_ffn_gate_loss_lambda", 0.0)
        lam = 0.0 if lam is None else float(lam)
        mode = str(getattr(self, "_ffn_gate_loss_mode", "l1"))

        if lam <= 0.0 or not hasattr(self, "_ffn_gates"):
            return 0.0

        if hasattr(self, "ffn_gate_sparsity_loss"):
            return self.ffn_gate_sparsity_loss(lambda_sparse=lam, mode=mode)

        loss = 0.0
        for g in self._ffn_gates:
            loss = loss + g.abs().mean()
        return lam * loss

    def freeze_sparsity_targets(self):
        heads, mids = [], []
        for _, _, sa in self._iter_cross_attn_modules():
            heads.append(int(sa.num_attention_heads))
        for layer in self.Qformer.bert.encoder.layer:
            if hasattr(layer, "intermediate_query"):
                mids.append(int(layer.intermediate_query.dense.out_features))
        self._sparsity_targets = {
            "heads": heads,
            "mids": mids,
            "vision_keep": float(getattr(self, "_vision_token_keep_ratio", 1.0)),
            "text_kv_keep": float(getattr(self, "_text_kv_prune_cfg", {}).get("keep_ratio", 1.0)) \
                            if hasattr(self, "_text_kv_prune_cfg") else 1.0,
        }

    def set_recovery_cost_loss_cfg(self, lambda_latency: float = 0.0, lambda_memory: float = 0.0):
        self._recovery_cost_lat = float(lambda_latency)
        self._recovery_cost_mem = float(lambda_memory)

    def _recovery_cost_loss(self):
        if not hasattr(self, "_sparsity_targets"):
            return 0.0
        tgt = self._sparsity_targets
        # 
        heads_now, mids_now = [], []
        for _, _, sa in self._iter_cross_attn_modules():
            heads_now.append(int(sa.num_attention_heads))
        for layer in self.Qformer.bert.encoder.layer:
            if hasattr(layer, "intermediate_query"):
                mids_now.append(int(layer.intermediate_query.dense.out_features))
        v_keep = float(getattr(self, "_vision_token_keep_ratio", 1.0))
        t_keep = float(getattr(self, "_text_kv_prune_cfg", {}).get("keep_ratio", 1.0)) \
                 if hasattr(self, "_text_kv_prune_cfg") else 1.0

        def over(a, b):  # elementwise overage average
            a = torch.tensor(a, dtype=torch.float32)
            b = torch.tensor(b, dtype=torch.float32)
            return torch.clamp(a - b, min=0).mean()

        lat_norm = over(heads_now, tgt["heads"]) + max(v_keep - tgt["vision_keep"], 0.0) + max(t_keep - tgt["text_kv_keep"], 0.0)
        mem_norm = over(mids_now, tgt["mids"]) + max(t_keep - tgt["text_kv_keep"], 0.0)

        return self._recovery_cost_lat * lat_norm + self._recovery_cost_mem * mem_norm

    @torch.no_grad()
    def export_slim_checkpoint(self, save_path: str):
        """
        - /
        """
        if hasattr(self, "finalize_qformer_ffn_gates"):
            try:
                self.finalize_qformer_ffn_gates()
            except Exception:
                pass

        marked_for_meta = {}
        try:
            marked = getattr(self, "_heads_marked_for_hard_prune", None)
            if isinstance(marked, dict) and len(marked) > 0:
                marked_for_meta = {
                    int(li): sorted({int(h) for h in heads})
                    for li, heads in marked.items() if heads
                }
                logging.info(f"[Export] hard-pruning Q-Former cross-attn heads before save: { {k: v for k, v in marked.items()} }")
                for li, heads in sorted(marked.items()):
                    if not heads:
                        continue
                    layer = self.Qformer.bert.encoder.layer[li]
                    layer.crossattention.prune_heads(set(int(h) for h in heads))
                    logging.info(f"[Export] Layer {li}: physically pruned {len(heads)} heads -> {sorted(list(set(int(h) for h in heads)))}")

                for _, _, sa in self._iter_cross_attn_modules():
                    if hasattr(sa, "set_permanent_head_mask"):
                        sa.set_permanent_head_mask(None)
                    if hasattr(sa, "set_runtime_head_mask"):
                        sa.set_runtime_head_mask(None)

                # 
                self._heads_marked_for_hard_prune = dict()
        except Exception as e:
            logging.warning(f"[Export] hard-prune at export failed (ignore & continue saving): {e}")

        # 2) 
        try:
            for _, _, sa in self._iter_cross_attn_modules():
                if hasattr(sa, "set_runtime_head_mask"):
                    sa.set_runtime_head_mask(None)
                if hasattr(sa, "_saved_attn_map"):
                    sa._saved_attn_map = None
                if hasattr(sa, "save_attention"):
                    sa.save_attention = False
        except Exception:
            pass
        for k in list(self.__dict__.keys()):
            if k.startswith("_vision_token_") or k.startswith("_text_kv_") or k.startswith("_headdrop"):
                setattr(self, k, None)

        slim_meta = {}
        try:
            cross_layers = list(self._iter_cross_attn_modules())
            slim_meta["qformer_crossattn_layer_indices"] = [int(li) for li, _, _ in cross_layers]
            slim_meta["qformer_crossattn_heads"] = [
                int(sa.num_attention_heads) for _, _, sa in cross_layers
            ]
            slim_meta["heads_pruned_from_original"] = marked_for_meta
        except Exception:
            pass
        try:
            slim_meta["qformer_ffn_mids"] = [
                int(layer.intermediate_query.dense.out_features)
                for layer in self.Qformer.bert.encoder.layer
            ]
        except Exception:
            pass

        torch.save({"model": self.state_dict(), "slim_meta": slim_meta}, save_path)
        logging.info(f"[Export] slim checkpoint saved to {save_path}; slim_meta={slim_meta}")

    def diagnose_ffn_representation(self, dataset, batch_size=8, num_batches=10, energy=0.9, collate_fn=None, device=None):
        """
        """

        device = device or next(self.parameters()).device
        self.eval()

        # ---- 1)  ----
        ranks = []
        with torch.no_grad():
            for li, lyr in enumerate(self.Qformer.bert.encoder.layer):
                w1 = lyr.intermediate_query.dense.weight.data   # [mid, hidden]
                w2 = lyr.output_query.dense.weight.data         # [hidden, mid]
                for tag, W in [("W1", w1), ("W2", w2)]:
                    try:
                        S = torch.linalg.svdvals(W)  # 
                        csum = torch.cumsum(S, dim=0)
                        thr = float(energy) * float(csum[-1])
                        r = int((csum < thr).sum().item()) + 1
                        ranks.append((li, tag, int(W.shape[0]), int(W.shape[1]), r))
                    except Exception:
                        pass
        logging.info("[Diag-FFN] weight effective ranks (layer,tag,out,in,rank@{:.2f}): {}".format(float(energy), ranks))

        # ---- 2)  ----
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
        acts = []
        with torch.no_grad():
            it = iter(loader)
            for _ in range(int(num_batches)):
                try:
                    samples = next(it)
                except StopIteration:
                    break
                for k,v in samples.items():
                    if hasattr(v, "to"):
                        samples[k] = v.to(device)
                tmp = []
                def _hook(_m, _inp, out):
                    tmp.append(out.detach())   # [B, T, mid]
                hks = []
                for lyr in self.Qformer.bert.encoder.layer:
                    hks.append(lyr.intermediate_query.register_forward_hook(_hook))
                try:
                    _ = self.forward(samples)  # 
                    if tmp:
                        A = torch.cat([t.reshape(-1, t.shape[-1]) for t in tmp], dim=0)  # [N, mid]
                        #  SVD
                        A = A - A.mean(0, keepdim=True)
                        U,S,Vh = torch.linalg.svd(A, full_matrices=False)
                        acts.append(S[:32].float().cpu())  #  32 
                finally:
                    for h in hks:
                        h.remove()
        if acts:
            Smean = torch.stack(acts, dim=0).mean(0)
            logging.info("[Diag-FFN] activation spectrum (top-32 mean singular values): {}".format([float(s) for s in Smean]))


    # --- [SAFE] finalize FFN by exact target mid dim ---
    @torch.no_grad()
    def finalize_qformer_ffn_to_mid(self, target_mid: int | list[int], group_size: int | None = None):
        if not hasattr(self, "_ffn_gates"):
            logging.warning("[FFN-Finalize@mid] no gates found; call attach_qformer_ffn_gates() first.")
            return
        gs = int(getattr(self, "_ffn_gate_group_size", 16) if group_size is None else group_size)
        layers = self.Qformer.bert.encoder.layer
        nL = len(layers)
        if isinstance(target_mid, (list, tuple)):
            assert len(target_mid) == nL, f"len(target_mid)={len(target_mid)} must equal #Qformer layers={nL}"
            mids = [int(m) for m in target_mid]
        else:
            mids = [int(target_mid)] * nL

        def _round_to_group(x: int, g: int) -> int:
            return int(math.ceil(x / g) * g)

        for li, lyr in enumerate(layers):
            want_mid = _round_to_group(max(1, mids[li]), gs)

            if not hasattr(lyr, "intermediate_query") or not hasattr(lyr, "output_query"):
                logging.warning(f"[FFN-Finalize@mid] layer {li} missing ffns, skip.")
                continue

            # 
            inter = lyr.intermediate_query.dense    # [hidden -> old_mid]
            out   = lyr.output_query.dense          # [old_mid -> hidden]
            hidden = inter.in_features
            old_mid = inter.out_features

            if want_mid == old_mid:
                continue

            keep = torch.arange(old_mid, device=inter.weight.device)
            if hasattr(self, "_ffn_gate_scores") and self._ffn_gate_scores is not None:
                scores = self._ffn_gate_scores[li]  # [old_mid//gs]
                scores = scores.repeat_interleave(gs)[:old_mid]
                _, idx = torch.topk(scores, k=want_mid, largest=True, sorted=True)
                keep = idx.sort().values
            else:
                with torch.no_grad():
                    energy = inter.weight.pow(2).sum(dim=1)  # [old_mid]
                    _, idx = torch.topk(energy, k=want_mid, largest=True, sorted=True)
                    keep = idx.sort().values

            new_inter = nn.Linear(hidden, want_mid, bias=(inter.bias is not None), dtype=inter.weight.dtype, device=inter.weight.device)
            new_out   = nn.Linear(want_mid, hidden, bias=(out.bias is not None),       dtype=out.weight.dtype,   device=out.weight.device)

            with torch.no_grad():
                new_inter.weight.copy_(inter.weight[keep, :])
                if inter.bias is not None:
                    new_inter.bias.copy_(inter.bias[keep])

                new_out.weight.copy_(out.weight[:, keep])
                if out.bias is not None:
                    new_out.bias.copy_(out.bias)

            lyr.intermediate_query.dense = new_inter
            lyr.output_query.dense       = new_out

            act_mod = getattr(lyr.intermediate_query, "intermediate_act_fn", None)
            while isinstance(act_mod, _ActWithGate) and hasattr(act_mod, "act"):
                act_mod = act_mod.act
            if act_mod is not None:
                lyr.intermediate_query.intermediate_act_fn = act_mod

        for k in list(self.__dict__.keys()):
            if k.startswith("_ffn_gate") or k in ("_ffn_gates", "_ffn_gate_scores"):
                try:
                    delattr(self, k)
                except Exception:
                    setattr(self, k, 0.0 if k.endswith("loss_lambda") else None)
        # 
        setattr(self, "_ffn_gate_loss_lambda", 0.0)


        logging.info(f"[FFN-Finalize@mid] done. target mids={mids}, group_size={gs}")

    @torch.no_grad()
    def scan_qformer_ffn_min_stable_mids(
        self,
        dataset,
        device,
        *,
        group_size: int = 16,
        num_batches: int = 8,
        batch_size: int = 4,
        num_workers: int = 0,
        loss_tol: float = 0.02,
        min_keep_groups: int = 1,
        collate_fn=None,
        default_text: str = "a photo of",
    ):
        """
        """
        dl = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
        self.eval()

        # 
        base_losses = []
        steps = 0
        for samples in dl:
            if steps >= int(num_batches):
                break
            for k, v in list(samples.items()):
                if hasattr(v, "to"):
                    samples[k] = v.to(device, non_blocking=True)
            if "text_input" not in samples:
                bs = samples["image"].size(0)
                samples["text_input"] = [default_text] * bs
            with amp_ctx:
                out = self.forward(samples)
                base_losses.append(float(out["loss"]))
            steps += 1
        if not base_losses:
            logging.warning("[FFN-Scan] no samples; return current mids.")
            #  mid
            mids_now = []
            for _, layer, inter, out, act, dmid in self._iter_qformer_ffn_modules():
                mids_now.append(int(dmid))
            return mids_now
        base_loss = sum(base_losses) / len(base_losses)
        logging.info(f"[FFN-Scan] base loss={base_loss:.6f}, batches={len(base_losses)}")

        layer_sorted_idx = {}
        layers_info = []
        gs = int(group_size)
        if not hasattr(self, "_ffn_gate_scores") or self._ffn_gate_scores is None:
            self._ffn_gate_scores = [None] * len(self.Qformer.bert.encoder.layer)

        for li, layer, inter, out, act, dmid in self._iter_qformer_ffn_modules():
            G = (dmid + gs - 1) // gs
            # 
            if hasattr(self, "_ffn_gates") and self._ffn_gates is not None and li < len(self._ffn_gates):
                gparam = self._ffn_gates[li].detach()
                #  sigmoid(g) 
                scores = torch.sigmoid(gparam).float()
                if li < len(self._ffn_gate_scores):
                    self._ffn_gate_scores[li] = scores.detach()
            elif hasattr(self, "_ffn_gate_scores") and self._ffn_gate_scores is not None and li < len(self._ffn_gate_scores):
                scores = self._ffn_gate_scores[li].detach().float()
            else:
                with torch.no_grad():
                    w = inter.weight.detach()
                    energy = w.pow(2).sum(dim=1)  # [dmid]
                    scores = energy.view(G, gs).mean(dim=1)
            _, group_order = torch.topk(scores, k=G, largest=True, sorted=True)
            idx = []
            for gidx in group_order.tolist():
                beg, end = gidx*gs, min((gidx+1)*gs, dmid)
                idx.extend(list(range(beg, end)))
            layer_sorted_idx[li] = torch.tensor(idx, device=device, dtype=torch.long)
            layers_info.append((li, dmid, G))

        hooks = []
        ablate_masks = {}
        def _register_mask_hook(li, act_module, dmid):
            def _mask_fwd(_m, _inp, _out):
                mask = ablate_masks.get(li, None)
                if mask is None:
                    return _out
                return _out * mask.view(1,1,-1)
            hk = act_module.register_forward_hook(_mask_fwd)
            hooks.append(hk)
        for li, layer, inter, out, act, dmid in self._iter_qformer_ffn_modules():
            _register_mask_hook(li, act, dmid)

        # 
        min_mids = [None] * len(self.Qformer.bert.encoder.layer)
        for (li, dmid, G) in layers_info:
            lo = max(int(min_keep_groups), 1)
            hi = int(G)
            best_keep_groups = G
            best_mask = None

            while lo <= hi:
                midg = (lo + hi) // 2
                keep_c = int(midg * gs)
                idx = layer_sorted_idx[li][:keep_c]
                mask = torch.zeros(dmid, device=device, dtype=torch.float32)
                mask[idx] = 1.0
                ablate_masks[li] = mask

                #  batch  masked 
                masked_losses = []
                steps = 0
                for samples in dl:
                    if steps >= int(num_batches):
                        break
                    ss = {}
                    for k, v in list(samples.items()):
                        ss[k] = v.to(device, non_blocking=True) if hasattr(v, "to") else v
                    if "text_input" not in ss:
                        bs = ss["image"].size(0)
                        ss["text_input"] = [default_text] * bs
                    with amp_ctx:
                        out = self.forward(ss)
                        masked_losses.append(float(out["loss"]))
                    steps += 1
                ablate_masks[li] = None  # 

                if not masked_losses:
                    delta = float("inf")
                else:
                    mloss = sum(masked_losses) / len(masked_losses)
                    delta = (mloss - base_loss) / max(base_loss, 1e-8)

                if delta <= float(loss_tol):
                    best_keep_groups = midg
                    best_mask = mask
                    hi = midg - 1
                else:
                    lo = midg + 1

            min_mid = int(best_keep_groups * gs)
            #  1 
            min_mid = max(min_mid, int(gs))
            min_mids[li] = min_mid

            if best_mask is not None:
                ablate_masks[li] = best_mask
            else:
                ablate_masks.pop(li, None)
            logging.info(f"[FFN-Scan] layer {li}: dmid={dmid}, G={G}, min_keep_groups={best_keep_groups}, min_mid={min_mid}")

        # hook
        for h in hooks:
            try: h.remove()
            except: pass

        return min_mids
    
    @torch.no_grad()
    def adjust_qformer_ffn_mids_global_loss(
        self,
        dataset,
        device,
        *,
        init_mids,
        group_size: int = 16,
        num_batches: int = 8,
        batch_size: int = 4,
        num_workers: int = 0,
        loss_tol: float = 0.03,
        collate_fn=None,
        default_text: str = "a photo of",
    ):
        """
        """
        # 
        if dataset is None:
            logging.warning("[FFN-GlobalAdjust] dataset is None; skip global adjust.")
            return list(init_mids)

        if not isinstance(init_mids, (list, tuple)):
            logging.warning("[FFN-GlobalAdjust] init_mids is not list/tuple; skip global adjust.")
            return list(init_mids)

        gs = max(1, int(group_size))

        dl = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=False,
        )

        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else nullcontext()
        self.eval()

        base_losses = []
        steps = 0
        for samples in dl:
            if steps >= int(num_batches):
                break
            for k, v in list(samples.items()):
                if hasattr(v, "to"):
                    samples[k] = v.to(device, non_blocking=True)
            if "text_input" not in samples:
                bs = samples["image"].size(0)
                samples["text_input"] = [default_text] * bs
            with amp_ctx:
                out = self.forward(samples)
                base_losses.append(float(out["loss"]))
            steps += 1

        if not base_losses:
            logging.warning("[FFN-GlobalAdjust] no samples; skip global adjust.")
            return list(init_mids)

        base_loss = sum(base_losses) / len(base_losses)
        logging.info(f"[FFN-GlobalAdjust] base loss={base_loss:.6f}, batches={len(base_losses)}")

        layer_sorted_idx = {}
        layers_info = []

        has_gate_scores = hasattr(self, "_ffn_gate_scores") and (self._ffn_gate_scores is not None)

        for li, layer, inter, out, act, dmid in self._iter_qformer_ffn_modules():
            dmid = int(dmid)
            if dmid <= 0:
                continue
            G = (dmid + gs - 1) // gs

            if has_gate_scores and li < len(self._ffn_gate_scores) and self._ffn_gate_scores[li] is not None:
                scores = self._ffn_gate_scores[li].detach().float()
            elif hasattr(self, "_ffn_gates") and self._ffn_gates is not None and li < len(self._ffn_gates):
                gparam = self._ffn_gates[li].detach()
                scores = torch.sigmoid(gparam).float()
            else:
                with torch.no_grad():
                    w = inter.weight.detach()
                    energy = w.pow(2).sum(dim=1)  # [dmid]
                    scores = energy.view(G, gs).mean(dim=1)

            _, group_order = torch.topk(scores, k=G, largest=True, sorted=True)
            idx = []
            for gidx in group_order.tolist():
                beg, end = gidx * gs, min((gidx + 1) * gs, dmid)
                idx.extend(list(range(beg, end)))
            layer_sorted_idx[li] = torch.tensor(idx, device=device, dtype=torch.long)
            layers_info.append((li, dmid, G))

        nL = len(self.Qformer.bert.encoder.layer)
        mids0 = [None] * nL
        for li in range(nL):
            if li < len(init_mids) and init_mids[li] is not None:
                try:
                    mids0[li] = int(init_mids[li])
                except Exception:
                    mids0[li] = None

        dmid_by_layer = {li: dmid for (li, dmid, _) in layers_info}
        mids0_clamped = []
        for li in range(nL):
            dmid = dmid_by_layer.get(li, None)
            m = mids0[li]
            if dmid is None:
                mids0_clamped.append(m)
                continue
            if m is None:
                m = dmid
            m = max(0, min(int(m), dmid))
            #  group_size
            if m > 0:
                m = int(math.ceil(m / gs) * gs)
                m = min(m, dmid)
            mids0_clamped.append(m)

        def _eval_loss_with_mids(mids_for_eval):
            ablate_masks = {}
            hooks = []

            def _register_mask_hook(li, act_module):
                def _mask_fwd(_m, _inp, _out):
                    mask = ablate_masks.get(li, None)
                    if mask is None:
                        return _out
                    return _out * mask.view(1, 1, -1)
                hk = act_module.register_forward_hook(_mask_fwd)
                hooks.append(hk)

            #  hook
            for li2, layer2, inter2, out2, act2, dmid2 in self._iter_qformer_ffn_modules():
                _register_mask_hook(li2, act2)

            #  mask
            for (li2, dmid2, G2) in layers_info:
                keep_c = 0
                if li2 < len(mids_for_eval) and mids_for_eval[li2] is not None:
                    keep_c = int(mids_for_eval[li2])
                keep_c = max(0, min(keep_c, dmid2))
                if keep_c >= dmid2:
                    continue
                if keep_c <= 0:
                    idx2 = None
                else:
                    idx2 = layer_sorted_idx[li2][:keep_c]
                mask = torch.zeros(dmid2, device=device, dtype=torch.float32)
                if idx2 is not None and len(idx2) > 0:
                    mask[idx2] = 1.0
                ablate_masks[li2] = mask

            #  DataLoader  loss
            losses = []
            steps_local = 0
            for samples2 in dl:
                if steps_local >= int(num_batches):
                    break
                ss = {}
                for k2, v2 in list(samples2.items()):
                    ss[k2] = v2.to(device, non_blocking=True) if hasattr(v2, "to") else v2
                if "text_input" not in ss:
                    bs2 = ss["image"].size(0)
                    ss["text_input"] = [default_text] * bs2
                with amp_ctx:
                    out2 = self.forward(ss)
                    losses.append(float(out2["loss"]))
                steps_local += 1

            #  hook
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass

            if not losses:
                return float("inf")
            return sum(losses) / len(losses)

        loss0 = _eval_loss_with_mids(mids0_clamped)
        delta0 = (loss0 - base_loss) / max(base_loss, 1e-8)
        logging.info(f"[FFN-GlobalAdjust] init mids loss={loss0:.6f}, delta={delta0:.4f}, tol={float(loss_tol):.4f}")

        if not (loss_tol is not None and float(loss_tol) > 0.0):
            return mids0_clamped

        if delta0 <= float(loss_tol):
            logging.info("[FFN-GlobalAdjust] init mids already within tolerance; no adjust.")
            return mids0_clamped

        beta_list = [1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
        best_mids = mids0_clamped
        best_delta = delta0

        for beta in beta_list:
            mids_beta = []
            for li in range(nL):
                dmid = dmid_by_layer.get(li, None)
                m0 = mids0_clamped[li]
                if dmid is None or m0 is None or m0 >= dmid:
                    mids_beta.append(m0)
                    continue
                m_new = int(math.ceil(m0 * float(beta) / gs) * gs)
                m_new = min(m_new, dmid)
                mids_beta.append(m_new)

            loss_b = _eval_loss_with_mids(mids_beta)
            delta_b = (loss_b - base_loss) / max(base_loss, 1e-8)
            logging.info(f"[FFN-GlobalAdjust] beta={beta:.3f}, loss={loss_b:.6f}, delta={delta_b:.4f}")

            if delta_b <= float(loss_tol):
                logging.info(f"[FFN-GlobalAdjust] choose beta={beta:.3f} (delta={delta_b:.4f} <= tol={float(loss_tol):.4f})")
                return mids_beta

            if delta_b < best_delta:
                best_delta = delta_b
                best_mids = mids_beta

        logging.warning(f"[FFN-GlobalAdjust] cannot meet tol={float(loss_tol):.4f}; use best beta (delta={best_delta:.4f})")
        return best_mids


            
    @torch.no_grad()
    def quick_random_ablate_check(self, dataset, device, *, max_batches=16, group_size=16, collate_fn=None, default_text="a photo of"):
        import random, contextlib
        if dataset is None:
            logging.warning("[Verify] dataset is None; skip."); return
        try:
            n = min(len(dataset), 128)
            idx = list(range(n))
        except Exception:
            idx = list(range(64))
        dl = DataLoader(Subset(dataset, idx), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

        base_losses, ablated_losses = [], []
        amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if torch.cuda.is_available() else contextlib.nullcontext()
        steps = 0
        self.eval()

        for samples in dl:
            if steps >= int(max_batches): break
            for k, v in list(samples.items()):
                if hasattr(v, "to"):
                    samples[k] = v.to(device, non_blocking=True)
            if "text_input" not in samples:
                bs = samples["image"].size(0)
                samples["text_input"] = [default_text] * bs

            with amp_ctx:
                out = self.forward(samples)
                base = float(out["loss"])
            base_losses.append(base)

            if hasattr(self, "_iter_qformer_ffn_modules"):
                for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
                    g  = max(1, int(group_size))
                    G  = (dmid + g - 1) // g
                    gi = random.randint(0, max(G - 1, 0))
                    beg, end = gi * g, min((gi + 1) * g, dmid)
                    mask = torch.ones(dmid, device=device, dtype=torch.float32)
                    mask[beg:end] = 0.0
                    def _mask_fwd(_m, _inp, _out):
                        return _out * mask.view(1, 1, -1)
                    hk = act_fn.register_forward_hook(_mask_fwd)
                    try:
                        with amp_ctx:
                            out2 = self.forward(samples)
                            loss2 = float(out2["loss"])
                    finally:
                        hk.remove()
                    ablated_losses.append(loss2)
            steps += 1

        if base_losses and ablated_losses:
            b = sum(base_losses)/len(base_losses); a = sum(ablated_losses)/len(ablated_losses)
            logging.info(f"[Verify@model] base={b:.4f}, ablated={a:.4f}, delta={a-b:.4f}")
        else:
            logging.warning("[Verify@model] not enough samples; skip.")

    # Ensure text_input is present for diagnostic batches.
    @torch.no_grad()
    def quick_random_ablate_check(self, dataset, device, *, max_batches=16, group_size=16, collate_fn=None, default_text="a photo of"):
        import random
        if dataset is None:
            logging.warning("[Verify] dataset is None; skip."); return
        N = min(len(dataset), 128) if hasattr(dataset, "__len__") else 64
        dl = DataLoader(Subset(dataset, list(range(N))), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

        def _ensure_text(samples):
            if "text_input" not in samples or samples["text_input"] is None:
                #  batch 
                bsz = 1
                for v in samples.values():
                    if torch.is_tensor(v): bsz = v.size(0); break
                samples["text_input"] = [default_text] * bsz
            return samples

        self.eval()
        loss_base, cnt = 0.0, 0
        for batch in dl:
            for k,v in list(batch.items()):
                if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)
            batch = _ensure_text(batch)
            out = self.forward(batch)
            loss_base += float(out["loss"].item()); cnt += 1
            if cnt >= max_batches: break

        hooks = []
        try:
            def _mk_hook(mask):
                def _h(_m,_i,out): return out * mask.view(1,1,-1)
                return _h
            for li, layer in enumerate(self.Qformer.bert.encoder.layer):
                inter = getattr(layer, "intermediate_query", None)
                act   = getattr(inter, "intermediate_act_fn", None) if inter is not None else None
                if inter is None or act is None: continue
                dmid = int(inter.dense.out_features)
                if dmid <= group_size: continue
                mask = torch.ones(dmid, device=device, dtype=inter.dense.weight.dtype)
                killG = max(1, dmid // group_size // 5)
                for gi in random.sample(range(dmid // group_size), k=killG):
                    s, e = gi*group_size, min((gi+1)*group_size, dmid); mask[s:e] = 0
                hooks.append(act.register_forward_hook(_mk_hook(mask)))

            loss_rand, cnt2 = 0.0, 0
            for batch in dl:
                for k,v in list(batch.items()):
                    if torch.is_tensor(v): batch[k] = v.to(device, non_blocking=True)
                batch = _ensure_text(batch)
                out = self.forward(batch)
                loss_rand += float(out["loss"].item()); cnt2 += 1
                if cnt2 >= max_batches: break
            logging.info(f"[Verify] loss(base)={loss_base/cnt:.4f} vs loss(random)={loss_rand/cnt2:.4f}")
        finally:
            for h in hooks:
                try: h.remove()
                except: pass


    # Compute a lightweight structure and cost summary.
    def summarize_pruned_structure_and_cost(self):
        """

        """
        

        heads = []
        mids = []
        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            sa = getattr(layer, "crossattention", None)
            h = getattr(getattr(sa, "self", None), "num_attention_heads", None)
            heads.append(int(h) if h is not None else None)
            mids.append(int(dmid))

        n_layers = len(heads)
        d = int(getattr(self.Qformer.config, "hidden_size", 768))
        Lq = int(getattr(self.Qformer.config, "query_length", 32))
        base_heads_each = int(getattr(self.Qformer.config, "num_attention_heads", 12) or 12)
        base_mid_each = int(getattr(self.Qformer.config, "intermediate_size", 3072) or 3072)
        head_dim = max(1, d // max(base_heads_each, 1))
        enc_width = int(getattr(self.Qformer.config, "encoder_width", getattr(self.visual_encoder, "num_features", d)) or d)

        base_heads = [base_heads_each for _ in range(n_layers)]
        base_mids = [base_mid_each for _ in range(n_layers)]

        vis_full_tokens = int(getattr(self, "_last_vtoken_full_N", 0) or 0)
        if vis_full_tokens <= 0:
            vis_full_tokens = int(getattr(self.visual_encoder, "num_patches", 256) or 256)

        keep_v_cfg = float(getattr(self, "_vision_token_keep_ratio", 1.0) or 1.0)
        keep_v_obs = None
        try:
            st_v = getattr(self, "_vision_token_usage_stats", None)
            if isinstance(st_v, dict) and float(st_v.get("total_raw_tokens", 0.0) or 0.0) > 0:
                keep_v_obs = float(st_v.get("total_used_tokens", 0.0) or 0.0) / float(st_v.get("total_raw_tokens", 1.0) or 1.0)
        except Exception:
            keep_v_obs = None
        keep_v_for_cost = float(keep_v_obs) if keep_v_obs is not None else keep_v_cfg
        vis_used_tokens_est = max(1, int(round(vis_full_tokens * keep_v_for_cost)))

        text_kv_cfg = getattr(self, "_text_kv_prune_cfg", None)
        text_kv_keep = 1.0
        text_kv_mode = None
        text_kv_enabled = False
        if isinstance(text_kv_cfg, dict) and bool(text_kv_cfg.get("enabled", False)):
            text_kv_enabled = True
            text_kv_mode = text_kv_cfg.get("mode", None)
            text_kv_keep = text_kv_cfg.get("total_keep_ratio", None)
            if text_kv_keep is None:
                text_kv_keep = text_kv_cfg.get("keep_ratio", 1.0)
            text_kv_keep = float(text_kv_keep if text_kv_keep is not None else 1.0)

        # q_proj + (k_proj,v_proj) + attention(score/value mix) + out_proj
        def _approx_cross_attn_flops(lq, lk, hidden, active_heads, head_dim_local, encoder_width_local):
            active_d = max(1, int(active_heads) * int(head_dim_local))
            q_proj = lq * hidden * active_d
            kv_proj = 2 * lk * encoder_width_local * active_d
            attn_core = 2 * lq * lk * active_d
            out_proj = lq * active_d * hidden
            return q_proj + kv_proj + attn_core + out_proj

        def _approx_ffn_flops(lq, hidden, mid):
            return 2 * lq * hidden * int(mid)

        base_cross = 0
        cur_cross = 0
        base_ffn = 0
        cur_ffn = 0
        active_dims = []

        for h in heads:
            h_cur = int(h if h is not None else base_heads_each)
            active_dims.append(int(h_cur * head_dim))
            base_cross += _approx_cross_attn_flops(Lq, vis_full_tokens, d, base_heads_each, head_dim, enc_width)
            cur_cross += _approx_cross_attn_flops(Lq, vis_used_tokens_est, d, h_cur, head_dim, enc_width)

        for m in mids:
            m_cur = int(m)
            base_ffn += _approx_ffn_flops(Lq, d, base_mid_each)
            cur_ffn += _approx_ffn_flops(Lq, d, m_cur)

        base_total = int(base_cross + base_ffn)
        cur_total = int(cur_cross + cur_ffn)

        def _ratio(cur, base):
            if float(base) <= 0:
                return 0.0
            return float(cur) / float(base)

        def _save_ratio(cur, base):
            if float(base) <= 0:
                return 0.0
            return max(0.0, 1.0 - float(cur) / float(base))

        head_keep_mean = (
            float(sum(int(h if h is not None else base_heads_each) for h in heads))
            / float(max(1, n_layers) * max(1, base_heads_each))
            if n_layers > 0 else 1.0
        )
        ffn_keep_mean = (
            float(sum(int(m) for m in mids))
            / float(max(1, n_layers) * max(1, base_mid_each))
            if n_layers > 0 else 1.0
        )

        report = {
            "qformer_heads_per_layer": heads,
            "qformer_heads_per_layer_baseline": base_heads,
            "qformer_ffn_mid_per_layer": mids,
            "qformer_ffn_mid_per_layer_baseline": base_mids,
            "vision_token_cfg": {
                "keep_ratio_cfg": float(keep_v_cfg),
                "keep_ratio_observed": float(keep_v_obs) if keep_v_obs is not None else None,
                "full_tokens": int(vis_full_tokens),
                "used_tokens_est": int(vis_used_tokens_est),
            },
            "text_kv_cfg": {
                "enabled": bool(text_kv_enabled),
                "mode": text_kv_mode,
                "keep_ratio_proxy": float(text_kv_keep),
            },
            "approx_flops": {
                "baseline": {
                    "cross_attn": int(base_cross),
                    "ffn": int(base_ffn),
                    "total": int(base_total),
                },
                "current": {
                    "cross_attn": int(cur_cross),
                    "ffn": int(cur_ffn),
                    "total": int(cur_total),
                },
                "saving_ratio": {
                    "cross_attn": float(_save_ratio(cur_cross, base_cross)),
                    "ffn": float(_save_ratio(cur_ffn, base_ffn)),
                    "total": float(_save_ratio(cur_total, base_total)),
                },
                "keep_ratio": {
                    "cross_attn": float(_ratio(cur_cross, base_cross)),
                    "ffn": float(_ratio(cur_ffn, base_ffn)),
                    "total": float(_ratio(cur_total, base_total)),
                },
            },
            "structure_keep_proxy": {
                "cross_attn_head_keep_mean": float(head_keep_mean),
                "ffn_mid_keep_mean": float(ffn_keep_mean),
                "cross_attn_active_dim_per_layer": [int(x) for x in active_dims],
            },
        }
        logging.info("[Prune-Report]\n" + json.dumps(report, indent=2))
        try:
            out_dir = Path(registry.get_path("result_dir"))
            (out_dir / "prune_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        except Exception:
            pass
        return report


    def _collect_cross_attn(self, *, expect_b: int, expect_n: int, want_grads: bool = False, qformer=None):
        """
        :
        attn: [B,H,Q,N]  None
        grad: [B,H,Q,N]  None
        """
        xs = []
        grads = [] if want_grads else None

        for _, _, sa in self._iter_cross_attn_modules(qformer=qformer):
            get = getattr(sa, "get_attention_map", None)
            attn = get() if callable(get) else getattr(sa, "attention_map", None)
            if attn is None or attn.dim() != 4:
                continue
            if attn.shape[0] != expect_b or attn.shape[-1] != expect_n:
                continue
            xs.append(attn.detach())

            if want_grads:
                getg = getattr(sa, "get_attn_gradients", None)
                g = getg() if callable(getg) else getattr(sa, "attn_gradients", None)
                if g is not None and g.dim() == 4 and tuple(g.shape) == tuple(attn.shape):
                    grads.append(g.detach())

        if not xs:
            return None, None

        try:
            stacked = torch.stack(xs, dim=0)          # [L,B,H,Q,N]
            attn_out = stacked.mean(dim=0)            # [B,H,Q,N]
        except Exception:
            base = xs[0].shape
            xs_same = [t for t in xs if tuple(t.shape) == tuple(base)]
            attn_out = torch.stack(xs_same, dim=0).mean(dim=0) if xs_same else None

        grad_out = None
        if want_grads and grads:
            try:
                gstack = torch.stack(grads, dim=0)    # [L,B,H,Q,N]
                grad_out = gstack.mean(dim=0)
            except Exception:
                base = grads[0].shape
                gs_same = [t for t in grads if tuple(t.shape) == tuple(base)]
                grad_out = torch.stack(gs_same, dim=0).mean(dim=0) if gs_same else None

        return attn_out, grad_out

    def _get_teacher_attn_grad_for_prune(self, samples):
        """
        """
        import torch

        teacher = getattr(self, "_vtoken_grad_teacher", None)
        if teacher is None:
            raise RuntimeError("teacher_attn_grad_external requires self._vtoken_grad_teacher")

        teacher_samples = dict(samples)

        if "text_input" not in teacher_samples:
            text = (
                teacher_samples.get("text_input", None)
                or teacher_samples.get("text", None)
                or teacher_samples.get("caption", None)
                or teacher_samples.get("captions", None)
            )

            if text is None:
                bs = None
                img = teacher_samples.get("image", None)
                if torch.is_tensor(img) and img.dim() > 0:
                    bs = int(img.size(0))
                else:
                    for v in teacher_samples.values():
                        if torch.is_tensor(v) and v.dim() > 0:
                            bs = int(v.size(0))
                            break
                bs = bs or 1

                default_text = getattr(teacher, "prompt", None) or getattr(self, "prompt", None) or "a photo of "
                text = [default_text] * bs

            elif isinstance(text, str):
                text = [text]
            elif isinstance(text, tuple):
                text = list(text)
            elif not isinstance(text, list):
                text = list(text)

            teacher_samples["text_input"] = text

        with torch.enable_grad():
            teacher.eval()
            teacher.zero_grad(set_to_none=True)

            #  cross-attn 
            try:
                if hasattr(teacher, "_toggle_cross_attn_save"):
                    teacher._toggle_cross_attn_save(True)
                if hasattr(teacher, "_clear_saved_attn_cache"):
                    teacher._clear_saved_attn_cache()
            except Exception:
                pass

            #  teacher  runtime prune
            prev_rt = getattr(teacher, "_vtoken_runtime_enabled", False)
            setattr(teacher, "_vtoken_runtime_enabled", False)

            try:
                out = teacher(teacher_samples)
                if isinstance(out, dict):
                    loss = out["loss"]
                elif hasattr(out, "loss"):
                    loss = out.loss
                else:
                    loss = out

                loss.backward()

                image = teacher_samples["image"]
                t_img = teacher.ln_vision(teacher.visual_encoder(image))
                t_atts = torch.ones(t_img.size()[:-1], dtype=torch.long, device=image.device)

                tfwd_id = int(getattr(teacher, "_vision_token_fwd_id", 0) or 0) + 1
                setattr(teacher, "_vision_token_fwd_id", tfwd_id)

                (
                    attn_for_prune,
                    grad_for_prune,
                    _attn_refresh,
                    _grad_refresh,
                    _want_grads,
                    _attn_src,
                    _did_refresh,
                    _refresh_reason,
                ) = teacher._get_cross_attn_for_prune_lagged(
                    image_embeds=t_img,
                    image_atts=t_atts,
                    v_keep=0.999999,
                    mode_for_attn="attn_grad",
                    fwd_id=tfwd_id,
                )

                teacher.zero_grad(set_to_none=True)
                return attn_for_prune, grad_for_prune

            finally:
                setattr(teacher, "_vtoken_runtime_enabled", prev_rt)



    def _get_cross_attn_for_prune_lagged(
        self,
        *,
        image_embeds: torch.Tensor,
        image_atts: torch.Tensor,
        v_keep: float,
        mode_for_attn: str,
        fwd_id: int,
    ):
        """


            attn_for_prune, grad_for_prune, attn_refresh, grad_refresh, want_grads, attn_src, did_refresh, refresh_reason
        """
        # ---  attn  ---
        mode_for_attn = str(mode_for_attn).lower()
        need_attn_for_scoring = mode_for_attn in (
            "attn_entropy", "attn", "entropy", "attn_mean",
            "attn_grad", "grad", "taylor_attn",
        )
        want_grads = mode_for_attn in ("attn_grad", "grad", "taylor_attn")

        if not (float(v_keep) < 1.0 and need_attn_for_scoring):
            #  cross-attn
            try:
                self._toggle_cross_attn_save(False)
            except Exception:
                pass
            return None, None, None, None, False, "none", False, ""

        B, N = int(image_embeds.shape[0]), int(image_embeds.shape[1])

        lag_onepass = bool(getattr(self, "_vision_token_lag_onepass", False))
        if bool(getattr(self, "_vision_token_lagged_two_pass", True)) is False:
            lag_onepass = True

        refresh_interval = int(getattr(self, "_vision_token_full_attn_refresh_interval", 0) or 0)
        refresh_interval = max(refresh_interval, 0)
        refresh_on_eval = bool(getattr(self, "_vision_token_full_attn_refresh_on_eval", False))

        strict_refresh_each_step = bool(getattr(self, "_vision_token_lag_strict_refresh_each_step", lag_onepass))

        attn_for_prune = None
        grad_for_prune = None
        attn_src = "none"

        if lag_onepass:
            cache_attn = getattr(self, "_last_xattn", None)
            cache_grad = getattr(self, "_last_xattn_grad", None)
            if torch.is_tensor(cache_attn) and cache_attn.dim() == 4 and int(cache_attn.shape[0]) == B and int(cache_attn.shape[-1]) == N:
                attn_for_prune = cache_attn
                if torch.is_tensor(cache_grad) and cache_grad.shape == cache_attn.shape:
                    grad_for_prune = cache_grad
                attn_src = "cache"
            else:
                attn_src = "cache-miss"

        did_refresh = False
        refresh_reason = ""
        if (not lag_onepass) and need_attn_for_scoring:
            did_refresh = True
            refresh_reason = "two-pass(always)"
        else:
            if attn_for_prune is None:
                did_refresh = True
                refresh_reason = "cache_miss"
            else:
                if lag_onepass and strict_refresh_each_step:
                    did_refresh = True
                    refresh_reason = "strict_next_step"
                elif refresh_interval > 0 and (int(fwd_id) % int(refresh_interval) == 0):
                    did_refresh = True
                    refresh_reason = f"interval={refresh_interval}"
                elif (not bool(self.training)) and refresh_on_eval:
                    did_refresh = True
                    refresh_reason = "eval"

        attn_refresh = None
        grad_refresh = None

        if did_refresh:
            query_tokens_warm = self.query_tokens.expand(B, -1, -1)

            #  cross-attn 
            self._toggle_cross_attn_save(True)

            if want_grads:
                with torch.enable_grad():
                    query_output_warm = self.Qformer.bert(
                        query_embeds=query_tokens_warm,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                    proxy = query_output_warm.last_hidden_state.float().pow(2).mean()

                    raw_maps = []
                    raw_sas = []
                    for _, _, sa in self._iter_cross_attn_modules():
                        get_raw = getattr(sa, "get_attention_map_raw", None)
                        raw = get_raw() if callable(get_raw) else getattr(sa, "_attention_map_raw", None)
                        if raw is None or (not torch.is_tensor(raw)) or raw.dim() != 4:
                            continue
                        if int(raw.shape[0]) != B or int(raw.shape[-1]) != N:
                            continue
                        raw_maps.append(raw)
                        raw_sas.append(sa)

                    if raw_maps:
                        grads = torch.autograd.grad(
                            proxy,
                            raw_maps,
                            retain_graph=False,
                            create_graph=False,
                            allow_unused=True,
                        )
                        for sa, g in zip(raw_sas, grads):
                            if g is None:
                                continue
                            saver = getattr(sa, "save_attn_gradients", None)
                            if callable(saver):
                                saver(g)
                            else:
                                sa.attn_gradients = g

                #  + proxy 
                attn_refresh, grad_refresh = self._collect_cross_attn(
                    expect_b=B,
                    expect_n=N,
                    want_grads=True,
                )

                if not bool(getattr(self, "_diag_tail_grad_force_attn_save", False)):
                    self._toggle_cross_attn_save(False)
                self._clear_saved_attn_cache()

            else:
                with torch.no_grad():
                    _ = self.Qformer.bert(
                        query_embeds=query_tokens_warm,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                attn_refresh, grad_refresh = self._collect_cross_attn(
                    expect_b=B,
                    expect_n=N,
                    want_grads=False,
                )

                if not bool(getattr(self, "_diag_tail_grad_force_attn_save", False)):
                    self._toggle_cross_attn_save(False)
                self._clear_saved_attn_cache()

            if attn_for_prune is None:
                attn_for_prune = attn_refresh
                grad_for_prune = grad_refresh if (want_grads and grad_refresh is not None) else None
                attn_src = "warmup"
            else:
                attn_src = "cache+refresh"

        else:
            try:
                self._toggle_cross_attn_save(False)
            except Exception:
                pass

        return attn_for_prune, grad_for_prune, attn_refresh, grad_refresh, want_grads, attn_src, did_refresh, refresh_reason


    # Compute effective ranks for weights and activations.
    @torch.no_grad()
    def compute_effective_rank_stats(self, dataset, max_batches=10, energy=0.9, collate_fn=None, num_workers=0):
        """
        """
        ranks_w = []
        energy = float(energy)

        # 
        for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
            W1 = inter_dense.weight.detach().float()  # [Dmid, Din]
            U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
            cume = (S.cumsum(0) / (S.sum() + 1e-12))
            r = int((cume < energy).sum().item()) + 1
            ranks_w.append(r)

        # 
        specs = []
        if dataset is not None and max_batches > 0:
            dl = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
            seen = 0
            acts = []
            handles = []
            try:
                for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
                    def _hook(_m, inp, out):
                        acts.append(out.detach().flatten(0,1))  # [B*L, Dmid]
                    handles.append(out_dense.register_forward_hook(_hook))

                for batch in dl:
                    if seen >= max_batches: break
                    for k,v in list(batch.items()):
                        if hasattr(torch, "is_tensor") and torch.is_tensor(v):
                            batch[k] = v.to(next(self.parameters()).device, non_blocking=True)
                    _ = self.forward(batch)  # 
                    seen += 1
            finally:
                for h in handles:
                    try: h.remove()
                    except: pass

            if acts:
                X = torch.cat(acts, dim=0)
                specs_layer = []
                offset = 0
                for li, layer, inter_dense, out_dense, act_fn, dmid in self._iter_qformer_ffn_modules():
                    specs_layer.append(None)
                specs = specs_layer
        out = {"ffn_effective_rank_w": ranks_w, "act_spectrum_note": "see logs/hook"}
        logging.info(f"[Rank] FFN weight effective ranks (90% energy): {ranks_w}")
        return out


    def enable_opt_quantization(self, mode: str = "bnb_8bit"):
        mode = str(mode).lower()

        if getattr(self, "_opt_quantized_on_init", False):
            logging.info(f"[Quant] already quantized at init with mode={self._opt_quant_mode}")
            return True

        logging.warning(
            f"[Quant] runtime quantization path is deprecated for switch-based workflow. "
            f"Please set model.opt_quant_mode='{mode}' before model construction."
        )
        return False



    def set_runtime_token_pruning(
        self,
        *,
        keep_ratio: float = 1.0,
        score_mode: str = "l2norm",
        tail_strategy: str = "drop",
        learned_teacher_mode: str | None = None,
        learned_lambda: float = 0.0,
        learned_temp: float | None = None,
        **_ignored_kwargs,
    ):
        """
        """
        kr = 1.0 if (keep_ratio is None) else float(keep_ratio)
        self._vision_token_keep_ratio = kr
        self._vision_token_score_mode = str(score_mode).lower()
        self._vision_token_tail_strategy = str(tail_strategy).lower()

        if learned_teacher_mode is not None:
            self._vision_token_learned_teacher_mode = str(learned_teacher_mode).lower()
        else:
            if not hasattr(self, "_vision_token_learned_teacher_mode"):
                self._vision_token_learned_teacher_mode = "entropy"

        try:
            self._vision_token_learned_lambda = float(learned_lambda or 0.0)
        except Exception:
            self._vision_token_learned_lambda = 0.0

        if learned_temp is None:
            self._vision_token_learned_temp = None
        else:
            try:
                self._vision_token_learned_temp = float(learned_temp)
            except Exception:
                self._vision_token_learned_temp = None

        mode = self._vision_token_score_mode

        #  cross-attn
        need_attn = mode in (
            "attn_entropy", "attn", "entropy", "attn_mean",
            "attn_grad", "grad", "taylor_attn",
        )

        if mode in ("learned", "mlp", "score_net"):
            tmode = str(getattr(self, "_vision_token_learned_teacher_mode", "entropy")).lower()
            if tmode in (
                "attn_entropy", "attn", "entropy", "attn_mean",
                "attn_grad", "grad", "taylor_attn",
            ):
                need_attn = True

        self._toggle_cross_attn_save(need_attn)
        return True


    
    def set_text_kv_pruning(
        self,
        trigger_len: int = 0,
        score_mode: str = "attn",
        imp_decay: float = 0.9,
        mode: str = "window",
        window_size: int = 256,
        keep_ratio: float | None = None,
        # legacy pool knobs (still supported)
        pool_bins: int = 64,
        keep_recent: int = 64,
        keep_recent_ratio: float | None = None,
        keep_recent_min: int = 0,
        # new: keep ratio over *all text history* (recommended)
        total_keep_ratio: float | None = None,
        tail_frac: float = 0.5,
        tail_min: int = 4,
        tail_max: int = 16,
        pool_min_bin_len: int = 1,
        tail_strategy: str = "drop",
        cap_len: int = 0,
        imp_smooth_k: int = 0,
        debug: bool = False,
        debug_interval: int = 200,
    ):
        """Text-side KV pruning (decode-time): window / pool.

        """

        first_install = not getattr(self, "_text_kv_patch_installed", False)
        self._text_kv_patch_installed = True

        def _clamp_ratio(x):
            try:
                x = float(x)
            except Exception:
                return None
            if x <= 0:
                return None
            return min(1.0, x)

        self._text_kv_prune_cfg = dict(
            enabled=True,
            mode=str(mode).lower(),
            trigger_len=int(trigger_len),
            score_mode=str(score_mode).lower(),
            imp_decay=float(imp_decay),
            window_size=int(window_size),
            keep_ratio=(None if keep_ratio is None else float(keep_ratio)),
            pool_bins=int(pool_bins),
            keep_recent=int(keep_recent),
            keep_recent_ratio=_clamp_ratio(keep_recent_ratio),
            keep_recent_min=int(keep_recent_min),
            total_keep_ratio=_clamp_ratio(total_keep_ratio),
            tail_frac=float(tail_frac),
            tail_min=int(tail_min),
            tail_max=int(tail_max),
            pool_min_bin_len=max(1, int(pool_min_bin_len)),
            tail_strategy=str(tail_strategy).lower(),
            cap_len=int(cap_len),
            imp_smooth_k=int(imp_smooth_k),
            debug=bool(debug),
            debug_interval=int(debug_interval),
        )

        self._text_kv_stats = getattr(self, "_text_kv_stats", None) or {
            "n_calls": 0,
            "n_pruned": 0,
            "n_noop": 0,
            "n_tail_token": 0,
            "n_merge": 0,
            "tail_T_sum": 0,
            "sum_L_before": 0,
            "sum_L_after": 0,
            "seen_L": [],
            "kept_L": [],
            "reasons": {},
            "imp_ent_sum": 0.0,
            "imp_ent_cnt": 0,
            "imp_top1_sum": 0.0,
        }


        logging.info(
            "[Text-KV] cfg updated: enabled=True, mode=%s, window=%s, keep_ratio=%s, total_keep_ratio=%s, cap_len=%s, imp_smooth_k=%s, debug=%s, dbg_int=%s, tail_frac=%s, tail_min=%s, tail_max=%s, pool_bins=%s, keep_recent=%s, keep_recent_ratio=%s, keep_recent_min=%s, pool_min_bin_len=%s, tail=%s",
            self._text_kv_prune_cfg["mode"],
            self._text_kv_prune_cfg["window_size"],
            self._text_kv_prune_cfg["keep_ratio"],
            self._text_kv_prune_cfg.get("total_keep_ratio", None),
            self._text_kv_prune_cfg.get("cap_len", 0),
            self._text_kv_prune_cfg.get("imp_smooth_k", 0),
            self._text_kv_prune_cfg.get("debug", False),
            self._text_kv_prune_cfg.get("debug_interval", 0),
            self._text_kv_prune_cfg.get("tail_frac", 0.5),
            self._text_kv_prune_cfg.get("tail_min", 4),
            self._text_kv_prune_cfg.get("tail_max", 16),
            self._text_kv_prune_cfg["pool_bins"],
            self._text_kv_prune_cfg["keep_recent"],
            self._text_kv_prune_cfg.get("keep_recent_ratio", None),
            self._text_kv_prune_cfg.get("keep_recent_min", 0),
            self._text_kv_prune_cfg.get("pool_min_bin_len", 1),
            self._text_kv_prune_cfg.get("tail_strategy", "drop"),
        )

        if first_install and not hasattr(self, "_orig_prepare_inputs_for_generation"):
            self._orig_prepare_inputs_for_generation = self.opt_model.prepare_inputs_for_generation

        # -------- attn_topk: patch opt_model.forward to capture last-layer self-attn --------
        if first_install and not hasattr(self, "_orig_opt_forward_for_text_kv"):
            self._orig_opt_forward_for_text_kv = self.opt_model.forward
            outer = self

            def _forward_with_textkv_importance(self_opt, *args, **kwargs):
                cfg = getattr(outer, "_text_kv_prune_cfg", None)
                mode_here = str(cfg.get("mode", "")).lower() if isinstance(cfg, dict) else ""
                enabled = bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False
                score_mode_here = str(cfg.get("score_mode", "attn")).lower() if isinstance(cfg, dict) else "attn"
                use_imp = enabled and (mode_here in ("attn_topk", "topk_attn", "importance")) and (score_mode_here not in ("learned", "scorer"))

                past = kwargs.get("past_key_values", None)

                tgt_len = None
                input_ids = kwargs.get("input_ids", None)
                inputs_embeds = kwargs.get("inputs_embeds", None)
                try:
                    if torch.is_tensor(input_ids) and input_ids.dim() == 2:
                        tgt_len = int(input_ids.size(1))
                    elif torch.is_tensor(inputs_embeds) and inputs_embeds.dim() == 3:
                        tgt_len = int(inputs_embeds.size(1))
                except Exception:
                    tgt_len = None

                need_capture = use_imp and (past is not None) and (tgt_len == 1)

                if need_capture:
                    kwargs["output_attentions"] = True
                    kwargs["return_dict"] = True

                out = outer._orig_opt_forward_for_text_kv(*args, **kwargs)

                if need_capture:
                    try:
                        attns = getattr(out, "attentions", None)
                        if attns is not None and len(attns) > 0 and attns[-1] is not None:
                            a = attns[-1]  #  [B,H,1,S]
                            if a.dim() == 4:
                                imp = a[:, :, -1, :].mean(dim=1)  # [B,S]
                            elif a.dim() == 3:
                                #  [B,1,S]  [B,H,S]
                                imp = a[:, -1, :] if a.size(1) == 1 else a.mean(dim=1)
                            else:
                                imp = None

                            if imp is not None:
                                outer._text_kv_last_imp = imp.detach()
                                try:
                                    smooth_k = int(cfg.get("imp_smooth_k", 0)) if isinstance(cfg, dict) else 0
                                except Exception:
                                    smooth_k = 0
                                if smooth_k and smooth_k > 1:
                                    buf = getattr(outer, "_text_kv_imp_buf", None)
                                    if (buf is None) or (not isinstance(buf, list)):
                                        buf = []
                                    buf.append(imp.detach())
                                    if len(buf) > smooth_k:
                                        buf = buf[-smooth_k:]
                                    outer._text_kv_imp_buf = buf
                                    try:
                                        outer._text_kv_imp_smooth = torch.stack([t.to(imp.device) for t in buf], dim=0).mean(dim=0).detach()
                                    except Exception:
                                        outer._text_kv_imp_smooth = imp.detach()
                                decay = float(cfg.get("imp_decay", 0.9)) if isinstance(cfg, dict) else 0.9
                                ema = getattr(outer, "_text_kv_imp_ema", None)
                                if (ema is None) or (not torch.is_tensor(ema)) or (ema.shape != imp.shape):
                                    outer._text_kv_imp_ema = imp.detach()
                                else:
                                    outer._text_kv_imp_ema = (ema.to(imp.device) * decay + imp.detach() * (1.0 - decay))
                    except Exception:
                        pass

                return out

            self.opt_model.forward = _forward_with_textkv_importance.__get__(self.opt_model, type(self.opt_model))


        # -------- helpers --------
        def _kv_len(k):
            return int(k.size(2)) if k.dim() == 4 else int(k.size(1))

        def _kv_slice(x, s, e):
            if x.dim() == 4:
                return x[:, :, s:e, :]
            return x[:, s:e, :]

        def _kv_cat(parts):
            # parts: list of tensors with same rank; cat on length dim
            if not parts:
                return None
            if parts[0].dim() == 4:
                return torch.cat(parts, dim=2)
            return torch.cat(parts, dim=1)
        
        def _kv_gather(x, idx_2d):
            """Gather past-KV along length dim with per-sample indices.
            x: [B,H,L,D] or [B,L,D]; idx_2d: [B,K] (long)
            """
            if x.dim() == 4:
                B_, H_, L_, D_ = x.shape
                idx = idx_2d[:, None, :, None].expand(B_, H_, idx_2d.size(1), D_)
                return x.gather(2, idx)
            else:
                B_, L_, D_ = x.shape
                idx = idx_2d[:, :, None].expand(B_, idx_2d.size(1), D_)
                return x.gather(1, idx)
                                                                
        def _stats_push(L_before, L_after, did_prune: bool, reason: str):
            st = self._text_kv_stats
            st["n_calls"] = int(st.get("n_calls", 0)) + 1
            if did_prune:
                st["n_pruned"] = int(st.get("n_pruned", 0)) + 1
            else:
                st["n_noop"] = int(st.get("n_noop", 0)) + 1
            st["sum_L_before"] = int(st.get("sum_L_before", 0)) + int(L_before)
            st["sum_L_after"] = int(st.get("sum_L_after", 0)) + int(L_after)
            try:
                st["seen_L"].append(int(L_before))
                st["kept_L"].append(int(L_after))
                if len(st["seen_L"]) > 2000:
                    st["seen_L"] = st["seen_L"][-2000:]
                    st["kept_L"] = st["kept_L"][-2000:]
            except Exception:
                pass
            if reason:
                st["reasons"][reason] = int(st["reasons"].get(reason, 0)) + 1

        def _ensure_2d_mask(attn_mask, B, need_len, device, dtype=torch.long):
            if attn_mask is None:
                return torch.ones((B, need_len), device=device, dtype=dtype)
            if attn_mask.dim() == 2:
                if attn_mask.size(1) >= need_len:
                    return attn_mask[:, :need_len].to(device=device)
                pad = torch.ones((B, need_len - attn_mask.size(1)), device=device, dtype=attn_mask.dtype)
                return torch.cat([attn_mask.to(device=device), pad], dim=1)
            return torch.ones((B, need_len), device=device, dtype=dtype)

        def _pool_variable_bins(x, head_start, head_len, out_bins):
            # x: [B,H,L,D] or [B,L,D]
            if out_bins <= 0 or head_len <= 0:
                return None
            out_bins = min(out_bins, head_len)
            # boundaries ensure exact out_bins, non-empty
            bounds = [ (i * head_len) // out_bins for i in range(out_bins + 1) ]
            # make sure strictly increasing
            for i in range(1, len(bounds)):
                if bounds[i] <= bounds[i-1]:
                    bounds[i] = bounds[i-1] + 1
            bounds[-1] = head_len
            pooled = []
            for i in range(out_bins):
                s = head_start + bounds[i]
                e = head_start + bounds[i+1]
                seg = _kv_slice(x, s, e)
                if x.dim() == 4:
                    pooled.append(seg.mean(dim=2, keepdim=True))
                else:
                    pooled.append(seg.mean(dim=1, keepdim=True))
            return _kv_cat(pooled)

        def _pool_mask_bins(mask_head, out_bins):
            # mask_head: [B, head_len]
            head_len = int(mask_head.size(1))
            out_bins = min(out_bins, head_len)
            bounds = [ (i * head_len) // out_bins for i in range(out_bins + 1) ]
            for i in range(1, len(bounds)):
                if bounds[i] <= bounds[i-1]:
                    bounds[i] = bounds[i-1] + 1
            bounds[-1] = head_len
            cols = []
            for i in range(out_bins):
                seg = mask_head[:, bounds[i]:bounds[i+1]]
                cols.append(seg.max(dim=1, keepdim=True).values)
            return torch.cat(cols, dim=1)  # [B, out_bins]

        def _drop_allzero_bins(kv, mask_bins):
            # kv: pooled head tensor; mask_bins: [B, bins]
            if kv is None or mask_bins is None:
                return kv, mask_bins
            col_keep = (mask_bins.sum(dim=0) > 0)
            if int(col_keep.sum()) <= 0:
                #  1
                col_keep = col_keep.clone()
                col_keep[0] = True
            idx = col_keep.nonzero(as_tuple=False).view(-1)
            if kv.dim() == 4:
                kv2 = kv.index_select(2, idx)
            else:
                kv2 = kv.index_select(1, idx)
            mask2 = mask_bins.index_select(1, idx)
            return kv2.contiguous(), mask2.contiguous()

        # -------- core prune: return (new_pkv, new_past_mask_2d) --------
        def _prune_past_kv_and_mask(past_key_values, attention_mask, input_len: int, query_len: int):
            if past_key_values is None:
                return None, None

            k0 = past_key_values[0][0]
            B = int(k0.size(0))
            device = k0.device
            L_past = _kv_len(k0)  # includes query + text history (no current token)

            mask_2d = _ensure_2d_mask(attention_mask, B, L_past + int(input_len), device=device)
            mask_past = mask_2d[:, :L_past]

            q_len = int(max(0, min(query_len, L_past)))
            hist_len = int(max(0, L_past - q_len))
            if hist_len <= 0:
                _stats_push(L_past + input_len, L_past + input_len, False, "no_hist")
                return past_key_values, mask_past

            query_mask = mask_past[:, :q_len]
            text_mask = mask_past[:, q_len:]

            mode_local = str(self._text_kv_prune_cfg.get("mode", "window")).lower()
            tail_strategy_local = str(self._text_kv_prune_cfg.get("tail_strategy", "drop")).lower()
            dbg = bool(self._text_kv_prune_cfg.get("debug", False))
            try:
                dbg_interval = int(self._text_kv_prune_cfg.get("debug_interval", 200) or 200)
            except Exception:
                dbg_interval = 200
            if dbg:
                c = int(getattr(self, "_text_kv_dbg_calls", 0)) + 1
                self._text_kv_dbg_calls = c
                do_dbg = (dbg_interval <= 0) or (c % dbg_interval == 0)
            else:
                do_dbg = False

            prompt_med = int(getattr(self, "_text_kv_prompt_len0_med", 0) or 0)
            gen_est = int(max(0, hist_len - prompt_med))
# -------- attn_topk (importance-based) --------
            if mode_local in ("attn_topk", "topk_attn", "importance"):
                window_size = int(self._text_kv_prune_cfg.get("window_size", 256))
                keep_recent_cfg = int(self._text_kv_prune_cfg.get("keep_recent", 0))
                cap_len_cfg = int(self._text_kv_prune_cfg.get("cap_len", 0) or 0)
                cfg_trigger = int(self._text_kv_prune_cfg.get("trigger_len", 0) or 0)
                if cfg_trigger <= 0:
                    cfg_trigger = window_size
                trigger_len = max(cfg_trigger, keep_recent_cfg)
                if cap_len_cfg > 0:
                    trigger_len = max(trigger_len, cap_len_cfg)

                try:
                    trigger_gap = int(self._text_kv_prune_cfg.get("trigger_gap", 0) or 0)
                except Exception:
                    trigger_gap = 0

                if cap_len_cfg > 0:
                    if trigger_gap <= 0:
                        trigger_gap = 4
                    if cfg_trigger <= cap_len_cfg:
                        trigger_len = max(trigger_len, cap_len_cfg + trigger_gap)

                if hist_len <= trigger_len:
                    if do_dbg:
                        try:
                            logging.debug(
                                "[Text-KV] no_shrink(trigger): L_past=%s, q_len=%s, hist_len=%s (prompt~%s + gen~%s) <= trigger_len=%s, cap_len=%s, keep_recent=%s",
                                int(L_past), int(q_len), int(hist_len), int(prompt_med), int(gen_est),
                                int(trigger_len), int(cap_len_cfg), int(keep_recent_cfg)
                            )
                        except Exception:
                            pass
                    _stats_push(L_past + input_len, L_past + input_len, False, "no_shrink")
                    return past_key_values, mask_past
                imp = None
                try:
                    smooth_k = int(self._text_kv_prune_cfg.get("imp_smooth_k", 0) or 0)
                except Exception:
                    smooth_k = 0
                if smooth_k and smooth_k > 1:
                    imp = getattr(self, "_text_kv_imp_smooth", None)
                if imp is None:
                    imp = getattr(self, "_text_kv_imp_ema", None)
                if imp is None:
                    imp = getattr(self, "_text_kv_last_imp", None)

                if imp is None:
                    try:
                        dec = getattr(getattr(self.opt_model, "model", None), "decoder", None)
                        layers = getattr(dec, "layers", None) if dec is not None else None
                        if layers is not None and len(layers) > 0:
                            for li in range(len(layers) - 1, -1, -1):
                                cand = getattr(layers[li].self_attn, "_last_attn_importance", None)
                                if cand is not None:
                                    imp = cand
                                    break
                    except Exception:
                        imp = None

                reason_tag = "attn_topk"

                score_mode_local = str(self._text_kv_prune_cfg.get("score_mode", "attn")).lower()
                if score_mode_local in ("learned", "scorer"):
                    scorer = getattr(self, "_text_kv_scorer", None)
                    if scorer is not None:
                        try:
                            k_imp = past_key_values[-1][0]
                            if k_imp.dim() == 4:
                                Bk = int(k_imp.size(0)); Hk = int(k_imp.size(1)); Lk = int(k_imp.size(2)); Dk = int(k_imp.size(3))
                                rep = k_imp.transpose(1, 2).contiguous().view(Bk, Lk, Hk * Dk)
                            else:
                                rep = k_imp
                            imp = scorer(rep.float()).squeeze(-1)  # [B, L_past]
                            reason_tag = "attn_topk_learned"
                        except Exception:
                            pass

                if imp is None or (not hasattr(imp, "dim")) or imp.dim() != 2 or int(imp.size(0)) != B:
                    imp = torch.arange(L_past, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
                    reason_tag = "attn_topk_recency_prior"

                if int(imp.size(1)) != int(L_past):
                    if int(imp.size(1)) == int(L_past + input_len):
                        imp = imp[:, :L_past]
                    elif int(imp.size(1)) == int(max(0, L_past - input_len)):
                        pad = imp.new_zeros((B, int(L_past) - int(imp.size(1))))
                        imp = torch.cat([imp, pad], dim=1)
                    else:
                        imp = torch.arange(L_past, device=device, dtype=torch.float32).view(1, -1).expand(B, -1)
                        reason_tag = "attn_topk_len_mismatch_prior"


                if int(imp.size(1)) != int(L_past):
                    if int(imp.size(1)) == int(L_past + input_len):
                        imp = imp[:, :L_past]
                    elif int(imp.size(1)) == int(max(0, L_past - input_len)):
                        pad = imp.new_zeros((B, int(L_past) - int(imp.size(1))))
                        imp = torch.cat([imp, pad], dim=1)
                    else:
                        _stats_push(L_past + input_len, L_past + input_len, False, "imp_len_mismatch")
                        return past_key_values, mask_past


                total_keep_ratio = self._text_kv_prune_cfg.get("total_keep_ratio", None)
                if total_keep_ratio is None:
                    kr = self._text_kv_prune_cfg.get("keep_ratio", None)
                    total_keep_ratio = (None if kr is None else float(kr))

                if total_keep_ratio is None:
                    if cap_len_cfg > 0:
                        desired_total = min(hist_len, cap_len_cfg)
                        reason_tag = reason_tag + "_cap"
                    else:
                        _stats_push(L_past + input_len, L_past + input_len, False, "no_total_ratio")
                        return past_key_values, mask_past
                else:
                    desired_total = int(math.ceil(hist_len * float(total_keep_ratio)))
                    if cap_len_cfg > 0:
                        desired_total = min(desired_total, cap_len_cfg)

                desired_total = max(1, min(hist_len, int(desired_total)))

                if do_dbg:
                    try:
                        logging.debug(
                            "[Text-KV] trigger: L_past=%s, q_len=%s, hist_len=%s (prompt~%s + gen~%s), desired_total=%s, ratio=%s, cap_len=%s, trigger_len=%s, keep_recent=%s, tail_strategy=%s, score_mode=%s, reason=%s",
                            int(L_past), int(q_len), int(hist_len), int(prompt_med), int(gen_est),
                            int(desired_total), str(total_keep_ratio), int(cap_len_cfg), int(trigger_len),
                            int(keep_recent_cfg), str(tail_strategy_local), str(score_mode_local), str(reason_tag)
                        )
                    except Exception:
                        pass
                tail_frac = float(self._text_kv_prune_cfg.get("tail_frac", 0.5))
                tail_min = int(self._text_kv_prune_cfg.get("tail_min", 4))
                tail_max = int(self._text_kv_prune_cfg.get("tail_max", 16))
                keep_recent_cfg = int(self._text_kv_prune_cfg.get("keep_recent", 0))

                want_tail_token = (tail_strategy_local == "tail_token")
                want_merge = (tail_strategy_local == "merge")
                merged_count = 1 if want_tail_token else 0

                tail_max_eff = max(tail_max, keep_recent_cfg)

                if keep_recent_cfg > 0:
                    desired_total = max(desired_total, min(hist_len, keep_recent_cfg + merged_count))

                tail_keep = int(round(desired_total * tail_frac))
                tail_keep = max(tail_min, tail_keep)
                if keep_recent_cfg > 0:
                    tail_keep = max(tail_keep, min(hist_len, keep_recent_cfg))
                tail_keep = min(tail_max_eff, tail_keep)
                tail_keep = min(hist_len, desired_total - merged_count, tail_keep)
                tail_keep = max(0, tail_keep)


                head_len = int(max(0, hist_len - tail_keep))
                head_budget = int(max(0, desired_total - merged_count - tail_keep))

                if do_dbg:
                    try:
                        drop_est = int(max(0, head_len - head_budget))
                        logging.debug(
                            "[Text-KV] split: desired_total=%s => tail_keep=%s, head_len=%s, head_budget=%s, merged_count=%s, drop_est=%s, tail_max_eff=%s",
                            int(desired_total), int(tail_keep), int(head_len), int(head_budget), int(merged_count), int(drop_est), int(tail_max_eff)
                        )
                    except Exception:
                        pass
                if head_len <= 0:
                    _stats_push(L_past + input_len, L_past + input_len, False, "no_head")
                    return past_key_values, mask_past

                if (not want_tail_token) and head_budget >= head_len and desired_total == hist_len:
                    _stats_push(L_past + input_len, L_past + input_len, False, "no_shrink")
                    return past_key_values, mask_past

                head_budget = max(0, min(head_len, head_budget))

                scores_head_raw = imp[:, q_len:q_len + head_len]  # [B, head_len]
                head_mask = text_mask[:, :head_len]               # [B, head_len]
                tail_mask_exact = text_mask[:, head_len:]         # [B, tail_keep]

                scores_head_sel = scores_head_raw.masked_fill(head_mask <= 0, float("-inf"))

                try:
                    st = self._text_kv_stats
                    _logits = scores_head_sel.float()
                    _valid = torch.isfinite(_logits).any(dim=1, keepdim=True)  # [B,1]
                    _p = torch.softmax(_logits, dim=1)                         # [B,head_len]
                    _p = torch.where(_valid, _p, torch.zeros_like(_p))
                    _pc = _p.clamp_min(1e-12)
                    _ent = -(_p * _pc.log()).sum(dim=1)                        # [B]
                    _top1 = _p.max(dim=1).values                               # [B]
                    st["imp_ent_sum"] = float(st.get("imp_ent_sum", 0.0)) + float(_ent.sum().item())
                    st["imp_ent_cnt"] = int(st.get("imp_ent_cnt", 0)) + int(_ent.numel())
                    st["imp_top1_sum"] = float(st.get("imp_top1_sum", 0.0)) + float(_top1.sum().item())
                except Exception:
                    pass


                if head_budget > 0:
                    topk_rel = torch.topk(scores_head_sel, k=head_budget, dim=1, largest=True).indices  # [B,k]
                    topk_rel, _ = topk_rel.sort(dim=1)
                    idx_head = topk_rel + q_len
                    kept_head_mask = head_mask.gather(1, topk_rel)
                else:
                    topk_rel = None
                    idx_head = None
                    kept_head_mask = head_mask[:, :0]


                drop_count = int(max(0, head_len - head_budget))

                insert_tail = bool(want_tail_token) and (head_len > 0) and (drop_count > 0)

                do_merge = bool(want_merge) and (head_len > 0) and (drop_count > 0)

                if insert_tail:
                    if head_budget > 0:
                        keep_flag = torch.zeros((B, head_len), device=device, dtype=torch.bool)
                        keep_flag.scatter_(1, topk_rel, True)
                        drop_flag = ~keep_flag
                    else:
                        drop_flag = torch.ones((B, head_len), device=device, dtype=torch.bool)

                    merged_mask = (head_mask * drop_flag.long()).max(dim=1, keepdim=True).values  # [B,1]
                else:
                    drop_flag = None
                    merged_mask = None

                new_pkv = []
                for (k, v) in past_key_values:
                    qk = _kv_slice(k, 0, q_len)
                    qv = _kv_slice(v, 0, q_len)

                    if head_budget > 0:
                        hk = _kv_gather(_kv_slice(k, 0, L_past), idx_head)
                        hv = _kv_gather(_kv_slice(v, 0, L_past), idx_head)
                    else:
                        hk = _kv_slice(k, 0, 0)
                        hv = _kv_slice(v, 0, 0)

                    if tail_keep > 0:
                        tk_keep = _kv_slice(k, q_len + head_len, q_len + hist_len)
                        tv_keep = _kv_slice(v, q_len + head_len, q_len + hist_len)
                    else:
                        tk_keep = _kv_slice(k, 0, 0)
                        tv_keep = _kv_slice(v, 0, 0)

                    if insert_tail:
                        oldk = _kv_slice(k, q_len, q_len + head_len)
                        oldv = _kv_slice(v, q_len, q_len + head_len)

                        logits = scores_head_raw
                        logits = logits.masked_fill(head_mask <= 0, float("-inf"))
                        logits = logits.masked_fill(~drop_flag, float("-inf"))

                        valid = torch.isfinite(logits).any(dim=1, keepdim=True)
                        w = torch.softmax(logits, dim=1)
                        w = torch.where(valid, w, torch.zeros_like(w))

                        if k.dim() == 4:
                            w4 = w[:, None, :, None]
                            mk = (oldk * w4).sum(dim=2, keepdim=True)
                            mv = (oldv * w4).sum(dim=2, keepdim=True)
                        else:
                            w3 = w[:, :, None]
                            mk = (oldk * w3).sum(dim=1, keepdim=True)
                            mv = (oldv * w3).sum(dim=1, keepdim=True)

                        new_k = _kv_cat([qk, mk, hk, tk_keep])
                        new_v = _kv_cat([qv, mv, hv, tv_keep])

                    elif do_merge:
                        oldk = _kv_slice(k, q_len, q_len + head_len)
                        oldv = _kv_slice(v, q_len, q_len + head_len)

                        keep_flag = torch.zeros((B, head_len), device=device, dtype=torch.bool)
                        if topk_rel is not None and int(topk_rel.numel()) > 0:
                            keep_flag.scatter_(1, topk_rel, True)
                        drop_flag2 = ~keep_flag

                        head_mask_bool = head_mask.to(dtype=torch.bool)
                        drop_mask = drop_flag2 & head_mask_bool
                        has_drop = drop_mask.any(dim=1)

                        if has_drop.any():
                            logits = scores_head_raw
                            logits = logits.masked_fill(~drop_mask, float("-inf"))
                            w = torch.softmax(logits, dim=1)
                            w = torch.where(has_drop[:, None], w, torch.zeros_like(w))

                            if k.dim() == 4:
                                w4 = w[:, None, :, None]
                                mk = (oldk * w4).sum(dim=2, keepdim=True)
                                mv = (oldv * w4).sum(dim=2, keepdim=True)

                                m4 = has_drop[:, None, None, None]
                                if head_budget > 0:
                                    hk0_new = 0.5 * hk[:, :, 0:1, :] + 0.5 * mk
                                    hv0_new = 0.5 * hv[:, :, 0:1, :] + 0.5 * mv
                                    hk[:, :, 0:1, :] = torch.where(m4, hk0_new, hk[:, :, 0:1, :])
                                    hv[:, :, 0:1, :] = torch.where(m4, hv0_new, hv[:, :, 0:1, :])
                                elif tail_keep > 0:
                                    tk0_new = 0.5 * tk_keep[:, :, 0:1, :] + 0.5 * mk
                                    tv0_new = 0.5 * tv_keep[:, :, 0:1, :] + 0.5 * mv
                                    tk_keep[:, :, 0:1, :] = torch.where(m4, tk0_new, tk_keep[:, :, 0:1, :])
                                    tv_keep[:, :, 0:1, :] = torch.where(m4, tv0_new, tv_keep[:, :, 0:1, :])
                            else:
                                w3 = w[:, :, None]
                                mk = (oldk * w3).sum(dim=1, keepdim=True)
                                mv = (oldv * w3).sum(dim=1, keepdim=True)

                                m3 = has_drop[:, None, None]
                                if head_budget > 0:
                                    hk0_new = 0.5 * hk[:, 0:1, :] + 0.5 * mk
                                    hv0_new = 0.5 * hv[:, 0:1, :] + 0.5 * mv
                                    hk[:, 0:1, :] = torch.where(m3, hk0_new, hk[:, 0:1, :])
                                    hv[:, 0:1, :] = torch.where(m3, hv0_new, hv[:, 0:1, :])
                                elif tail_keep > 0:
                                    tk0_new = 0.5 * tk_keep[:, 0:1, :] + 0.5 * mk
                                    tv0_new = 0.5 * tv_keep[:, 0:1, :] + 0.5 * mv
                                    tk_keep[:, 0:1, :] = torch.where(m3, tk0_new, tk_keep[:, 0:1, :])
                                    tv_keep[:, 0:1, :] = torch.where(m3, tv0_new, tv_keep[:, 0:1, :])

                        new_k = _kv_cat([qk, hk, tk_keep])
                        new_v = _kv_cat([qv, hv, tv_keep])

                    else:
                        new_k = _kv_cat([qk, hk, tk_keep])
                        new_v = _kv_cat([qv, hv, tv_keep])

                    new_pkv.append((new_k, new_v))


                if insert_tail:
                    new_text_mask = torch.cat([merged_mask, kept_head_mask, tail_mask_exact], dim=1)
                    try:
                        self._text_kv_stats["n_tail_token"] = int(self._text_kv_stats.get("n_tail_token", 0)) + 1
                        self._text_kv_stats["tail_T_sum"] = int(self._text_kv_stats.get("tail_T_sum", 0)) + int(drop_count)
                    except Exception:
                        pass
                else:
                    new_text_mask = torch.cat([kept_head_mask, tail_mask_exact], dim=1)
                    if want_merge and int(drop_count) > 0:
                        try:
                            self._text_kv_stats["n_merge"] = int(self._text_kv_stats.get("n_merge", 0)) + 1
                            self._text_kv_stats["tail_T_sum"] = int(self._text_kv_stats.get("tail_T_sum", 0)) + int(drop_count)
                        except Exception:
                            pass

                new_past_mask = torch.cat([query_mask, new_text_mask], dim=1)
                L_after = q_len + int(new_text_mask.size(1))
                did_prune = (L_after < L_past)
                if do_dbg:
                    try:
                        text_kept = int(new_text_mask.size(1))
                        logging.debug(
                            "[Text-KV] result: did_prune=%s, L_past=%s -> L_after=%s, text_kept=%s/%s, insert_tail=%s, want_merge=%s, reason=%s",
                            bool(did_prune), int(L_past), int(L_after), int(text_kept), int(hist_len),
                            bool(insert_tail), bool(want_merge), str(reason_tag)
                        )
                    except Exception:
                        pass
                _stats_push(L_past + input_len, L_after + input_len, did_prune, reason_tag)
                return new_pkv, new_past_mask


            # -------- window --------
            if mode_local == "window":
                kr = self._text_kv_prune_cfg.get("keep_ratio", None)
                if kr is not None:
                    keep_hist = int(math.ceil(hist_len * float(kr)))
                else:
                    keep_hist = int(self._text_kv_prune_cfg.get("window_size", 256))
                keep_hist = max(1, min(hist_len, keep_hist))
                remove = int(hist_len - keep_hist)
                if remove <= 0:
                    _stats_push(L_past + input_len, L_past + input_len, False, "no_shrink")
                    return past_key_values, mask_past

                removed_mask = text_mask[:, :remove]
                kept_mask = text_mask[:, remove:]
                tail_mask = removed_mask.max(dim=1, keepdim=True).values  # [B,1]

                new_pkv = []
                for (k, v) in past_key_values:
                    # prefix(query)
                    qk = _kv_slice(k, 0, q_len)
                    qv = _kv_slice(v, 0, q_len)
                    # kept recent
                    ks = _kv_slice(k, q_len + remove, q_len + hist_len)
                    vs = _kv_slice(v, q_len + remove, q_len + hist_len)

                    if tail_strategy_local in ("tail_token", "merge"):
                        old = _kv_slice(k, q_len, q_len + remove)
                        oldv = _kv_slice(v, q_len, q_len + remove)
                        if k.dim() == 4:
                            tk = old.mean(dim=2, keepdim=True)
                            tv = oldv.mean(dim=2, keepdim=True)
                        else:
                            tk = old.mean(dim=1, keepdim=True)
                            tv = oldv.mean(dim=1, keepdim=True)
                        scale = math.log1p(float(remove))
                        tk = tk * tk.new_tensor(scale)
                        tv = tv * tv.new_tensor(scale)

                        if tail_strategy_local == "merge":
                            # merge into the *oldest kept* token (right after query)
                            new_k = _kv_cat([qk, ks])
                            new_v = _kv_cat([qv, vs])
                            # merge into position q_len (first kept)
                            try:
                                if new_k.dim() == 4:
                                    new_k[:, :, q_len:q_len+1, :] = 0.5 * new_k[:, :, q_len:q_len+1, :] + 0.5 * tk
                                    new_v[:, :, q_len:q_len+1, :] = 0.5 * new_v[:, :, q_len:q_len+1, :] + 0.5 * tv
                                else:
                                    new_k[:, q_len:q_len+1, :] = 0.5 * new_k[:, q_len:q_len+1, :] + 0.5 * tk
                                    new_v[:, q_len:q_len+1, :] = 0.5 * new_v[:, q_len:q_len+1, :] + 0.5 * tv
                            except Exception:
                                pass
                        else:
                            # insert tail token after query
                            new_k = _kv_cat([qk, tk, ks])
                            new_v = _kv_cat([qv, tv, vs])
                    else:
                        new_k = _kv_cat([qk, ks])
                        new_v = _kv_cat([qv, vs])

                    new_pkv.append((new_k, new_v))

                new_pkv = tuple(new_pkv)

                # build new past mask
                if tail_strategy_local == "tail_token":
                    new_text_mask = torch.cat([tail_mask, kept_mask], dim=1)
                    try:
                        self._text_kv_stats["n_tail_token"] = int(self._text_kv_stats.get("n_tail_token", 0)) + 1
                        self._text_kv_stats["tail_T_sum"] = int(self._text_kv_stats.get("tail_T_sum", 0)) + int(remove)
                    except Exception:
                        pass
                else:
                    new_text_mask = kept_mask
                    if tail_strategy_local == "merge":
                        try:
                            self._text_kv_stats["n_merge"] = int(self._text_kv_stats.get("n_merge", 0)) + 1
                            self._text_kv_stats["tail_T_sum"] = int(self._text_kv_stats.get("tail_T_sum", 0)) + int(remove)
                        except Exception:
                            pass

                new_past_mask = torch.cat([query_mask, new_text_mask], dim=1)
                L_after = q_len + int(new_text_mask.size(1))
                _stats_push(L_past + input_len, L_after + input_len, True, "window")
                return new_pkv, new_past_mask

            # -------- pool --------
            if mode_local == "pool":
                total_keep_ratio = self._text_kv_prune_cfg.get("total_keep_ratio", None)
                tail_frac = float(self._text_kv_prune_cfg.get("tail_frac", 0.5))
                tail_min = int(self._text_kv_prune_cfg.get("tail_min", 4))
                tail_max = int(self._text_kv_prune_cfg.get("tail_max", 16))
                pool_bins = int(self._text_kv_prune_cfg.get("pool_bins", 64))
                keep_recent = int(self._text_kv_prune_cfg.get("keep_recent", 64))
                keep_recent_ratio = self._text_kv_prune_cfg.get("keep_recent_ratio", None)
                keep_recent_min = int(self._text_kv_prune_cfg.get("keep_recent_min", 0))
                pool_min_bin_len = int(self._text_kv_prune_cfg.get("pool_min_bin_len", 1))

                if total_keep_ratio is not None:
                    desired_total = int(math.ceil(hist_len * float(total_keep_ratio)))
                    desired_total = max(1, min(hist_len, desired_total))
                    tail_keep = int(round(desired_total * tail_frac))
                    tail_keep = max(tail_min, tail_keep)
                    tail_keep = min(tail_max, tail_keep)
                    tail_keep = min(desired_total, hist_len, tail_keep)
                    head_out = int(desired_total - tail_keep)
                else:
                    if keep_recent_ratio is not None:
                        tail_keep = int(math.ceil(hist_len * float(keep_recent_ratio)))
                        tail_keep = max(keep_recent_min, tail_keep)
                    else:
                        tail_keep = int(keep_recent)
                    tail_keep = max(0, min(hist_len, tail_keep))
                    head_out = int(min(pool_bins, hist_len - tail_keep))

                head_len = int(max(0, hist_len - tail_keep))
                if head_len <= 0 or head_out <= 0 or head_out >= head_len:
                    _stats_push(L_past + input_len, L_past + input_len, False, "no_shrink")
                    return past_key_values, mask_past

                # enforce min bin length => max bins
                max_bins = max(1, head_len // max(1, pool_min_bin_len))
                head_out = min(head_out, max_bins)
                head_out = max(1, min(head_out, head_len))

                head_mask = text_mask[:, :head_len]
                tail_mask = text_mask[:, head_len:]

                new_pkv = []
                pooled_mask = _pool_mask_bins(head_mask, head_out)

                for (k, v) in past_key_values:
                    qk = _kv_slice(k, 0, q_len)
                    qv = _kv_slice(v, 0, q_len)

                    # pool older head part
                    head_k = _pool_variable_bins(k, q_len, head_len, head_out)
                    head_v = _pool_variable_bins(v, q_len, head_len, head_out)

                    # drop bins where all batch are padding(0)
                    head_k, pooled_mask2 = _drop_allzero_bins(head_k, pooled_mask)
                    head_v, _ = _drop_allzero_bins(head_v, pooled_mask)

                    # exact keep tail
                    tail_k = _kv_slice(k, q_len + head_len, q_len + hist_len)
                    tail_v = _kv_slice(v, q_len + head_len, q_len + hist_len)

                    new_k = _kv_cat([qk, head_k, tail_k])
                    new_v = _kv_cat([qv, head_v, tail_v])
                    new_pkv.append((new_k, new_v))

                new_pkv = tuple(new_pkv)

                # recompute pooled_mask in case all-zero bins dropped
                pooled_bins = int(pooled_mask2.size(1))
                new_text_mask = torch.cat([pooled_mask2, tail_mask], dim=1)
                new_past_mask = torch.cat([query_mask, new_text_mask], dim=1)

                L_after = q_len + pooled_bins + int(tail_mask.size(1))
                did_prune = (L_after < L_past)
                _stats_push(L_past + input_len, L_after + input_len, did_prune, "pool")
                return new_pkv, new_past_mask

            # unknown
            _stats_push(L_past + input_len, L_past + input_len, False, f"unknown_mode:{mode_local}")
            return past_key_values, mask_past

        # -------- monkey patch prepare_inputs_for_generation --------
        if first_install:
            def _prepare_inputs_for_generation_with_kv(
                self_model,
                input_ids=None,
                query_embeds=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                use_cache=None,
                **kwargs,
            ):
                """

                - attention_mask  new_past_len + 1
                """

                cfg = getattr(self, "_text_kv_prune_cfg", None)
                enabled = bool(cfg.get("enabled", False)) if isinstance(cfg, dict) else False

                if not enabled:
                    try:
                        return self._orig_prepare_inputs_for_generation(
                            input_ids=input_ids,
                            query_embeds=query_embeds,
                            past_key_values=past_key_values,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            **kwargs,
                        )
                    except TypeError:
                        #  past  past_key_values
                        return self._orig_prepare_inputs_for_generation(
                            input_ids=input_ids,
                            query_embeds=query_embeds,
                            past=past_key_values,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            **kwargs,
                        )

                if past_key_values is None:
                    qe = query_embeds if query_embeds is not None else kwargs.get("query_embeds", None)
                    if qe is None:
                        qe = getattr(self.opt_model, "_blip2_cached_query_embeds", None)

                    try:
                        self._q_len_for_kv = int(qe.shape[1]) if torch.is_tensor(qe) else 0
                    except Exception:
                        self._q_len_for_kv = 0

                    self._text_kv_abs_pos = None
                    try:
                        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                            self._text_kv_abs_pos = attention_mask.long().sum(-1, keepdim=True) - 1
                    except Exception:
                        self._text_kv_abs_pos = None

                    try:
                        if torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
                            ql = int(getattr(self, "_q_len_for_kv", 0) or 0)
                            lens = attention_mask.long().sum(-1)
                            pl = (lens - ql).clamp(min=0)
                            self._text_kv_prompt_len0_mean = float(pl.float().mean().item()) if pl.numel() > 0 else 0.0
                            self._text_kv_prompt_len0_med = int(pl.median().item()) if pl.numel() > 0 else 0
                    except Exception:
                        self._text_kv_prompt_len0_mean = None
                        self._text_kv_prompt_len0_med = None

                    if bool(cfg.get("debug", False)):
                        try:
                            logging.debug(
                                "[Text-KV] prefill: q_len=%s, prompt_len0_mean=%.2f, prompt_len0_med=%s, trigger_len=%s, cap_len=%s, keep_recent=%s, mode=%s, score_mode=%s",
                                int(getattr(self, "_q_len_for_kv", 0) or 0),
                                float(getattr(self, "_text_kv_prompt_len0_mean", 0.0) or 0.0),
                                int(getattr(self, "_text_kv_prompt_len0_med", 0) or 0),
                                int(cfg.get("trigger_len", 0) or 0),
                                int(cfg.get("cap_len", 0) or 0),
                                int(cfg.get("keep_recent", 0) or 0),
                                str(cfg.get("mode", "")),
                                str(cfg.get("score_mode", "")),
                            )
                        except Exception:
                            pass
                    try:
                        self._text_kv_last_imp = None
                        self._text_kv_imp_ema = None
                    except Exception:
                        pass

                    try:
                        out = self._orig_prepare_inputs_for_generation(
                            input_ids=input_ids,
                            query_embeds=qe,
                            past_key_values=None,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            **kwargs,
                        )
                    except TypeError:
                        out = self._orig_prepare_inputs_for_generation(
                            input_ids=input_ids,
                            query_embeds=qe,
                            past=None,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=use_cache,
                            **kwargs,
                        )

                    if isinstance(out, dict):
                        out.pop("position_ids", None)
                    return out

                q_len = int(getattr(self, "_q_len_for_kv", 0))

                if input_ids is not None and input_ids.size(1) > 1:
                    input_ids_last = input_ids[:, -1:]
                else:
                    input_ids_last = input_ids

                position_ids_last = None
                try:
                    if getattr(self, "_text_kv_abs_pos", None) is not None:
                        position_ids_last = self._text_kv_abs_pos + 1
                        self._text_kv_abs_pos = position_ids_last
                except Exception:
                    position_ids_last = None

                new_pkv, new_past_mask = _prune_past_kv_and_mask(
                    past_key_values, attention_mask, input_len=1, query_len=q_len
                )
                if new_pkv is None:
                    new_pkv = past_key_values

                attn2d = attention_mask
                try:
                    if torch.is_tensor(new_past_mask) and new_past_mask.dim() == 2:
                        B = int(new_past_mask.size(0))
                        cur_mask = new_past_mask.new_ones((B, 1), dtype=new_past_mask.dtype)
                        attn2d = torch.cat([new_past_mask, cur_mask], dim=1)
                except Exception:
                    attn2d = attention_mask

                model_inputs = {
                    "input_ids": input_ids_last,
                    "query_embeds": None,
                    "past_key_values": new_pkv,
                    "attention_mask": attn2d,
                    "use_cache": True if use_cache is None else use_cache,
                    "inputs_embeds": None,
                }
                if position_ids_last is not None:
                    model_inputs["position_ids"] = position_ids_last

                for k in ("output_attentions", "output_hidden_states", "return_dict"):
                    if k in kwargs:
                        model_inputs[k] = kwargs[k]

                return model_inputs


            #  patch
            self.opt_model.prepare_inputs_for_generation = _prepare_inputs_for_generation_with_kv.__get__(
                self.opt_model, type(self.opt_model)
            )
            try:
                if hasattr(self.opt_model, "config"):
                    self.opt_model.config.use_cache = True
            except Exception:
                pass
            logging.info("[Text-KV] patch installed successfully (mask+pos aligned).")

    def disable_text_kv_pruning(self):
        if hasattr(self, "_orig_prepare_inputs_for_generation"):
            self.opt_model.prepare_inputs_for_generation = self._orig_prepare_inputs_for_generation
        self._text_kv_prune_cfg = {"enabled": False}


    def attach_qformer_ffn_gates(self, group_size: int = 16):
        """
        """
        if getattr(self, "_ffn_gates_attached", False) and hasattr(self, "_ffn_gates"):
            logging.info("[FFN-Gate][qry] gates already attached; reuse existing gates.")
            return self._ffn_gates

        if not hasattr(self, "_ffn_orig_mids"):
            self._ffn_orig_mids = []
            for li, layer in enumerate(self.Qformer.bert.encoder.layer):
                if hasattr(layer, "intermediate_query"):
                    self._ffn_orig_mids.append(int(layer.intermediate_query.dense.out_features))
                else:
                    self._ffn_orig_mids.append(None)
            logging.info(f"[FFN-Gate][qry] recorded original mid dims: {self._ffn_orig_mids}")

        gates = []
        attached = 0
        for li, layer in enumerate(self.Qformer.bert.encoder.layer):
            if not (hasattr(layer, "intermediate_query") and hasattr(layer, "output_query")):
                logging.warning(f"[FFN-Gate][qry] layer {li}: no intermediate_query/output_query; skip.")
                continue
            inter_q = layer.intermediate_query
            w = inter_q.dense.weight
            dmid  = int(w.shape[0])
            act = inter_q.intermediate_act_fn
            wrapped = _ActWithGate(act, dmid, group_size, device=w.device, dtype=w.dtype)
            inter_q.intermediate_act_fn = wrapped
            gates.append(wrapped.gate)
            attached += 1
            logging.info(f"[FFN-Gate][qry] attach layer {li}: d_mid={dmid}, group={group_size}")

        if attached == 0:
            logging.warning("[FFN-Gate][qry] nothing attached (no query-FFN found)")
            return []

        self._ffn_gate_group_size = int(group_size)
        self._ffn_gates = gates
        setattr(self, "_ffn_gates_attached", True)
        logging.info(f"[FFN-Gate][qry] attached gates to {attached} query-FFN(s), group_size={int(group_size)}")

        return gates


    def ffn_gate_sparsity_loss(self, lambda_sparse: float = 1e-4, mode: str = "l1"):
        """


        """
        if not hasattr(self, "_ffn_gates"):
            return 0.0

        step = int(getattr(self, "_ffn_gate_step", 0) or 0)
        setattr(self, "_ffn_gate_step", step + 1)

        warm = getattr(self, "_ffn_gate_warmup_steps", 0.0)
        try:
            warm = float(warm)
        except Exception:
            warm = 0.0
        if warm > 0:
            scale = min(1.0, float(step + 1) / warm)
        else:
            scale = 1.0

        lam_eff = float(0.0 if lambda_sparse is None else lambda_sparse) * float(scale)
        if lam_eff <= 0.0:
            return 0.0

        p = getattr(self, "_ffn_gate_penalty_p", 1.0)
        try:
            p = float(p)
        except Exception:
            p = 1.0
        if p <= 0.0:
            p = 1.0

        mode_l = str(mode).lower()
        eps = 1e-6
        loss = 0.0

        if mode_l in ("l1", "mask_l1", "l1_mask", "lp", "mask_lp"):
            for g in self._ffn_gates:
                m = torch.sigmoid(g).clamp(min=eps, max=1.0 - eps)
                if abs(p - 1.0) < 1e-8:
                    loss = loss + m.mean()
                else:
                    loss = loss + m.pow(p).mean()

        elif mode_l in ("raw_l1", "l1_raw", "raw_lp", "lp_raw"):
            for g in self._ffn_gates:
                a = g.abs().clamp(min=eps)
                if abs(p - 1.0) < 1e-8:
                    loss = loss + a.mean()
                else:
                    loss = loss + a.pow(p).mean()

        elif mode_l in ("group_lasso", "gl", "mask_gl", "group_lasso_mask"):
            for g in self._ffn_gates:
                m = torch.sigmoid(g).clamp(min=eps, max=1.0 - eps)
                loss = loss + torch.sqrt((m ** 2).sum() + 1e-8) / float(m.numel())

        elif mode_l in ("raw_gl", "group_lasso_raw"):
            for g in self._ffn_gates:
                loss = loss + torch.sqrt((g ** 2).sum() + 1e-8) / float(g.numel())

        else:
            for g in self._ffn_gates:
                m = torch.sigmoid(g).clamp(min=eps, max=1.0 - eps)
                if abs(p - 1.0) < 1e-8:
                    loss = loss + m.mean()
                else:
                    loss = loss + m.pow(p).mean()

        return lam_eff * loss
    
    def set_ffn_gate_loss_cfg(self, lambda_sparse: float | None = None, mode: str = "l1", **kwargs):
        """

        - set_ffn_gate_loss_cfg(lmbda=..., group_size=..., p=..., warmup=...)

        """
        # 
        lmbda = kwargs.pop("lmbda", None)
        if lambda_sparse is None and lmbda is not None:
            lambda_sparse = lmbda

        self._ffn_gate_loss_lambda = float(0.0 if lambda_sparse is None else lambda_sparse)
        self._ffn_gate_loss_mode = str(mode)

        if "group_size" in kwargs and kwargs["group_size"] is not None:
            try:
                self._ffn_gate_group_size = int(kwargs["group_size"])
            except Exception:
                pass

        if "p" in kwargs and kwargs["p"] is not None:
            try:
                self._ffn_gate_penalty_p = float(kwargs["p"])
            except Exception:
                self._ffn_gate_penalty_p = 1.0
        else:
            if not hasattr(self, "_ffn_gate_penalty_p"):
                self._ffn_gate_penalty_p = 1.0

        warm = None
        if "warmup" in kwargs and kwargs["warmup"] is not None:
            warm = kwargs["warmup"]
        elif "warmup_steps" in kwargs and kwargs["warmup_steps"] is not None:
            warm = kwargs["warmup_steps"]
        if warm is not None:
            try:
                self._ffn_gate_warmup_steps = float(warm)
            except Exception:
                self._ffn_gate_warmup_steps = 0.0
        else:
            if not hasattr(self, "_ffn_gate_warmup_steps"):
                self._ffn_gate_warmup_steps = 0.0

        self._ffn_gate_step = 0


    @torch.no_grad()
    def finalize_qformer_ffn_gates(
        self,
        threshold: float = None,
        target_sparsity: float = None,
        min_keep_groups: int = 1,         # 
        max_prune_ratio: float = 0.9
    ):
        """
        -  intermediate_query.dense / output_query.dense
        """
        if not hasattr(self, "Qformer") or not hasattr(self.Qformer, "bert"):
            logging.warning("[FFN-Finalize][qry] Q-Former not found; skip.")
            return
        if not hasattr(self, "_ffn_gates"):
            logging.info("[FFN-Finalize][qry] skip: no gates attached (idempotent).")
            return

        group_size = getattr(self, "_ffn_gate_group_size", 16)
        layers = self.Qformer.bert.encoder.layer
        n_layers = len(layers)

        for li, layer in enumerate(layers):
            if not (hasattr(layer, "intermediate_query") and hasattr(layer, "output_query")):
                continue
            inter_mod = layer.intermediate_query
            out_mod   = layer.output_query

            # 
            actw = getattr(inter_mod, "intermediate_act_fn", None)
            if actw is None or not hasattr(actw, "gate"):
                continue

            gate = torch.sigmoid(actw.gate).detach().cpu()
            G = gate.numel()
            if G == 0:
                # 
                if hasattr(actw, "act"):
                    inter_mod.intermediate_act_fn = actw.act
                continue

            # 
            pruned_groups = []
            if target_sparsity is not None:
                sp = float(max(0.0, min(1.0, target_sparsity)))
                K = int(math.floor(G * sp))
                K = min(K, G - max(1, int(min_keep_groups)))
                K = min(K, int(math.floor(G * max_prune_ratio)))
                if K > 0:
                    pruned_groups = torch.topk(gate, k=K, largest=False, sorted=False).indices.tolist()
                used_mode = f"target_sparsity={sp} -> K={K}/{G}"
            elif threshold is not None:
                thr = float(threshold)
                pruned_groups = (gate < thr).nonzero(as_tuple=False).view(-1).tolist()
                maxK = int(math.floor(G * max_prune_ratio))
                if len(pruned_groups) > maxK:
                    order = torch.argsort(gate, dim=0, descending=False)
                    pruned_groups = order[:maxK].tolist()
                if G - len(pruned_groups) < max(1, int(min_keep_groups)):
                    pruned_groups = pruned_groups[: max(0, G - max(1, int(min_keep_groups)))]
                used_mode = f"threshold={thr:.4f} -> K={len(pruned_groups)}/{G}"
            else:
                if hasattr(actw, "act"):
                    inter_mod.intermediate_act_fn = actw.act
                logging.warning(f"[FFN-Finalize][qry] layer {li}: no pruning mode given; gate removed only.")
                continue

            # 
            dmid = int(inter_mod.dense.out_features)
            keep_groups = sorted(list(set(range(G)) - set(pruned_groups)))
            kept_idx = []
            for gi in keep_groups:
                start = gi * group_size
                end   = min((gi + 1) * group_size, dmid)
                kept_idx.extend(range(start, end))

            kept_idx = torch.as_tensor(kept_idx, dtype=torch.long)

            # 
            W1 = inter_mod.dense.weight      # [mid, in]
            b1 = inter_mod.dense.bias        # [mid]
            W2 = out_mod.dense.weight        # [out, mid]
            b2 = out_mod.dense.bias          # [out]

            device = W1.device
            dtype  = W1.dtype

            kept_idx = kept_idx.to(device=device)

            if kept_idx.numel() <= 0:
                kept_idx = torch.arange(0, min(group_size, dmid), dtype=torch.long, device=device)

            W1_new = W1.index_select(0, kept_idx).contiguous()
            b1_new = b1.index_select(0, kept_idx).contiguous() if b1 is not None else None
            W2_new = W2.index_select(1, kept_idx).contiguous()

            mid_new = int(kept_idx.numel())
            inter_new = nn.Linear(W1_new.size(1), mid_new, bias=(b1 is not None)).to(device, dtype=dtype)
            out_new   = nn.Linear(mid_new, W2_new.size(0), bias=(b2 is not None)).to(device, dtype=dtype)

            inter_new.weight.copy_(W1_new)
            if b1_new is not None: inter_new.bias.copy_(b1_new)
            out_new.weight.copy_(W2_new)
            if b2 is not None:     out_new.bias.copy_(b2)

            inter_mod.dense = inter_new
            out_mod.dense   = out_new

            if hasattr(actw, "act"):
                inter_mod.intermediate_act_fn = actw.act

            logging.info(f"[FFN-Finalize][qry] layer {li}: {used_mode}; mid {dmid} -> {mid_new} "
                        f"(groups pruned={len(pruned_groups)})")

        try:
            del self._ffn_gates
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ensure_runtime_head_mask_api(self):
        """
        """

        for _, _, sa in self._iter_cross_attn_modules():

            if not hasattr(sa, "set_runtime_head_mask"):
                sa._runtime_head_mask = None  #  None / [H] / [H,1,1] / [1,H,1,1]

                def _setter(mask, _sa=sa):
                    _sa._runtime_head_mask = mask
                sa.set_runtime_head_mask = _setter

            if not hasattr(sa, "get_runtime_head_mask"):
                sa.get_runtime_head_mask = (lambda _sa=sa: getattr(_sa, "_runtime_head_mask", None))

            if not hasattr(sa, "set_permanent_head_mask"):
                sa._permanent_head_mask = None
                def _set_perm_head_mask(m, _sa=sa):
                    _sa._permanent_head_mask = m
                sa.set_permanent_head_mask = _set_perm_head_mask
            if not hasattr(sa, "get_permanent_head_mask"):
                sa.get_permanent_head_mask = (lambda _sa=sa: getattr(_sa, "_permanent_head_mask", None))

            if not hasattr(sa, "clear_head_output_cache"):
                def _clear_head_output_cache(_sa=sa):
                    _sa._saved_head_ctx = None
                    _sa._saved_head_ctx_grad = None
                sa.clear_head_output_cache = _clear_head_output_cache

            if not hasattr(sa, "get_head_output"):
                sa.get_head_output = (lambda _sa=sa: getattr(_sa, "_saved_head_ctx", None))
            if not hasattr(sa, "get_head_output_grad"):
                sa.get_head_output_grad = (lambda _sa=sa: getattr(_sa, "_saved_head_ctx_grad", None))

            if not hasattr(sa, "_orig_forward"):
                sa._orig_forward = sa.forward

                def _forward_with_mask(*args, _sa=sa, **kwargs):
                    out = _sa._orig_forward(*args, **kwargs)

                    if isinstance(out, tuple):
                        context = out[0]
                    else:
                        context = out

                    if context is None:
                        return out

                    B, Q, C = context.shape

                    H = None
                    if hasattr(_sa, "num_attention_heads"):
                        H = int(_sa.num_attention_heads)
                    elif hasattr(_sa, "num_heads"):
                        H = int(_sa.num_heads)

                    if (not H) or (H <= 0):
                        if hasattr(_sa, "attention_head_size"):
                            head_dim = int(_sa.attention_head_size)
                            H = max(1, C // head_dim)
                        else:
                            return out

                    if C % H != 0:
                        return out
                    head_dim = C // H

                    # context [B,Q,C] -> ctx_heads [B,H,Q,Dh]
                    try:
                        ctx_heads = context.contiguous().view(B, Q, H, head_dim).permute(0, 2, 1, 3).contiguous()
                    except Exception:
                        return out

                    def _normalize_mask(mm, H, dtype, device):
                        if mm is None:
                            return None
                        m = torch.as_tensor(mm, dtype=dtype, device=device)
                        if m.dim() == 1:
                            flat = m
                        elif m.dim() == 3:
                            flat = m.reshape(-1)
                        elif m.dim() == 4:
                            #  [1,H,1,1] / [B,H,1,1] / [1,1,H,1]
                            if m.shape[1] == H:
                                return m[:1, :H, :1, :1]
                            if m.shape[2] == H:
                                return m[:1, :1, :H, :1].permute(0, 2, 1, 3).contiguous()
                            flat = m.reshape(-1)
                        else:
                            flat = m.reshape(-1)
                        if flat.numel() > H:
                            flat = flat[:H]
                        elif flat.numel() < H:
                            flat = torch.cat([flat, torch.ones(H - flat.numel(), dtype=dtype, device=device)], dim=0)
                        return flat.view(1, H, 1, 1)

                    perm = _normalize_mask(getattr(_sa, "_permanent_head_mask", None), H, ctx_heads.dtype, ctx_heads.device)
                    runt = _normalize_mask(getattr(_sa, "_runtime_head_mask", None), H, ctx_heads.dtype, ctx_heads.device)

                    eff_ctx_heads = ctx_heads if perm is None else (ctx_heads * perm)
                    try:
                        _sa._saved_head_ctx = eff_ctx_heads.detach()
                    except Exception:
                        _sa._saved_head_ctx = eff_ctx_heads
                    _sa._saved_head_ctx_grad = None
                    if eff_ctx_heads.requires_grad:
                        def _save_ctx_grad(g, _sa=_sa):
                            try:
                                _sa._saved_head_ctx_grad = g.detach()
                            except Exception:
                                _sa._saved_head_ctx_grad = g
                        eff_ctx_heads.register_hook(_save_ctx_grad)

                    # runtime mask  ablation / headdrop
                    final_ctx_heads = eff_ctx_heads if runt is None else (eff_ctx_heads * runt)

                    # [B,H,Q,Dh] -> [B,Q,C]
                    new_context = final_ctx_heads.permute(0, 2, 1, 3).contiguous().view(B, Q, C)

                    if isinstance(out, tuple):
                        return (new_context,) + out[1:]
                    else:
                        return new_context

                sa.forward = _forward_with_mask

        return True



    @torch.no_grad()
    def _replace_linear_in_attention(self, attn, rank: int, stats: dict):
        changed = 0
        if hasattr(attn, "self"):
            sa = attn.self
            for name in ["query", "key", "value"]:
                if hasattr(sa, name) and isinstance(getattr(sa, name), nn.Linear):
                    old = getattr(sa, name)
                    new = _linear_to_lowrank(old, rank)
                    setattr(sa, name, new)
                    changed += 1
        stats["qkv"] = stats.get("qkv", 0) + changed
        return changed


    @torch.no_grad()
    def _replace_linear_in_ffn(self, layer, rank: int, stats: dict):
        changed = 0
        if hasattr(layer, "intermediate_query") and hasattr(layer.intermediate_query, "dense"):
            lin = layer.intermediate_query.dense
            if isinstance(lin, nn.Linear):
                layer.intermediate_query.dense = _linear_to_lowrank(lin, rank)
                changed += 1
        if hasattr(layer, "output_query") and hasattr(layer.output_query, "dense"):
            lin = layer.output_query.dense
            if isinstance(lin, nn.Linear):
                layer.output_query.dense = _linear_to_lowrank(lin, rank)
                changed += 1
        stats["ffn"] = stats.get("ffn", 0) + changed
        return changed


    def apply_lowrank_to_qformer(self, rank: int = 64, scope=("qkv", "ffn")):
        stats = {}
        total = 0
        for li, layer in enumerate(self.Qformer.bert.encoder.layer):
            if "qkv" in scope and getattr(layer, "has_cross_attention", False):
                total += self._replace_linear_in_attention(layer.crossattention, rank, stats)
            if "ffn" in scope:
                total += self._replace_linear_in_ffn(layer, rank, stats)
        self._lowrank_stats = dict(replaced=total, rank=int(rank), detail=stats, scope=list(scope))
        logging.info(f"[LowRank] replaced {total} Linear -> LowRank(r={rank}) "
                    f"(qkv={stats.get('qkv',0)}, ffn={stats.get('ffn',0)})")
        return total


    def merge_similar_qformer_heads(
        self,
        dataset,
        device,
        *,
        num_batches: int = 8,
        batch_size: int = 4,
        sim_threshold: float = 0.97,
        method: str = "cosine",           # "cosine" | "cka"
        max_merge_per_layer: int = 1,
        num_workers: int = 0,
        collate_fn=None,
        default_text: Optional[str] = None,
        protect_strongest: bool = True,
        save_similarity_matrix: bool = False,
        save_similarity_per_layer: bool = False,
        similarity_save_dir: Optional[str] = None,
        similarity_save_prefix: str = "cross_attn_head_similarity",
        similarity_save_png: bool = True,
        similarity_save_dtype: str = "float16",
        similarity_pt_path: Optional[str] = None,
        min_heads_per_layer: int = 1,
        register_for_hard_prune: bool = True,
        final_keep_ratio_cap: Optional[float] = None,
        merge_mode: str = "mask",
        fusion_context_threshold: float = 0.90,
        fusion_alpha_min: float = 0.0,
        fusion_alpha_max: float = 2.0,
        fusion_zero_drop_output: bool = True,
        fusion_require_context: bool = True,
    ):
        """
        Stage 1 uses the attention-similarity matrix to propose redundant head pairs.

        merge_mode="mask" preserves the previous behaviour and only applies a permanent
        head mask. merge_mode="output_fusion" first calibrates the current per-head
        context outputs, absorbs the dropped head's output-projection contribution into
        the retained head, and then masks the dropped head. Stage 2 can subsequently
        recompute k-fold importance on the fused model and fill the remaining global
        pruning budget.

        If final_keep_ratio_cap is set, Stage 1 never removes more heads than the final
        global budget permits.
        """

        default_text = (default_text or getattr(self, "prompt", "") or "a photo of")
        pruned: Dict[int, List[int]] = {}
        layer_avg_patterns: Dict[int, torch.Tensor] = {}
        layer_similarity: Dict[int, torch.Tensor] = {}
        layer_strength: Dict[int, torch.Tensor] = {}
        loaded_similarity = None
        steps = 0
        batches_with_maps = 0

        self._heads_marked_for_hard_prune = getattr(self, "_heads_marked_for_hard_prune", dict())

        if similarity_pt_path:
            try:
                loaded_similarity = _load_head_similarity_payload(similarity_pt_path)
                method_from_file = str(loaded_similarity.get("method", method) or method)
                if str(method).lower() != str(method_from_file).lower():
                    logging.info(
                        f"[Head-Merge] override method from similarity file: cfg={method} -> file={method_from_file}"
                    )
                method = method_from_file
                sim_global = loaded_similarity["similarity"].detach().float().cpu()
                for item in loaded_similarity.get("layer_ranges", []) or []:
                    li = int(item.get("layer_idx", -1))
                    st = int(item.get("start", 0))
                    ed = int(item.get("end", 0))
                    if li < 0 or ed <= st:
                        continue
                    layer_similarity[li] = sim_global[st:ed, st:ed].clone()
                logging.info(
                    f"[Head-Merge] using offline similarity file: path={similarity_pt_path}, "
                    f"shape={tuple(sim_global.shape)}, method={method}, "
                    f"layers={sorted(layer_similarity.keys())}"
                )
            except Exception as e:
                logging.warning(f"[Head-Merge] failed to load similarity file '{similarity_pt_path}': {e}")
                loaded_similarity = None
                layer_similarity = {}

        if loaded_similarity is None:
            if dataset is None:
                logging.warning("[Head-Merge] dataset is None and no similarity_pt_path provided; skip.")
                self._head_merge_info = dict(
                    threshold=sim_threshold,
                    method=method,
                    merged=pruned,
                    similarity_pt_path=similarity_pt_path,
                    min_heads_per_layer=int(min_heads_per_layer),
                    final_keep_ratio_cap=final_keep_ratio_cap,
                )
                return pruned

            dl = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=False,
            )

            avg_maps: Dict[int, List[torch.Tensor]] = {}

            def _collect_current_cross_attn_maps(expect_b: int, expect_n: int):
                """
                Collect cross-attention maps from the current Q-Former pass.
                Only maps whose batch size and visual-token length match the expected
                current path are accepted, so stale full-token maps cannot be mixed with
                pruned-token maps.
                """
                maps = []
                for _li, _ca, _sa in self._iter_cross_attn_modules():
                    attn = getattr(_sa, "_saved_attn_map", None)
                    if attn is None:
                        get_map = getattr(_sa, "get_attention_map", None)
                        attn = get_map() if callable(get_map) else None
                    if attn is None or (not torch.is_tensor(attn)) or attn.dim() != 4:
                        continue
                    if int(attn.shape[0]) != int(expect_b) or int(attn.shape[-1]) != int(expect_n):
                        continue
                    maps.append((int(_li), attn.detach().float()))
                return maps

            def _run_qformer_for_head_similarity(samples):
                """
                Run only visual encoder + Q-Former for head-similarity collection.
                This avoids calling self(samples), whose runtime vtoken branch may turn
                cross-attention saving off or clear the attention cache before the
                similarity collector reads it.
                """
                image = samples.get("image", None)
                if image is None:
                    raise KeyError("samples must contain 'image' for head similarity collection.")

                with self.maybe_autocast():
                    image_embeds = self.ln_vision(self.visual_encoder(image))

                image_atts = torch.ones(
                    image_embeds.size()[:-1],
                    dtype=torch.long,
                    device=image_embeds.device,
                )

                B = int(image_embeds.shape[0])
                N_full = int(image_embeds.shape[1])

                v_keep = getattr(self, "_vision_token_keep_ratio", 1.0)
                v_keep = 1.0 if v_keep is None else float(v_keep)

                score_mode = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
                tail_strategy = str(getattr(self, "_vision_token_tail_strategy", "drop")).lower()

                mode_for_attn = score_mode
                if score_mode in ("learned", "mlp", "score_net"):
                    mode_for_attn = str(
                        getattr(self, "_vision_token_learned_teacher_mode", "entropy")
                    ).lower()

                need_attn_for_vtoken = (
                    v_keep < 1.0
                    and mode_for_attn in (
                        "attn_entropy",
                        "attn",
                        "entropy",
                        "attn_mean",
                        "attn_grad",
                        "grad",
                        "taylor_attn",
                    )
                )

                attn_for_score = None

                # First pass: full-token Q-Former attention for runtime vtoken scoring.
                # This pass is used only to score/select visual tokens, not for the
                # final head-similarity matrix.
                if need_attn_for_vtoken:
                    self._toggle_cross_attn_save(True)
                    query_tokens_full = self.query_tokens.expand(B, -1, -1)
                    with torch.no_grad():
                        _ = self.Qformer.bert(
                            query_embeds=query_tokens_full,
                            encoder_hidden_states=image_embeds,
                            encoder_attention_mask=image_atts,
                            return_dict=True,
                        )

                    warm_maps = [m for _, m in _collect_current_cross_attn_maps(expect_b=B, expect_n=N_full)]
                    if len(warm_maps) > 0:
                        base_shape = tuple(warm_maps[0].shape)
                        warm_maps = [m for m in warm_maps if tuple(m.shape) == base_shape]
                        if len(warm_maps) > 0:
                            attn_for_score = torch.stack(warm_maps, dim=0).mean(dim=0)

                # Apply the same runtime visual-token setting as the current run.
                # If keep=1.0 this branch is skipped, so the full-token path is measured.
                if v_keep < 1.0 and hasattr(self, "_select_vision_tokens"):
                    image_embeds, image_atts = self._select_vision_tokens(
                        image_embeds,
                        image_atts,
                        attn_bhqk=attn_for_score,
                        attn_grad_bhqk=None,
                        keep_ratio=v_keep,
                        mode=score_mode,
                        tail_strategy=tail_strategy,
                    )

                image_atts = image_atts.to(device=image_embeds.device, dtype=torch.long)
                B_after = int(image_embeds.shape[0])
                N_after = int(image_embeds.shape[1])

                # Second pass: this is the actual path whose cross-attention head
                # patterns are saved. For vtoken-on it uses the selected visual-token
                # sequence; for vtoken-off it uses the full visual-token sequence.
                self._toggle_cross_attn_save(True)
                query_tokens = self.query_tokens.expand(B_after, -1, -1)
                with torch.no_grad():
                    _ = self.Qformer.bert(
                        query_embeds=query_tokens,
                        encoder_hidden_states=image_embeds,
                        encoder_attention_mask=image_atts,
                        return_dict=True,
                    )

                return _collect_current_cross_attn_maps(expect_b=B_after, expect_n=N_after)

            try:
                amp_ctx = nullcontext()
                for samples in dl:
                    if steps >= num_batches:
                        break

                    for k, v in list(samples.items()):
                        if torch.is_tensor(v):
                            samples[k] = v.to(device, non_blocking=True)

                    if "text_input" not in samples:
                        bs = None
                        if "image" in samples and hasattr(samples["image"], "size"):
                            try:
                                bs = samples["image"].size(0)
                            except Exception:
                                bs = None
                        if bs is None:
                            for v in samples.values():
                                if torch.is_tensor(v) and v.dim() > 0:
                                    bs = int(v.size(0))
                                    break
                        bs = bs or batch_size
                        samples["text_input"] = [default_text] * bs

                    with amp_ctx:
                        maps = _run_qformer_for_head_similarity(samples)

                    got_map = False
                    for li, attn in maps:
                        pat = _avg_head_pattern(attn.detach().float())
                        if torch.isnan(pat).any() or torch.isinf(pat).any():
                            continue
                        avg_maps.setdefault(int(li), []).append(pat)
                        got_map = True

                    if got_map:
                        batches_with_maps += 1
                    steps += 1
            finally:
                self._toggle_cross_attn_save(False)

            for li, mats in sorted(avg_maps.items()):
                if not mats:
                    continue
                A = torch.stack(mats, dim=0).mean(dim=0)
                if torch.isnan(A).any() or torch.isinf(A).any():
                    continue
                layer_avg_patterns[int(li)] = A.detach().float()
                layer_strength[int(li)] = A.detach().float().norm(dim=-1)
                layer_similarity[int(li)] = _pairwise_head_similarity(A.detach().float(), method=method)

        logging.info(
            f"[Head-Sim] collection summary: steps={steps}, batches_with_maps={batches_with_maps}, "
            f"layers={sorted(layer_similarity.keys())}"
        )
        if save_similarity_matrix and not (layer_avg_patterns or loaded_similarity is not None):
            logging.warning("[Head-Sim] no cross-attn maps collected; skip saving similarity matrix.")

        if save_similarity_matrix and layer_avg_patterns:
            try:
                save_dir = similarity_save_dir or os.path.join(os.getcwd(), "head_similarity")
                prefix = str(similarity_save_prefix or "cross_attn_head_similarity")
                global_parts: List[torch.Tensor] = []
                head_meta: List[Dict[str, int]] = []
                layer_ranges: List[Dict[str, int]] = []
                cursor = 0

                for li, A in sorted(layer_avg_patterns.items()):
                    A_cpu = A.detach().cpu()
                    H = int(A_cpu.shape[0])
                    start = cursor
                    end = cursor + H
                    layer_ranges.append({"layer_idx": int(li), "start": int(start), "end": int(end)})
                    for h in range(H):
                        head_meta.append({
                            "global_index": int(cursor + h),
                            "layer_idx": int(li),
                            "head_idx": int(h),
                        })
                    cursor = end
                    global_parts.append(A_cpu)

                    if save_similarity_per_layer:
                        sims_layer = _pairwise_head_similarity(A_cpu, method=method)
                        _save_head_similarity_artifacts(
                            sims_layer,
                            save_dir=save_dir,
                            prefix=f"{prefix}_layer{int(li):02d}",
                            method=method,
                            head_meta=[{"global_index": int(h), "layer_idx": int(li), "head_idx": int(h)} for h in range(H)],
                            layer_ranges=[{"layer_idx": int(li), "start": 0, "end": int(H)}],
                            save_png=bool(similarity_save_png),
                            save_dtype=similarity_save_dtype,
                        )

                if global_parts:
                    A_global = torch.cat(global_parts, dim=0)
                    sims_global = _pairwise_head_similarity(A_global, method=method)
                    paths = _save_head_similarity_artifacts(
                        sims_global,
                        save_dir=save_dir,
                        prefix=f"{prefix}_global",
                        method=method,
                        head_meta=head_meta,
                        layer_ranges=layer_ranges,
                        save_png=bool(similarity_save_png),
                        save_dtype=similarity_save_dtype,
                    )
                    bytes_per = torch.tensor([], dtype=torch.float16 if str(similarity_save_dtype).lower() in ("float16", "fp16", "half") else torch.float32).element_size()
                    est_mb = (int(sims_global.numel()) * int(bytes_per)) / (1024.0 * 1024.0)
                    logging.info(
                        f"[Head-Sim] saved global matrix: shape={tuple(sims_global.shape)}, method={method}, "
                        f"dir={save_dir}, est_tensor={est_mb:.4f} MB, pt={paths['pt']}"
                    )
            except Exception as e:
                logging.warning(f"[Head-Sim] save skipped: {e}")

        modules = list(self._iter_cross_attn_modules())
        if not modules:
            self._head_merge_info = dict(
                threshold=sim_threshold,
                method=method,
                merged=pruned,
                similarity_pt_path=similarity_pt_path,
                min_heads_per_layer=int(min_heads_per_layer),
                final_keep_ratio_cap=final_keep_ratio_cap,
            )
            return pruned

        def _active_heads_for_layer(li, sa):
            active = self._get_active_qformer_head_indices(sa)
            H = int(getattr(sa, "num_attention_heads", 0) or 0)
            return [int(h) for h in active if 0 <= int(h) < H]

        total_heads_all = 0
        total_active_all = 0
        for li, _, sa in modules:
            H = int(getattr(sa, "num_attention_heads", 0) or 0)
            total_heads_all += H
            total_active_all += len(_active_heads_for_layer(int(li), sa))
        already_pruned = max(int(total_heads_all - total_active_all), 0)
        final_budget = None
        if final_keep_ratio_cap is not None:
            keep = min(max(float(final_keep_ratio_cap), 0.0), 1.0)
            final_target_total = int(round(total_heads_all * (1.0 - keep)))
            final_budget = max(final_target_total - already_pruned, 0)

        # Build same-layer attention-similarity candidate pairs.  Candidate selection is
        # deliberately separated from the action (mask or fusion), so both modes use the
        # same offline similarity evidence and the same final pruning budget.
        merge_mode = str(merge_mode or "mask").lower().strip()
        if merge_mode in ("fusion", "fuse", "true_fusion", "output", "output-projection"):
            merge_mode = "output_fusion"
        elif merge_mode not in ("mask", "soft_mask", "output_fusion"):
            raise ValueError(f"Unsupported head merge_mode={merge_mode!r}")
        if merge_mode == "soft_mask":
            merge_mode = "mask"

        floor = max(int(min_heads_per_layer), 1)
        pair_pool: Dict[int, List[Tuple[int, int, float]]] = {}
        layer_caps: Dict[int, int] = {}

        for li, _, sa in modules:
            li = int(li)
            sims = layer_similarity.get(li, None)
            if sims is None:
                continue
            H = int(getattr(sa, "num_attention_heads", 0) or 0)
            if H <= 1:
                continue
            active_heads = _active_heads_for_layer(li, sa)
            if len(active_heads) <= 1:
                continue
            per_layer_cap = max(len(active_heads) - floor, 0)
            if per_layer_cap <= 0 or int(max_merge_per_layer) == 0:
                continue
            hard_cap = per_layer_cap if int(max_merge_per_layer) < 0 else min(int(max_merge_per_layer), per_layer_cap)
            if hard_cap <= 0:
                continue

            sims_local = sims.detach().float().cpu()
            pairs = [
                (int(i), int(j), float(sims_local[int(i), int(j)].item()))
                for idx_i, i in enumerate(active_heads)
                for j in active_heads[idx_i + 1:]
                if float(sims_local[int(i), int(j)].item()) >= float(sim_threshold)
            ]
            pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
            if pairs:
                pair_pool[li] = pairs
                layer_caps[li] = int(hard_cap)

        if not pair_pool:
            logging.info(
                f"[Head-Merge] no similar heads found (thr={sim_threshold}, method={method}, mode={merge_mode})."
            )
            self._head_merge_info = dict(
                threshold=sim_threshold,
                method=method,
                mode=merge_mode,
                merged=pruned,
                similarity_pt_path=similarity_pt_path,
                min_heads_per_layer=int(min_heads_per_layer),
                final_keep_ratio_cap=final_keep_ratio_cap,
            )
            return pruned

        fusion_stats: Dict[int, torch.Tensor] = {}
        fusion_steps = 0

        if merge_mode == "output_fusion":
            if dataset is None:
                if bool(fusion_require_context):
                    raise ValueError("output_fusion requires a calibration dataset")
                logging.warning(
                    "[Head-Fusion] calibration dataset is None; falling back to attention-only soft masking."
                )
                merge_mode = "mask"
            else:
                # The existing runtime wrapper exposes the true per-head context output
                # [B,H,Q,Dh] before output.dense.  No Qformer.py modification is needed.
                self._ensure_runtime_head_mask_api()
                calibration_loader = DataLoader(
                    dataset,
                    batch_size=int(batch_size),
                    shuffle=False,
                    num_workers=int(num_workers),
                    pin_memory=True,
                    collate_fn=collate_fn,
                    drop_last=False,
                )

                was_training = bool(self.training)
                saved_save_attention = {
                    int(li): bool(getattr(sa, "save_attention", False))
                    for li, _, sa in modules
                }
                try:
                    self.eval()
                    self._toggle_cross_attn_save(False)
                    for _, _, sa in modules:
                        if hasattr(sa, "set_runtime_head_mask"):
                            sa.set_runtime_head_mask(None)
                        if hasattr(sa, "clear_head_output_cache"):
                            sa.clear_head_output_cache()

                    with torch.no_grad():
                        for samples in calibration_loader:
                            if fusion_steps >= int(num_batches):
                                break
                            if not isinstance(samples, dict):
                                continue
                            for k, v in list(samples.items()):
                                if torch.is_tensor(v):
                                    samples[k] = v.to(device, non_blocking=True)
                            image = samples.get("image", None)
                            if image is None:
                                continue

                            with self.maybe_autocast():
                                image_embeds = self.ln_vision(self.visual_encoder(image))
                            image_atts = torch.ones(
                                image_embeds.size()[:-1],
                                dtype=torch.long,
                                device=image_embeds.device,
                            )
                            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                            _ = self.Qformer.bert(
                                query_embeds=query_tokens,
                                encoder_hidden_states=image_embeds,
                                encoder_attention_mask=image_atts,
                                return_dict=True,
                            )

                            got_any = False
                            for li, _, sa in modules:
                                li = int(li)
                                if li not in pair_pool:
                                    continue
                                get_ctx = getattr(sa, "get_head_output", None)
                                ctx = get_ctx() if callable(get_ctx) else getattr(sa, "_saved_head_ctx", None)
                                if ctx is None or (not torch.is_tensor(ctx)) or ctx.dim() != 4:
                                    continue
                                # [B,H,Q,Dh] -> [H, B*Q*Dh], then accumulate the
                                # head-wise Gram matrix.  Its diagonal is head energy and
                                # its off-diagonal entries are cross-head inner products.
                                H = int(ctx.shape[1])
                                flat = ctx.detach().float().permute(1, 0, 2, 3).contiguous().view(H, -1)
                                gram = flat @ flat.transpose(0, 1)
                                if li not in fusion_stats:
                                    fusion_stats[li] = gram.cpu()
                                else:
                                    fusion_stats[li] += gram.cpu()
                                got_any = True
                            if got_any:
                                fusion_steps += 1
                finally:
                    for li, _, sa in modules:
                        try:
                            sa.save_attention = saved_save_attention.get(int(li), False)
                        except Exception:
                            pass
                    if was_training:
                        self.train()
                    else:
                        self.eval()

                logging.info(
                    f"[Head-Fusion] context calibration finished: steps={fusion_steps}, "
                    f"layers={sorted(fusion_stats.keys())}, context_thr={float(fusion_context_threshold):.4f}"
                )
                if fusion_steps <= 0 and bool(fusion_require_context):
                    raise RuntimeError("head fusion collected no valid per-head context outputs")
                if fusion_steps <= 0:
                    logging.warning(
                        "[Head-Fusion] no context outputs collected; falling back to attention-only soft masking."
                    )
                    merge_mode = "mask"

        selected_records: List[Dict[str, Union[int, float, str]]] = []

        if merge_mode == "output_fusion":
            eps = 1e-12
            alpha_lo = float(min(fusion_alpha_min, fusion_alpha_max))
            alpha_hi = float(max(fusion_alpha_min, fusion_alpha_max))

            for li, pairs in sorted(pair_pool.items()):
                gram = fusion_stats.get(int(li), None)
                if gram is None:
                    continue
                H = int(gram.shape[0])
                used_heads: Set[int] = set()
                accepted = 0
                for i, j, attn_sim in pairs:
                    if i in used_heads or j in used_heads:
                        continue
                    if not (0 <= i < H and 0 <= j < H):
                        continue
                    e_i = max(float(gram[i, i].item()), 0.0)
                    e_j = max(float(gram[j, j].item()), 0.0)
                    cross = float(gram[i, j].item())
                    ctx_cos = cross / max((e_i * e_j) ** 0.5, eps)
                    ctx_cos = max(min(ctx_cos, 1.0), -1.0)
                    if bool(fusion_require_context) and ctx_cos < float(fusion_context_threshold):
                        logging.info(
                            f"[Head-Fusion] reject L{li} H{i}<->H{j}: "
                            f"attn_sim={attn_sim:.6f}, context_cos={ctx_cos:.6f}"
                        )
                        continue

                    # Keep the stronger context head.  The least-squares scalar maps
                    # keep-context to drop-context, so W_drop can be absorbed into W_keep.
                    if protect_strongest:
                        keep_h, drop_h = (i, j) if e_i >= e_j else (j, i)
                    else:
                        keep_h, drop_h = (min(i, j), max(i, j))
                    e_keep = max(float(gram[keep_h, keep_h].item()), eps)
                    e_drop = max(float(gram[drop_h, drop_h].item()), eps)
                    cross_kd = float(gram[keep_h, drop_h].item())
                    alpha = cross_kd / e_keep
                    alpha = max(alpha_lo, min(alpha_hi, alpha))
                    residual = max(
                        e_drop - 2.0 * alpha * cross_kd + (alpha * alpha) * e_keep,
                        0.0,
                    )
                    rel_error = residual / e_drop

                    selected_records.append({
                        "mode": "output_fusion",
                        "layer": int(li),
                        "keep": int(keep_h),
                        "drop": int(drop_h),
                        "alpha": float(alpha),
                        "attn_similarity": float(attn_sim),
                        "context_cosine": float(ctx_cos),
                        "relative_error": float(rel_error),
                    })
                    used_heads.update((int(i), int(j)))
                    accepted += 1
                    if accepted >= int(layer_caps.get(li, 0)):
                        break
        else:
            # Backward-compatible soft-mask mode.  Use non-overlapping pairs and, when
            # online head strength is available, protect the strongest head in the layer.
            for li, pairs in sorted(pair_pool.items()):
                sa = self.Qformer.bert.encoder.layer[int(li)].crossattention.self
                active_heads = _active_heads_for_layer(int(li), sa)
                strength = layer_strength.get(int(li), None)
                if torch.is_tensor(strength) and int(strength.numel()) == int(getattr(sa, "num_attention_heads", 0)):
                    strength = strength.detach().float().cpu()
                else:
                    strength = None
                protected: Set[int] = set()
                if protect_strongest and active_heads:
                    strongest = (
                        max(active_heads, key=lambda h: (float(strength[int(h)].item()), -int(h)))
                        if strength is not None
                        else min(active_heads)
                    )
                    protected.add(int(strongest))

                used_heads: Set[int] = set()
                accepted = 0
                for i, j, attn_sim in pairs:
                    if i in used_heads or j in used_heads:
                        continue
                    if i in protected and j in protected:
                        continue
                    if i in protected:
                        drop_h = j
                    elif j in protected:
                        drop_h = i
                    elif strength is not None:
                        drop_h = i if float(strength[i]) <= float(strength[j]) else j
                    else:
                        drop_h = max(i, j)
                    selected_records.append({
                        "mode": "mask",
                        "layer": int(li),
                        "keep": int(j if drop_h == i else i),
                        "drop": int(drop_h),
                        "alpha": 0.0,
                        "attn_similarity": float(attn_sim),
                        "context_cosine": float("nan"),
                        "relative_error": float("nan"),
                    })
                    used_heads.update((int(i), int(j)))
                    accepted += 1
                    if accepted >= int(layer_caps.get(li, 0)):
                        break

        selected_records.sort(
            key=lambda r: (
                -float(r["attn_similarity"]),
                -float(r["context_cosine"]) if math.isfinite(float(r["context_cosine"])) else 0.0,
                int(r["layer"]),
                int(r["drop"]),
            )
        )
        if final_budget is not None:
            selected_records = selected_records[:max(int(final_budget), 0)]

        if not selected_records:
            logging.info(
                f"[Head-Merge] no candidates survived action checks "
                f"(thr={sim_threshold}, mode={merge_mode}, context_thr={fusion_context_threshold})."
            )
            self._head_merge_info = dict(
                threshold=sim_threshold,
                method=method,
                mode=merge_mode,
                merged=pruned,
                similarity_pt_path=similarity_pt_path,
                min_heads_per_layer=int(min_heads_per_layer),
                final_keep_ratio_cap=final_keep_ratio_cap,
                fusion_records=[],
            )
            return pruned

        selected_by_layer: Dict[int, List[Dict[str, Union[int, float, str]]]] = {}
        for record in selected_records:
            selected_by_layer.setdefault(int(record["layer"]), []).append(record)

        self._ensure_runtime_head_mask_api()
        applied_fusions: List[Dict[str, Union[int, float, str]]] = []
        for li, records in sorted(selected_by_layer.items()):
            layer = self.Qformer.bert.encoder.layer[int(li)]
            sa = layer.crossattention.self
            out_dense = layer.crossattention.output.dense
            H_now = int(getattr(sa, "num_attention_heads", 0) or 0)
            head_dim = int(getattr(sa, "attention_head_size", 0) or 0)
            device0 = next(sa.parameters()).device
            dtype0 = next(sa.parameters()).dtype
            base = getattr(sa, "_permanent_head_mask", None)
            if (base is None) or (int(base.numel()) != H_now):
                base = torch.ones(H_now, 1, 1, device=device0, dtype=dtype0)
            else:
                base = base.to(device0, dtype=dtype0).view(H_now, 1, 1)

            dropped = []
            for record in records:
                keep_h = int(record["keep"])
                drop_h = int(record["drop"])
                if not (0 <= keep_h < H_now and 0 <= drop_h < H_now) or keep_h == drop_h:
                    continue

                if str(record["mode"]) == "output_fusion":
                    if not isinstance(out_dense, nn.Linear):
                        raise TypeError(
                            f"Head fusion requires nn.Linear output.dense, got {type(out_dense)} at layer {li}"
                        )
                    expected_in = H_now * head_dim
                    if int(out_dense.weight.shape[1]) != expected_in:
                        raise RuntimeError(
                            f"Unexpected output.dense shape at layer {li}: {tuple(out_dense.weight.shape)}, "
                            f"expected in_features={expected_in}"
                        )
                    ks = slice(keep_h * head_dim, (keep_h + 1) * head_dim)
                    ds = slice(drop_h * head_dim, (drop_h + 1) * head_dim)
                    alpha = float(record["alpha"])
                    with torch.no_grad():
                        out_dense.weight[:, ks].add_(out_dense.weight[:, ds], alpha=alpha)
                        if bool(fusion_zero_drop_output):
                            out_dense.weight[:, ds].zero_()
                    applied_fusions.append(dict(record))
                    logging.info(
                        f"[Head-Fusion] L{li}: H{drop_h} -> H{keep_h}, alpha={alpha:.6f}, "
                        f"attn_sim={float(record['attn_similarity']):.6f}, "
                        f"context_cos={float(record['context_cosine']):.6f}, "
                        f"rel_error={float(record['relative_error']):.6f}, "
                        f"zero_drop_output={bool(fusion_zero_drop_output)}"
                    )

                base[drop_h, 0, 0] = 0.0
                dropped.append(drop_h)

            if dropped:
                sa.set_permanent_head_mask(base)
                sa.set_runtime_head_mask(None)
                pruned[int(li)] = sorted(set(int(x) for x in dropped))
                if register_for_hard_prune:
                    marked = set(int(x) for x in self._heads_marked_for_hard_prune.get(int(li), []))
                    marked |= set(pruned[int(li)])
                    self._heads_marked_for_hard_prune[int(li)] = sorted(marked)
                logging.info(
                    f"[Head-Merge] Layer {int(li)}: mode={merge_mode}, inactive_heads={pruned[int(li)]} "
                    f"(H unchanged={H_now}, thr={sim_threshold}, method={method}"
                    f"{', offline_pt=' + str(similarity_pt_path) if similarity_pt_path else ''})"
                )

        self._head_fusion_info = {
            "mode": merge_mode,
            "context_threshold": float(fusion_context_threshold),
            "alpha_min": float(fusion_alpha_min),
            "alpha_max": float(fusion_alpha_max),
            "zero_drop_output": bool(fusion_zero_drop_output),
            "calibration_steps": int(fusion_steps),
            "records": applied_fusions,
        }

        logging.info(
            f"[Head-Merge][threshold] mode={merge_mode} thr={sim_threshold:.6g} "
            f"total_heads={total_heads_all} total_active={total_active_all} "
            f"already_pruned={already_pruned} final_keep_cap={final_keep_ratio_cap} "
            f"additional_budget={final_budget} selected_total={sum(len(v) for v in pruned.values())}"
        )

        self._head_merge_info = dict(
            threshold=sim_threshold,
            method=method,
            mode=merge_mode,
            merged=pruned,
            similarity_pt_path=similarity_pt_path,
            min_heads_per_layer=int(min_heads_per_layer),
            final_keep_ratio_cap=final_keep_ratio_cap,
            fusion_records=applied_fusions,
            fusion_context_threshold=float(fusion_context_threshold),
            fusion_calibration_steps=int(fusion_steps),
        )
        return pruned

    def enable_random_head_dropout(self, prob: float = 0.1, drop_prob: Optional[float] = None):
        """
        """
        if drop_prob is not None:
            prob = drop_prob
        self._ensure_runtime_head_mask_api()
        self._headdrop_prob = float(prob)
        self._headdrop_hits = 0

        def _pre_step_mask():
            for _, _, sa in self._iter_cross_attn_modules():
                H = int(sa.num_attention_heads)
                mask = torch.ones(H, 1, 1, device=next(sa.parameters()).device)
                if self._headdrop_prob > 0:
                    drop = torch.rand(H, device=mask.device) < self._headdrop_prob
                    if drop.all():  # 
                        drop[torch.randint(0, H, (1,)).item()] = False
                    mask[drop, 0, 0] = 0.0
                    if drop.any():
                        sa.set_runtime_head_mask(mask)
                        self._headdrop_hits += int(drop.sum().item() > 0)
                    else:
                        sa.set_runtime_head_mask(None)
                else:
                    sa.set_runtime_head_mask(None)

        self._headdrop_step_fn = _pre_step_mask
        logging.info(f"[HeadDrop] enabled with p={self._headdrop_prob}.")
        return True

    def _infer_visual_token_counts_for_flops(self):
        """
        - v_full:  token 
        """
        n_full = int(getattr(self, "_vision_token_last_n_full", 0) or 0)
        n_after = int(getattr(self, "_vision_token_last_n_after", 0) or 0)

        st = getattr(self, "_vision_token_usage_stats", None)
        if isinstance(st, dict):
            calls = int(st.get("calls", 0) or 0)
            total_raw = float(st.get("total_raw_tokens", 0.0) or 0.0)
            total_used = float(st.get("total_used_tokens", 0.0) or 0.0)

            if n_full <= 0 and calls > 0 and total_raw > 0:
                n_full = int(round(total_raw / float(calls)))
            if n_after <= 0 and calls > 0 and total_used > 0:
                n_after = int(round(total_used / float(calls)))

        v_current = n_after if n_after > 0 else n_full

        return {
            "v_full": int(n_full),
            "v_current": int(v_current),
            "v_after": int(n_after),
        }

    def _get_qformer_flops_breakdown(self):
        """
        1) query-FFN
        """
        out = {
            "mids": [],
            "q_len": int(getattr(self, "num_query_token", 32)),
            "hidden": int(getattr(self.Qformer.config, "hidden_size", 0)),
            "heads": [],
            "head_dims": [],
            "q_ffn": 0.0,
            "xattn_const": 0.0,
            "xattn_var_current": 0.0,
            "xattn_var_full": 0.0,
            "current_total": None,
            "full_total": None,
            "saved": None,
            "saved_pct": None,
            "v_full": 0,
            "v_current": 0,
        }

        q_len = out["q_len"]
        hidden = out["hidden"]

        counts = self._infer_visual_token_counts_for_flops()
        v_full = int(counts["v_full"])
        v_current = int(counts["v_current"])
        out["v_full"] = v_full
        out["v_current"] = v_current

        cross_specs = []

        for li, layer in enumerate(self.Qformer.bert.encoder.layer):
            # ---- Query FFN ----
            if hasattr(layer, "intermediate_query") and hasattr(layer.intermediate_query, "dense"):
                mid = int(layer.intermediate_query.dense.out_features)
                out["mids"].append(mid)
                # 2 * (up proj + down proj) 
                out["q_ffn"] += 4.0 * hidden * mid * q_len
            else:
                out["mids"].append(None)

            # ---- Cross Attention ----
            if getattr(layer, "has_cross_attention", False):
                sa = layer.crossattention.self

                n_heads = int(getattr(sa, "num_attention_heads", 0) or 0)
                all_head_size = int(getattr(sa, "all_head_size", hidden) or hidden)
                head_dim = int(getattr(sa, "attention_head_size", 0) or (all_head_size // max(n_heads, 1)))

                out["heads"].append(n_heads)
                out["head_dims"].append(head_dim)
                cross_specs.append((n_heads, all_head_size, head_dim))

                out["xattn_const"] += 2.0 * q_len * hidden * all_head_size
                out["xattn_const"] += 2.0 * q_len * all_head_size * hidden

        if v_full > 0:
            for _, all_head_size, _ in cross_specs:
                # K/V projection
                out["xattn_var_full"] += 4.0 * v_full * hidden * all_head_size
                # QK^T + Attn@V
                out["xattn_var_full"] += 4.0 * q_len * v_full * all_head_size

        if v_current > 0:
            for _, all_head_size, _ in cross_specs:
                out["xattn_var_current"] += 4.0 * v_current * hidden * all_head_size
                out["xattn_var_current"] += 4.0 * q_len * v_current * all_head_size

        if v_current > 0:
            out["current_total"] = out["q_ffn"] + out["xattn_const"] + out["xattn_var_current"]

        if v_full > 0:
            out["full_total"] = out["q_ffn"] + out["xattn_const"] + out["xattn_var_full"]

        if (out["current_total"] is not None) and (out["full_total"] is not None):
            out["saved"] = out["full_total"] - out["current_total"]
            denom = max(out["full_total"], 1e-12)
            out["saved_pct"] = 100.0 * out["saved"] / denom

        return out

    def _get_vtoken_flops_breakdown(
        self,
        *,
        image_embeds_full: torch.Tensor,
        image_embeds_used: torch.Tensor,
        attn_bhqk: Optional[torch.Tensor] = None,
        score_mode: str = "l2norm",
        tail_strategy: str = "drop",
        did_refresh: bool = False,
        used_grad: bool = False,
    ) -> Dict[str, float]:
        """

        1) attn:
        2) select:
        - topk/select  proxy
        - gather selected tokens
        - merge / tail_token  tail reduction proxy
        3) cache:
        - attn / grad cache  copy proxy

        """

        out = {
            "attn": 0.0,
            "select": 0.0,
            "cache": 0.0,
            "total": 0.0,
        }

        if image_embeds_full is None or image_embeds_used is None:
            return out

        B = int(image_embeds_full.size(0))
        N_full = int(image_embeds_full.size(1))
        C = int(image_embeds_full.size(2))
        N_used = int(image_embeds_used.size(1))
        N_drop = max(N_full - N_used, 0)

        mode = str(score_mode or "l2norm").lower()
        tail = str(tail_strategy or "drop").lower()

        # -------------------------------------------------
        # (A) attn FLOPs
        # -------------------------------------------------
        #     VToken  attn map 
        try:
            qfb = self._get_qformer_flops_breakdown()
            qformer_full_flops = float(qfb.get("full_total", 0.0) or 0.0)
        except Exception:
            qformer_full_flops = 0.0

        if did_refresh and qformer_full_flops > 0.0:
            out["attn"] += qformer_full_flops * B

        # 2) attention map  score reduction  proxy
        #    attn_bhqk: [B, H, Q, N]
        if attn_bhqk is not None and torch.is_tensor(attn_bhqk) and attn_bhqk.dim() == 4:
            Bb, H, Q, N = [int(x) for x in attn_bhqk.shape]

            if mode in ("attn_entropy", "entropy"):
                # soft-normalize / entropy / q-weighted reduce  proxy
                out["attn"] += float(6 * Bb * H * Q * N + 2 * Bb * Q * N)

            elif mode in ("attn", "attn_mean"):
                out["attn"] += float(3 * Bb * H * Q * N + 1 * Bb * Q * N)

            elif mode in ("attn_grad", "grad", "taylor_attn"):
                # attn * grad + reduce
                mul_term = 4 if used_grad else 3
                out["attn"] += float(mul_term * Bb * H * Q * N + 1 * Bb * Q * N)

        # -------------------------------------------------
        # (B) select FLOPs
        # -------------------------------------------------
        select_flops = 0.0

        # 1) score 
        if mode in ("l2norm", "l2"):
            # token norm:  proxy
            select_flops += float(2 * B * N_full * C)

        elif mode in ("attn_entropy", "entropy", "attn", "attn_mean", "attn_grad", "grad", "taylor_attn"):
            pass

        else:
            select_flops += float(B * N_full * C)

        # 2) topk / sort proxy
        if N_full > 1:
            select_flops += float(B * N_full * math.log2(max(N_full, 2)))

        # 3) gather selected tokens
        select_flops += float(B * N_used * C)

        # 4) tail merge / tail_token  reduction proxy
        if N_drop > 0:
            if tail == "merge":
                #  tail 
                select_flops += float(2 * B * N_drop * C)
            elif tail == "tail_token":
                select_flops += float(1 * B * N_drop * C)
            elif tail == "drop":
                pass
            else:
                select_flops += float(1 * B * N_drop * C)

        out["select"] = float(select_flops)

        # -------------------------------------------------
        # (C) cache FLOPs
        # -------------------------------------------------
        cache_flops = 0.0
        if attn_bhqk is not None and torch.is_tensor(attn_bhqk) and attn_bhqk.dim() == 4:
            Bb, H, Q, N = [int(x) for x in attn_bhqk.shape]
            cache_flops += float(Bb * H * Q * N)
            if used_grad:
                cache_flops += float(Bb * H * Q * N)

        out["cache"] = float(cache_flops)
        out["total"] = float(out["attn"] + out["select"] + out["cache"])
        return out

    def log_pruning_summary(self, stage: str = "eval"):
        """

        """
        heads = []
        try:
            for _, _, sa in self._iter_cross_attn_modules():
                heads.append(int(sa.num_attention_heads))
        except Exception:
            pass

        lines = [f"- CrossAttn heads per layer: {heads}"]

        # Head-Merge
        try:
            info = getattr(self, "_head_merge_info", None)
            if isinstance(info, dict) and info:
                total = 0
                merged = info.get("merged", {})
                if isinstance(merged, dict):
                    total = sum(len(v) for v in merged.values() if isinstance(v, (list, tuple)))
                lines.append(
                    "- Head-Merge: done, total_pruned="
                    f"{total}, threshold={info.get('threshold')}, method={info.get('method')}"
                )
        except Exception:
            pass

        # LowRank
        try:
            st = getattr(self, "_lowrank_stats", None)
            if isinstance(st, dict) and st:
                lines.append(
                    f"- LowRank: enabled, replaced={st.get('replaced',0)}, rank={st.get('rank')}, scope={st.get('scope')}"
                )
        except Exception:
            pass

        try:
            v_keep = getattr(self, "_vision_token_keep_ratio", None)
            v_keep = 1.0 if v_keep is None else float(v_keep)
            score_mode = str(getattr(self, "_vision_token_score_mode", "l2norm")).lower()
            tail_strategy = str(getattr(self, "_vision_token_tail_strategy", "drop")).lower()
            lines.append(f"- Vision-Token: keep={v_keep:.3f}, score_mode={score_mode}, tail={tail_strategy}")
        except Exception:
            pass


        try:
            cfg = getattr(self, "_text_kv_prune_cfg", None)
            if isinstance(cfg, dict) and bool(cfg.get("enabled", False)):
                mode = str(cfg.get("mode", "window"))
                tail = str(cfg.get("tail_strategy", "drop"))
                window = cfg.get("window_size", None)
                keep_ratio = cfg.get("keep_ratio", None)
                pool_bins = cfg.get("pool_bins", None)
                keep_recent = cfg.get("keep_recent", None)

                st = getattr(self, "_text_kv_stats", {}) or {}
                calls = int(st.get("n_calls", 0) or 0)
                pruned = int(st.get("n_pruned", 0) or 0)
                n_tt = int(st.get("n_tail_token", 0) or 0)
                n_mg = int(st.get("n_merge", 0) or 0)
                t_sum = int(st.get("tail_T_sum", 0) or 0)

                lines.append(
                    f"- Text-KV: enabled, mode={mode}, tail={tail}, window={window}, keep_ratio={keep_ratio}, "
                    f"pool_bins={pool_bins}, keep_recent={keep_recent}, calls={calls}, pruned={pruned}, "
                    f"tail_token={n_tt}, merge={n_mg}, tail_T_sum={t_sum}"
                )
                try:
                    reasons = st.get("reasons", {}) if isinstance(st, dict) else {}
                    if isinstance(reasons, dict) and len(reasons) > 0:
                        top_items = sorted(reasons.items(), key=lambda kv: -int(kv[1]))[:6]
                        reasons_str = ", ".join([f"{k}={int(v)}" for k, v in top_items])
                        lines.append(f"  - Text-KV reasons: {reasons_str}")
                except Exception:
                    pass

                try:
                    _calls = int(st.get("n_calls", 0) or 0)
                    _sb = int(st.get("sum_L_before", 0) or 0)
                    _sa = int(st.get("sum_L_after", 0) or 0)
                    if _calls > 0 and _sb > 0:
                        avg_before = float(_sb) / float(_calls)
                        avg_after = float(_sa) / float(_calls)
                        avg_keep = (avg_after / avg_before) if avg_before > 0 else 1.0
                        lines.append(f"  - Text-KV len(avg): before={avg_before:.1f}, after={avg_after:.1f}, keep={avg_keep:.3f}")

                        def _pct(xs, q):
                            if not isinstance(xs, list) or len(xs) == 0:
                                return None
                            ys = sorted([int(x) for x in xs if isinstance(x, (int, float))])
                            if len(ys) == 0:
                                return None
                            pos = int(round((len(ys) - 1) * float(q)))
                            pos = max(0, min(len(ys) - 1, pos))
                            return ys[pos]

                        seen = st.get("seen_L", [])
                        kept = st.get("kept_L", [])
                        p50_s = _pct(seen, 0.50); p90_s = _pct(seen, 0.90)
                        p50_k = _pct(kept, 0.50); p90_k = _pct(kept, 0.90)
                        if (p50_s is not None) and (p50_k is not None):
                            lines.append(f"  - Text-KV len(p50/p90): seen={p50_s}/{p90_s}, kept={p50_k}/{p90_k}")

                    ent_cnt = int(st.get("imp_ent_cnt", 0) or 0)
                    if ent_cnt > 0:
                        ent_mean = float(st.get("imp_ent_sum", 0.0)) / float(ent_cnt)
                        top1_mean = float(st.get("imp_top1_sum", 0.0)) / float(ent_cnt)
                        lines.append(f"  - Text-KV imp: entropy={ent_mean:.3f}, top1_p={top1_mean:.3f} (head)")
                except Exception:
                    pass
            else:
                lines.append("- Text-KV: disabled")
        except Exception:
            pass

        try:
            gs = getattr(self, "_text_kv_gen_stats", None)
            if isinstance(gs, dict) and int(gs.get("n_seq", 0) or 0) > 0:
                n = int(gs.get("n_seq", 0) or 0)
                s1 = float(gs.get("sum_new", 0) or 0)
                s2 = float(gs.get("sum_new_sq", 0) or 0)
                mean = s1 / float(n)
                var = max(0.0, s2 / float(n) - mean * mean)
                std = math.sqrt(var)
                eos_rate = float(gs.get("n_has_eos", 0) or 0) / float(n)
                hit_rate = float(gs.get("n_hit_max", 0) or 0) / float(n)
                t = float(gs.get("sum_decode_time", 0.0) or 0.0)
                tok_per_s = (s1 / t) if t > 1e-8 else 0.0

                lens = gs.get("lens_tail", [])
                if isinstance(lens, list) and len(lens) > 0:
                    lens_sorted = sorted([int(x) for x in lens])
                    def _p(q):
                        pos = int(round((len(lens_sorted) - 1) * float(q)))
                        pos = max(0, min(len(lens_sorted) - 1, pos))
                        return lens_sorted[pos]
                    p50 = _p(0.50); p90 = _p(0.90)
                    lines.append(f"- Gen-Len: avg_new={mean:.2f}+/-{std:.2f}, p50/p90={p50}/{p90}, eos_rate={eos_rate:.3f}, hit_max={hit_rate:.3f}, tok/s={tok_per_s:.1f}, n_seq={n}")
                else:
                    lines.append(f"- Gen-Len: avg_new={mean:.2f}+/-{std:.2f}, eos_rate={eos_rate:.3f}, hit_max={hit_rate:.3f}, tok/s={tok_per_s:.1f}, n_seq={n}")
        except Exception:
            pass

        # Append generation diagnostics when enabled.
        try:
            gp = self._get_generate_stage_profile_dict() if hasattr(self, "_get_generate_stage_profile_dict") else None
            s4t = gp.get("stage4_total_sec", {}) or {}
            s4m = gp.get("stage4_ms_seq", {}) or {}
            s4s = gp.get("stage4_share_tracked", {}) or {}

            if len(s4t) > 0:
                lines.append(
                    "  - Gen-Stage4(total): "
                    f"visual_encoder={float(s4t.get('visual_encoder', 0.0)):.4f}s, "
                    f"runtime_vtoken={float(s4t.get('runtime_vtoken', 0.0)):.4f}s, "
                    f"qformer={float(s4t.get('qformer', 0.0)):.4f}s, "
                    f"llm_decode={float(s4t.get('llm_decode', 0.0)):.4f}s"
                )
                lines.append(
                    "  - Gen-Stage4(ms/seq): "
                    f"visual_encoder={float(s4m.get('visual_encoder', 0.0)):.2f}, "
                    f"runtime_vtoken={float(s4m.get('runtime_vtoken', 0.0)):.2f}, "
                    f"qformer={float(s4m.get('qformer', 0.0)):.2f}, "
                    f"llm_decode={float(s4m.get('llm_decode', 0.0)):.2f}"
                )
                lines.append(
                    "  - Gen-Stage4(share): "
                    f"visual_encoder={float(s4s.get('visual_encoder', 0.0)):.3f}, "
                    f"runtime_vtoken={float(s4s.get('runtime_vtoken', 0.0)):.3f}, "
                    f"qformer={float(s4s.get('qformer', 0.0)):.3f}, "
                    f"llm_decode={float(s4s.get('llm_decode', 0.0)):.3f}"
                )
            if isinstance(gp, dict):
                tot = gp["total_sec"]
                avg = gp["avg_call_sec"]
                msq = gp["ms_seq"]

                vtot = gp["vtoken_total_sec"]
                vavg = gp["vtoken_avg_call_sec"]
                vmsq = gp["vtoken_ms_seq"]
                nv_tot = gp.get("vtoken_normal_total_sec", {}) or {}
                nv_avg = gp.get("vtoken_normal_avg_call_sec", {}) or {}
                nv_msq = gp.get("vtoken_normal_ms_seq", {}) or {}
                rt_tot = gp.get("runtime_vtoken_total_sec", {}) or {}
                rt_avg = gp.get("runtime_vtoken_avg_call_sec", {}) or {}
                rt_msq = gp.get("runtime_vtoken_ms_seq", {}) or {}

                lines.append(
                    "- Gen-Stage(total): "
                    f"visual={tot['t_visual']:.4f}s, "
                    f"vtoken={tot['t_vtoken']:.4f}s, "
                    f"qformer={tot['t_qformer']:.4f}s, "
                    f"opt_prep={tot['t_opt_prep']:.4f}s, "
                    f"generate={tot['t_generate']:.4f}s, "
                    f"post_decode={tot['t_post_decode']:.4f}s, "
                    f"tracked={gp['tracked_total_sec']:.4f}s"
                )
                lines.append(
                    "  - VToken-Normal(total): "
                    f"visual={float(nv_tot.get('t_vtoken_normal_visual', 0.0) or 0.0):.4f}s, "
                    f"qformer={float(nv_tot.get('t_vtoken_normal_qformer', 0.0) or 0.0):.4f}s, "
                    f"total={float(nv_tot.get('t_vtoken_normal_total', 0.0) or 0.0):.4f}s"
                )
                lines.append(
                    "  - Runtime-VToken(total): "
                    f"total={float(rt_tot.get('t_runtime_total', 0.0) or 0.0):.4f}s = "
                    f"attn={float(rt_tot.get('t_runtime_attn', 0.0) or 0.0):.4f}s + "
                    f"select={float(rt_tot.get('t_runtime_select', 0.0) or 0.0):.4f}s + "
                    f"cache={float(rt_tot.get('t_runtime_cache', 0.0) or 0.0):.4f}s"
                )

                lines.append(
                    "- Gen-Stage(avg/call): "
                    f"visual={avg['t_visual']:.4f}s, "
                    f"vtoken={avg['t_vtoken']:.4f}s, "
                    f"qformer={avg['t_qformer']:.4f}s, "
                    f"opt_prep={avg['t_opt_prep']:.4f}s, "
                    f"generate={avg['t_generate']:.4f}s, "
                    f"post_decode={avg['t_post_decode']:.4f}s"
                )
                lines.append(
                    "  - VToken-Normal(avg/call): "
                    f"visual={float(nv_avg.get('t_vtoken_normal_visual', 0.0) or 0.0):.4f}s, "
                    f"qformer={float(nv_avg.get('t_vtoken_normal_qformer', 0.0) or 0.0):.4f}s, "
                    f"total={float(nv_avg.get('t_vtoken_normal_total', 0.0) or 0.0):.4f}s"
                )
                lines.append(
                    "  - Runtime-VToken(avg/call): "
                    f"total={float(rt_avg.get('t_runtime_total', 0.0) or 0.0):.4f}s = "
                    f"attn={float(rt_avg.get('t_runtime_attn', 0.0) or 0.0):.4f}s + "
                    f"select={float(rt_avg.get('t_runtime_select', 0.0) or 0.0):.4f}s + "
                    f"cache={float(rt_avg.get('t_runtime_cache', 0.0) or 0.0):.4f}s"
                )

                lines.append(
                    "  - Gen-Stage(ms/seq): "
                    f"visual={msq['t_visual']:.2f}, "
                    f"vtoken={msq['t_vtoken']:.2f}, "
                    f"qformer={msq['t_qformer']:.2f}, "
                    f"opt_prep={msq['t_opt_prep']:.2f}, "
                    f"generate={msq['t_generate']:.2f}, "
                    f"post_decode={msq['t_post_decode']:.2f}"
                )
                qformer_inner_total = (
                    float(tot.get("t_qformer_cross", 0.0) or 0.0)
                    + float(tot.get("t_qformer_ffn", 0.0) or 0.0)
                    + float(tot.get("t_qformer_other", 0.0) or 0.0)
                )

                if qformer_inner_total > 0.0:
                    lines.append(
                        "  - QFormer(total): "
                        f"cross={float(tot.get('t_qformer_cross', 0.0) or 0.0):.4f}s, "
                        f"ffn={float(tot.get('t_qformer_ffn', 0.0) or 0.0):.4f}s, "
                        f"other={float(tot.get('t_qformer_other', 0.0) or 0.0):.4f}s, "
                        f"sum={qformer_inner_total:.4f}s"
                    )
                    lines.append(
                        "  - QFormer(avg/call): "
                        f"cross={float(avg.get('t_qformer_cross', 0.0) or 0.0):.4f}s, "
                        f"ffn={float(avg.get('t_qformer_ffn', 0.0) or 0.0):.4f}s, "
                        f"other={float(avg.get('t_qformer_other', 0.0) or 0.0):.4f}s"
                    )
                    lines.append(
                        "  - QFormer(ms/seq): "
                        f"cross={float(msq.get('t_qformer_cross', 0.0) or 0.0):.2f}, "
                        f"ffn={float(msq.get('t_qformer_ffn', 0.0) or 0.0):.2f}, "
                        f"other={float(msq.get('t_qformer_other', 0.0) or 0.0):.2f}"
                    )
                    
                lines.append(
                    "  - VToken-Normal(ms/seq): "
                    f"visual={float(nv_msq.get('t_vtoken_normal_visual', 0.0) or 0.0):.2f}, "
                    f"qformer={float(nv_msq.get('t_vtoken_normal_qformer', 0.0) or 0.0):.2f}, "
                    f"total={float(nv_msq.get('t_vtoken_normal_total', 0.0) or 0.0):.2f}"
                )
                lines.append(
                    "  - Runtime-VToken(ms/seq): "
                    f"total={float(rt_msq.get('t_runtime_total', 0.0) or 0.0):.2f} = "
                    f"attn={float(rt_msq.get('t_runtime_attn', 0.0) or 0.0):.2f} + "
                    f"select={float(rt_msq.get('t_runtime_select', 0.0) or 0.0):.2f} + "
                    f"cache={float(rt_msq.get('t_runtime_cache', 0.0) or 0.0):.2f}"
                )
        except Exception:
            pass

        # HeadDrop
        try:
            if hasattr(self, "_headdrop_prob"):
                lines.append(f"- HeadDrop: p={float(self._headdrop_prob):.3f}, hits={int(getattr(self,'_headdrop_hits',0))}")
        except Exception:
            pass

        # Q-Former FFN + CrossAttn  FLOPs
        try:
            fb = self._get_qformer_flops_breakdown()
            lines.append(f"- Q-Former FFN mid per layer (query): {fb['mids']}")

            if (fb.get("current_total") is not None) and (fb.get("full_total") is not None):
                lines.append(
                    f"- Q-Former FLOPs/sample (current): total={fb['current_total']/1e9:.4f}G = "
                    f"q_ffn={fb['q_ffn']/1e9:.4f}G + "
                    f"xattn_const={fb['xattn_const']/1e9:.4f}G + "
                    f"xattn_var={fb['xattn_var_current']/1e9:.4f}G"
                )
                lines.append(
                    f"- Q-Former FLOPs/sample (full vision): total={fb['full_total']/1e9:.4f}G = "
                    f"q_ffn={fb['q_ffn']/1e9:.4f}G + "
                    f"xattn_const={fb['xattn_const']/1e9:.4f}G + "
                    f"xattn_var={fb['xattn_var_full']/1e9:.4f}G"
                )
                lines.append(
                    f"- Q-Former FLOPs saved vs full vision: "
                    f"{fb['saved']/1e9:.4f}G/sample ({fb['saved_pct']:.2f}%)"
                )

                # ===== VToken-Normal / Runtime-VToken FLOPs =====
                try:
                    vtf_seq = gp.get("vtoken_flops_per_seq", {}) or {}
                    rt_f_attn = float(vtf_seq.get("f_vtoken_attn", 0.0) or 0.0)
                    rt_f_select = float(vtf_seq.get("f_vtoken_select", 0.0) or 0.0)
                    rt_f_cache = float(vtf_seq.get("f_vtoken_cache", 0.0) or 0.0)
                    rt_f_total = float(vtf_seq.get("f_vtoken_total", 0.0) or 0.0)

                    q_cur = float(fb.get("current_total", 0.0) or 0.0)
                    lines.append(
                        "- VToken-Normal FLOPs/sample (QFormer-side approx): "
                        f"total={q_cur/1e9:.4f}G"
                    )
                    lines.append(
                        "- Runtime-VToken FLOPs/sample (approx): "
                        f"total={rt_f_total/1e9:.4f}G = "
                        f"attn={rt_f_attn/1e9:.4f}G + "
                        f"select={rt_f_select/1e9:.4f}G + "
                        f"cache={rt_f_cache/1e9:.4f}G"
                    )
                    lines.append(
                        "- VToken-Normal + Runtime FLOPs/sample (approx): "
                        f"total={(q_cur + rt_f_total)/1e9:.4f}G = "
                        f"normal={q_cur/1e9:.4f}G + "
                        f"runtime={rt_f_total/1e9:.4f}G"
                    )
                except Exception:
                    pass

                lines.append(
                    f"- CrossAttn detail: q={fb['q_len']}, "
                    f"v_full={fb['v_full']}, v_used={fb['v_current']}, "
                    f"heads={fb['heads']}, head_dim={fb['head_dims']}"
                )
            else:
                lines.append(
                    f"- Q-Former FLOPs/sample (partial): "
                    f"q_ffn={fb['q_ffn']/1e9:.4f}G, q={fb['q_len']}, "
                    f"v_full={fb.get('v_full', 0)}, v_used={fb.get('v_current', 0)}"
                )
        except Exception:
            pass

        try:
            st_tail = getattr(self, "_tail_grad_profile", None)
            if isinstance(st_tail, dict) and st_tail.get("count", 0) > 0:
                n = float(st_tail["count"])
                keep_frac = float(st_tail.get("keep", 0.0) / n)
                tail_frac = float(st_tail.get("tail", 0.0) / n)
                lines.append(
                    f"- Tail-Merge grad share (approx): keep={keep_frac:.3f}, tail={tail_frac:.3f} (avg over {int(n)} batches)"
                )
        except Exception:
            pass

        usage_line = None
        try:
            st = getattr(self, "_vision_token_usage_stats", None)
            if isinstance(st, dict) and st.get("calls", 0) > 0:
                total_raw = float(st.get("total_raw_tokens", 0.0))
                total_used = float(st.get("total_used_tokens", 0.0))
                calls = int(st.get("calls", 0))
                avg_keep = (total_used / total_raw) if total_raw > 0 else 0.0
                usage_line = f"[Vision-Token@usage] calls={calls}, total_raw={int(total_raw)}, total_used={int(total_used)}, avg_keep={avg_keep:.4f}"
                lines.append(f"- Vision-Token usage: calls={calls}, avg_keep={avg_keep:.4f} (raw={int(total_raw)}, used={int(total_used)})")
        except Exception:
            pass

        usedl2_line = None
        try:
            st = getattr(self, "_vision_token_usage_stats", None)
            if isinstance(st, dict):
                s1 = st.get("used_l2_sum", None)
                s2 = st.get("used_l2_sq_sum", None)
                c  = st.get("used_l2_count", None)

                if hasattr(s1, "item"): s1 = float(s1.item())
                if hasattr(s2, "item"): s2 = float(s2.item())
                if hasattr(c, "item"):  c  = int(c.item())

                if isinstance(c, int) and c > 0 and (s1 is not None) and (s2 is not None):
                    mean = s1 / c
                    var  = s2 / c - mean * mean
                    if var < 0: var = 0.0
                    std = math.sqrt(var)
                    usedl2_line = f"[Vision-Token@usedL2] mean={mean:.6f}, std={std:.6f}, count={c}"
                    lines.append(f"- Used token L2: mean={mean:.6f}, std={std:.6f}, count={c}")
        except Exception:
            pass

        banner = "\n".join([
            f"======  Pruning-Summary ({stage})  ======",
            *lines,
            "======================================",
        ])
        logging.info("\n" + banner + "\n")

        try:
            if usage_line is not None:
                logging.info(usage_line)
            if usedl2_line is not None:
                logging.info(usedl2_line)
        except Exception:
            pass

    def _set_qformer_sdpa_enabled(self, enabled: bool):
        enabled = bool(enabled)
        for m in self.Qformer.modules():
            if m.__class__.__name__ == "BertSelfAttention":
                setattr(m, "sdpa_enabled", enabled)

    def _set_opt_sdpa_enabled(self, enabled: bool):
        enabled = bool(enabled)
        for m in self.opt_model.modules():
            if m.__class__.__name__ == "OPTAttention":
                setattr(m, "sdpa_enabled", enabled)


class _ActWithGate(nn.Module):
    def __init__(self, act_fn, d_mid: int, group_size: int, *, device=None, dtype=None):
        super().__init__()
        self.act = act_fn
        self.group_size = int(group_size)
        G = int(math.ceil(d_mid / self.group_size))
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float32
        self.gate = nn.Parameter(torch.zeros(G, device=device, dtype=dtype))

    def forward(self, x):
        y = self.act(x)
        gate = torch.sigmoid(self.gate)
        gate_full = torch.repeat_interleave(gate, self.group_size, dim=0)[: y.shape[-1]]
        return y * gate_full.to(dtype=y.dtype)
