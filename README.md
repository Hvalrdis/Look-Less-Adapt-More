# RVT: Recovery-Time Visual-Token Filtering

This repository packages the implementation used for **Recovery-Time Visual-Token Filtering (RVT)** in the accompanying BLIP-2 Q-Former compression study. RVT reduces the visual evidence seen during recovery of an FFN-pruned Q-Former, then restores the full visual-token sequence before cross-attention head pruning. The public package is an overlay for the upstream [LAVIS](https://github.com/salesforce/LAVIS) tree; it does not redistribute model weights, COCO, or private experiment outputs.

## Method

The released path is:

1. Train sigmoid gates on query-side FFN channel groups (group size 16), then globally materialize a target keep ratio while retaining at least one group per layer.
2. Recover the FFN-pruned Q-Former with one of three matched paths: **Full Token**, **Random**, or **RVT**.
3. For RVT, average cross-attention maps across layers and heads, weight query distributions with a bounded softmax of negative attention entropy, and keep the top visual tokens. The full-token refresh pass is detached from the recovery loss; discarded tokens are normalized and score-weighted before a scaled residual is merged into the last retained token.
4. Estimate cross-attention head importance with Grad×Act over five folds, combine fold pruning candidates by intersection-and-fill, and physically export the pruned Q-Former.

The production implementation is in `src/lavis/models/blip2_models/blip2_opt.py`. Dependency-light scoring kernels are in `src/rvt_core.py` and are useful for CPU smoke tests.

## Repository layout

```text
configs/                         Reproducible paper-stage YAML templates (OPT-2.7B, OPT-6.7B, FlanT5-XL)
figures/                         Paper figures reused for documentation
results/                         Paper-reported cross-backbone sweep data (CSV)
scripts/train.py                 LAVIS training entry point
scripts/evaluate.py              LAVIS captioning evaluation entry point
src/rvt_core.py                  Standalone RVT scoring and mask kernels
src/lavis/models/blip2_models/   BLIP-2, Q-Former, OPT, and RVT model patch
src/lavis/runners/               Iteration runner and checkpoint handling
LICENSE                          BSD-3-Clause notice for the LAVIS-derived code
licenses/                        Apache-2.0 license and third-party notices
```

## Environment

The research environment used Python 3.10, the `salesforce-lavis` **1.0.1** package family, PyTorch 2.4.1 (CUDA 12.1), TorchVision 0.19.1, Transformers 4.33.2, PyYAML 6.0.2, FairScale 0.4.4, and WebDataset 1.0.2. The original source checkout did not retain a reliable Git commit ID. We identify the upstream environment by the published package version and provide the modified LAVIS modules as an overlay.

Install dependencies in a dedicated environment, then copy the overlay files into the installed LAVIS package in that environment. The following step **modifies the isolated environment's installed LAVIS package**; it never touches an upstream server checkout. Keep an unmodified environment separately if needed.

```bash
conda create -n rvt python=3.10 -y
conda activate rvt
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install salesforce-lavis==1.0.1

# Install the research modifications into this dedicated environment only.
python - <<'PY_OVERLAY'
import importlib.util
import pathlib
import shutil
source = pathlib.Path('src/lavis').resolve()
spec = importlib.util.find_spec('lavis')
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError('Install salesforce-lavis==1.0.1 before applying the overlay')
target = pathlib.Path(next(iter(spec.submodule_search_locations))).resolve()
for patch in sorted(source.rglob('*.py')):
    dest = target / patch.relative_to(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patch, dest)
    print(f'Applied {dest.relative_to(target)}')
PY_OVERLAY

python - <<'PY_VERIFY'
import lavis.models.blip2_models.blip2_opt as module
print('Installed RVT model:', module.__file__)
PY_VERIFY
```

Run smoke tests in a CPU-only environment with `python scripts/validate_release.py`. Full training needs the original upstream task registration, COCO files, pretrained checkpoints, and an appropriately sized CUDA GPU. Install `bitsandbytes` separately if you enable quantized OPT loading, using a version compatible with your CUDA environment.

## Data and model preparation

Download COCO images and the Karpathy caption annotations using their original licenses. Place them under `data/coco/` or edit the `build_info` paths in the YAML files. Obtain the official BLIP-2 OPT-2.7B, OPT-6.7B, or FlanT5-XL checkpoints from the upstream model providers. Checkpoints are intentionally excluded from this archive.

Set `model.pretrained` and, when applicable, `run.resume_ckpt_path` to local files. In the public templates, `configs/opt27b_main.yaml` and `configs/opt27b_head_prune.yaml` intentionally contain `model.pretrained: null`; replace it before training. For example:

```bash
cp configs/opt27b_main.yaml configs/local_main.yaml
python scripts/train.py --cfg-path configs/local_main.yaml \
  --options model.pretrained=/absolute/path/to/ffn_recovered_checkpoint.pth

python scripts/train.py --cfg-path configs/opt27b_head_prune.yaml \
  --options model.pretrained=/absolute/path/to/rvt_recovered_checkpoint.pth
```

The FFN gate template starts from the official BLIP-2 OPT-2.7B URL. Never commit checkpoints, cache directories, dataset images, or generated logs.

The cross-backbone templates are `configs/opt67b_cross_backbone.yaml` and `configs/flant5xl_cross_backbone.yaml`. They encode the Appendix matched pre-head recovery settings: OPT-6.7B uses visual-token keep ratio 0.55, 50 iterations, batch size 8, and initial learning rate `5e-6`; FlanT5-XL uses keep ratio 0.70, 450 iterations, batch size 8 with two-step accumulation, and initial learning rate `2.5e-6`. Both templates leave `model.pretrained` null by design. The selected 0.75 head-keep post-recovery settings are reported in the Appendix and must be applied in the subsequent head-pruning stage.

OPT-6.7B uses the released `blip2_opt` overlay. FlanT5-XL is registered by upstream LAVIS as `blip2_t5` with `model_type: caption_coco_flant5xl`; use the matching LAVIS 1.0.1 `blip2_t5.py` and `modeling_t5.py` modules in the upstream checkout. The RVT overlay in this archive is OPT-focused. The FlanT5 YAML documents the paper validation settings, but the T5-specific RVT integration is not included. Therefore, this package does not provide a validated T5 RVT training entry point.

## Training and evaluation

All commands are launched from the repository root after the LAVIS overlay is installed:

```bash
# FFN gate scoring (paper selection: 3,000 iterations)
python scripts/train.py --cfg-path configs/opt27b_ffn_gate.yaml

# Matched recovery / RVT stage (edit model.pretrained to the materialized FFN checkpoint)
python scripts/train.py --cfg-path configs/opt27b_main.yaml

# Five-fold Grad×Act head scoring and physical export
python scripts/train.py --cfg-path configs/opt27b_head_prune.yaml

# Caption evaluation on the configured split
python scripts/evaluate.py --cfg-path configs/opt27b_main.yaml

# Cross-backbone matched validation templates
python scripts/train.py --cfg-path configs/opt67b_cross_backbone.yaml \
  --options model.pretrained=/absolute/path/to/opt67b_recovery_checkpoint.pth
# FlanT5-XL uses a T5-specific RVT implementation not included in this archive.
# See configs/flant5xl_cross_backbone.yaml for documented settings.
```

For a dry configuration check without a GPU, run `python scripts/validate_release.py`. The training commands intentionally retain the upstream LAVIS data/task registration and checkpoint-loading logic; they are not expected to complete without COCO, model weights, and a CUDA-capable installation.

## Main paper settings

| Setting | OPT-2.7B primary study |
|---|---:|
| Image resolution | 364 |
| FFN group size / keep ratio | 16 / 0.35 |
| RVT visual-token keep ratio | 0.45 |
| RVT score / tail handling | attention entropy / residual merge |
| Head importance / keep ratio | Grad×Act, five-fold intersection-and-fill / 0.75 |
| Caption decoding | beam 5, minimum 8, maximum 30 tokens |
| Main seeds | 42, 7, 52 |

## Reported results

On the COCO Karpathy test split, the final RVT + SDPA path reports BLEU-1/2/3/4 = **0.8400/0.6982/0.5594/0.4404**, METEOR = **0.3097**, ROUGE-L = **0.6164**, CIDEr = **1.4406**, and SPICE = **0.2429** (three-seed mean). The matched Full Token + SDPA path reports CIDEr **1.4366** and BLEU-4 **0.4395**.

The materialized final configuration reduces local query-module FLOPs from **9.0597 G to 6.9757 G (23.0%)** and Q-Former parameters from **105.14 M to 63.28 M (39.8%)**. Under the fixed A100 profiling protocol, total model-stage latency is **53.24 ± 0.35 ms/sequence** with SDPA versus **53.24 ± 0.50 ms/sequence** for the baseline. Enabling RVT at inference is a diagnostic, not the final static path: it adds **1.20 ± 0.04 ms/sequence** and raises total latency to **54.13 ± 0.42 ms/sequence**.

Matched validation shows that the mean RVT–Full Token aggregate difference over head keep ratios 0.60–0.80 is **+0.00643** for OPT-2.7B, **+0.00107** for OPT-6.7B, and **+0.00117** for FlanT5-XL. The cross-backbone margins are small and can change direction; RVT is not a universal inference-time token-pruning method.

The accompanying figures are `figures/method_overview.pdf`, `figures/cross_backbone_delta.png`, and `figures/runtime_breakdown.pdf`.

## Reported sweep data

`results/cross_backbone_sweep_all_metrics.csv` contains the paper-reported head-keep sweep across OPT-2.7B, OPT-6.7B, and FlanT5-XL. The documentation also reuses the paper's method, cross-backbone comparison, and runtime figures. These are reported results; the CPU smoke tests do not reproduce or independently validate them.

## Reproducibility notes and limitations

Validation data select gate checkpoints, keep ratios, recovery endpoints, and pruning masks; the held-out COCO test split is used only after those choices are fixed. The complete structural and runtime analysis is centered on OPT-2.7B. OPT-6.7B and FlanT5-XL are validation-only cross-backbone comparisons. Reproducing the paper requires substantial GPU memory, COCO preprocessing, the original BLIP-2 checkpoints, and long-running training; no full training or evaluation is run by the release checks.

The release contains no passwords, server addresses, access tokens, checkpoints, model weights, datasets, caches, or private output directories. Results above are copied from the paper, not recomputed by the CPU smoke test.

## Citation

The associated manuscript is currently being prepared for ACIIDS 2027. The citation will be updated with formal proceedings details upon publication.

```bibtex
@misc{wang2026rvt,
  author = {Wang, Yifan},
  title  = {Looking Less to Adapt More: Recovery-Time Visual-Token Filtering for Compressed Multimodal Bridges},
  year   = {2026},
  note   = {Manuscript, ACIIDS 2027 submission}
}
```

## Third-party software

The model and runner patch preserve the upstream LAVIS BSD-3-Clause notice. `modeling_opt.py` retains the Apache-2.0 notice from the Hugging Face/Fairseq-derived OPT implementation. See `LICENSE` and `licenses/THIRD_PARTY.md`.
