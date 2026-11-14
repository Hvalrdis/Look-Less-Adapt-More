# Looking Less to Adapt More

Code for *Recovery-Time Visual-Token Filtering (RVT)* in BLIP-2 Q-Former compression. The implementation uses [Salesforce LAVIS](https://github.com/salesforce/LAVIS) and includes the modified model, Q-Former and training code.

## Method

The experiments first prune query-side FFN channel groups and recover the pruned Q-Former with the full visual-token sequence. The recovered model provides a common starting checkpoint for three matched recovery paths, Full Token, Random and RVT. All three paths use the same training budget. Full Token uses the complete visual sequence, Random samples the same number of tokens as RVT, and RVT selects tokens using cross-attention scores.

During RVT recovery, attention-entropy scoring ranks the visual tokens. The selected tokens are used for the recovery update, with a residual merge for the unselected tokens. A full-token pass refreshes the attention scores at each recovery step. The scoring pass is detached from the recovery loss.

After this stage, cross-attention head importance is estimated from the complete visual sequence using five-fold Grad×Act scores and intersection-and-fill selection. Head pruning and the subsequent recovery use full visual-token input. The final model retains the pruned FFN channels and cross-attention heads.

[Compression protocol](figures/method_overview.pdf)

## Code

```text
configs/                       Experiment settings
scripts/train.py                Training
scripts/evaluate.py             Caption evaluation
scripts/check.py                Code and configuration checks
src/lavis/models/blip2_models/  BLIP-2, Q-Former and OPT changes
src/lavis/runners/              Training and evaluation runners
src/rvt_core.py                 Token selection and pruning functions
figures/method_overview.pdf     Compression protocol
```

The RVT recovery implementation is in `src/lavis/models/blip2_models/blip2_opt.py`. The related attention and FFN changes are in `Qformer.py`, `blip2_qformer.py` and `blip2.py`.

## Installation

The experiments used Python 3.10, PyTorch 2.4.1, TorchVision 0.19.1, Transformers 4.33.2 and LAVIS 1.0.1. Additional dependencies are listed in `requirements.txt`.

```bash
conda create -n rvt python=3.10 -y
conda activate rvt
pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install salesforce-lavis==1.0.1
pip install -r requirements.txt
```

The modified LAVIS files replace the corresponding files in the installed package. The following command applies the changes in the active Python environment.

```bash
python - <<'PY_OVERLAY'
import importlib.util
from pathlib import Path
import shutil

source = Path('src/lavis')
spec = importlib.util.find_spec('lavis')
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError('Install salesforce-lavis before copying the files.')
target_root = Path(next(iter(spec.submodule_search_locations)))
for file in source.rglob('*.py'):
    target = target_root / file.relative_to(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, target)
PY_OVERLAY
```

The OPT implementation imports `bitsandbytes`. Install a version compatible with the local CUDA environment. The remaining LAVIS modules are supplied by the installed package.

## Training and evaluation

The experiments use MS COCO images and Karpathy caption annotations. Set the dataset paths in the YAML files and provide the checkpoint for each stage through `model.pretrained`. The training sequence starts with FFN gate training and FFN pruning, followed by standard FFN recovery, the matched recovery paths, cross-attention head pruning and post-head-pruning recovery.

The main OPT-2.7B configuration files are `opt27b_ffn_gate.yaml`, `opt27b_main.yaml` and `opt27b_head_prune.yaml`. For example, after preparing the required checkpoint and dataset paths, RVT recovery can be started with

```bash
python scripts/train.py --cfg-path configs/opt27b_main.yaml \
  --options model.pretrained=/path/to/ffn_recovered.pth
```

Head pruning starts from the chosen pre-head recovery checkpoint.

```bash
python scripts/train.py --cfg-path configs/opt27b_head_prune.yaml \
  --options model.pretrained=/path/to/rvt_recovered.pth
```

The OPT-6.7B settings are recorded in `configs/opt67b_cross_backbone.yaml`. The FlanT5-XL settings are recorded in `configs/flant5xl_cross_backbone.yaml`; running the latter requires the corresponding T5-specific RVT implementation. Full training and evaluation require the pretrained models, COCO data and a CUDA GPU.

The Python files, YAML configurations and the main token-selection calculations can be checked with

```bash
python scripts/check.py
```

## Citation

Yifan Wang, *Looking Less to Adapt More: Recovery-Time Visual-Token Filtering for Compressed Multimodal Bridges*. Manuscript prepared for ACIIDS 2027.

## License

The LAVIS-derived files retain the BSD-3-Clause license and copyright notices. `modeling_opt.py` includes Apache-2.0 code. See `LICENSE` and `licenses/THIRD_PARTY.md` for the source licenses.
