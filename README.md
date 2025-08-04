<div align="center">

# GTR: Improving Large 3D Reconstruction Models through Geometry and Texture Refinement

<a href="https://arxiv.org/abs/2406.05649"><img src="https://img.shields.io/badge/ArXiv-2406.05649-brightgreen"></a>
<a href="https://snap-research.github.io/GTR"><img src="https://img.shields.io/badge/Project-github.io-blue"></a>
<a href="https://huggingface.co/snap-research/gtr"><img src="https://img.shields.io/badge/HuggingFace-gtr-orange"></a>

</div>

---

![Demo Visuals](demo_visuals.gif)

# Installation

We recommend using `Python>=3.10`, `PyTorch==2.7.0`, and `CUDA>=12.4`.
```bash
conda create --name gtr python=3.10
conda activate gtr
pip install -U pip

pip install torch==2.7.0 torchvision==0.22.0 torchmetrics==1.2.1 --index-url https://download.pytorch.org/whl/cu124
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

# How to Use

Please download model checkpoint from [here](https://drive.google.com/file/d/1ITVqdDLmY5EISj4vsZ2O4sN5mZv9fUfB/view?usp=sharing), and then put it under the `ckpts/` directory.

We provide multiview grid data examples under `./examples/` generated using [Zero123++](https://github.com/SUDO-AI-3D/zero123plus). Our inference script loads pretrained checkpoint, runs fast texture refinement,  reconstructs the textured mesh from multiview grid data and exports the mesh. There will be 3 files in the output folder, including exported mesh in `.obj` format, rotating gif visuals of mesh and rotating gif visuals of NeRF.

To infer on multiview data from other sources, simply change camera parameters [here](https://github.sc-corp.net/Snapchat/GTR/blob/main/scripts/prepare_mv.py#L153-L157) accordingly to match the multiview data.

```bash
# Preprocessing
python3 scripts/prepare_mv.py --in_dir ./examples/cute_horse.png --out_dir ./examples/cute_horse

# Inference
python3 scripts/inference.py --ckpt_path ckpts/full_checkpoint.pth --in_dir ./examples/cute_horse --out_dir ./outputs/cute_horse 
```
