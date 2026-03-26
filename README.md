<div align="center">
<a id="readme-top"></a>
<h1> Seeing Space and Motion: Enhancing Latent Actions with Geometric and Dynamic Awareness for Vision-Language-Action Models
 </h1>

**SSM-VLA** is a Robust Vision-Language-Action Framework with **Geometric** Perception and Explicit **Dynamics** Reasoning.

</div>


# Data Preparation
For data preparation of latent action model, please refer to [Moto](https://github.com/TencentARC/Moto).

# Quick Start
Install dependencies and run training with accelerate:

```bash
# install requirements for F-LAM
pip install -r f-lam/requirements.txt

# run training (single-GPU)
accelerate launch train_lam.py --config_path lam_calvin.yaml

# run multi-GPU (8 processes)
accelerate launch --num_processes=8 --num_machines=1 train_lam.py --config_path lam_calvin.yaml
```

# 📝To Do
- [x] F-LAM code
- [ ] VLA code

# Acknowledgement
We would like to thank [Moto-GPT](https://github.com/TencentARC/Moto) for their great work which inspired data preparation and implementation details.