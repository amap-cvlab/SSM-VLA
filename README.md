<div align="center">
<h1> 
Seeing Space and Motion: Enhancing Latent Actions with Geometric and Dynamic Awareness for Vision-Language-Action Models
</h1>

![image](assets/teaser.png?raw=true)

</div>


## Data Preparation
For data preparation of latent action model, please refer to [Moto](https://github.com/TencentARC/Moto).

## Quick Start
Install dependencies and run training with accelerate:

```bash
# install requirements for F-LAM
cd f-lam
pip install -r requirements.txt

# train on single-GPU
accelerate launch train_lam.py --config_path lam_calvin.yaml

# train on multi-GPU (8 processes)
accelerate launch --num_processes=8 --num_machines=1 train_lam.py --config_path lam_calvin.yaml
```

## To Do
- [x] F-LAM code
- [ ] VLA code

## Acknowledgement
We would like to thank [Moto-GPT](https://github.com/TencentARC/Moto) for their great work which inspired data preparation and implementation details.