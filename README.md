# Med-TTT

Official code repository for Med-TTT: Vision Test-Time Training model for Medical Image Segmentation. [![arXiv](https://img.shields.io/badge/arXiv-2410.02523-brightgreen.svg)](https://arxiv.org/abs/2410.02523)

## Updates
04 Oct 2024 (v1.0.0):
* Paper released
* Public version is available now
* More datasets will come soon

24 Sep 2024 (v0.0.0):
* Uploaded the prototype version of Med-TTT
* Public version and paper coming soon

## Datasets
### ISIC datsets
ISIC datasets is prepared as the [VM-UNet](https://github.com/JCruan519/VM-UNet) did.
- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio, can be found here {[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm) or [GoogleDrive](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view?usp=sharing)}. 

- After downloading the datasets, you are supposed to put them into './isic17/' and '.a/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png
## Installation
```bash
conda create -n medttt python=3.8
conda activate medttt
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install "transformers[torch]"
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs tqdm
```

## Train
```bash
cd Med-TTT
python train.py 
```

## Evaluate
```bash
cd Med-TTT
python test-result.py 
```

## Paper
If you use our work in your research, please cite our paper as follows:

```bibtex
@misc{xu2024medtttvisiontesttimetraining,
      title={Med-TTT: Vision Test-Time Training model for Medical Image Segmentation}, 
      author={Jiashu Xu},
      year={2024},
      eprint={2410.02523},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2410.02523}, 
}
```
# ✨Acknowledgments✨
We thank the authors of [VM-UNet](https://github.com/JCruan519/VM-UNet) and [Learn at Test Time](https://github.com/test-time-training/ttt-lm-pytorch) for their open-source codes.
