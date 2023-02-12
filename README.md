# SRGAN-Pytorch

# Overview

This repository is implementation of the [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/pdf/1609.04802v5.pdf) at CVPR 2017

## Abstract
Despite the breakthroughs in accuracy and speed of single image super-resolution using faster and deeper convolutional neural networks, one central problem remains largely unsolved: how do we recover the finer texture details when we super-resolve at large upscaling factors? The behavior of optimization-based super-resolution methods is principally driven by the choice of the objective function. Recent work has largely focused on minimizing the mean squared reconstruction error. The resulting estimates have high peak signal-to-noise ratios, but they are often lacking high-frequency details and are perceptually unsatisfying in the sense that they fail to match the fidelity expected at the higher resolution. In this paper, we present SRGAN, a generative adversarial network (GAN) for image super-resolution (SR). To our knowledge, it is the first framework capable of inferring photo-realistic natural images for 4x upscaling factors. To achieve this, we propose a perceptual loss function which consists of an adversarial loss and a content loss. The adversarial loss pushes our solution to the natural image manifold using a discriminator network that is trained to differentiate between the super-resolved images and original photo-realistic images. In addition, we use a content loss motivated by perceptual similarity instead of similarity in pixel space. Our deep residual network is able to recover photo-realistic textures from heavily downsampled images on public benchmarks. An extensive mean-opinion-score (MOS) test shows hugely significant gains in perceptual quality using SRGAN. The MOS scores obtained with SRGAN are closer to those of the original high-resolution images than to those obtained with any state-of-the-art method.

# Environments
The environmental settings are described below. (I cannot gaurantee if it works on other environments)
```
numpy==1.21.6
opencv_python==4.7.0.68
Pillow==9.4.0
PyYAML==6.0
torch==1.12.1+cu102
torchsummary==1.5.1
torchvision==0.13.1+cu102
tqdm==4.64.1
```

# Dataset
This repository is trained on **DIV2K** dataset, download from: [link](https://data.vision.ee.ethz.ch/cvl/DIV2K) \
And I validate model on Set5, Set14, and BSD100 dataset download from: [link](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD) \
if you want to make rich dataset, you can follow `setup/prepare_dataset.py` for detail.

The data structure are described below here:
```
DIV2K
  |
  |--train
  |     |--img1.png
  |     |--img2.png
  |     |-- ...
  |
  |--valid
  |     |--img1.png
  |     |--img2.png
  |     |-- ...
```

# How to train, validation and inference
comming soon ...

# Result
Source of original paper results: https://arxiv.org/pdf/1609.04802v5.pdf. The result of this project when evaluation with scale `X4` is described below here:

| Eval dataset | Eval. Mat | SRGAN (original) | SRGAN (this project) |
|--------------|-----------|------------------|----------------------|
| Set5         | PSNR/SSIM | 29.40 / 0.8472   | 29.22 / 0.8464       |
| Set14        | PSNR/SSIM | 26.02 / 0.7397   | 26.69 / 0.7315       |
| BSD100       | PSNR/SSIM | 25.16 / 0.6688   | 26.21 / 0.6903       |

# Contributor
Tien Thong Doan
