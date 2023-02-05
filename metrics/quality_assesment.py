import cv2
import numpy as np
from sewar.full_ref import psnr, ssim

# PSNR and SSIM are implemented in sewar package

def define_quality_asseesment(sr_image, hr_image, mode='psnr'):
    assert mode in ['psnr', 'ssim']
    
    if mode == 'psnr':
        return psnr(sr_image, hr_image)
    elif mode == 'ssim':
        return ssim(sr_image, hr_image)
    else:
        raise NotImplementedError