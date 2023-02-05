import cv2
import warnings
import numpy as np
from sewar.full_ref import psnr, ssim

# PSNR and SSIM are implemented in sewar package

def _check_image(raw_image: np.ndarray, dst_image: np.ndarray):
    """Check whether the size and type of the two images are the same
    Args:
        raw_image (np.ndarray): image data to be compared, BGR format, data range [0, 255]
        dst_image (np.ndarray): reference image data, BGR format, data range [0, 255]
    """
    # check image scale
    assert raw_image.shape == dst_image.shape, \
        f"Supplied images have different sizes {str(raw_image.shape)} and {str(dst_image.shape)}"

    # check image type
    if raw_image.dtype != dst_image.dtype:
        warnings.warn(f"Supplied images have different dtypes{str(raw_image.shape)} and {str(dst_image.shape)}")


def define_quality_asseesment(sr_image, hr_image, mode='psnr'):
    assert mode in ['psnr', 'ssim']
    
    sr_image = sr_image.copy().astype(np.int64)
    hr_image = hr_image.copy().astype(np.int64)
    
    _check_image(sr_image, hr_image)
    
    if mode == 'psnr':
        return psnr(sr_image, hr_image)
    elif mode == 'ssim':
        return ssim(sr_image, hr_image)[1]
    else:
        raise NotImplementedError
    

# code to test the quality assesment function
if __name__ == '__main__':
    hr_image = np.random.randint(220, 255, (256, 256, 3))
    sr_image = np.random.randint(25, 155, (256, 256, 3))
    
    psnr_score = define_quality_asseesment(sr_image, hr_image, mode='psnr')
    ssim_score = define_quality_asseesment(sr_image, hr_image, mode='ssim')
    
    print("PSNR: ", psnr_score)
    print("SSIM: ", ssim_score)