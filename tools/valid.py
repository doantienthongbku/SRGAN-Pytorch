import os
import torch
import cv2
from utils import *
from metrics.quality_assesment import PSNR, SSIM


config = read_cfg('config/config.yaml')
model_path = config['test']['g_model_weights_path']