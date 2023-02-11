import torch
import torch.nn as nn
import time
from metrics.meter import AverageMeter, ProgressMeter


def validation(config, g_model: nn.Module, valid_prefetcher, epoch: int,
               writer, psnr_model, ssim_model, mode: str):
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(valid_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")
    
    # convert mode to eval
    g_model.eval()
    batch_index = 0
    
    valid_prefetcher.reset()
    batch_data = valid_prefetcher.next()
    end = time.time()
    
    with torch.no_grad():
        while batch_data is not None:
            # tranfer data to GPU
            hr = batch_data['hr'].to(config['device'], non_blocking=True)
            lr = batch_data['lr'].to(config['device'], non_blocking=True)
            
            # generate super resolution image
            sr = g_model(lr)
            
            # compute psnr and ssim
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            
            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Record training log information
            if batch_index % config['train']['valid_print_frequency'] == 0:
                progress.display(batch_index + 1)
                
            batch_data = valid_prefetcher.next()
            batch_index += 1
            
    # print metrics
    progress.display_summary()

    if mode == "valid" or mode == "test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg
            
    