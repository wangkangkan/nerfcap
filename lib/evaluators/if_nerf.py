import platform
import sys

import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
#from skimage import measure
import os
import cv2
from termcolor import colored


class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

        self.allerrmse = np.ones((14,300),dtype=np.float64) 
        self.allerrpsnr = np.ones((14,300),dtype=np.float64) 
        self.allerrssim = np.ones((14,300),dtype=np.float64) 
        self.boundbox = np.ones((14*300,4),dtype=np.float64) 

    def psnr_metric(self, img_pred, img_gt):
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, img_pred, img_gt, batch):
        if not cfg.eval_whole_img:
            mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
            H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
            mask_at_box = mask_at_box.reshape(H, W)
            # crop the object region
            x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
            img_pred = img_pred[y:y + h, x:x + w]
            img_gt = img_gt[y:y + h, x:x + w]

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.system('mkdir -p {}'.format(result_dir))

        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()

        path = '{}/{:d}/'.format(result_dir, view_index)
        os.makedirs(path, exist_ok=True)
        cv2.imwrite('{}/frame{:04d}_view{:04d}.png'.format(path, frame_index,view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        path = '{}/{:d}/gt/'.format(result_dir, view_index)
        os.makedirs(path, exist_ok=True)    
        cv2.imwrite('{}/frame{:04d}_view{:04d}_gt.png'.format(path, frame_index,view_index),
            (img_gt[..., [2, 1, 0]] * 255))

        # compute the ssim
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        self.boundbox[view_index*300+frame_index-700,:] = [x, y, w, h] 
        return ssim

    def evaluate(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        mask_at_box = mask_at_box.reshape(H, W)

        # convert the pixels into an image
        white_bkgd = int(cfg.white_bkgd)
        img_pred = np.zeros((H, W, 3)) + white_bkgd
        img_pred[mask_at_box] = rgb_pred
        img_gt = np.zeros((H, W, 3)) + white_bkgd
        img_gt[mask_at_box] = rgb_gt
        if cfg.eval_whole_img:
            rgb_pred = img_pred
            rgb_gt = img_gt

        mse = np.mean((rgb_pred - rgb_gt)**2)
        self.mse.append(mse)

        psnr = self.psnr_metric(rgb_pred, rgb_gt)
        self.psnr.append(psnr)

        rgb_pred = img_pred
        rgb_gt = img_gt
        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)
        self.ssim.append(ssim)

        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        self.allerrmse[view_index, frame_index-700] = mse
        self.allerrpsnr[view_index, frame_index-700] = psnr       
        self.allerrssim[view_index, frame_index-700] = ssim

    def summarize(self):
        result_dir = cfg.result_dir
        print(colored('summarize results saved at {}'.format(result_dir), 'yellow'))

        result_path = os.path.join(cfg.result_dir, 'metrics.npy')
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}
        # np.save(result_path, metrics)
        self.mse = []
        self.psnr = []
        self.ssim = []
        np.savetxt(os.path.join(cfg.result_dir, 'allerrmse_general_700.txt'),self.allerrmse)
        np.savetxt(os.path.join(cfg.result_dir, 'allerrpsnr_general_700.txt'),self.allerrpsnr)
        np.savetxt(os.path.join(cfg.result_dir, 'allerrssim_general_700.txt'),self.allerrssim)
        