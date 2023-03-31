from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
import time

from lib.decode import decode
from lib.model import create_model, load_model
from lib.pt_renderer import PtRender
from lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  construct_meshes,
)

from scipy.io import loadmat, savemat
import pdb


def opts():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--input_res', default=512, type=int)
  parser.add_argument('--arch', default='resnet_50', type=str)
  parser.add_argument('--load_model', default='model/final.pth', type=str)
  parser.add_argument('--BFM', default='BFM/mSEmTFK68etc.chj', type=str)
  parser.add_argument('--video_path', default='video/1.mp4', type=str)
  parser.add_argument('--output', default='result', type=str)
  parser.add_argument('--gpu_id', default=0, type=int)
  parser.add_argument('--start', default=0, type=int)
  parser.add_argument('--end', default=846, type=int)

  return parser.parse_args()

def main(opt):

    device = torch.device('cuda:{}'.format(opt.gpu_id))
    opt.device = device

    print('Creating model...')
    # opt.input_res = 256
    render = PtRender(opt).to(device).eval()
    opt.heads = {'hm': 1, 'params': 257}
    model = create_model(opt.arch, opt.heads)

    if opt.load_model != '':
        model = load_model(model, opt.load_model)
        model.to(device).eval()

    with torch.no_grad():
        #videos = sorted(glob.glob('/home6/yiran/TalkingFaceRev/talkingheadData/Motion/motion_frames/*/*/'))
        videos = sorted(glob.glob('/home6/yiran/TalkingFaceRev/talkingheadData/Motion2/motion_frames/*/*/'))
        print('len videos: {}'.format(len(videos)))
        for ii in range(opt.start,opt.end):
            video = videos[ii]
            base = 'Motion/{}/{}'.format(video.split('/')[-3],video.split('/')[-2])

            os.makedirs(os.path.join(opt.output, base), exist_ok = True)
            t0 = time.time()
            for i in range(len(glob.glob('{}/frame*.jpg'.format(video)))):
                matfile = os.path.join(opt.output, base, '{}.mat'.format(str(i).zfill(5)))
                if os.path.exists(matfile):
                    continue

                image = cv2.imread('{}/frame{}.jpg'.format(video,i))
                #print('{}/frame{}.jpg'.format(video,i))
                pre_img, meta = preprocess(image.copy(), opt.input_res, device)

                output, topk_scores, topk_inds, topk_ys, topk_xs = decode(pre_img, model)
                params = _tranpose_and_gather_feat(output['params'], topk_inds)

                B, C, _ = params.size()
                if C == 0:
                    print('{}/frame{}.jpg'.format(video,i),'no face!')
                    #cv2.imwrite(outfile, image)
                    continue

                # 3DMM formation
                # split coefficients
                id_coeff, ex_coeff, tex_coeff, coeff = render.Split_coeff(params.view(-1, params.size(2)))
                render.set_RotTransLight(coeff, topk_inds.view(-1))
                savemat(matfile,{'params':params.cpu().numpy(), 'topk_inds':topk_inds.cpu().numpy()})
            t1 = time.time()
            print('{} {} takes {:02f} seconds'.format(ii, base, t1-t0))

if __name__ == '__main__':
  opt = opts()
  main(opt)
