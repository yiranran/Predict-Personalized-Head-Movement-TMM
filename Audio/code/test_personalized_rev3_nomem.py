import os, sys
import argparse
import glob
import shutil
import cv2
import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
import librosa
import python_speech_features
from scipy.signal import argrelextrema
sys.path.append('../../WM3DR')
from lib.pt_renderer import PtRender
from lib.utils import (
  _tranpose_and_gather_feat,
  get_frames,
  preprocess,
  construct_meshes,
)
from torch.autograd import Variable
from render_util import render_fast
import pdb
sys.path.append('../../Deep3DFaceReconstruction')
from preprocess_img import Preprocess, Preprocess2

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

    Lm3D = loadmat('../../Deep3DFaceReconstruction/BFM/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
    Lm3D = Lm3D[[1,2,0,3,4],:]

    return Lm3D

def opts():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch_size', default=1, type=int)
  parser.add_argument('--input_res', default=512, type=int)
  parser.add_argument('--BFM', default='./BFM/mSEmTFK68etc.chj', type=str)
  parser.add_argument('--audiobasen', type=str)
  parser.add_argument('--person', type=str)
  parser.add_argument('--gpu_id', default=0, type=int)
  parser.add_argument('--debug', default=0, type=int)
  parser.add_argument('--option', default=13, type=int)
  parser.add_argument('--ganepoch', default=60, type=int)

  return parser.parse_args()

def render_params(render, params, topk_inds, result_render_path, result_mask_path, device, h=224, w=224):
    B, C, _ = params.size()

    # 3DMM formation
    # split coefficients
    id_coeff, ex_coeff, tex_coeff, coeff = render.Split_coeff(params.view(-1, params.size(2)))
    render.set_RotTransLight(coeff, topk_inds.view(-1))

    # reconstruct shape
    canoShape_ = render.Shape_formation(id_coeff, ex_coeff)
    rotShape = render.RotTrans(canoShape_)

    Albedo = render.Texture_formation(tex_coeff)

    Texture, lighting = render.Illumination(Albedo, canoShape_)
    Texture = torch.clamp(Texture, 0, 1)

    rotShape = rotShape.view(B, C, -1, 3)
    Texture = Texture.view(B, C, -1, 3)

    # Pytorch3D render
    meshes = construct_meshes(rotShape, Texture, render.BFM.tri.view(1, -1), device)

    rendered, gpu_masks, depth = render(meshes) # RGB
    rendered = rendered.squeeze(0).detach().cpu().numpy()
    gpu_masks = gpu_masks.squeeze(0).unsqueeze(-1).cpu().numpy()

    # resize to original image
    rendered = cv2.resize(rendered, (max(h, w), max(h, w)))[:h, :w]
    gpu_masks = cv2.resize(gpu_masks, (max(h, w), max(h, w)), interpolation=cv2.INTER_NEAREST)[:h, :w, np.newaxis]
    
    cv2.imwrite(result_render_path, (rendered[..., ::-1] * 255).astype(np.uint8))
    cv2.imwrite(result_mask_path, (gpu_masks[..., ::-1] * 255).astype(np.uint8))

def getsingle5(srcdir,name,lm_path,multi=1):
    srcroot = os.getcwd()
    imgs = glob.glob(os.path.join(srcroot,srcdir,'*_blend2.png'))
    print('srcdir',os.path.join(srcroot,srcdir,'*_blend2.png'))
    src2 = os.path.join(srcroot,srcdir,'crop')
    os.makedirs(src2, exist_ok=True)

    # crop
    lm3D = load_lm3d()
    lm = np.loadtxt(lm_path)
    trans_params = None
    print(len(imgs))
    for i in range(len(imgs)):
        img = Image.open(imgs[i])
        if i == 0:
            input_img,_,transform_params = Preprocess(img,lm,lm3D)
        else:
            input_img,_ = Preprocess2(img,lm,transform_params)
        input_img = np.squeeze(input_img)
        im = Image.fromarray(input_img[:,:,::-1])
        im.save(os.path.join(src2,os.path.basename(imgs[i])))

    os.makedirs('../../render-to-video/datasets/list/testSingle',exist_ok=True)
    f1 = open('../../render-to-video/datasets/list/testSingle/%s.txt'%name,'w')
    imgs = sorted(glob.glob(os.path.join(src2,'*_blend2.png')))
    if multi:
        imgs = imgs[2:]
    for im in imgs:
        print(im, file=f1)
    f1.close()
    return src2,lm,transform_params

def read_vid(vid):
    cap = cv2.VideoCapture(vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = cap.read()
    images = []
    while success:
        images.append(cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC))
        success, image = cap.read()
    return images

def read_vid2(vid,lm_path):
    cap = cv2.VideoCapture(vid)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    success, image = cap.read()
    images = []
    lm3D = load_lm3d()
    lm = np.loadtxt(lm_path)
    trans_params = None
    while success:
        image = Image.fromarray(image[:,:,::-1])
        if trans_params is None:
            image,_,transform_params = Preprocess(image,lm,lm3D)
        else:
            image,_ = Preprocess2(image,lm,transform_params)
        image = np.squeeze(image)
        images.append(cv2.resize(image,(256,256),interpolation=cv2.INTER_CUBIC))
        success, image = cap.read()
    return images

def read_audio(audioname):
    speech, sr = librosa.load(audioname, sr=16000)
    speech = np.insert(speech, 0, np.zeros(1920))
    speech = np.append(speech, np.zeros(1920))
    mfcc = python_speech_features.mfcc(speech, 16000, winstep=0.01)
    return mfcc

def load_coef(coef_folder):
    res = []
    filelist = glob.glob(coef_folder+'/*.mat')
    train_frames = 300
    for i in range(train_frames):
        if os.path.exists(coef_folder+'/%05d.mat'%i):
            data = loadmat(coef_folder+'/%05d.mat'%i)['params'][0]
            if data.shape[0] != 1:
                data = data[0:1,:]
            res.append(data)
        else:
            return []
    return np.stack(res, axis=0)

def coefNet_pose(audioname, coef_folder, device, version='Motion846_contraloss4_autogradhidden_hn_conti_10epochs'):
    cttNet = torch.load('../model/{}/latest_cttMotionNet.pth'.format(version), map_location=device)
    iddNet = torch.load('../model/{}/latest_iddNet.pth'.format(version), map_location=device)
    cttNet.to(device)
    cttNet.eval()
    iddNet.to(device)
    iddNet.eval()
        
    with torch.no_grad(): 
        mfcc = Variable(torch.FloatTensor(read_audio(audioname)).to(device))
        mfcc = mfcc.unsqueeze(0)
        
        coef = Variable(torch.FloatTensor(load_coef(coef_folder)).to(device))
        print('loading encoder input from', coef_folder)
        coef_ang = coef[:, 0, 224:227]
        coef_tsl = coef[:, 0, 254:257]

        coef = torch.cat([coef_ang, coef_tsl], 1)
        coef = coef.unsqueeze(0)
        
        if 'MotionRotate' not in version:
            idd,_,_ = iddNet(coef)
        else:
            idd,_,_ = iddNet(coef_ang.unsqueeze(0))
        coef_predict = cttNet(mfcc, idd)
    return coef_predict[0].detach().cpu().numpy(), coef[0].detach().cpu().numpy()

def nearest(real, query):
    diff = np.abs(np.tile(query, [real.shape[0],1]) - real)
    cost = np.sum(diff[:,:3], axis=1)
    I = np.argmin(cost)
    return I

def smooth(x,window_len=11,window='hanning'):
	if x.ndim != 1:
		raise(ValueError, "smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise(ValueError, "Input vector needs to be bigger than window size.")
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	if window == 'flat':
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')

	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[int(window_len/2):-int(window_len/2)]

def bg_matching2(pose_predict, pose_real, speed=1):
    print('bg_matching2 ...')
    N = pose_predict.shape[0]
    num = pose_real.shape[0]
    thre = N/10.
    # cal keyframes
    y = [0,0,0]
    Ids = [0,0,0]
    y0 = [0,0,0]
    n_peaks = 0
    for k in [2]:
        y[k] = smooth(pose_predict[:,k],window_len=7)
        y0[k] = y[k]
        # local maxima
        maxIds = argrelextrema(y[k],np.greater)
        # local minima
        minIds = argrelextrema(y[k],np.less)
        Ids[k] = np.concatenate((maxIds[0],minIds[0]))
        Ids[k] = np.sort(Ids[k])
        n_peaks += Ids[k].shape[0]
        y[k] = y0[k][Ids[k]]
    print(n_peaks, Ids[2].shape, thre)
    while n_peaks > thre:
        n_peaks = 0
        for k in [2]:
            maxIds = argrelextrema(y[k],np.greater,order=2)
            minIds = argrelextrema(y[k],np.less,order=2)
            Ids[k] = np.concatenate((Ids[k][maxIds],Ids[k][minIds]))
            Ids[k] = np.sort(Ids[k])
            n_peaks += Ids[k].shape[0]
            y[k] = y0[k][Ids[k]]
        print(n_peaks, Ids[2].shape, thre)
    Ids = Ids[2]
    Ids = np.insert(Ids,0,0)
    Ids = np.append(Ids,N-1)
    Ids = np.sort(np.unique(Ids))
    print(Ids)
    # cal nearest
    Is = np.zeros(Ids.shape)
    I = nearest(pose_real[:,:3], pose_predict[0,:3])
    Is[0] = I
    print(Ids[0], I)
    for i in range(1,Ids.shape[0]):
        period = Ids[i] - Ids[i-1]
        period_st = max(0,int(I-speed*period))
        period_ed = min(num,I+int(speed*period))
        pose_realt = pose_real[period_st:period_ed,:3]
        delta = pose_predict[Ids[i],:3] - pose_predict[Ids[i-1],:3]
        target = pose_real[I,:3] + delta
        In = nearest(pose_realt, target)
        I = period_st + In
        print(Ids[i], I, Ids[i-1])
        Is[i] = I
    assigns = [0] * N
    for i in range(Ids.shape[0]-1):
        l = Ids[i+1] - Ids[i]
        assigns[Ids[i]] = int(Is[i])
        for j in range(1,l):
            assigns[Ids[i]+j] = int(round(float(j)/l*(Is[i+1]-Is[i]) + Is[i]))
    assigns[Ids[-1]] = int(Is[-1])
    return assigns, Ids
      
if __name__ == "__main__":
    opt = opts()

    person = opt.person
    audiobasen = opt.audiobasen
    gpu_id = opt.gpu_id
    debug = opt.debug
    option = opt.option
    ganepoch = opt.ganepoch

    if os.path.exists(os.path.join('../audio/',audiobasen+'.wav')):
        in_file = os.path.join('../audio/',audiobasen+'.wav')
    elif os.path.exists(os.path.join('../audio/',audiobasen+'.mp3')):
        in_file = os.path.join('../audio/',audiobasen+'.mp3')
    else:
        print('audio file not exists, please put in %s'%os.path.join(os.getcwd(),'../audio'))
        exit(-1)
    
    audio_exp_name = 'atcnet_pose0_con3/'+person
    audioepoch = 99
    audiomodel = os.path.join(audio_exp_name,audiobasen+'_%d'%audioepoch)
    sample_dir = os.path.join('../results3_nomem/',audiomodel)

    if option == 13: #[op5 + WM3DRFull]
        ganmodel='seq_p2p/%s_13'%person;post='_full9'
        audionet = '../model/atcnet_pose01/atcnet_lstm_199.pth'#!!
        sample_dir += '_coeffix_generalexp01_wm3drfull_finetune'
        seq='rseq13_'+person+'_'+audiobasen+post # use same seq as withmem
    os.makedirs(sample_dir, exist_ok=True)

    ## 1.1 audio to expression
    if not os.path.exists(sample_dir+'/00000.npy'):
        add = '--model_name {} --pose 1 --relativeframe 0'.format(audionet)
        print('python atcnet_test1.py --device_ids %d %s --sample_dir %s --in_file %s' % (gpu_id,add,sample_dir,in_file))
        os.system('python atcnet_test1.py --device_ids %d %s --sample_dir %s --in_file %s' % (gpu_id,add,sample_dir,in_file))
    
    ## 1.2 audio to pose
    coef_src_dir = os.path.join('../../WM3DR/result', '{}'.format(person))
    device = torch.device('cuda:{}'.format(gpu_id))
    opt.device = device
    pose_predict, pose_real = coefNet_pose(in_file,coef_src_dir,device)
    assigns, Ids = bg_matching2(pose_predict, pose_real)
    if debug:
        # render original predicted poses
        coefs_tar = sorted(glob.glob(os.path.join(sample_dir,'*.npy')))
        exp_predict = np.zeros([len(coefs_tar), 64])
        for i in range(len(coefs_tar)):
            exp_predict[i] = np.load(os.path.join(sample_dir,'{:05d}.npy'.format(i)))[:64]
        render_fast(pose_predict, exp_predict, in_file, os.path.join(sample_dir,'pose_predict'), gpu_id, Ids)

    
    ## 2.render to save_dir
    render = PtRender(opt).to(device).eval()
    coefs_tar = sorted(glob.glob(os.path.join(sample_dir,'*.npy')))
    src_len = 300
    coef_src = []
    for i in range(src_len):
        coef_src.append(loadmat(os.path.join(coef_src_dir, '{:05d}.mat'.format(i))))
    ori_dir = os.path.join('../../Data', '{}'.format(person))
    rep = 'frame{:d}.png'
    save_dir = os.path.join(sample_dir,'R_%s_reassign2'%person)
    bg_dir = os.path.join(sample_dir,'bg_reassign2')
    os.makedirs(save_dir, exist_ok = True)
    os.makedirs(bg_dir, exist_ok = True)
    im0 = cv2.imread(os.path.join(ori_dir, rep.format(0)))
    print(os.path.join(ori_dir, rep.format(0)))
    h0, w0, _ = im0.shape
    print('rendering ...')
    for i in range(len(coefs_tar)):
        src_rank = assigns[i] # calc from pose_predict
        shutil.copy(os.path.join(ori_dir, rep.format(src_rank)), os.path.join(bg_dir,'{:05d}.png'.format(i)))

        params = coef_src[src_rank]['params']
        topk_inds = coef_src[src_rank]['topk_inds']
        coef_tar = np.load(os.path.join(sample_dir,'{:05d}.npy'.format(i)))
        params[:,:,80:144] = coef_tar[:64] # paste exp coeff

        result_render_path = os.path.join(save_dir,'{:05d}.png'.format(i))
        result_mask_path = result_render_path[:-4] + '_mask.png'
        if not os.path.exists(result_render_path) or not os.path.exists(result_mask_path):
            render_params(render, torch.from_numpy(params).to(device), torch.from_numpy(topk_inds).to(device), result_render_path, result_mask_path, device, h0, w0)
    if debug:
        if not os.path.exists(save_dir+'.mov'):
            command = 'ffmpeg -framerate 25  -i ' + save_dir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + save_dir+'.mp4'
            os.system(command)
            cmd = 'ffmpeg -i ' + save_dir+'.mp4'  + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + save_dir+'.mov'
            os.system(cmd)

            command = 'ffmpeg -framerate 25  -i ' + bg_dir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + sample_dir+'/选取背景结果.mp4'
            os.system(command)
    
    ## 3.blend rendered with background
    srcdir = save_dir
    if shutil.which('matlab'): 
        cmd = "cd ../results; matlab -nojvm -nosplash -nodesktop -nodisplay -r \"alpha_blend_vbg2('" + bg_dir + "','" + srcdir + "'); quit;\""
    elif shutil.which('octave'):
        cmd = "cd ../results; octave --eval \"pkg load image; alpha_blend_vbg2('" + bg_dir + "','" + srcdir + "'); quit;\""
    else:
        raise Exception('No matlab or octave installation found!')
        
    os.system(cmd)
    tempdir = "temp_{}_{}".format(person, audiobasen)
    if debug:
        if not os.path.exists(srcdir+'-1.mp4'):
            command = 'ffmpeg -framerate 25  -i ' + srcdir + '/%05d_blend2.png -c:v libx264 -y -vf format=yuv420p ' + srcdir+'-1.mp4'
            os.system(command)
    if debug:
        if not os.path.exists(sample_dir+'/recon-0.mp4') or not os.path.exists(sample_dir+'/recon-0.mov'):
            os.system("rm -rf " + tempdir)
            os.system("mkdir " + tempdir)
            fusedir = '../../WM3DR/result/{}/crop/'.format(person)
            for i in range(400):
                shutil.copy("{}/{:05d}.png".format(fusedir,i),"{}/{:05d}.png".format(tempdir,i))
            command = 'ffmpeg -framerate 25  -i ' + './' + tempdir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + sample_dir+'/recon-0.mp4'
            os.system(command)
            in_file1 = '../../Data/{}.wav'.format(person)
            if not os.path.exists(in_file1):
                cmd = 'ffmpeg -i ../../Data/{}.mp4 {}'.format(person, in_file1)
                os.system(cmd)
            cmd = 'ffmpeg -i ' + sample_dir+'/recon-0.mp4'  + ' -i ' + in_file1 + ' -vcodec copy  -acodec copy -y  ' + sample_dir+'/recon-0.mov'
            os.system(cmd)
        if not os.path.exists(sample_dir+'/recon-1.mp4') or not os.path.exists(sample_dir+'/recon-1.mov'):
            os.system("rm -rf " + tempdir)
            os.system("mkdir " + tempdir)
            fusedir1 = '../../WM3DR/result/{}/crop/'.format(person)
            for i in range(400):
                shutil.copy("{}/{:05d}_wm3dr_bm.png".format(fusedir1,i),"{}/{:05d}.png".format(tempdir,i))
            command = 'ffmpeg -framerate 25  -i ' + './' + tempdir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + sample_dir+'/recon-1.mp4'
            os.system(command)
            in_file1 = '../../Data/{}.wav'.format(person)
            if not os.path.exists(in_file1):
                cmd = 'ffmpeg -i ../../Data/{}.mp4 {}'.format(person, in_file1)
                os.system(cmd)
            cmd = 'ffmpeg -i ' + sample_dir+'/recon-1.mp4'  + ' -i ' + in_file1 + ' -vcodec copy  -acodec copy -y  ' + sample_dir+'/recon-1.mov'
            os.system(cmd)
        if not os.path.exists(sample_dir+'/real-0.mp4'):
            os.system("rm -rf " + tempdir)
            os.system("mkdir " + tempdir)
            realdir = '../../Data/{}/'.format(person)
            for i in range(400):
                shutil.copy(os.path.join(ori_dir, rep.format(i)),"{}/{:05d}.png".format(tempdir,i))
            command = 'ffmpeg -framerate 25  -i ' + './' + tempdir + '/%05d.png -c:v libx264 -y -vf format=yuv420p ' + sample_dir+'/real-0.mp4'
            os.system(command)
    os.system("rm "+sample_dir+"/*.npy")
    
    ## 4.gan
    save_dir = os.path.join(sample_dir,'R_%s_reassign2'%person)
    video_name = os.path.join(sample_dir,'%s_%swav_results%s.mp4'%(person,audiobasen,post))
    if not os.path.exists(video_name.replace('.mp4','.mov')):
        sample_dir2 = '../../render-to-video/results/%s/test_%d/images%s_nomem/'%(ganmodel,ganepoch,seq)
        srcroot = os.getcwd()
        gan_input_dir = os.path.join(srcroot,save_dir,'crop')
        command = 'cd ../../render-to-video; python test.py --dataroot %s --name %s --netG unetac_256 --model test --Nw 3 --norm batch --dataset_mode single_multi --use_memory 0 --attention 1 --num_test 10000 --epoch %d --gpu_ids %d --imagefolder images%s_nomem'%(seq,ganmodel,ganepoch,gpu_id,seq)
        print(command)
        os.system(command)

        prefix = os.path.basename(gan_input_dir)
        os.system('cp '+sample_dir2+'/'+prefix+'-00002_blend2_fake.png '+sample_dir2+'/'+prefix+'-00000_blend2_fake.png')
        os.system('cp '+sample_dir2+'/'+prefix+'-00002_blend2_fake.png '+sample_dir2+'/'+prefix+'-00001_blend2_fake.png')
        
        command = 'ffmpeg -loglevel panic -framerate 25  -i ' + sample_dir2 + '/' + prefix + '-%05d_blend2_fake.png -c:v libx264 -y -vf format=yuv420p ' + video_name
        os.system(command)

        command = 'ffmpeg -loglevel panic -i ' + video_name + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')
        os.system(command)
        os.remove(video_name)
        print('saved to',video_name.replace('.mp4','.mov'))
    
    if debug:
        vid1 = save_dir+'.mov'
        vid2 = video_name.replace('.mp4','.mov')
        video_name1 = video_name.replace('.mp4','_concat.mp4')
        if not os.path.exists(video_name1.replace('.mp4','.mov')):
            lm_path = os.path.join('../../Data',person,'frame0.txt')
            images1 = read_vid2(vid1,lm_path)
            images2 = read_vid(vid2)
            print(len(images1),len(images2))
            
            os.system("rm -rf " + tempdir)
            os.system("mkdir " + tempdir)
            for i in range(len(images1)):
                imcombine = np.concatenate((images1[i],images2[i]),axis=1)
                cv2.imwrite('./' + tempdir + '/concat{:05d}.png'.format(i),imcombine)
            command = 'ffmpeg -framerate 25  -i ' + './' + tempdir + '/concat%05d.png -c:v libx264 -y -vf format=yuv420p ' + video_name1
            os.system(command)
            command = 'ffmpeg -loglevel panic -i ' + video_name1 + ' -i ' + in_file + ' -vcodec copy  -acodec copy -y  ' + video_name1.replace('.mp4','.mov')
            os.system(command)
            os.remove(video_name1)
            os.system("rm -rf " + tempdir)
            print('saved to',video_name1.replace('.mp4','.mov'))
    print('output is',video_name.replace('.mp4','.mov'))