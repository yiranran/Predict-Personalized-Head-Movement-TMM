import os, sys, glob
import pdb
from PIL import Image
import numpy as np
from scipy.io import loadmat
sys.path.append('../Deep3DFaceReconstruction')
from preprocess_img import Preprocess, Preprocess2

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

    Lm3D = loadmat('../Deep3DFaceReconstruction/BFM/similarity_Lm3D_all.mat')
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31,37,40,43,46,49,55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
    Lm3D = Lm3D[[1,2,0,3,4],:]

    return Lm3D

def get_news5(n):
    trainN=300; testN=100
    video = str(n);name = str(n)+'_wm3drfull_win3';start = 0;
    print(video,name)

    rootdir = os.path.join(os.getcwd(),'../WM3DR/result/')
    srcdir = os.path.join(os.getcwd(),'../Data',video)
    srcdir2 = os.path.join(os.getcwd(),'../WM3DR/result/',video,'bm')
    srcdir3 = os.path.join(os.getcwd(),'../WM3DR/result/',video,'crop')
    os.makedirs(srcdir3, exist_ok=True)

    cmd = "cd "+rootdir+"; matlab -nojvm -nosplash -nodesktop -nodisplay -r \"alpha_blend_wm3dr('" + video + "'," + str(start) + "," + str(trainN+testN) + "); quit;\""
    os.system(cmd)
    
    lm3D = load_lm3d()
    lm_path = os.path.join('../Data',video,'frame0.txt')
    lm = np.loadtxt(lm_path)
    trans_params = None
    for i in range(start, start+trainN+testN):
        if os.path.exists(os.path.join(srcdir3,'%05d_wm3dr_bm.png'%i)) and os.path.exists(os.path.join(srcdir3,'%05d.png'%i)):
            continue
        img = Image.open(os.path.join(srcdir2,'%05d_wm3dr_bm.png'%i))
        if trans_params is None:
            input_img,_,transform_params = Preprocess(img,lm,lm3D)
        else:
            input_img,_ = Preprocess2(img,lm,transform_params)
        input_img = np.squeeze(input_img)
        im = Image.fromarray(input_img[:,:,::-1])
        im.save(os.path.join(srcdir3,'%05d_wm3dr_bm.png'%i))

        if os.path.exists(os.path.join(srcdir,'%05d.png'%i)):
            img = Image.open(os.path.join(srcdir,'%05d.png'%i))
        else:
            img = Image.open(os.path.join(srcdir,'frame%d.png'%i))
        input_img,_ = Preprocess2(img,lm,transform_params)
        input_img = np.squeeze(input_img)
        im = Image.fromarray(input_img[:,:,::-1])
        im.save(os.path.join(srcdir3,'%05d.png'%i))


    if not os.path.exists('datasets/list/trainA'):
        os.makedirs('datasets/list/trainA')
    if not os.path.exists('datasets/list/trainB'):
        os.makedirs('datasets/list/trainB')
    f1 = open('datasets/list/trainA/%s.txt'%name,'w')
    f2 = open('datasets/list/trainB/%s.txt'%name,'w')
    if 'win3' in name:
        start1 = start + 2
    else:
        start1 = start
    for i in range(start1,start+trainN):
        print(os.path.join(srcdir3,'%05d_wm3dr_bm.png'%i),file=f1)
        print(os.path.join(srcdir3,'%05d.png'%i),file=f2)
    f1.close()
    f2.close()
    if not os.path.exists('datasets/list/testA'):
        os.makedirs('datasets/list/testA')
    if not os.path.exists('datasets/list/testB'):
        os.makedirs('datasets/list/testB')
    f1 = open('datasets/list/testA/%s.txt'%name,'w')
    f2 = open('datasets/list/testB/%s.txt'%name,'w')
    for i in range(start+trainN,start+trainN+testN):
        print(os.path.join(srcdir3,'%05d_wm3dr_bm.png'%i),file=f1)
        print(os.path.join(srcdir3,'%05d.png'%i),file=f2)
    f1.close()
    f2.close()
    return name

def save_each_60(folder):
    pths = sorted(glob.glob(folder+'/*.pth'))
    for pth in pths:
        epoch = os.path.basename(pth).split('_')[0]
        if epoch == '60':
            continue
        os.remove(pth)

def save_each_10(folder):
    pths = sorted(glob.glob(folder+'/*.pth'))
    for pth in pths:
        epoch = os.path.basename(pth).split('_')[0]
        if epoch in ['10','20','30','40','50','60']:
            continue
        os.remove(pth)


if __name__ == '__main__':
    n = int(sys.argv[1])
    gpu_id = int(sys.argv[2])

    # first wm3dr recon, then crop
    name = get_news5(n) 
    print(name)

    # prepare arcface feature
    cmd = 'cd arcface/; python test_batch.py --imglist trainB/%s.txt --gpu %d' % (name,gpu_id)
    os.system(cmd)
    cmd = 'cd arcface/; python test_batch.py --imglist testB/%s.txt --gpu %d' % (name,gpu_id)
    os.system(cmd)

    # fine tune the mapping
    n = str(n)

    # finetune
    # the initial state is:
    # /home6/yiran/Audio-driven-TalkingFace-HeadPose/render-to-video/checkpoints/seq_p2p/0_net_G.pth
    cmd = 'python train.py --dataroot %s --name seq_p2p/%s_13-2 --model pix2pix --continue_train --epoch 0 --epoch_count 1 --netG unetac_256 --dataset_mode aligned_multi --attention 1 --lambda_mask 2 --Nw 3 --lr 0.0001 --display_env seq_%s_13 --gpu_ids %d --niter 60 --niter_decay 0' % (name,n,n,gpu_id)
    os.system(cmd)
    save_each_10('checkpoints/seq_p2p/%s_13-2'%(n))

    epoch = 60
    cmd = 'python test.py --dataroot %s --name seq_p2p/%s_13-2 --model pix2pix --netG unetac_256 --dataset_mode aligned_multi --attention 1 --Nw 3 --norm batch --num_test 200 --epoch %d --gpu_ids %d --imagefolder images%d' % (name,n,epoch,gpu_id,epoch)
    os.system(cmd)
