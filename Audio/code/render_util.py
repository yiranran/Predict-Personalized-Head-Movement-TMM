import os
import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf 
from reconstruct_mesh import Reconstruction_for_render, Render_layer

rootdir = '../../Deep3DFaceReconstruction/'

class BFM():
    def __init__(self):
        model_path = rootdir+'BFM/BFM_model_front.mat'
        model = loadmat(model_path)
        self.meanshape = model['meanshape'] # mean face shape 
        self.idBase = model['idBase'] # identity basis
        self.exBase = model['exBase'] # expression basis
        self.meantex = model['meantex'] # mean face texture
        self.texBase = model['texBase'] # texture basis
        self.point_buf = model['point_buf'] # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
        self.tri = model['tri'] # vertex index for each triangle face, starts from 1
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1 # 68 face landmark index, starts from 0


def render_fast(coefs_head, coefs_lip, audioname, outname, gpu_id, Ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if not os.path.exists(outname):
        os.mkdir(outname)
    facemodel = BFM()
    with tf.Graph().as_default() as graph:

        faceshaper = tf.placeholder(name = "face_shape_r", shape = [1,35709,3], dtype = tf.float32)
        facenormr = tf.placeholder(name = "face_norm_r", shape = [1,35709,3], dtype = tf.float32)
        facecolor = tf.placeholder(name = "face_color", shape = [1,35709,3], dtype = tf.float32)
        rendered = Render_layer(faceshaper,facenormr,facecolor,facemodel,1)

        rstimg = tf.placeholder(name = 'rstimg', shape = [224,224,4], dtype=tf.uint8)
        encode_png = tf.image.encode_png(rstimg)
        border = np.zeros((224,224,4))
        border[20:22,20:204,2:] = 255
        border[-22:-20:,20:204,2:] = 255
        border[20:204,20:22,2:] = 255
        border[20:204,-22:-20,2:] = 255

        coefs_head[:2,:] = coefs_head[2,:]
        coefs_head[-2:,:] = coefs_head[-3,:]
        
        with tf.Session() as sess:
            for i in range(coefs_lip.shape[0]):
                coef = np.zeros((1,257))
                coef[0,80:144] = coefs_lip[i][:64]
                if coefs_head[i].shape[0] == 6:
                    coef[0,224:227] = coefs_head[i][-6:-3]
                    coef[0,254:257] = coefs_head[i][-3:]
                elif coefs_head[i].shape[0] == 3:
                    coef[0,224:227] = coefs_head[i][:3]
                face_shape_r,face_norm_r,face_color,tri = Reconstruction_for_render(coef,facemodel)
                final_images = sess.run(rendered, feed_dict={faceshaper: face_shape_r.astype('float32'), facenormr: face_norm_r.astype('float32'), facecolor: face_color.astype('float32')})
                result_image = final_images[0, :, :, :]
                result_image = np.clip(result_image, 0., 1.).copy(order='C')
                if i not in Ids:
                    result_bytes = sess.run(encode_png,{rstimg: result_image*255.0})
                else:
                    result_bytes = sess.run(encode_png,{rstimg: result_image*255.0 + border})
                print(result_image.shape)
                result_output_path = '%s/%d_render.png'%(outname, i)
                with open(result_output_path, 'wb') as output_file:
                    output_file.write(result_bytes)
    write_video(outname, audioname, coefs_head.shape[0])
    print(outname)
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8,9"

def write_video(outname, audioname, n):
    fps = 25
    img_size = (224, 224)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    
    mp4outname = outname+'.mp4'
    video_writer = cv2.VideoWriter(mp4outname, fourcc, fps, img_size)
    for i in range(n):
        frame = cv2.imread('%s/%d_render.png'%(outname, i))

        video_writer.write(frame)
    video_writer.release()
    
    cmd = 'ffmpeg -i ' + mp4outname  + ' -i ' + audioname + ' -vcodec copy  -acodec copy -y  ' + mp4outname.replace('.mp4','.mov') 
    os.system(cmd)
    cmd = 'rm ' + mp4outname
    os.system(cmd)
