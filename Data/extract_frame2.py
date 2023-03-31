import cv2
import os, sys
import glob
import dlib
import numpy as np
import time
from scipy.io import loadmat
from PIL import Image
sys.path.append('../Deep3DFaceReconstruction')
from preprocess_img import Preprocess

# load landmarks for standard face, which is used for image preprocessing
def load_lm3d():

	Lm3D = loadmat('../Deep3DFaceReconstruction/BFM/similarity_Lm3D_all.mat')
	Lm3D = Lm3D['lm']

	# calculate 5 facial landmarks using 68 landmarks
	lm_idx = np.array([31,37,40,43,46,49,55]) - 1
	Lm3D = np.stack([Lm3D[lm_idx[0],:],np.mean(Lm3D[lm_idx[[1,2]],:],0),np.mean(Lm3D[lm_idx[[3,4]],:],0),Lm3D[lm_idx[5],:],Lm3D[lm_idx[6],:]], axis = 0)
	Lm3D = Lm3D[[1,2,0,3,4],:]

	return Lm3D

# load input images and corresponding 5 landmarks
def load_img(img_path,lm_path):

	image = Image.open(img_path)
	lm = np.loadtxt(lm_path)

	return image,lm

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def detect_image(imagename, savepath, detector, predictor):
    image = cv2.imread(imagename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
        eyel = np.round(np.mean(shape[36:42,:],axis=0)).astype("int")
        eyer = np.round(np.mean(shape[42:48,:],axis=0)).astype("int")
        nose = shape[33]
        mouthl = shape[48]
        mouthr = shape[54]
        if savepath != "":
            message = '%d %d\n%d %d\n%d %d\n%d %d\n%d %d\n' % (eyel[0],eyel[1],
                eyer[0],eyer[1],nose[0],nose[1],
                mouthl[0],mouthl[1],mouthr[0],mouthr[1])
            with open(savepath, 'w') as s_file:
                s_file.write(message)
            return
    
def detect_dir(folder, lm3D, detector, predictor, crop):
    # only detect the first frame
    for file in sorted(glob.glob(folder+"/frame0.png")):
        detect_image(file, file[:-4]+'.txt', detector, predictor)
    if not crop:
        return
    txt_list = sorted(glob.glob(os.path.join(folder,'*.txt')))
    if len(txt_list) == 0:
        return
    out_folder = folder + '_crop'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for i in range(len(glob.glob(folder+"/*.png"))):
        #print(file,txt_list[0])
        input_file = folder + "/frame{}.png".format(i)
        img,lm = load_img(input_file,txt_list[0])
        # preprocess input image
        input_img,lm_new,transform_params = Preprocess(img,lm,lm3D)
        input_img = np.squeeze(input_img)
        im = Image.fromarray(input_img[:,:,::-1])
        cropped_output_path = out_folder + "/{:05d}.png".format(i)
        im.save(cropped_output_path)

t1 = time.time()
mp4 = sys.argv[1]
crop = int(sys.argv[2])
videoname = mp4
outdir = mp4[:-4]
lm3D = load_lm3d()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Deep3DFaceReconstruction/shape_predictor_68_face_landmarks.dat')
    
cap = cv2.VideoCapture(videoname)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
success, image = cap.read()
postfix = ".png"
if not os.path.exists(outdir):
    os.makedirs(outdir)
count = 0
while success and count<400:
    cv2.imwrite("%s/frame%d%s"%(outdir,count,postfix),image)
    success, image = cap.read()
    count += 1
detect_dir(outdir, lm3D, detector, predictor, crop)
t2 = time.time()
