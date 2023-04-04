import os
import multiprocessing
from os.path import join, isfile
import numpy as np
import scipy.ndimage
import skimage.color
from PIL import Image
import sklearn.cluster
from sklearn.cluster import KMeans
import util
import imageio
from opts import get_opts

def extract_filter_responses(opts, img):
    filter_scales=opts.filter_scales
    img=np.dstack([img]*3) if len(img.shape)<3 else img[:,:,:3] if img.shape[2]>3 else img
    img_lab=skimage.color.rgb2lab(img)
    filter_responses=[]
    for s in filter_scales:
        filters=[scipy.ndimage.gaussian_filter(img_lab[:,:,i],sigma=s,mode='reflect') for i in range(3)]
        filter_responses.extend(filters)
        filters=[scipy.ndimage.gaussian_laplace(img_lab[:,:,i],sigma=s,mode='reflect') for i in range(3)]
        filter_responses.extend(filters)
        filters=[scipy.ndimage.gaussian_filter(img_lab[:,:,i],sigma=s,order=[1,0],mode='reflect') for i in range(3)]
        filter_responses.extend(filters)
        filters=[scipy.ndimage.gaussian_filter(img_lab[:,:,i],sigma=s,order=[0,1],mode='reflect') for i in range(3)]
        filter_responses.extend(filters)
    filter_responses=np.dstack(filter_responses)
    return filter_responses

def compute_dictionary_one_image(args):
    opts=get_opts()
    arguments=args
    img=imageio.imread(f'../data/{arguments[2]}')
    img=img.astype('float')/255
    filter_responses=extract_filter_responses(opts,img)
    filter_shape=np.shape(filter_responses)
    sampled_response=np.reshape(filter_responses,(filter_shape[0]*filter_shape[1],filter_shape[2]))
    sampled_response=sampled_response[np.random.randint(filter_shape[0]*filter_shape[1],size=arguments[1]),:]
    return sampled_response

def compute_dictionary(opts, n_worker=1):
    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K
    alpha=opts.alpha
    train_files = open(join(data_dir, "C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\train_files.txt")).read().splitlines()
    train_data=np.asarray(train_files)
    n_train=train_data.shape[0]
    args=[(i,alpha,train_data[i]) for i in range(n_train)]
    with multiprocessing.Pool(processes=n_worker) as pool:
        responses = pool.map(compute_dictionary_one_image, args)
    response=np.concatenate(responses,axis=0)
    kmeans=sklearn.cluster.KMeans(n_clusters=K).fit(response)
    dictionary=kmeans.cluster_centers_
    np.save('C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\Dictionary\\dictionary.npy', dictionary)
    return dictionary

def get_visual_words(opts, img, dictionary):
    filter_responses=extract_filter_responses(opts,img)
    flat_responses=filter_responses.reshape(-1,filter_responses.shape[-1])
    euclidean_distances=scipy.spatial.distance.cdist(flat_responses,dictionary)
    wordmap=np.argmin(euclidean_distances,axis=1).reshape(img.shape[:2])
    return wordmap
