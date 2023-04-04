import os
import math
import multiprocessing
from os.path import join
from copy import copy
import sklearn

import numpy as np
from PIL import Image

import visual_words
import imageio
from opts import get_opts

def get_feature_from_wordmap(opts, wordmap):
    K = opts.K
    shape=np.shape(wordmap)
    data=np.reshape(wordmap,(shape[0]*shape[1],1))
    hist1,bin_edge=np.histogram(data,bins=np.linspace(0,K,num=K+1,endpoint=True))
    hist=hist1/np.linalg.norm(hist1,ord=1)
    hist=np.reshape(hist1/np.linalg.norm(hist1,ord=1),(1,K))
    return np.reshape(hist1/np.linalg.norm(hist1,ord=1),(1,K))


def get_feature_from_wordmap_SPM(opts, wordmap):
  
    K = opts.K
    L = opts.L
    shape=np.shape(wordmap)
    weights=[2**(-(L-1)) if l in (0,1) else 2**(l-L) for l in range(L)]
    histograms=[]
    for i,weight in enumerate(weights):
        layer=L-i-1
        sec_H=shape[0]//2**layer
        sec_W=shape[1]//2**layer
        for row in range(2**layer):
            for col in range(2**layer):
                histogram=get_feature_from_wordmap(opts, wordmap[sec_H*row:sec_H*(row+1), sec_W*col:sec_W*(col+1)])
                histograms.append(histogram*weight)
    return np.hstack(histograms)

def get_image_feature(opts, img_path, dictionary):
    img=imageio.imread(f'../data/{img_path}')
    img=img.astype('float')/255
    feature=get_feature_from_wordmap_SPM(opts,visual_words.get_visual_words(opts,img,dictionary))
    return feature

def build_recognition_system(opts, n_worker=1):
    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, "train_files.txt")).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, "train_labels.txt"), np.int32)
    dictionary = np.load(join(out_dir, 'C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\Dictionary\\dictionary.npy'))

    train_files=np.asarray(train_files)
    n_train=train_files.shape[0]
    labels=np.asarray(train_labels)
    features=[]
    for i in range(n_train):
        feature=get_image_feature(opts,train_files[i],dictionary) 
        features.append(feature) 
    features=np.vstack(features)
    np.savez("trained_system.npz",features=features,labels=labels,dictionary=dictionary,SPM_layer_num=SPM_layer_num)

def distance_to_set(word_hist, histograms):
    return np.sum(np.minimum(word_hist,histograms),axis = 1)

def evaluate_recognition_system(opts, n_worker=1):

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, "trained_system.npz"))
    train_labels = trained_system['labels']
    features = trained_system['features']

    dictionary = trained_system["dictionary"]

    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system["SPM_layer_num"]

    test_files = open(join(data_dir, "test_files.txt")).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, "test_labels.txt"), np.int32)

    test_data=np.asarray(test_files).astype(str)
    n_test=len(test_files)
    predict_labels=[]
    for i in range(n_test):
        test_feature=get_image_feature(opts,test_data[i],dictionary)
        predict_labels.append(train_labels[np.argmax(distance_to_set(test_feature,features))])
    miss={}
    for i in range(len(predict_labels)):
        if predict_labels[i]!=test_labels[i]:
            miss[i]=join(opts.data_dir,test_data[i])
    confusion=sklearn.metrics.confusion_matrix(test_labels, predict_labels)
    accuracy=sklearn.metrics.accuracy_score(test_labels, predict_labels)
    return confusion, accuracy, test_labels, predict_labels, miss