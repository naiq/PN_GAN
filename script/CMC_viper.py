import numpy as np
import caffe
import os
import cv2
import pickle
import matplotlib.pyplot as plt

def _cmc_core(D, G, P):
    m, n = D.shape
    order = np.argsort(D, axis=0)
    match = (G[order] == P)
    return (match.sum(axis=1) * 1.0 / n).cumsum()

def cmc(distmat, glabels=None, plabels=None, ds=None, repeat=None):
    m, n = distmat.shape
    if glabels is None and plabels is None:
        glabels = np.arange(0, m)
        plabels = np.arange(0, n)
    if isinstance(glabels, list):
        glabels = np.asarray(glabels)
    if isinstance(plabels, list):
        plabels = np.asarray(plabels)
    ug = np.unique(glabels)
    if ds is None:
        ds = ug.size
    if repeat is None:
        if ds == ug.size and ug.size == len(glabels):
            repeat = 1
        else:
            repeat = 100
    ret = 0
    for __ in xrange(repeat):
        G = np.random.choice(ug, ds, replace=False)
        p_inds = [i for i in xrange(len(plabels)) if plabels[i] in G]
        P = plabels[p_inds]
        D = np.zeros((ds, P.size))
        for i, g in enumerate(G):
            samples = np.where(glabels == g)[0]
            j = np.random.choice(samples)
            D[i, :] = distmat[j, p_inds]
        ret += _cmc_core(D, G, P)
    return ret / repeat

def compute_gallery_probe(data_path, test_file):
    test_idx = []
    for line in open(test_file, 'r').readlines():
        line = line.split()
        test_idx.append(int(line[0]))

    # get sorted files
    query_file = get_sort_file(data_path + 'cam_a/', test_idx)
    test_file = get_sort_file(data_path + 'cam_b/', test_idx)
    probe = query_file
    gallery = test_file

    return gallery, probe

def get_sort_file(path, idx):
    files = os.listdir(path)
    files = [i for i in files if int(i[0:3]) in idx]
    files.sort(key=lambda x:int(x[0:3]))
    return files

if __name__ == '__main__':
    all_metric = ['avg', 'max', 'concat']
    all_res = []
    for m in all_metric:
        pkl_file_path = '../feature/dist_' + m + '.pkl'
        data_path = '/DATACENTER/1/qxl/PRID/VIPeR/' # path to VIPeR dataset, need to be modified
        test_file = 'test_idx.txt'

        # sort query/test
        gallery, probe = compute_gallery_probe(data_path, test_file)

        # load distmat
        pkl_file = open(pkl_file_path, 'rb')
        distmat = pickle.load(pkl_file)

        # compute CMC
        glabels = [i[0:3] for i in gallery]
        plabels = [i[0:3] for i in probe]
        res = cmc(distmat, glabels=glabels, plabels=plabels, ds=None, repeat=None)
        # print res
        # print m
        all_res.append([res[0], res[4], res[9]])

    # print Rank-1, Rank-5, and Rank-10 for each metric learning
    for i, r in enumerate(all_res):
        print all_metric[i], r

