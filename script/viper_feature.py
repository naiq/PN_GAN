import numpy as np
import caffe
import os, random
import cv2
import pickle
import scipy.io as sio

def get_sort_file(path, idx):
    files = os.listdir(path)
    files = [i for i in files if int(i[0:3]) in idx]
    files.sort(key=lambda x:int(x[0:3]))
    return files

def Calculate_query_test_ID(query_dir, test_dir, test_file, feature_dir):
    test_idx = []
    for line in open(test_file, 'r').readlines():
        line = line.split()
        test_idx.append(int(line[0]))

    # get sorted files
    query_file = get_sort_file(query_dir, test_idx)
    test_file = get_sort_file(test_dir, test_idx)
    query = query_file
    test = test_file

    testID = [int(i.split('_')[0]) for i in test]
    sio.savemat(feature_dir+'testID.mat', {'testID':testID})
    queryID = [int(i.split('_')[0]) for i in query]
    sio.savemat(feature_dir+'queryID.mat', {'queryID':queryID})
    testCAM = [2 for i in test] # the images from camb are labeld as CAM 2
    sio.savemat(feature_dir+'testCAM.mat', {'testCAM':testCAM})
    queryCAM = [1 for i in query] # the images from camb are labeld as CAM 1
    sio.savemat(feature_dir+'queryCAM.mat', {'queryCAM':queryCAM})

    return query, test

def Calculate_8_pose_feature(im, dir, gen_dir, net_p, flag):
    for i in range(1,9):
        if flag:
            p = cv2.imread(gen_dir + 'c2_' + im.split('.')[0]+'_to_'+str(i)+'.png').astype(np.float32)
        else:
            p = cv2.imread(gen_dir + 'c1_' + im.split('.')[0]+'_to_'+str(i)+'.png').astype(np.float32)
        for c, mean in enumerate([103.939, 116.779, 123.68]):
            p[:, :, c] -= mean
        p_resize = cv2.resize(p, (112,224)).astype(np.float32)
        net_p.blobs['data'].data[i-1] = p_resize.transpose(2,0,1)
    out_p = net_p.forward()['fea']
    out_p = out_p / np.sqrt((np.sum(np.square(out_p), axis=1))).reshape(out_p.shape[0],1)

    return out_p

def Calculate_single_feature(images, output, dir, file_format, generate_dir, model_p, deploy, gpu_id, flag):
    output_p = output + '_p' + file_format
    if not os.path.isfile(output_p):
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        net_p = caffe.Net(deploy, model_p, caffe.TEST)
        net_p.blobs['data'].reshape(8,3,224,112)

        N = len(images)
        features_p = np.zeros((8,1024, N))
        for i, im in enumerate(images):
            out_p = Calculate_8_pose_feature(im, dir, generate_dir, net_p, flag)
            features_p[:,:,i] = out_p
            print str(i+1) + '/' + str(N), im
        if file_format == '.mat':
            sio.savemat(output_p, {'fea':features_p})
        elif file_format == '.pkl':
            save_file = open(output_p, 'wb')
            pickle.dump(features_p, save_file)
            save_file.close()
    else: # only for .pkl file
        pkl_file = open(output_p, 'rb')
        features_p = pickle.load(pkl_file)
    return features_p

def Calculate_distmat(dist_out, file_format, query_p_feature, test_p_feature, query, test):

    f_i_avg = np.mean(test_p_feature, axis=0)
    f_j_avg = np.mean(query_p_feature, axis=0)
    f_i_max = np.max(test_p_feature, axis=0)
    f_j_max = np.max(query_p_feature, axis=0)
    f_i_concat = test_p_feature.reshape(1024*8, len(test))
    f_j_concat = query_p_feature.reshape(1024*8, len(query))

    distmat_avg = feature_distance_euclidean(f_i_avg, f_j_avg)
    distmat_max = feature_distance_euclidean(f_i_max, f_j_max)
    distmat_concat = feature_distance_euclidean(f_i_concat, f_j_concat)

    all_dist = [distmat_avg, distmat_max, distmat_concat]
    all_metric = ['_avg', '_max', '_concat']

    if file_format == '.mat':
        for i, m in enumerate(all_metric):
            sio.savemat(dist_out + m + file_format, {'dist':all_dist[i]})
    elif file_format == '.pkl':
        for i, m in  enumerate(all_metric):
            save_file = open(dist_out + m + file_format, 'wb')
            pickle.dump(all_dist[i], save_file)
            save_file.close()

def feature_distance_euclidean(probe, query):
    Np = probe.shape[1]
    Nq = query.shape[1]
    pmag = np.sum(np.square(probe), axis=0)
    qmag = np.sum(np.square(query), axis=0)
    distmat = np.tile(qmag, [Np,1]) + np.tile(pmag.reshape(Np,1), [1, Nq]) - 2 * np.dot(probe.T, query)
    return distmat

if __name__ == '__main__':

    # caffe config
    model_p = '../model/model.caffemodel'
    deploy = '../model/deploy.prototxt'
    gpu_id = 3

    # data path
    test_file = 'test_idx.txt'
    feature_dir = '../feature/'
    generate_dir = '../dataset/'
    query_dir = '/DATACENTER/1/qxl/PRID/VIPeR/cam_a/' # path to cam_a of VIPeR, need to be modified
    test_dir = '/DATACENTER/1/qxl/PRID/VIPeR/cam_b/' # path to cam_b of VIPeR, need to be modified

    # config
    file_format = '.pkl' # optional: .mat or .pkl
    dist_out = feature_dir + 'dist'
    query_out = feature_dir + 'Hist_query'
    test_out = feature_dir + 'Hist_test'

    # calcuate query/test ID and CAM
    query, test = Calculate_query_test_ID(query_dir, test_dir, test_file, feature_dir)

    # calculate query features
    Hist_query_p = Calculate_single_feature(query, query_out, query_dir, file_format, generate_dir, model_p, deploy, gpu_id, 0)
    print 'Finish calculating single query features...'

    # calculate test features
    Hist_test_p = Calculate_single_feature(test, test_out, test_dir, file_format, generate_dir, model_p, deploy, gpu_id, 1)
    print 'Finish calculating single test features...'

    # calculate distmat
    distmat = Calculate_distmat(dist_out, '.mat', Hist_query_p, Hist_test_p, query, test)
    distmat = Calculate_distmat(dist_out, '.pkl', Hist_query_p, Hist_test_p, query, test)

    print len(query)
    print len(test)

