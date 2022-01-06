
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 10:30:35 2019

@author: kuangen
"""
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import mat4py as m4p
from random import shuffle
import pandas as pd
import copy 
import glob

def load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Raw/', X_dim = 4, 
                   is_one_hot = False, is_normalized = False, 
                   is_resize = False, leave_one_num = -1, sub_num = 8, 
                   feature_length = 5625, sensor_num = 0):
    
    idx_vec = list(range(sub_num))
    if -1 == leave_one_num:
        shuffle(idx_vec)
        idx_train = idx_vec[:5]
        idx_test = idx_vec[5:-1]
    else:
        idx_test = [copy.deepcopy(idx_vec[leave_one_num])]
        idx_vec.pop(leave_one_num)
        idx_train = idx_vec
    
    # dataset: 
    # x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
    # x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
    dataset = []
    for i in range(6):
        dataset.append(np.array([], dtype=np.float32).reshape(0,feature_length))
        dataset.append(np.array([], dtype=np.float32).reshape(0))
    for idx in idx_train:
        data_read = load_one_AB_mat(data_path, idx = idx)
        for j in range(6):
            dataset[j] = np.concatenate((dataset[j], data_read[j]), axis = 0)
    for idx in idx_test:
        data_read = load_one_AB_mat(data_path, idx = idx)
        for j in range(6):
            dataset[j + 6] = np.concatenate((dataset[j + 6], data_read[j]), axis = 0)
    for i in range(6):
        if is_resize:
            dataset[2 * i] = dataset[2 * i].reshape((-1, 
                   45, int(feature_length / 45)))
        if 0 != sensor_num:
            dataset[2 * i] = dataset[2 * i][:,9*(sensor_num-1):9 * sensor_num,:]
        if is_normalized:
            dataset[2 * i] = preprocessing.scale(dataset[2 * i])
        for j in range(X_dim - len(dataset[2 * i].shape)):
            dataset[2 * i] = np.expand_dims(dataset[2 * i], axis = 1)
        if is_one_hot:
            dataset[2 * i + 1] = one_hot(dataset[2 * i + 1], 
                      n_classes = int(1 + np.max(dataset[2 * i + 1])))
        
    return tuple(dataset)

def save_UCI_DSADS(x_mat, y_mat, file_path = 'data/1_dataset_UCI_DSADS/Raw/', 
                   val_size = 0.15, test_size = 0.15):
    x_shape = x_mat.shape
    x_mat = x_mat.reshape((x_shape[0], x_shape[1], x_shape[2], -1))
    seg_num = x_shape[0]*x_shape[2]
    idx_vec = np.arange(seg_num)
    for s in range(8):
        np.random.shuffle(idx_vec)
        x_mat_s = x_mat[:,s,:,:].reshape((seg_num, -1))
        y_mat_s = y_mat[:,s,:].reshape(seg_num)
        x_train, x_test, y_train, y_test = train_test_split(
                x_mat_s[idx_vec,:], y_mat_s[idx_vec], test_size= (val_size + test_size))
        x_test, x_val, y_test, y_val = train_test_split(
                x_test, y_test, test_size= test_size / (val_size + test_size))
        data = {'x_train': x_train.tolist(), 'x_val': x_val.tolist(), 'x_test': x_test.tolist(), 
            'y_train': y_train.tolist(), 'y_val': y_val.tolist(), 'y_test': y_test.tolist()
            };
        m4p.savemat(file_path + str(s) + '.mat', data)

def read_UCI_DSADS(file_path = 'data/1_dataset_UCI_DSADS/data/'):
    x_mat = np.zeros((19, 8, 60, 125, 45))
    y_mat = np.zeros((19, 8, 60))
    for y in range(19):
        y_str = 'a%02d' % (y+1)
        for s in range(8):
            s_str = 'p%d' % (s+1)
            file_name_list = glob.glob(file_path + y_str + '/' + s_str + '/' 
                                       + '*.txt')
            for f in range(len(file_name_list)):
                x_mat[y, s, f, :, :] = np.loadtxt(file_name_list[f],delimiter = ',')
                y_mat[y, s, f] = y
    return x_mat, y_mat

def one_hot(y, n_classes=7):
    # Function to encode neural one-hot output labels from number indexes 
    # e.g.: 
    # one_hot(y=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y = y.reshape(len(y))
    return np.eye(n_classes)[np.array(y, dtype=np.int32)]  # Returns FLOATS

def read_csv_one_hot(data_path, idx_x, val_size = 0.15, test_size = 0.15):
    mydata = genfromtxt(data_path, delimiter=',')
    X = mydata[1:,idx_x]
    X = np.expand_dims(X, axis = -1)
    X = np.expand_dims(X, axis = -1)
    y = np.floor_divide(mydata[1:,-1] % 100 , 10)
    y = one_hot(y, n_classes = 1 + np.max(y))
    # 70 % to training, and 15 % each to test and val sets
    x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size= (val_size + test_size))
    x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size= test_size / (val_size + test_size))
    return x_train, y_train, x_val, y_val, x_test, y_test

def read_csv_data(data_path, idx_x = None):
    if idx_x is None:
        idx_x = np.arange(0, 368)
    mydata = genfromtxt(data_path, delimiter=',')
    y = np.floor_divide(mydata[1:, -1] % 100, 10)
    y_pre = np.floor_divide(mydata[1:, -1], 1000)
    X = mydata[1:, idx_x]
    data = {'X': X, 'y_pre': y_pre, 'y': y}
    return data


def read_csv(data_path, idx_x, X_dim = 4, val_size = 0.15, test_size = 0.15, 
             is_walking = False):
    mydata = genfromtxt(data_path, delimiter=',')
    y = np.floor_divide(mydata[1:,-1] % 100 , 10)
    X = mydata[1:,idx_x]
    if is_walking:
        for k in [0, 6]:
            X = X[y != k,:]
            y = y[y != k]
        y = y - 1
    for i in range(X_dim - len(X.shape)):
        X = np.expand_dims(X, axis = -1)
    # 70 % to training, and 15 % each to test and val sets
    x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size= (val_size + test_size))
    x_test, x_val, y_test, y_val = train_test_split(
            x_test, y_test, test_size= test_size / (val_size + test_size))
    return x_train, y_train, x_val, y_val, x_test, y_test

def calc_thigh_idx():
    idx_x = []
    for i in range(6):
        idx_IMU_1 = np.arange(7, 13, dtype=int) * ((i+1)) - 1
        idx_IMU_2 = np.arange(19, 31, dtype=int) * ((i+1)) - 1
        idx_x = np.concatenate((idx_x, idx_IMU_1, idx_IMU_2), axis = -1)
    for i in range(20):
        idx_EMG = np.arange(4, 8, dtype=int) * ((i+1)) + 228 - 1
        idx_x = np.concatenate((idx_x, idx_EMG), axis = -1)
    
    return idx_x.astype(np.int32)

def save_mat(data_name, is_walking = False):
    idx_x = np.arange(0,368)
    data_s_path = '0_dataset\AB156\Features\AB156_Features_300.csv'
    x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test = \
    read_csv(data_s_path,idx_x, X_dim = 2, is_walking = is_walking)
    # target domain
    data_t_path = '0_dataset\AB186\Features\AB186_Features_300.csv'
    x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
    read_csv(data_t_path,idx_x, X_dim = 2, is_walking = is_walking)
    data = {'x_s_train': x_s_train.tolist(), 'x_s_val': x_s_val.tolist(), 'x_s_test': x_s_test.tolist(), 
            'y_s_train': y_s_train.tolist(), 'y_s_val': y_s_val.tolist(), 'y_s_test': y_s_test.tolist(),
            'x_t_train': x_t_train.tolist(), 'x_t_val': x_t_val.tolist(), 'x_t_test': x_t_test.tolist(),
            'y_t_train': y_t_train.tolist(), 'y_t_val': y_t_val.tolist(), 'y_t_test': y_t_test.tolist()};
    m4p.savemat(data_name, data)
    
def save_one_AB_mat(data_name, data_path, is_walking = False):
    idx_x = np.arange(0,368)
    x_train, y_train, x_val, y_val, x_test, y_test = \
    read_csv(data_path,idx_x, X_dim = 2, is_walking = is_walking)
    data = {'x_train': x_train.tolist(), 'x_val': x_val.tolist(), 'x_test': x_test.tolist(), 
            'y_train': y_train.tolist(), 'y_val': y_val.tolist(), 'y_test': y_test.tolist()
            };
    m4p.savemat(data_name, data)
    
def save_all_AB_mat():
    num_AB =['AB156',
             'AB185',
             'AB186',
             'AB188',
             'AB189',
             'AB190',
             'AB191',
             'AB192',
             'AB193',
             'AB194'
             ]
    for i in range(10):
        save_one_AB_mat('0_dataset/AB_dataset/AB_' + str(i) + '.mat', 
                        data_path = '0_dataset/' + num_AB[i] + '/Features/' + num_AB[i] + '_Features_300.csv',
                        is_walking = False)

def save_resize_idx_mat():
    data_path = 'data\\0_dataset\\AB156\\Features\\AB156_Features_300.csv'
    names = pd.read_csv(data_path, nrows=1).columns.tolist()
    leg_names = ['Ipsi ', 'Contra ']
    sensor_names = ['Ankle', 'TA', 'MG', 'SOL', 'Shank', 'Knee', 'BF', 
                   'ST', 'VL', 'RF', 'Thigh']
    EMG_names = ['TA', 'MG', 'SOL', 'BF', 'ST', 'VL', 'RF']
    IMU_names = ['Shank','Thigh', 'Waist']
    
    indices_mat = np.zeros((0, 12), dtype=int)
    for r in range(2):
        for c in range(11):
            if 0 == r:
                sensor_name = sensor_names[c]
            else:
                sensor_name = sensor_names[10 - c]
            indices = np.array([i for i, name in enumerate(names)
                                if leg_names[r] + sensor_name in name]).reshape((1, -1))
        
            if sensor_name in EMG_names:
                indices = np.concatenate((indices,-1 * np.ones((1, 2), dtype=int)), 
                                         axis = -1) 
            if sensor_name in IMU_names:
                indices = indices.reshape((3, -1))
            indices_mat = np.concatenate((indices_mat, indices), axis = 0)
        if 0 == r:
            sensor_name = 'Waist'
            indices = np.array([i for i, name in enumerate(names) 
            if sensor_name in name]).reshape((1, -1))
            indices = indices.reshape((3, -1))      
            indices_mat = np.concatenate((indices_mat, indices), axis = 0)


    data = {'idx_mat': indices_mat.tolist()};
    m4p.savemat('data\\AB_dataset\\AB_idx.mat', data)

def resize_feature(x, data_path):
    data = m4p.loadmat(data_path + 'idx.mat')
    idx_mat = np.array(data['idx_mat']) + 1
    zero_vec = np.zeros((len(x),1))
    x_concat = np.concatenate((zero_vec,x), axis = 1)
    return x_concat[:,idx_mat]

   

def load_st_AB_mat(data_path = '0_dataset/AB_dataset/AB_', X_dim = 4, 
                   is_one_hot = False, is_normalized = False, 
                   is_resize = False, leave_one_num = -1,
                   sensor_num = 0):
    
    idx_vec = list(range(10))
    if -1 == leave_one_num:
        shuffle(idx_vec)
        idx_train = idx_vec[:5]
        idx_test = idx_vec[5:-1]
    else:
        idx_test = [copy.deepcopy(idx_vec[leave_one_num])]
        idx_vec.pop(leave_one_num)
        idx_train = idx_vec
        
        
    # dataset: 
    # x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
    # x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
    dataset = []
    for i in range(6):
        dataset.append(np.array([], dtype=np.float32).reshape(0,368))
        dataset.append(np.array([], dtype=np.float32).reshape(0))
    for idx in idx_train:
        data_read = load_one_AB_mat(data_path, idx = idx)
        for j in range(6):
            dataset[j] = np.concatenate((dataset[j], data_read[j]), axis = 0)
#    for i in range(len(idx_test)):
#        data_read = load_one_AB_mat(data_path, idx = idx_test[i])
    for idx in idx_test:
        data_read = load_one_AB_mat(data_path, idx = idx)
        for j in range(6):
            dataset[j + 6] = np.concatenate((dataset[j + 6], data_read[j]), axis = 0)
    for i in range(6):
        if is_resize:
            dataset[2 * i] = resize_feature(dataset[2 * i], data_path)
        if 0 != sensor_num:
            emg_idx = np.r_[np.arange(1,4), np.arange(8,12), np.arange(21,25),
                            np.arange(29,32)]
            imu_idx = np.r_[np.arange(4,7), np.arange(12,21), np.arange(26,29)]
            angle_idx = np.r_[0, 7, 25, 32]
            sensor_idx = [emg_idx, imu_idx, angle_idx, np.r_[emg_idx, imu_idx],
                          np.r_[emg_idx, angle_idx], np.r_[imu_idx, angle_idx]]
            dataset[2 * i] = dataset[2 * i][:,sensor_idx[sensor_num-1],:]
        if is_normalized:
            dataset[2 * i] = preprocessing.scale(dataset[2 * i])
        for j in range(X_dim - len(dataset[2 * i].shape)):
            dataset[2 * i] = np.expand_dims(dataset[2 * i], axis = 1)
        if is_one_hot:
            dataset[2 * i + 1] = one_hot(dataset[2 * i + 1], 
                      n_classes = int(1 + np.max(dataset[2 * i + 1])))
        
    return tuple(dataset)

def load_one_AB_mat(data_path = '0_dataset/AB_dataset/AB_', idx = 0):
    data = m4p.loadmat(data_path + str(idx) + '.mat')
    return [np.array(data['x_train']), np.array(data['y_train']),
           np.array(data['x_val']), np.array(data['y_val']),
           np.array(data['x_test']), np.array(data[ 'y_test'])]

def load_mat(data_name, X_dim = 4, is_one_hot = True, is_normalized = False):
    data = m4p.loadmat(data_name)
    data_array =   [np.array(data['x_s_train']), np.array(data['y_s_train']),\
           np.array(data['x_s_val']), np.array(data['y_s_val']),\
           np.array(data['x_s_test']), np.array(data['y_s_test']),\
           np.array(data['x_t_train']), np.array(data['y_t_train']),\
           np.array(data['x_t_val']), np.array(data['y_t_val']),\
           np.array(data['x_t_test']), np.array(data['y_t_test'])] 
    for i in range(6):
        if is_normalized:
            data_array[2 * i] = preprocessing.scale(data_array[2 * i])
        for j in range(X_dim - len(data_array[2 * i].shape)):
            data_array[2 * i] = np.expand_dims(data_array[2 * i], axis = -1)
        if is_one_hot:
            data_array[2 * i + 1] = one_hot(data_array[2 * i + 1], 
                      n_classes = int(1 + np.max(data_array[2 * i + 1])))
    return tuple(data_array)


if __name__ == '__main__':
    # x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
    # x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
    #     load_UCI_mat(data_path='data/DSADS/Raw/',
    #                  feature_length=125 * 45, is_resize=True, leave_one_num=1)
    # print(x_s_train.shape, y_s_train.shape)
    save_all_AB_mat()