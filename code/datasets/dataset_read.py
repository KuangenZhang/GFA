import sys
import numpy as np
import pickle
import torch
sys.path.append('../loader')
from datasets.unaligned_data_loader import UnalignedDataLoader


def dataset_read(batch_size, leave_one_idx = 0, dataset_name ='DSADS', sensor_idx = 0):
    S_train = {}
    T_train = {}

    S_test = {}
    T_test = {}

    X_s_train, X_s_test, y_s_train, y_s_test, \
    X_t_train, X_t_test, y_t_train, y_t_test = load_data(leave_one_idx=leave_one_idx, dataset_name=dataset_name,
                                                         sensor_idx=sensor_idx)

    S_train['imgs'] = X_s_train
    S_train['labels'] = y_s_train
    T_train['imgs'] = X_t_train
    T_train['labels'] = y_t_train

    S_test['imgs'] = X_s_test
    S_test['labels'] = y_s_test
    T_test['imgs'] = X_t_test
    T_test['labels'] = y_t_test

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S_train, T_train, batch_size, batch_size)
    data_train = train_loader.load_data()

    test_loader = UnalignedDataLoader()
    test_loader.initialize(S_test, T_test, batch_size, batch_size)
    data_test = test_loader.load_data()
    # print('Target validation shape: {}'.format(T_val['labels'].shape[0] + T_test['labels'].shape[0]))
    return data_train, data_test


def load_data(leave_one_idx = 0, dataset_name = 'DSADS', sensor_idx = 0):
    if 'UTD' in dataset_name:
        dataset = np.load('data/{}.npy'.format(dataset_name), allow_pickle=True)
        return dataset['X_s_train'], dataset['X_s_test'], dataset['y_s_train'], dataset['y_s_test'], \
               dataset['X_t_train'], dataset['X_t_test'], dataset['y_t_train'], dataset['y_t_test']
    else:
        dataset = np.load('data/{}.npy'.format(dataset_name), allow_pickle=True).item()
        X_train, X_test, y_train, y_test = dataset['X_train'], dataset['X_test'], dataset['y_train'], dataset['y_test']
        subject_indices = set(range(X_train.shape[0]))
        leave_r_out_indices = list(subject_indices - set([leave_one_idx]))
        is_feature = 'feature' in dataset_name
        X_s_train, X_t_train = split_source_and_target(X_train, leave_r_out_indices, leave_one_idx, is_feature=is_feature)
        X_s_test, X_t_test = split_source_and_target(X_test, leave_r_out_indices, leave_one_idx, is_feature=is_feature)
        if is_feature:
            feature_indices = sensor_idx_to_feature_indices(dataset_name, sensor_idx)
            X_s_train = X_s_train[..., feature_indices]
            X_t_train = X_t_train[..., feature_indices]
            X_s_test = X_s_test[..., feature_indices]
            X_t_test = X_t_test[..., feature_indices]
        y_s_train, y_t_train = split_source_and_target(y_train, leave_r_out_indices, leave_one_idx, is_feature=is_feature)
        y_s_test, y_t_test = split_source_and_target(y_test, leave_r_out_indices, leave_one_idx, is_feature=is_feature)
        print(X_s_train.shape, X_s_test.shape, y_s_train.shape, y_s_test.shape,
              X_t_train.shape, X_t_test.shape, y_t_train.shape, y_t_test.shape)
        return X_s_train, X_s_test, y_s_train, y_s_test, \
               X_t_train, X_t_test, y_t_train, y_t_test


def split_source_and_target(X, leave_r_out_indices, leave_one_idx, is_feature = True):
    X_s = np.concatenate(X[leave_r_out_indices], axis=0) # (data_num, time_steps, sensor_num)
    X_t = X[leave_one_idx]
    if is_feature:
        X_s = X_s.reshape((X_s.shape[0], -1)).squeeze()
        X_t = X_t.reshape((X_t.shape[0], -1)).squeeze()
    return X_s, X_t


def sensor_idx_to_feature_indices(dataset_name ='DSADS_feature', sensor_idx = 0):
    if 'ENABL3S' in dataset_name:
        '''
        sensor idx: [0: All, 1: IMU, 2: Angle, 3: EMG, 4: IMU + Angle, 5: IMU+EMG, 6: Angle+EMG]
        '''
        imu_indices = np.arange(0, 180)
        angle_indices = np.arange(180, 228)
        emg_indices = np.arange(228, 368)
        feature_indices = [np.r_[imu_indices, angle_indices, emg_indices],
                          imu_indices,
                          angle_indices,
                          emg_indices,
                          np.r_[imu_indices, angle_indices],
                          np.r_[imu_indices, emg_indices],
                          np.r_[angle_indices, emg_indices]]
        return feature_indices[sensor_idx]
    else:
        '''
        sensor idx: [0: All, 1: Remove torso, 2: Remove right arm, 3: Remove left arm, 4: Remove right leg, 5: Remove left leg]
        '''
        feature_indices = np.arange(0, 45*6)
        if 0 == sensor_idx:
            return feature_indices
        else:
            return np.delete(feature_indices, feature_indices[(sensor_idx-1)*54:sensor_idx*54])

