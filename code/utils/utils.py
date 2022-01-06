import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

'''Process all subject data of DSADS dataset'''
def prepare_DSADS_data():
    x_mat, y_mat = read_UCI_DSADS()
    prepare_leave_one_subject_out_dataset(x_mat, y_mat)

def read_UCI_DSADS(file_path = 'data/DSADS/data/', is_save_data = False):
    if is_save_data:
        '''
        x_mat: (class_num, subject_num, data_num, time_steps, sensor_num)
        y_mat: (class_num, subject_num, data_num)
        '''
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
        np.save('data/DSADS/data.npy', {'x_mat': x_mat, 'y_mat': y_mat})

    data = np.load('data/DSADS/data.npy', allow_pickle=True).item()
    x_mat = np.transpose(data['x_mat'], axes=(1, 0, 2, 3, 4)).reshape((8, -1, 125, 45))
    y_mat = np.transpose(data['y_mat'], axes=(1, 0, 2)).reshape((8, -1))
    print('x_mat size: {}, y_mat size: {}'.format(x_mat.shape, y_mat.shape))
    return x_mat, y_mat

def prepare_leave_one_subject_out_dataset(x_mat, y_mat):
    subject_num = x_mat.shape[0]
    subject_indices = set(range(subject_num))
    for r in range(subject_num):
        leave_r_out_indices = list(subject_indices - set([r]))
        X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
            x_mat[leave_r_out_indices].reshape((-1, x_mat.shape[-2], x_mat.shape[-1])),
            y_mat[leave_r_out_indices].reshape((-1)), test_size=0.3)
        X_t_train, X_t_test, y_t_train, y_t_test = train_test_split(x_mat[r],
                                                                    y_mat[r], test_size=0.3)
        print(X_s_train.shape, X_s_test.shape, y_s_train.shape, y_s_test.shape,
              X_t_train.shape, X_t_test.shape, y_t_train.shape, y_t_test.shape)
        dataset = {'X_s_train':X_s_train, 'X_s_test':X_s_test,
                   'y_s_train':y_s_train, 'y_s_test':y_s_test,
                   'X_t_train':X_t_train, 'X_t_test':X_t_test,
                   'y_t_train':y_t_train, 'y_t_test':y_t_test}
        np.save('data/DSADS/DA/target_{}'.format(r), dataset)


'''Process all subject data of ENABL3S dataset'''
def prepare_ENABL3S_data():
    x_mat, y_mat = read_ENABL3S_data()
    write_ENABL3S_dataset(x_mat, y_mat)

def read_ENABL3S_data(file_path = 'data/ENABL3S/', is_save_data = True):
    if is_save_data:
        '''
            x_mat: (subject_num, data_num, time_steps, sensor_num)
            y_mat: (subject_num, data_num)
        '''
        subject_names = []
        for content in os.listdir(file_path):
            if '.' not in content and 'AB' in content:
                subject_names.append(content)
        subject_names = sorted(subject_names)
        subject_num = len(subject_names)
        sensor_num = 52
        x_mat_all = np.zeros(10, dtype=np.object)
        y_mat_all = np.zeros(10, dtype=np.object)
        for s in range(subject_num):
            x_mat_list = [] #(data_num, time_steps, sensor_num)
            y_mat_list = [] # (data_num)
            file_names = glob.glob('{}/{}/Processed/*.csv'.format(file_path, subject_names[s]))
            for file_name in sorted(file_names):
                print(file_name)
                data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
                x_data = data[:, :sensor_num]
                y_data = data[:,  sensor_num]
                gait_event_indices = data[:, 53:(53+8):2]
                gait_event_indices = np.reshape(gait_event_indices, (-1))
                gait_event_indices = gait_event_indices[~np.isnan(gait_event_indices)].astype(int)
                for gait_event_idx in gait_event_indices:
                    x = x_data[gait_event_idx-1000:gait_event_idx]
                    y = y_data[gait_event_idx]
                    x_mat_list.append(x)
                    y_mat_list.append(y)
            x_mat = np.stack(x_mat_list, axis=0)
            y_mat = np.stack(y_mat_list, axis=0)
            x_mat_all[s] = x_mat
            y_mat_all[s] = y_mat
        np.save('{}/data.npy'.format(file_path), {'x_mat': x_mat_all, 'y_mat': y_mat_all})
        for s in range(subject_num):
            print(x_mat_all[s].shape, y_mat_all[s].shape)
    '''
    load data and split to train and test
    '''
    data = np.load('{}/data.npy'.format(file_path), allow_pickle=True).item()
    x_mat = data['x_mat']
    y_mat = data['y_mat']
    return x_mat, y_mat


def write_ENABL3S_dataset(x_mat, y_mat):
    '''

    Args:
        x_mat: object array (subject_num), each object (data_num, time_steps, sensor_num)
        y_mat: object array (subject_num), each object (data_num)
    Returns:
        dataset: {X_train, y_train, X_test, y_test}
        X_train, X_test: object array (subject_num), each object (data_num, time_steps, sensor_num)
        y_train, y_test: object array (subject_num), each object (data_num)
    '''
    X_train = np.zeros(10, dtype=np.object)
    y_train = np.zeros(10, dtype=np.object)
    X_test = np.zeros(10, dtype=np.object)
    y_test = np.zeros(10, dtype=np.object)
    subject_num = x_mat.shape[0]
    for i in range(subject_num):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(
            x_mat[i], y_mat[i], test_size=0.3)
    np.save('data/ENABL3S.npy', {'X_train': X_train, 'y_train': y_train,
                                 'X_test': X_test, 'y_test': y_test})
    print('Write ENABL3S dataset successfully!')

'''Process all subject data of ENABL3S dataset'''
def prepare_ENABL3S_feature_data():
    x_mat, y_mat = read_ENABL3S_feature_data()
    write_feature_dataset(x_mat, y_mat)

def read_ENABL3S_feature_data(file_path = 'data/ENABL3S/', is_save_data = True):
    if is_save_data:
        '''
            x_mat: (subject_num, data_num, feature_num)
            y_mat: (subject_num, data_num)
        '''
        subject_names = []
        for content in os.listdir(file_path):
            if '.' not in content and 'AB' in content:
                subject_names.append(content)
        subject_names = sorted(subject_names)
        subject_num = len(subject_names)
        x_mat_all = np.zeros(10, dtype=np.object)
        y_mat_all = np.zeros(10, dtype=np.object)
        for s in range(subject_num):
            file_name = glob.glob('{}/{}/Features/*_300.csv'.format(file_path, subject_names[s]))[0]
            print(file_name)
            data = np.genfromtxt(file_name, delimiter=',', skip_header=1)
            x_mat_all[s] = data[:, :-2]
            y_mat_all[s] = np.remainder(np.floor(data[:, -1] / 10), 10).astype(int)  # get the second digit from the right of the number
        np.save('{}/feature.npy'.format(file_path), {'x_mat': x_mat_all, 'y_mat': y_mat_all})
        for s in range(subject_num):
            print(x_mat_all[s].shape, y_mat_all[s].shape)
    '''
    load data and split to train and test
    '''
    data = np.load('{}/feature.npy'.format(file_path), allow_pickle=True).item()
    x_mat = data['x_mat']
    y_mat = data['y_mat']
    return x_mat, y_mat

def write_feature_dataset(x_mat, y_mat, file_name ='ENABL3S_feature'):
    '''

    Args:
        x_mat: object array (subject_num), each object (data_num, feature_num)
        y_mat: object array (subject_num), each object (data_num)
    Returns:
        dataset: {X_train, y_train, X_test, y_test}
        X_train, X_test: object array (subject_num), each object (data_num, feature_num)
        y_train, y_test: object array (subject_num), each object (data_num)
    '''
    subject_num = x_mat.shape[0]
    X_train = np.zeros(subject_num, dtype=np.object)
    y_train = np.zeros(subject_num, dtype=np.object)
    X_test = np.zeros(subject_num, dtype=np.object)
    y_test = np.zeros(subject_num, dtype=np.object)
    for i in range(subject_num):
        X_train[i], X_test[i], y_train[i], y_test[i] = train_test_split(
            x_mat[i], y_mat[i], test_size=0.3)
    np.save('data/{}.npy'.format(file_name),
            {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test})
    print('Write {} dataset successfully!'.format(file_name))


'''Process all subject data of ENABL3S dataset'''
def prepare_DSADS_feature_data():
    x_mat, y_mat = read_UCI_DSADS()
    feature_mat = extract_DSADS_feature(x_mat)
    write_feature_dataset(feature_mat, y_mat, file_name='DSADS_feature')
    # write_ENABL3S_feature_dataset(x_mat, y_mat)


def extract_DSADS_feature(x_mat):
    x_mat = np.transpose(x_mat, (0, 1, -1, -2))
    print('x_mat shape: {}'.format(x_mat.shape))
    feature_mat = np.zeros(x_mat.shape[0:-1]+ (6,))  #(subject_num, data_num, sensor_num, feature_num)
    feature_mat[..., 0] = np.max(x_mat, axis=-1)
    feature_mat[..., 1] = np.min(x_mat, axis=-1)
    feature_mat[..., 2] = np.mean(x_mat, axis=-1)
    feature_mat[..., 3] = np.std(x_mat, axis=-1)
    feature_mat[..., 4] = x_mat[..., 0]
    feature_mat[..., 5] = x_mat[..., -1]
    print(feature_mat.shape)
    return feature_mat



def features_to_tnse(features):
    features = features.squeeze()
    tsne = TSNE(perplexity=40, n_components=2, n_iter=300,
                learning_rate=100).fit_transform(features)
    tsne_min, tsne_max = np.min(tsne, 0), np.max(tsne, 0)
    tsne = (tsne - tsne_min) / (tsne_max - tsne_min)
    return tsne


def main():
    # prepare_DSADS_data()
    prepare_DSADS_feature_data()
    # prepare_ENABL3S_feature_data()
    # prepare_ENABL3S_data()

if __name__ == '__main__':
    main()