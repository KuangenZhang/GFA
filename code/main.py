'''In this file, we can prepare the data, change the hyperparameters, and change the index of the leaved subject'''
from __future__ import print_function
import argparse
import torch
import os
import numpy as np
import zipfile
import utils.utils as utils
from solver import Solver
from solver import classification


def convert_args_to_bool(args):
    args.eval_only = (args.eval_only in ['True', True])
    return args


def save_features(args, leave_one_idx = 1):
    torch.cuda.empty_cache()
    model_name = '{}_{}_target_subject_{}'.format(
        args.dataset, args.method_name, leave_one_idx)
    print(model_name)
    solver = Solver(args, leave_one_idx=leave_one_idx, model_name=model_name)

    input_features, features, labels, domains = solver.extract_features(idx_G=0)
    print(input_features.shape, features.shape, labels.shape, domains.shape)

    labels = labels.astype(np.int)
    domains = domains.astype(np.int)

    tsne_features = {'input_features': input_features, 'features': features,
                     'input_tsne': utils.features_to_tnse(input_features), 'feature_tsne': utils.features_to_tnse(features),
                     'labels': labels, 'domains': domains}
    result_data_dir = '{}/tsne'.format(args.result_dir)
    if not os.path.exists(result_data_dir):
        os.mkdir(result_data_dir)
    np.save('{}/tsne_features_{}.npy'.format(result_data_dir, model_name), tsne_features)


def main_eval(args):
    print(args)
    if 'DSADS' in args.dataset:
        sub_id = 3
    else:
        sub_id = 4
    # for i in range(sub_num):
    for i in [sub_id]:
        print('Test ', i)
        save_features(args, leave_one_idx=i)

def main(args):
    print(args)
    model_name = '{}_{}'.format(args.dataset, args.method_name)
    if 'DSADS' in args.dataset:
        test_num = 8
    elif 'UTD' in args.dataset:
        test_num = 1
    else:
        test_num = 10
    acc_s = np.zeros(test_num)
    acc_t = np.zeros(test_num)
    predict_result_vec = np.zeros(test_num, dtype=np.object)
    loss_dict_all = {'acc_s': np.zeros((int(args.max_epoch), test_num)),
                     'acc_t': np.zeros((int(args.max_epoch), test_num)),
                 'loss_mean': np.zeros((int(args.max_epoch), test_num)),
                 'loss_var':  np.zeros((int(args.max_epoch), test_num)),
                 'loss_d_kl': np.zeros((int(args.max_epoch), test_num))}
    for i in range(test_num):
        (acc_s[i], acc_t[i]), loss_dict, predict_result_vec[i] = classification(args, leave_one_idx=i)
        update_loss_dict(loss_dict_all, loss_dict, i)
    if not args.eval_only:
        if not os.path.exists(args.result_dir):
            os.mkdir(args.result_dir)
        if not os.path.exists('{}/loss'.format(args.result_dir)):
            os.mkdir('{}/loss'.format(args.result_dir))
        np.savetxt("{}/loss/final_acc_{}.csv".format(args.result_dir, model_name),
                   np.transpose(np.c_[acc_s, acc_t]), delimiter=",")
        np.save("{}/loss/loss_dict_all_{}.npy".format(args.result_dir, model_name), loss_dict_all)
    source_result_text = '{}: Mean of test acc in the source domain: {:.1f}%'.format(model_name, 100 * np.mean(acc_s))
    target_result_text = '{}: Mean of test acc in the target domain: {:.1f}%'.format(model_name, 100 * np.mean(acc_t))
    print(source_result_text)
    print(target_result_text)
    return source_result_text, target_result_text

def update_loss_dict(loss_dict_all, loss_dict, i):
    for key in loss_dict_all.keys():
        loss_dict_all[key][:, i] = loss_dict[key]

def extract_checkpoint_and_data():
    if not os.path.exists('checkpoint') and not os.path.exists('data'):
        print('Extract GFA_checkpoint_and_data.\n')
        with zipfile.ZipFile("GFA_checkpoint_and_data.zip", "r") as zip_ref:
            zip_ref.extractall('')

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--result_dir', type=str, default=r'results', metavar='N')
    parser.add_argument('--checkpoint_dir', type=str, default=r'checkpoint', metavar='N')
    parser.add_argument('--dataset', default='DSADS_feature',
                        help='dataset, including DSADS_feature, ENABL3S_feature, UTD')
    parser.add_argument('--sensor_idx', default=0)
    parser.add_argument('--method_name', default='GFA',
                        help='check the method name')
    parser.add_argument('--eval_only', default=True,
                        help='evaluation only option')
    parser.add_argument('--epoch_s', type=int, default=30)
    parser.add_argument('--k_u', type=float, default=10)
    parser.add_argument('--k_v', type=float, default=0.1)
    parser.add_argument('--k_d', type=float, default=1)
    parser.add_argument('--n_c', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate (default: 0.0002)')
    parser.add_argument('--max_epoch', type=int, default=100, metavar='N',
                        help='how many epochs')
    parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                        help='random seed (default: 10)')

    args = parser.parse_args()
    args = convert_args_to_bool(args)

    '---------------Extract data and checkpoint -----------------'
    extract_checkpoint_and_data()
    '------------------------------------------------------------'

    if torch.cuda.is_available():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if 'UTD' in args.dataset:
        args.epoch_s = 50
        args.max_epoch = 150
    source_result_text_list = []
    target_result_text_list = []
    for dataset_name in ['DSADS_feature', 'ENABL3S_feature', 'UTD']:
        args.dataset = dataset_name
        source_result_text, target_result_text = main(args)
        source_result_text_list.append(source_result_text)
        target_result_text_list.append(target_result_text)
    for i in range(len(source_result_text_list)):
        print(source_result_text_list[i])
        print(target_result_text_list[i])

