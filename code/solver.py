'''In this file, we write an unsupervised domain adaptation algorithm to decrease the error of estimating
the foot placements of target subject'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
from torch.autograd import Variable
from model.model import Generator, Classifier, DomainPredictor
from datasets.dataset_read import dataset_read, load_data
from tqdm import tqdm

device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def classification(args, leave_one_idx = -1):
    torch.cuda.empty_cache()
    model_name = '{}_{}_target_subject_{}'.format(args.dataset, args.method_name, leave_one_idx)
    solver = Solver(args, leave_one_idx=leave_one_idx, model_name=model_name)
    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    acc_s_vec = np.zeros(args.max_epoch)
    acc_t_vec = np.zeros(args.max_epoch)
    loss_mean_vec = np.zeros(args.max_epoch)
    loss_var_vec = np.zeros(args.max_epoch)
    loss_d_kl_vec = np.zeros(args.max_epoch)

    if not args.eval_only:
        for t in tqdm(range(args.max_epoch)):
            solver.train(t)
            _, acc_t_vec[t], feat_t = solver.test(is_target=True)
            _, acc_s_vec[t], feat_s = solver.test(is_target=False)
            _, acc_s_train, _ = solver.test(is_test=False)
            loss_mean_vec[t] = calc_mean_mse_loss(feat_s, feat_t)
            loss_var_vec[t] = calc_var_mse_loss(feat_s, feat_t)
            loss_d_kl_vec[t] = calc_GFA_divergence_loss(feat_s, feat_t)

            solver.save_model()
            print('Source training acc: {:.1f}%. Source test acc: {:.1f}%. Target test acc: {:.1f}%. '
                  'loss_mean: {:.4f}, loss_var: {:.4f}, loss_d_kl: {:.4f}'.format(
                100 * acc_s_train, 100 * acc_s_vec[t], 100 * acc_t_vec[t],
                loss_mean_vec[t], loss_var_vec[t], loss_d_kl_vec[t]))

    loss_dict = {'acc_s': acc_s_vec,
                 'acc_t': acc_t_vec,
                 'loss_mean': loss_mean_vec,
                 'loss_var': loss_var_vec,
                 'loss_d_kl': loss_d_kl_vec}
    solver.calc_forward_time()
    return solver.test_ensemble(), loss_dict, solver.predict()


def Solver(args, leave_one_idx=0, model_name=''):
    method_name = args.method_name
    print('Model {}'.format(method_name))
    return eval(method_name)(args, leave_one_idx, model_name)


class MCD(object):
    def __init__(self, args, leave_one_idx=0, model_name='', num_G=1, num_C=2, num_D = 0):
        self.args = args
        self.batch_size = args.batch_size
        self.num_k = 2
        self.checkpoint_dir = args.checkpoint_dir
        self.lr = args.lr

        self.num_G = num_G  # number of generators
        self.num_C = num_C  # number of regressors
        self.num_D = num_D  # number of domain classifier
        self.leave_one_idx = leave_one_idx
        self.model_name = model_name

        '''Load dataset'''
        self.X_s_train, self.X_s_test, self.y_s_train, self.y_s_test, \
        self.X_t_train, self.X_t_test, self.y_t_train, self.y_t_test = \
            load_data(leave_one_idx=leave_one_idx, dataset_name=self.args.dataset, sensor_idx=self.args.sensor_idx)
        self.data_train, self.data_test = \
            dataset_read(batch_size=args.batch_size, leave_one_idx=leave_one_idx, dataset_name=self.args.dataset,
                         sensor_idx=self.args.sensor_idx)

        self.net_dict = self.init_model()
        if args.eval_only:
            self.eval_model()
            self.load_model()
        else:
            self.set_optimizer(which_opt=args.optimizer, lr=args.lr)

    def init_model(self):
        G_list = []
        for i in range(self.num_G):
            G_list.append(Generator(dataset=self.args.dataset, sensor_idx=self.args.sensor_idx).to(device))
        C_list = []
        for i in range(self.num_C):
            C_list.append(Classifier(self.args.dataset, sensor_idx=self.args.sensor_idx, n_c=self.args.n_c).to(device))
        D_list = []
        for i in range(self.num_C):
            D_list.append(DomainPredictor(self.args.dataset).to(device))
        return {'G': G_list, 'C': C_list, 'D': D_list}

    def save_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                torch.save(self.net_dict[key][i], '{}/{}_{}_{}.pt'.format(
                    self.args.checkpoint_dir, self.model_name, key, i))

    def load_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                file_name = '{}/{}_{}_{}.pt'.format(
                    self.args.checkpoint_dir, self.model_name, key, i)
                if 'cpu' in device_name:
                    self.net_dict[key][i] = torch.load(file_name,
                        map_location=lambda storage, loc: storage)
                else:
                    self.net_dict[key][i] = torch.load(file_name)

        print('Loaded file: {}'.format(file_name))


    def train_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].train()

    def eval_model(self):
        for key in self.net_dict.keys():
            for i in range(len(self.net_dict[key])):
                self.net_dict[key][i].eval()

    def step_model(self, keys):
        for key in keys:
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].step()
        self.reset_grad()

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        self.opt_dict = {}
        for key in self.net_dict.keys():
            self.opt_dict.update({key: []})
            for i in range(len(self.net_dict[key])):
                if which_opt == 'momentum':
                    opt = optim.SGD(self.net_dict[key][i].parameters(),
                                    lr=lr,
                                    momentum=momentum)
                elif which_opt == 'adam':
                    opt = optim.Adam(self.net_dict[key][i].parameters(),
                                     lr=lr)
                else:
                    raise Exception("Unrecognized optimization method.")
                self.opt_dict[key].append(opt)

    def reset_grad(self):
        for key in self.opt_dict.keys():
            for i in range(len(self.opt_dict[key])):
                self.opt_dict[key][i].zero_grad()

    def calc_smoother_loss(self, X_s, X_t, output_s, output_t):
        X = torch.cat([X_s, X_t], dim=0)
        output = torch.cat([output_s, output_t], dim=0)  # (2 * batch_size, output_length)
        _, output_noise_list = self.calc_output_list(X * (1 - 5e-2 + Variable(1e-1 * torch.rand(X.shape).float().to(device))))
        return mse(output, output_noise_list[0])

    def calc_output_list(self, X):
        # return parallel_calc_output_list(self.net_dict['G'], self.net_dict['C'], img)
        feat_list = [None for _ in range(self.num_G)]
        output_list = [None for _ in range(self.num_C)]
        num_C_for_G = int(self.num_C / self.num_G)
        for r in range(len(self.net_dict['G'])):
            feat_list[r] = self.net_dict['G'][r](X)
        for r in range(len(self.net_dict['G'])):
            for c in range(num_C_for_G):
                output_list[r * num_C_for_G + c] = self.net_dict['C'][r * num_C_for_G + c](feat_list[r])
        return feat_list, output_list

    def calc_mean_output(self, X):
        _, output_list = self.calc_output_list(X)
        output = torch.stack(output_list, dim=0)
        output_mean = torch.mean(output, dim=0)
        return output_mean

    def voting_output(self, X):
        feat_list, output_list = self.calc_output_list(X)
        output_vec = torch.stack(output_list, dim=0) #(ensemble_num, batch_size, class_num)
        pred_ensemble = output_vec.data.max(dim=-1)[1] #(ensemble_num, batch_size)
        pred_ensemble = torch.mode(pred_ensemble, dim=0)[0] #(batch_size)
        return pred_ensemble, feat_list

    def calc_classifier_discrepancy_loss(self, output_list):
        loss = 0.0
        num_C_for_G = int(self.num_C/self.num_G)
        for r in range(self.num_G):
            mean_output_t = torch.mean(
                torch.stack(output_list[num_C_for_G * r:num_C_for_G * (r + 1)]), dim=0)
            for c in range(num_C_for_G):
                loss += discrepancy(output_list[num_C_for_G * r + c], mean_output_t)
        return loss

    def train_DA(self, X_s, X_t, y_s, t):
        # 1: Minimize the classification error of source data
        feat_s_list, output_s_list = self.calc_output_list(X_s)
        loss = calc_source_loss(output_s_list, y_s)
        loss.backward()
        self.step_model(['G', 'C'])
        # 2: Maximize the discrepancy of classifiers
        _, output_s_list = self.calc_output_list(X_s)
        _, output_t_list = self.calc_output_list(X_t)
        loss = calc_source_loss(output_s_list, y_s) - self.calc_classifier_discrepancy_loss(output_t_list)
        loss.backward()
        self.step_model(['C'])

        # 3: Minimize the discrepancy of classifiers by training feature extractor
        for i in range(self.num_k):
            feat_t_list, output_t_list = self.calc_output_list(X_t)
            loss = self.calc_classifier_discrepancy_loss(output_t_list)
            loss.backward()
            self.step_model(['G'])

    def train(self, t):
        self.train_model()
        torch.cuda.manual_seed(1)
        for batch_idx, data in enumerate(self.data_train):
            X_t = data['T'].float()
            X_s = data['S'].float()
            y_s = data['S_label'].long()
            X_s = Variable(X_s.to(device))
            X_t = Variable(X_t.to(device))
            y_s = Variable(y_s.to(device))
            self.reset_grad()
            self.train_DA(X_s, X_t, y_s, t)

    def train_target(self):
        self.train_model()
        torch.cuda.manual_seed(1)
        for batch_idx, data in enumerate(self.data_train):
            X_t = data['T'].float()
            y_t = data['T_label'].float()
            X_t = Variable(X_t.to(device))
            y_t = Variable(y_t.to(device))

            self.reset_grad()

            _, output_t_list = self.calc_output_list(X_t)
            loss = self.calc_source_loss(output_t_list, y_t)
            loss.backward()
            self.step_model(['G', 'C'])

    def test_ensemble(self):
        self.load_model()
        y_s_test_predict, acc_s, _ = self.test(is_target=False)
        # print('Final source prediction acc: {:.1f}%'.format(100 * acc_s))
        y_t_test_predict, acc_t, _ = self.test(is_target=True)
        # print('Final target prediction acc: {:.1f}%'.format(100 * acc_t))
        return acc_s, acc_t

    def predict(self, is_target = True, is_test = True):
        if is_test:
            dataset = self.data_test
        else:
            dataset = self.data_train
        y_predict_list = []
        y_list = []
        X_list = []
        self.eval_model()
        for batch_idx, data in enumerate(dataset):
            X_t = data['T'].float()
            X_s = data['S'].float()
            y_t = data['T_label'].long()
            y_s = data['S_label'].long()
            X_s = Variable(X_s.to(device))
            X_t = Variable(X_t.to(device))
            if is_target:
                X = X_t
                y = y_t
            else:
                X = X_s
                y = y_s
            print(y_t.shape, y_s.shape, y.shape)
            y_predict_i = self.calc_mean_output(X) # probability distribution: (batch, class num)
            y_predict_list.append(y_predict_i.data.cpu().numpy())
            X_list.append(X.data.cpu().numpy())
            y_list.append(y)
            if len(y) < self.args.batch_size:
                break
        y_predict = np.concatenate(y_predict_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        X = np.concatenate(X_list, axis=0)

        predict_result = {'y_predict': y_predict, 'y': y, 'X': X}
        return predict_result


    def test(self, is_target = True, is_test = True):
        if is_test:
            dataset = self.data_test
        else:
            dataset = self.data_train
        y_predict_list = []
        y_list = []
        feat_list = []
        self.eval_model()
        for batch_idx, data in enumerate(dataset):
            X_t = data['T'].float()
            X_s = data['S'].float()
            y_t = data['T_label'].long()
            y_s = data['S_label'].long()
            X_s = Variable(X_s.to(device))
            X_t = Variable(X_t.to(device))
            if is_target:
                X = X_t
                y = y_t
            else:
                X = X_s
                y = y_s

            y_predict_i, feat_vec = self.voting_output(X=X)
            feat_list.append(feat_vec[0])
            y_predict_list.append(y_predict_i.data.cpu().numpy())
            y_list.append(y)
        y_predict = np.concatenate(y_predict_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        feat = torch.cat(feat_list, dim=0)
        acc = np.mean(y_predict == y)
        return y_predict, acc, feat

    def calc_forward_time(self):
        X = self.X_t_test[[0]]
        y = self.y_t_test[[0]]
        time_start = time.time()
        X = Variable(torch.from_numpy(X).float().to(device))
        y_predict, feat_list = self.voting_output(X=X)
        y_predict = y_predict.data.cpu().numpy()
        acc = np.mean(y_predict == y)
        print('Acc: {}, Forward time: {:.2f} ms'.format(acc, 1e3 * (time.time() - time_start)))



    def extract_features(self, idx_G = 0):
        self.eval_model()

        X_s = self.X_s_train
        y_s = self.y_s_train.squeeze()
        domain_s = np.zeros(y_s.shape)
        X_t = self.X_t_train
        y_t = self.y_t_train.squeeze()
        domain_t = np.ones(y_t.shape)

        X_np = np.vstack([X_s, X_t])
        X = Variable(torch.from_numpy(X_np).float().to(device))

        feature_vec = self.net_dict['G'][idx_G](X)
        y = np.concatenate([y_s, y_t])
        domain_vec = np.concatenate([domain_s, domain_t])
        input_feature_vec = X_np.reshape((len(y), -1))
        return input_feature_vec, feature_vec.cpu().detach().numpy(), y, domain_vec


class GFA(MCD):
    def __init__(self, args, leave_one_idx=0, model_name='', num_G=1, num_R=1, num_D=0):
        MCD.__init__(self, args, leave_one_idx, model_name, num_G, num_R, num_D)
        if 'DSADS' in self.args.dataset:
            self.k_u = 10
            self.k_v = 0.1
            self.k_d = 1
        elif 'UTD' in self.args.dataset:
            self.k_u = 1
            self.k_v = 0.01
            self.k_d = 0.1
        else:
            self.k_u = 5
            self.k_v = 0.05
            self.k_d = 0.1
        if 'UTD' in self.args.dataset:
            self.k_e = 0.1
        else:
            self.k_e = 0
        self.epoch_s = self.args.epoch_s
        self.var_val = 1

    def train_DA(self, X_s, X_t, y_s, t):
        # 1: Minimize the classification error of source data
        if 'DSADS' in self.args.dataset or 'ENABL3S' in self.args.dataset:
            X_s = X_s*(1-1e-1+Variable(2e-1*torch.rand(X_s.shape).float().to(device)))
            X_t = X_t*(1-1e-1+Variable(2e-1*torch.rand(X_t.shape).float().to(device)))

        feat_s_list, output_s_list = self.calc_output_list(X_s)
        feat_t_list, output_t_list = self.calc_output_list(X_t)

        feat_s = feat_s_list[0]
        feat_t = feat_t_list[0]
        loss = calc_source_loss(output_s_list, y_s)
        if t == self.epoch_s:
            self.var_val = torch.mean(torch.var(feat_s, dim=0, keepdim=True)).data.cpu()
        if t > self.epoch_s:
            '''
                k = min(sigma_t^2, sigma_s^2)/(u_s - u_t)^2
            '''
            k = (min(torch.mean(torch.var(feat_s, dim=0)), torch.mean(torch.var(feat_t, dim=0))) \
                 / torch.mean(torch.square(torch.mean(feat_s, dim=0) - torch.mean(feat_t, dim=0)))).data.cpu().item()
            if self.k_u ** 2 * k < self.args.k_v:
                self.k_v = self.k_u ** 2 * k
                print('r_v: {}'.format(self.k_v))
            loss += self.k_u * calc_mean_mse_loss(feat_s, feat_t) \
                    + self.k_v * calc_const_var_mse_loss(feat_s, feat_t, self.var_val) \
                    + self.k_d * calc_GFA_divergence_loss(feat_s, feat_t) + self.k_e * calc_output_ent(output_t_list)
        loss.backward()
        self.step_model(['G', 'C'])


def mse(out1, out2, dim=-1):
    '''
    :param out1: (batch_size, feature_length)
    :param out2: (batch_size, feature_length)
    :return:
    '''
    return torch.mean(torch.sum(torch.square(out1-out2), dim=dim))

def np_mse(out1, out2, dim=-1):
    return np.mean(np.sum(np.square(out1 - out2), axis=dim))


def calc_source_loss(output_s_list, y_s):
    criterion = nn.CrossEntropyLoss().to(device)
    loss = 0.0
    for output_s in output_s_list:
        loss += criterion(output_s, y_s)
    return loss


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2, dim=-1)))


def calc_feature_discrepency_loss(feat_list):
    criterion_consistency = nn.L1Loss().to(device)
    mean_feat = torch.mean(torch.stack(feat_list), dim=0)
    loss = 0.0
    for feat_s in feat_list:
        loss += criterion_consistency(feat_s, mean_feat)
    return loss


def ent(out_t):
    out_t = F.softmax(out_t)
    loss_ent = -torch.mean(torch.sum(out_t * (torch.log(out_t + 1e-5)), dim=-1))
    return loss_ent

def calc_output_ent(output_t_list):
    loss = 0.0
    for output_t in output_t_list:
        loss += ent(output_t)
    return loss

def calc_mean_mse_loss(feat_s, feat_t, is_tensor = False):
    return mse(torch.mean(feat_s, dim=0), torch.mean(feat_t, dim=0))

def calc_var_mse_loss(feat_s, feat_t):
    return mse(torch.mean(torch.var(feat_s, dim=0)), torch.mean(torch.var(feat_t, dim=0)))

def calc_const_var_mse_loss(feat_s, feat_t, var_val):
    return mse(torch.mean(torch.var(feat_s, dim=0)), np.squeeze(var_val)) +\
           mse(torch.mean(torch.var(feat_t, dim=0)), np.squeeze(var_val))

def generate_Gaussian_distribution(feature_mat):
    mu = torch.mean(feature_mat, dim=0).data.cpu()
    sigma = torch.std(feature_mat, dim=0).data.cpu()
    z = np.random.normal(mu, sigma, feature_mat.size())
    zn = Variable(torch.cuda.FloatTensor(z))
    return zn

def calc_GFA_divergence_loss(feat_s, feat_t):
    '''
                    A batch of samples of each feature channel should follow a Gaussian distribution.
                    Output the KL divergence of each element and calculate the mean of all elements.
                    F.kl_div(log(a), b) = mean(b * log(b/a))
                    zn_s[:, 0] ~ N(torch.mean(feat_s_list[0], dim=0)[0], torch.std(feat_s_list[0], dim=0)[0])
                '''
    zn_s = generate_Gaussian_distribution(feat_s)
    zn_t = generate_Gaussian_distribution(feat_t)
    zn_s = torch.sort(zn_s, dim=0)[0]
    zn_t = torch.sort(zn_t, dim=0)[0]
    feat_s = torch.sort(feat_s, dim=0)[0]
    feat_t = torch.sort(feat_t, dim=0)[0]
    loss_kld_s = calc_d_kl(zn_s, feat_s)
    loss_kld_t = calc_d_kl(zn_t, feat_t)
    return loss_kld_s + loss_kld_t


def calc_d_kl(x, y):
    x = F.softmax(x, dim=0)
    y = F.softmax(y, dim=0)
    return torch.mean(torch.sum(y*(torch.log(y)-torch.log(x)), dim=0))
