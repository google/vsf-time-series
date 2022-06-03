# Copyright 2022 Google LLC
# Copyright (c) 2020 Zonghan Wu

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

""" Primary utilities """
import pickle
import numpy as np
import os
import math
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import networkx as nx
import time


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class DataLoaderM(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.num_nodes = xs.shape[2]
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj = load_pickle(pkl_filename)
    return adj



class StandardScaler():
    """
    Standard the input
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def transform(self, data):
        return (data - self.mean) / self.std
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def load_dataset(args, dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    total_num_nodes = None
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

        print("Shape of ", category, " input = ", data['x_' + category].shape)

        total_num_nodes = data['x_' + category].shape[2]
        data['total_num_nodes'] = total_num_nodes

    if args.predefined_S:
        count = math.ceil(total_num_nodes * (args.predefined_S_frac / 100))
        oracle_idxs = np.random.choice( np.arange(total_num_nodes), size=count, replace=False )
        data['oracle_idxs'] = oracle_idxs
        for category in ['train', 'val', 'test']:
            data['x_' + category] = data['x_' + category][:, :, oracle_idxs, :]
            data['y_' + category] = data['y_' + category][:, :, oracle_idxs, :]

    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])

    data['train_loader'] = DataLoaderM(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoaderM(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoaderM(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data



def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)

def masked_rmse(preds, labels, null_val=np.nan):
    mse_loss, per_instance = masked_mse(preds=preds, labels=labels, null_val=null_val)
    return torch.sqrt(mse_loss), torch.sqrt(per_instance)


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss), torch.mean(loss, dim=1)


def metric(pred, real):
    mae, mae_per_instance = masked_mae(pred,real,0.0)[0].item(), masked_mae(pred,real,0.0)[1]
    rmse, rmse_per_instance = masked_rmse(pred,real,0.0)[0].item(), masked_rmse(pred,real,0.0)[1]
    return mae, rmse, mae_per_instance, rmse_per_instance



def get_node_random_idx_split(args, num_nodes, lb, ub):
    count_percent = np.random.choice( np.arange(lb, ub+1), size=1, replace=False )[0]
    count = math.ceil(num_nodes * (count_percent / 100))

    current_node_idxs = np.random.choice( np.arange(num_nodes), size=count, replace=False )
    return current_node_idxs


def zero_out_remaining_input(testx, idx_current_nodes, device):
    zero_val_mask = torch.ones_like(testx).bool()#.to(device)
    zero_val_mask[:, :, idx_current_nodes, :] = False
    inps = testx.masked_fill_(zero_val_mask, 0.0)
    return inps



def obtain_instance_prototypes(args, x_train):
    stride = x_train.shape[0] // args.num_prots
    prototypes = []
    for i in range(args.num_prots):
        a = x_train[ i*stride : (i+1)*stride ]
        prot = a[ np.random.randint(0, a.shape[0]) ] # randint will give a single interger here
        prot = np.expand_dims(prot, axis=0)
        prototypes.append(prot)

    prototypes = np.concatenate(tuple(prototypes), axis=0)
    print("\nShape of instance prototypes = ", prototypes.shape, "\n")
    prototypes = torch.FloatTensor(prototypes).to(args.device)
    return prototypes


def obtain_relevant_data_from_prototypes(args, testx, instance_prototypes, idx_current_nodes):
    rem_idx_subset = torch.LongTensor(np.setdiff1d(np.arange(args.num_nodes), idx_current_nodes)).to(args.device)
    idx_current_nodes = torch.LongTensor(idx_current_nodes).to(args.device)

    data_idx_train = instance_prototypes[:, :, idx_current_nodes, :].unsqueeze(0).repeat(testx.shape[0], 1, 1, 1, 1)
    a = testx[:, :, idx_current_nodes, :].transpose(3, 1).unsqueeze(1).repeat(1, instance_prototypes.shape[0], 1, 1, 1)
    assert data_idx_train.shape == a.shape

    raw_diff = data_idx_train - a
    diff = torch.pow( torch.absolute( raw_diff ), args.dist_exp_value ).view(testx.shape[0], instance_prototypes.shape[0], -1)
    diff = torch.mean(diff, dim=-1)

    min_values, topk_idxs = torch.topk(diff, args.num_neighbors_borrow, dim=-1, largest=False)
    b_size = testx.shape[0]
    original_instance = testx.clone()
    testx = testx.repeat(args.num_neighbors_borrow, 1, 1, 1)
    orig_neighs = []

    for j in range(args.num_neighbors_borrow):
        nbs = instance_prototypes[topk_idxs[:, j].view(-1)].transpose(3, 1)
        orig_neighs.append( nbs )
        desired_vals = nbs[:, :, rem_idx_subset, :]
        start, end = j*b_size, (j+1)*b_size
        _local = testx[start:end]
        _local[:, :, rem_idx_subset, :] = desired_vals
        testx[start:end] = _local

    orig_neighs = torch.cat(orig_neighs, dim=0)
    return testx, min_values, orig_neighs, topk_idxs, original_instance



def load_node_feature(path):
    fi = open(path)
    x = []
    for li in fi:
        li = li.strip()
        li = li.split(",")
        e = [float(t) for t in li[1:]]
        x.append(e)
    x = np.array(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    z = torch.tensor((x-mean)/std,dtype=torch.float)
    return z


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


def obtain_discrepancy_from_neighs(preds, orig_neighs_forecasts, args, idx_current_nodes):
    orig_neighs_forecasts = orig_neighs_forecasts.transpose(1, 3)
    orig_neighs_forecasts = orig_neighs_forecasts[:, 0, :, :]
    orig_neighs_forecasts = orig_neighs_forecasts[:, idx_current_nodes, :]

    orig_neighs_forecasts = torch.chunk(orig_neighs_forecasts, args.num_neighbors_borrow)
    orig_neighs_forecasts = [ a.unsqueeze(1) for a in orig_neighs_forecasts ]
    orig_neighs_forecasts = torch.cat(orig_neighs_forecasts, dim=1)

    len_tensor = torch.FloatTensor( np.arange(1, preds.shape[-1]+1) ).to(args.device).view(1, 1, 1, -1).repeat(
                              preds.shape[0], args.num_neighbors_borrow, preds.shape[2], 1) # tensor of time step indexes
    distance = torch.absolute( (preds - orig_neighs_forecasts) / len_tensor ).view(preds.shape[0], args.num_neighbors_borrow, -1)
    distance = torch.mean(distance, dim=-1)
    return distance, orig_neighs_forecasts
