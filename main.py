#!/usr/bin/python
# -*- coding: utf8 -*-
"""
Main function of FedGmTE-Net framework
for jointly predicting multiple trajectories from a single input graph at baseline using federation.
---------------------------------------------------------------------

  FedGmTE_Net(input_t0_clients, M_tn_loaders_clients, F_tn_loaders_clients, num_clients, num_fold, opts, num_samples_per_client)
          Inputs:
                  input_t0_clients:  for each client represents the data acquired at t0 which are the input to the network
                            ==> it is a PyTorch dataloader returning elements from source dataset batch by batch
                  M_tn_loaders_clients:  for each client PyTorch dataloader representing the modality 'M' (i.e., low-resolution)
                            acquired at multiple timepoints.
                  F_tn_loaders_clients:  for each client PyTorch dataloaders representing the modality 'F' (i.e., super-resolution)
                            acquired at multiple timepoints.
                  num_clients: total number of clients for federation
                  num_fold: current fold number (cross-validation used)
                  opts:         a python object (parser) storing all arguments needed to run the code such as hyper-parameters
                  number_samples_per_client: number of samples each client has
          Output:
                  model:        our FedGmTE-Net model

Sample use for training:
  model = FedGmTE_Net(input_t0_clients, M_tn_loaders_clients, F_tn_loaders_clients, num_clients, num_fold, opts, num_samples_per_client)
  model.train()

Sample use for testing:
  model = FedGmTE_Net(input_t0, M_tn_loaders, F_tn_loaders, num_clients, num_fold, opts)
  metrics_LR_clients, metrics_SR_clients = model.test()
          Output:
                  metrics_LR_clients: All evaluation metrics for the LR modality for each client
                  metrics_SR_clients: All evaluation metrics for the SR modality for each client

---------------------------------------------------------------------
Please cite the above paper if you use this code.
All rights reserved.
"""
import argparse
import yaml
import numpy as np
from torch.backends import cudnn
from data_loader import *
from utils import *
from plotting import *
from prediction import FedGmTE_Net
from dataset import prepare_data, complete_dataset

parser = argparse.ArgumentParser()
# Initialisation
# Basic opts.
parser.add_argument('--nb_timepoints', type=int, default=3, help='how many timepoint we have in a trajectory')
parser.add_argument('--gen_log_dir', type=str, default='logs/')
parser.add_argument('--gen_checkpoint_dir', type=str, default='models/')
parser.add_argument('--gen_result_dir', type=str, default='results/')
parser.add_argument('--result_root', type=str, default='result')
parser.add_argument('--gen_plot_dir', type=str, default='plots/')
parser.add_argument('--lr_dim', type=int, default=35, help='low resolution matrix dimension')
parser.add_argument('--sr_dim', type=int, default=116, help='super resolution matrix dimension')
# GCN model opts
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--hidden1', type=int, default=100)
parser.add_argument('--hidden2', type=int, default=50)
parser.add_argument('--hidden3', type=int, default=16)
parser.add_argument('--LRout', type=int, default=595)
parser.add_argument('--SRout', type=int, default=6670)
# Training opts.
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
parser.add_argument('--num_workers', type=int, default=0, help='num_workers to load data.')
parser.add_argument('--num_iters', type=int, default=200, help='number of total iterations for training')
parser.add_argument('--log_step', type=int, default=5)
parser.add_argument('--early_stop', type=bool, default=True, help='use early stop or not')
parser.add_argument('--patience', type=float, default=10, help="early stop patience")
parser.add_argument('--n_folds', type=int, default=4, help="number of clients in federation learning")
parser.add_argument('--val_ratio', type=float, default=0.4, help="validation set ratio used for early stop")
parser.add_argument('--tp_coef', type=float, default=0.001, help="KL Loss Coefficient")
# Test opts.
parser.add_argument('--test_iters', type=int, default=200, help='test model from this step')
# Federation
parser.add_argument('--num_local_iters', type=list, default=[5, 5, 5], help="local iterations for federation for each client")
parser.add_argument('--num_global_iters', type=int, default=40, help="global iterations for federation")
parser.add_argument('--federate', type=bool, default=True, help="use federation")
# Auxiliary Regulariser
parser.add_argument('--use_aux_reg', type=bool, default=False, help="use auxiliary regularizer during training")
parser.add_argument('--reg_strength', type=float, default=0.01, help="auxiliary regularizer strength")

opts = parser.parse_args()

### Modes:
# 0: NoFedGmTE-Net
# 1: FedGmTE-Net
# 2: FedGmTE-Net*

### Evaluation metrics:
# mae: MAE (graph)
# ns: MAE (NS)

### Data types:
# simulate_multi: simulated dataset

modes_dict = {0:"nofed", 1:"fed", 2:"fed+"}
eval_metrics = ["mae", "ns"]
data_type = "simulate_multi"
iid = True
def main(mode):
    opts.complete_ratio = 0.6
    opts.tp_coef = 0.001
    opts.metrics = eval_metrics

    if mode == 0:
        opts.federate = False
        opts.use_aux_reg = False
    elif mode == 1:
        opts.federate = True
        opts.use_aux_reg = False
    elif mode == 2:
        opts.federate = True
        opts.use_aux_reg = True
        opts.reg_strength = 0.5

    # Early stop
    if not opts.early_stop:
        opts.val_ratio = 0
    else:
        opts.val_ratio = 0.4

    extension = f"/{iid}/{data_type}/{modes_dict[mode]}/"

    opts.log_dir = os.path.join(opts.result_root + extension, opts.gen_log_dir)
    opts.checkpoint_dir = os.path.join(opts.result_root + extension, opts.gen_checkpoint_dir)
    opts.result_dir = os.path.join(opts.result_root + extension, opts.gen_result_dir)
    opts.plot_dir = os.path.join(opts.result_root + extension,opts.gen_plot_dir)

    # For fast training.
    cudnn.benchmark = True

    if torch.cuda.is_available():
        print("Running on GPU")
    else:
        print("Running on CPU")

    # Load data
    data_lr, data_sr = prepare_data(data_type=data_type)

    # Vectorize
    vec_data_lr = []
    for _, sample in enumerate(data_lr):
        vec_sample = []
        for _, t_sample in enumerate(sample):
            vec = vectorize(t_sample)
            vec_sample.append(vec)
        vec_data_lr.append(vec_sample)
    vec_data_lr = np.array(vec_data_lr)

    vec_data_sr = []
    for _, sample in enumerate(data_sr):
        vec_sample = []
        for _, t_sample in enumerate(sample):
            vec = vectorize(t_sample)
            vec_sample.append(vec)
        vec_data_sr.append(vec_sample)
    vec_data_sr = np.array(vec_data_sr)

    # Create directories if not exist.
    create_dirs_if_not_exist([opts.log_dir, opts.checkpoint_dir, opts.result_dir, opts.plot_dir])

    # log opts.
    with open(os.path.join(opts.result_root, 'opts.yaml'), 'w') as f:
        f.write(yaml.dump(vars(opts)))

        metrics_LR_folds = []
        metrics_SR_folds = []

        LR_losses_folds = []
        SR_losses_folds = []
        total_losses_folds = []

        preds_LR = []
        preds_SR = []

        # Cross Validation
        for num_fold in range(opts.n_folds):
            # Train test split
            torch.cuda.empty_cache()
            print(f"********* FOLD {num_fold} *********")
            train_lr, test_lr = get_nfold_split(vec_data_lr, number_of_folds=opts.n_folds, current_fold_id=num_fold)
            train_sr, test_sr = get_nfold_split(vec_data_sr, number_of_folds=opts.n_folds, current_fold_id=num_fold)

            num_clients = opts.n_folds - 1

            # Train a model for each fold
            print('============================')
            print(f"Train with {modes_dict[mode]}")
            print('============================')
            input_t0_clients = []
            M_tn_loaders_clients = []
            F_tn_loaders_clients = []

            # Randomly discard samples
            sample_availability_table = random_table(train_lr.shape[0], opts.nb_timepoints, ratio=opts.complete_ratio)
            # Discard unavailable samples
            for i in range(len(sample_availability_table)):
                for t in range(opts.nb_timepoints):
                    if sample_availability_table[i][t] == 0:
                        train_lr[i][t] = None
                        train_sr[i][t] = None

            len_client_data = train_lr.shape[0] // num_clients

            print("Number of samples: ",train_lr.shape[0])

            # Non iid (based on t=0)
            if not iid:
                labels = kmeans(train_lr[:, 0, :], num_clients)

            num_samples_per_client = np.zeros(num_clients)

            for k in range(num_clients):
                if iid:
                    # Uniform split
                    train_lr_client = train_lr[len_client_data*k:len_client_data*(k+1)]
                    train_sr_client = train_sr[len_client_data*k:len_client_data*(k+1)]
                else:
                    # Non iid split
                    train_lr_client = train_lr[labels == k]
                    train_sr_client = train_sr[labels == k]

                # Number of samples
                num_samples_per_client[k] = len(train_lr_client)

                # Complete missing data
                train_lr_client = complete_dataset(train_lr_client, n_time=opts.nb_timepoints)
                train_sr_client = complete_dataset(train_sr_client, n_time=opts.nb_timepoints)

                #----READ MODALITY 1 AT T0
                data_t0 = train_lr_client[:, 0, :]
                input_t0 = get_loader(data_t0, data_t0.shape[0], opts.num_workers)

                #----READ MULTI-TRAJECTORY DATA FROM T1 to TN
                M_tn_loaders = []
                F_tn_loaders = []
                for timepoint in range(0, opts.nb_timepoints):
                    M_data_tn = train_lr_client[:, timepoint, :]
                    F_data_tn = train_sr_client[:, timepoint, :]

                    M_tn_loader = get_loader(M_data_tn, M_data_tn.shape[0], opts.num_workers)
                    F_tn_loader = get_loader(F_data_tn, F_data_tn.shape[0], opts.num_workers)

                    M_tn_loaders.append(M_tn_loader)
                    F_tn_loaders.append(F_tn_loader)

                input_t0_clients.append(input_t0)
                M_tn_loaders_clients.append(M_tn_loaders)
                F_tn_loaders_clients.append(F_tn_loaders)

            model = FedGmTE_Net(input_t0_clients, M_tn_loaders_clients, F_tn_loaders_clients, num_clients, num_fold, opts, num_samples_per_client=num_samples_per_client)
            LR_losses_clients, SR_losses_clients, total_losses_clients = model.train()
            LR_losses_folds.append(LR_losses_clients)
            SR_losses_folds.append(SR_losses_clients)
            total_losses_folds.append(total_losses_clients)

            # Test models
            print('============================')
            print("Test")
            print('============================')
            input_t0_clients = []
            M_tn_loaders_clients = []
            F_tn_loaders_clients = []

            #----READ MODALITY 1 AT T0
            data_t0 = test_lr[:, 0, :]
            input_t0 = get_loader(data_t0, data_t0.shape[0], opts.num_workers)

            #----READ MULTI-TRAJECTORY DATA FROM T1 to TN
            M_tn_loaders = []
            F_tn_loaders = []
            for timepoint in range(0, opts.nb_timepoints):
                M_data_tn = test_lr[:, timepoint, :]
                F_data_tn = test_sr[:, timepoint, :]

                M_tn_loader = get_loader(M_data_tn, M_data_tn.shape[0], opts.num_workers)
                F_tn_loader = get_loader(F_data_tn, F_data_tn.shape[0], opts.num_workers)

                M_tn_loaders.append(M_tn_loader)
                F_tn_loaders.append(F_tn_loader)

            input_t0_clients.append(input_t0)
            M_tn_loaders_clients.append(M_tn_loaders)
            F_tn_loaders_clients.append(F_tn_loaders)

            input_t0_clients *= num_clients
            M_tn_loaders_clients *= num_clients
            F_tn_loaders_clients *= num_clients

            model = FedGmTE_Net(input_t0_clients, M_tn_loaders_clients, F_tn_loaders_clients, num_clients, num_fold, opts)

            metrics_LR_clients, metrics_SR_clients = model.test()
            metrics_LR_folds.append(metrics_LR_clients)
            metrics_SR_folds.append(metrics_SR_clients)

            # Predicted trajectories
            pred_LR, pred_SR, _, _ = model.forward()
            preds_LR.append(pred_LR)
            preds_SR.append(pred_SR)

            if num_fold == 0:
                predicted_trajectory_LR_clients, predicted_trajectory_SR_clients, real_trajectory_LR_clients, real_trajectory_SR_clients = model.forward()
                # Predictions - client 0, sample 0
                num_sample = 2
                for t in range(opts.nb_timepoints):
                    # real LR
                    save_path = os.path.join(opts.result_dir, f'LR real - t_{t}')
                    plot_cbt(antiVectorize(real_trajectory_LR_clients[0][t][num_sample], opts.lr_dim), t, save_path, vmin=0, vmax=max(real_trajectory_LR_clients[0][t][num_sample]))

                    # predicted LR
                    save_path = os.path.join(opts.result_dir, f'LR prediction - t_{t}')
                    plot_cbt(antiVectorize(predicted_trajectory_LR_clients[0][t][num_sample], opts.lr_dim), t, save_path, vmin=0, vmax=max(real_trajectory_LR_clients[0][t][num_sample]))

                    # real SR
                    save_path = os.path.join(opts.result_dir, f'SR real - t_{t}')
                    plot_cbt(antiVectorize(real_trajectory_SR_clients[0][t][num_sample], opts.sr_dim), t, save_path, vmin=0, vmax=max(real_trajectory_SR_clients[0][t][num_sample]))

                    # predicted SR
                    save_path = os.path.join(opts.result_dir, f'SR prediction - t_{t}')
                    plot_cbt(antiVectorize(predicted_trajectory_SR_clients[0][t][num_sample], opts.sr_dim), t, save_path, vmin=0, vmax=max(real_trajectory_SR_clients[0][t][num_sample]))

    delete_dirs_if_exist([opts.log_dir, opts.checkpoint_dir, opts.plot_dir])

    return LR_losses_folds, SR_losses_folds, total_losses_folds, metrics_LR_folds, metrics_SR_folds, preds_LR, preds_SR

if __name__ == '__main__':
    LR_losses_modes = []
    SR_losses_modes = []
    total_losses_modes = []
    metrics_LR_modes = []
    metrics_SR_modes = []

    preds_LR_modes = []
    preds_SR_modes = []

    extension = f"/{iid}/{data_type}/"

    create_dirs_if_not_exist([opts.result_root + extension + "loss"])
    for metric in eval_metrics:
        create_dirs_if_not_exist([opts.result_root + extension + metric])

    for mode in modes_dict.keys():
        LR_losses_folds, SR_losses_folds, total_losses_folds, metrics_LR_folds, metrics_SR_folds, preds_LR, preds_SR = main(mode)
        LR_losses_modes.append(LR_losses_folds)
        SR_losses_modes.append(SR_losses_folds)
        total_losses_modes.append(total_losses_folds)

        metrics_LR_modes.append(metrics_LR_folds)
        metrics_SR_modes.append(metrics_SR_folds)

        preds_LR_modes.append(preds_LR)
        preds_SR_modes.append(preds_SR)

    LR_losses_modes = np.array(LR_losses_modes, dtype=object)
    SR_losses_modes = np.array(SR_losses_modes, dtype=object)
    total_losses_modes = np.array(total_losses_modes, dtype=object)
    metrics_LR_modes = np.array(metrics_LR_modes)
    metrics_SR_modes = np.array(metrics_SR_modes)
    preds_LR_modes = np.array(preds_LR_modes)
    preds_SR_modes = np.array(preds_SR_modes)

    num_clients = opts.n_folds - 1
    # Loss plots
    for i in range(opts.n_folds):
        for k in range(num_clients):
            LR_losses_client = LR_losses_modes[:, i, k]
            SR_losses_client = SR_losses_modes[:, i, k]
            total_losses_client = total_losses_modes[:, i, k]
            save_path = os.path.join(opts.result_root + extension + "loss", 'Fold_{}_Client_{}_Loss'.format(i, k))
            plot_loss(LR_losses_client, SR_losses_client, total_losses_client, list(modes_dict.values()), save_path)

    # Bar charts
    methods = modes_dict.values()
    timepoints = []
    for t in range(opts.nb_timepoints):
        timepoints.append(f"t{t}")
    for k in range(num_clients):
        for i, metric in enumerate(eval_metrics):
            if metric == "t":
                continue
            # MAE
            print('============================')
            print(metric)
            print('============================')
            # LR
            print('============================')
            print("LR")
            print('============================')
            save_path = os.path.join(opts.result_root + extension + metric, 'Client {} {} LR'.format(k, metric))
            plot_mae(metrics_LR_modes[:, :, k, :, i], timepoints, methods, save_path, np.amin(metrics_LR_modes[:, :, :, :, i]))
            # SR
            print('============================')
            print("SR")
            print('============================')
            save_path = os.path.join(opts.result_root + extension + metric, 'Client {} {} SR'.format(k, metric))
            plot_mae(metrics_SR_modes[:, :, k, :, i], timepoints, methods, save_path, np.amin(metrics_SR_modes[:, :, :, :, i]))
            # Average
            print('============================')
            print("Total")
            print('============================')
            save_path = os.path.join(opts.result_root + extension + metric, 'Client {} {} Total'.format(k, metric))
            mean_metrics = (metrics_LR_modes + metrics_SR_modes) / 2
            plot_mae(mean_metrics[:, :, k, :, i], timepoints, methods, save_path, np.amin(mean_metrics[:, :, :, :, i]))
