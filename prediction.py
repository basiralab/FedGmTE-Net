import os
import time
import datetime
import itertools
import copy
import torch
from sklearn.metrics import mean_absolute_error
from model import *
from metrics import *
from utils import *

class FedGmTE_Net(object):
    """
    Build FedGmTE-Net model for training and testing
    """

    def __init__(self, input_t0_clients, M_tn_loaders_clients, F_tn_loaders_clients, num_clients, num_fold, opts, num_samples_per_client=None):

        self.input_t0_clients = input_t0_clients

        self.M_tn_loaders_clients = M_tn_loaders_clients
        self.F_tn_loaders_clients = F_tn_loaders_clients

        self.num_clients = num_clients
        self.num_samples_per_client = num_samples_per_client
        self.num_fold = num_fold

        self.opts = opts

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Adjacecny matrices for clients
        self.adjs = []
        for k in range(num_clients):
            t0_iter = iter(self.input_t0_clients[k])
            t0_morph_encoder = next(t0_iter)
            t0_M_encoder = t0_morph_encoder[0].to(self.device)

            adj = construct_similarity_adjacency_matrix(t0_M_encoder).to(self.device)

            self.adjs.append(adj)

        # build models
        self.build_model()

    def build_model(self):
        """
        Build encoder and decoder networks and initialize optimizer.
        """
        encoders = []
        decoders_LR = []
        decoders_SR = []

        for _ in range(self.num_clients):
            encoder =  Encoder(self.opts.LRout, self.opts.hidden1, self.opts.hidden2,
                                    self.opts.dropout).to(self.device)

            decoder_LR = Decoder_LR(self.opts.hidden2, self.opts.hidden1, self.opts.LRout,
                                        self.opts.dropout, self.opts.nb_timepoints).to(self.device)

            decoder_SR = Decoder_SR(self.opts.hidden2, self.opts.hidden1,
                                        self.opts.SRout, self.opts.dropout, self.opts.nb_timepoints).to(self.device)

            encoders.append(encoder)
            decoders_LR.append(decoder_LR)
            decoders_SR.append(decoder_SR)

        self.Encoders = encoders
        self.Decoders_LR = decoders_LR
        self.Decoders_SR = decoders_SR

        # build optimizer
        optimizers = []
        for k in range(self.num_clients):
            param_list = [self.Encoders[k].parameters()] + [self.Decoders_LR[k].parameters()] + [self.Decoders_SR[k].parameters()]
            optimizer = torch.optim.Adam(itertools.chain(*param_list),
                                                    self.opts.lr, [self.opts.beta1, self.opts.beta2])
            optimizers.append(optimizer)
        self.optimizers = optimizers

        # shared model parameters
        if self.opts.federate:
            self.global_Encoder = Encoder(self.opts.LRout, self.opts.hidden1, self.opts.hidden2,
                                    self.opts.dropout).to(self.device)

            self.global_Decoder_LR = Decoder_LR(self.opts.hidden2, self.opts.hidden1, self.opts.LRout,
                                        self.opts.dropout, self.opts.nb_timepoints).to(self.device)

            self.global_Decoder_SR = Decoder_SR(self.opts.hidden2, self.opts.hidden1,
                                        self.opts.SRout, self.opts.dropout, self.opts.nb_timepoints).to(self.device)

            self.set_global_parameters()

    # Get flat parameters
    def get_flat_params(self, model):
        par_flat = [p.view(-1) for p in model.parameters()]
        par_flat = torch.cat(par_flat, dim = 0)
        return par_flat

    # Used for federation
    def set_global_parameters(self):
        for k in range(self.num_clients):
            # # Update models
            self.Encoders[k].load_state_dict(self.global_Encoder.state_dict())
            self.Decoders_LR[k].load_state_dict(self.global_Decoder_LR.state_dict())
            self.Decoders_SR[k].load_state_dict(self.global_Decoder_SR.state_dict())

    def aggregate_model_params(self, models, h=None):
        target_state_dict = copy.deepcopy(models[0].state_dict())
        total_clients_samples = np.sum(self.num_samples_per_client)

        model_state_dict_list = [copy.deepcopy(models[k].state_dict()) for k in range(1, self.num_clients)]

        for key in target_state_dict:
            if target_state_dict[key].data.dtype == torch.float32:
                target_state_dict[key].data = (self.num_samples_per_client[0] / total_clients_samples) * target_state_dict[key].data.clone()
                for k, model_state_dict in enumerate(model_state_dict_list):
                    target_state_dict[key].data += (self.num_samples_per_client[k+1] / total_clients_samples) * model_state_dict[key].data.clone()

        return target_state_dict

    # Used for federation
    def update_global_parameters(self):
        global_enc_params = self.aggregate_model_params(self.Encoders)
        global_dec_lr_params = self.aggregate_model_params(self.Decoders_LR)
        global_dec_sr_params = self.aggregate_model_params(self.Decoders_SR)

        self.global_Encoder.load_state_dict(global_enc_params)
        self.global_Decoder_LR.load_state_dict(global_dec_lr_params)
        self.global_Decoder_SR.load_state_dict(global_dec_sr_params)

    def restore_model(self, k):
        """
        Restore the trained networks.
        """
        print('Loading the trained models for client {}'.format(k))

        Encoder_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Encoder_Client_{}.ckpt'.format(self.num_fold, k))
        self.Encoders[k].load_state_dict(torch.load(Encoder_path, map_location=lambda storage, loc: storage))

        Decoder_LR_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Decoder_LR_Client_{}.ckpt'.format(self.num_fold, k))
        self.Decoders_LR[k].load_state_dict(torch.load(Decoder_LR_path, map_location=lambda storage, loc: storage))

        Decoder_SR_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Decoder_SR_Client_{}.ckpt'.format(self.num_fold, k))
        self.Decoders_SR[k].load_state_dict(torch.load(Decoder_SR_path, map_location=lambda storage, loc: storage))

    def reset_grad(self, k):
        """
        Reset the gradient buffer.
        """
        self.optimizers[k].zero_grad()

    def loss_FedGmTE_Net(self, real, predicted, k, t, train=True, morph=True):
        # real: timepoint x sample x feat_vec
        val_i = max(1, math.floor(len(real[t]) * self.opts.val_ratio))

        # Set data for train/validation
        if train:
            real = [tensor[val_i:] for tensor in real]
            predicted = [tensor[val_i:] for tensor in predicted]
        else:
            real = [tensor[:val_i] for tensor in real]
            predicted = [tensor[:val_i] for tensor in predicted]

        # For antivectorize for node strength
        if morph:
            mat_dim = self.opts.lr_dim
        else:
            mat_dim = self.opts.sr_dim

        self.MAE = torch.nn.L1Loss().to(self.device)
        self.tp = torch.nn.MSELoss().to(self.device)

        loss = 0
        n_samples = len(real[t])
        for i in range(n_samples):
            local_loss = self.MAE(predicted[t][i], real[t][i]) + self.opts.tp_coef * self.tp(antiVectorize_tensor(predicted[t][i], mat_dim, device=self.device).sum(dim=-1),
                                                                                                    antiVectorize_tensor(real[t][i], mat_dim, device=self.device).sum(dim=-1))
            loss += 1 / n_samples * local_loss

        return loss

    def train(self):
        """
        Train our networks for different clients
        """
        LR_losses = []
        SR_losses = []
        total_losses = []

        for k in range(self.num_clients):
            LR_losses.append([])
            SR_losses.append([])
            total_losses.append([])

        # Without federation
        if not self.opts.federate:
            num_iters = self.opts.num_iters
            for k in range(self.num_clients):
                _, _, _, LR_losses_client, SR_losses_client, total_losses_client = self.train_client(k, num_iters)
                # Save Loss figures
                LR_losses[k].extend(LR_losses_client)
                SR_losses[k].extend(SR_losses_client)
                total_losses[k].extend(total_losses_client)

        # With federation
        else:
            num_iters = self.opts.num_global_iters

            # Early stop
            best_val_loss_clients = np.full(self.num_clients, np.inf)
            best_epochs_clients = np.zeros(self.num_clients)
            for i in range(num_iters):
                print('============================')
                print(f"Global iteration number: {i}")
                print('============================')

                # Stop training when all of the models stabalise
                stop_training = []

                for k in range(self.num_clients):
                    best_epoch_client, best_val_loss_client, stop_training_client, LR_losses_client, SR_losses_client, total_losses_client = self.train_client(k,
                    num_iters=self.opts.num_local_iters[k], global_iter=i, best_val_loss=best_val_loss_clients[k], best_epoch=best_epochs_clients[k])

                    # Early stop
                    best_epochs_clients[k] = best_epoch_client
                    best_val_loss_clients[k] = best_val_loss_client

                    LR_losses[k].extend(LR_losses_client)
                    SR_losses[k].extend(SR_losses_client)
                    total_losses[k].extend(total_losses_client)

                    stop_training.append(stop_training_client)

                # Early stop
                if all(stop_training):
                    break

                print("-----------Model Aggregation--------------")
                # Aggregate
                # Set models params equal to global model
                self.update_global_parameters()
                self.set_global_parameters()

        return LR_losses, SR_losses, total_losses

    def calculate_auxiliary_loss(self, k, M_tgt_GT, F_tgt_GT):
        LR_loss = 0
        val_LR_loss = 0
        SR_loss = 0
        val_SR_loss = 0
        adj = self.adjs[k]
        M_fake_i = []
        F_fake_i = []

        for timepoint in range(0, self.opts.nb_timepoints):
            embedding = self.Encoders[k](M_tgt_GT[timepoint], adj)
            M_fake_t = self.Decoders_LR[k].forward_once(embedding, adj)
            F_fake_t = self.Decoders_SR[k].forward_once(embedding, adj)

            M_fake_i.append(M_fake_t)
            F_fake_i.append(F_fake_t)

        for timepoint in range(0, self.opts.nb_timepoints):
            val_loss_ti_LR = self.loss_FedGmTE_Net(M_tgt_GT, M_fake_i, k, timepoint, train=False, morph=True)
            val_loss_ti_SR = self.loss_FedGmTE_Net(F_tgt_GT, F_fake_i, k, timepoint, train=False, morph=False)
            val_LR_loss += (val_loss_ti_LR)
            val_SR_loss += (val_loss_ti_SR)

            loss_ti_LR = self.loss_FedGmTE_Net(M_tgt_GT, M_fake_i, k, timepoint, train=True, morph=True)
            loss_ti_SR = self.loss_FedGmTE_Net(F_tgt_GT, F_fake_i, k, timepoint, train=True, morph=False)
            LR_loss += (loss_ti_LR)
            SR_loss += (loss_ti_SR)

        val_LR_loss = torch.mean(val_LR_loss)
        val_SR_loss = torch.mean(val_SR_loss)
        LR_loss = torch.mean(LR_loss)
        SR_loss = torch.mean(SR_loss)

        val_total_loss = (val_LR_loss + val_SR_loss) / 2
        total_loss = (LR_loss + SR_loss) / 2

        return total_loss, val_total_loss

    def train_client(self, k, num_iters, global_iter=0, best_val_loss=np.inf, best_epoch=0):
        t0_iter = iter(self.input_t0_clients[k])

        tn_morph_iters = []
        for loader in self.M_tn_loaders_clients[k]:
            tn_morph_iters.append(iter(loader))

        tn_func_iters = []
        for loader in self.F_tn_loaders_clients[k]:
            tn_func_iters.append(iter(loader))

        # Start training.
        start_time = time.time()
        start_iters = 0
        LR_losses = []
        SR_losses = []
        total_losses = []
        print(f" 1. Train the client {k} Network for LR and SR")
        stop_training = False
        for i in range(start_iters, num_iters):
            print("-------------iteration-{}-------------".format(i))
            # =================================================================================== #
            #                             1. Preprocess input data                                #
            # =================================================================================== #

            #---ENCODER---#
            # Prepare the input to the encoder of the Network
            # It is a matrix of real complete subjects with low-resolution graphs at t0
            try:
                t0_morph_encoder = next(t0_iter)
            except:
                t0_iter = iter(self.input_t0_clients[k])
                t0_morph_encoder = next(t0_iter)

            t0_M_encoder = t0_morph_encoder[0].to(self.device)

            #---Graph trajectory decoder-1-----#
            # Prepare the real data to compute the loss
            # It is a matrix of real complete subjects with trajectory of low-resolution graphs (t0 ... tn)
            M_tgt_GT = []
            for tn_morph_idx in range(len(tn_morph_iters)):
                try:
                    M_tgt_GT_i = next(tn_morph_iters[tn_morph_idx])
                    M_tgt_GT.append(M_tgt_GT_i)
                except:
                    tn_morph_iters[tn_morph_idx] = iter(self.M_tn_loaders_clients[k][tn_morph_idx])
                    M_tgt_GT_i = next(tn_morph_iters[tn_morph_idx])
                    M_tgt_GT.append(M_tgt_GT_i)
            for tn_morph_idx in range(len(M_tgt_GT)):
                M_tgt_GT[tn_morph_idx] = M_tgt_GT[tn_morph_idx][0].to(self.device)

            #---Graph trajectory decoder-2----#
            # Prepare the real data to compute the loss
            # It is a matrix of real complete subjects with trajectory of super-resolution graphs (t0 ... tn)
            F_tgt_GT = []
            for tn_func_idx in range(len(tn_func_iters)):
                try:
                    F_tgt_GT_i = next(tn_func_iters[tn_func_idx])
                    F_tgt_GT.append(F_tgt_GT_i)
                except:
                    tn_func_iters[tn_func_idx] = iter(self.F_tn_loaders_clients[k][tn_func_idx])
                    F_tgt_GT_i = next(tn_func_iters[tn_func_idx])
                    F_tgt_GT.append(F_tgt_GT_i)

            for tn_func_idx in range(len(F_tgt_GT)):
                F_tgt_GT[tn_func_idx] = F_tgt_GT[tn_func_idx][0].to(self.device)

            # =================================================================================== #
            #          2. Train the Network for multi-trajectory evolution prediction             #
            # =================================================================================== #
            LR_loss = 0
            val_LR_loss = 0
            adj = self.adjs[k]
            embedding = self.Encoders[k](t0_M_encoder, adj)
            M_fake_i = self.Decoders_LR[k](embedding, adj)

            SR_loss = 0
            val_SR_loss = 0
            F_fake_i = self.Decoders_SR[k](embedding, adj)

            for timepoint in range(0, self.opts.nb_timepoints):
                # Early stop
                val_loss_ti_LR = self.loss_FedGmTE_Net(M_tgt_GT, M_fake_i, k, timepoint, train=False, morph=True)
                val_loss_ti_SR = self.loss_FedGmTE_Net(F_tgt_GT, F_fake_i, k, timepoint, train=False, morph=False)
                val_LR_loss += (val_loss_ti_LR)
                val_SR_loss += (val_loss_ti_SR)

                loss_ti_LR = self.loss_FedGmTE_Net(M_tgt_GT, M_fake_i, k, timepoint, train=True, morph=True)
                loss_ti_SR = self.loss_FedGmTE_Net(F_tgt_GT, F_fake_i, k, timepoint, train=True, morph=False)
                LR_loss += (loss_ti_LR)
                SR_loss += (loss_ti_SR)

            # Used for early stop
            val_total_loss = (val_LR_loss + val_SR_loss) / 2

            total_loss = (LR_loss + SR_loss) / 2
            if self.opts.use_aux_reg:
                train_aux_loss, val_aux_loss = self.calculate_auxiliary_loss(k, M_tgt_GT, F_tgt_GT)
                total_loss += self.opts.reg_strength * train_aux_loss
                val_total_loss += self.opts.reg_strength * val_aux_loss

            self.reset_grad(k)
            total_loss.backward()
            self.optimizers[k].step()

            # Logging.
            loss = {}
            loss[f'loss/Client {k}'] = total_loss.item()

            LR_losses.append(LR_loss.item())
            SR_losses.append(SR_loss.item())
            total_losses.append(total_loss.item())

            # =================================================================================== #
            #                                   3. Miscellaneous                                  #
            # =================================================================================== #
            # print out training information.
            if (i + 1) % self.opts.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i + 1, num_iters)
                for tag, value in loss.items():
                    log += ", {}: {:.4f}".format(tag, value)
                print(log)

            # Early stop check
            if self.opts.early_stop:
                curr_epoch = global_iter * num_iters + i
                if val_total_loss < best_val_loss:
                    best_val_loss = val_total_loss
                    best_epoch = curr_epoch
                elif curr_epoch > 50 and curr_epoch - best_epoch >= self.opts.patience:
                    # If the validation loss hasn't improved for the patience parameter number of epochs, stop training
                    print(f'Early stopping at epoch {curr_epoch + 1}')
                    stop_training = True
                    break

        # save model checkpoints.
        Encoder_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Encoder_Client_{}.ckpt'.format(self.num_fold, k))
        torch.save(self.Encoders[k].state_dict(), Encoder_path)

        Decoder_LR_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Decoder_LR_Client_{}.ckpt'.format(self.num_fold, k))
        torch.save(self.Decoders_LR[k].state_dict(), Decoder_LR_path)

        Decoder_SR_path = os.path.join(self.opts.checkpoint_dir, 'Fold_{}_Decoder_SR_Client_{}.ckpt'.format(self.num_fold, k))
        torch.save(self.Decoders_SR[k].state_dict(), Decoder_SR_path)

        print('Saved model checkpoints for client {} into {}...'.format(k, self.opts.checkpoint_dir))

        print('============================')
        print(f"End of Training for client {k}")
        print('============================')

        return best_epoch, best_val_loss, stop_training, LR_losses, SR_losses, total_losses

    # =================================================================================== #
    #                              6. Test with a new dataset                             #
    # =================================================================================== #
    def test(self):
        """
        Test all trained clients networks of our FedGmTE-Net
        """
        metrics_LR_clients = []
        metrics_SR_clients = []

        for k in range(self.num_clients):
            metrics_LR = []
            metrics_SR = []
            print(f" 1. Test the client {k} Network for LR and SR")

            self.restore_model(k)
            self.Encoders[k].eval()
            self.Decoders_LR[k].eval()
            self.Decoders_SR[k].eval()

            t0_M_encoder = next(iter(self.input_t0_clients[k]))
            t0_M_encoder = t0_M_encoder[0].to(self.device)

            tn_morph_iters = []
            for loader in self.M_tn_loaders_clients[k]:
                tn_morph_iters.append(iter(loader))
            M_tgt_GT = []
            for tn_morph_idx in range(len(tn_morph_iters)):
                M_tgt_GT_i = next(tn_morph_iters[tn_morph_idx])
                M_tgt_GT.append(M_tgt_GT_i)
            for tn_morph_idx in range(len(M_tgt_GT)):
                M_tgt_GT[tn_morph_idx] = M_tgt_GT[tn_morph_idx][0].to(self.device)

            tn_func_iters = []
            for loader in self.F_tn_loaders_clients[k]:
                tn_func_iters.append(iter(loader))
            F_tgt_GT = []
            for tn_func_idx in range(len(tn_func_iters)):
                F_tgt_GT_i = next(tn_func_iters[tn_func_idx])
                F_tgt_GT.append(F_tgt_GT_i)
            for tn_func_idx in range(len(F_tgt_GT)):
                F_tgt_GT[tn_func_idx] = F_tgt_GT[tn_func_idx][0].to(self.device)

            with torch.no_grad():
                adj = self.adjs[k]
                embedding = self.Encoders[k](t0_M_encoder, adj)
                M_fake_i = self.Decoders_LR[k](embedding, adj)
                F_fake_i = self.Decoders_SR[k](embedding, adj)
                for timepoint in range(0, self.opts.nb_timepoints):
                    # Ignore if no ground truth values
                    morph_mask = ~torch.isnan(M_tgt_GT[timepoint].cpu()).any(dim=1)
                    func_mask = ~torch.isnan(F_tgt_GT[timepoint].cpu()).any(dim=1)
                    M_tgt_GT[timepoint] = M_tgt_GT[timepoint][morph_mask]
                    M_fake_i[timepoint] = M_fake_i[timepoint][morph_mask]
                    F_tgt_GT[timepoint] = F_tgt_GT[timepoint][func_mask]
                    F_fake_i[timepoint] = F_fake_i[timepoint][func_mask]

                    # metrics
                    metrics_LR_t = []
                    metrics_SR_t = []
                    for metric in self.opts.metrics:
                        if metric == "mae":
                            metric_LR = mean_absolute_error(M_tgt_GT[timepoint].cpu(), M_fake_i[timepoint].cpu())
                            metric_SR = mean_absolute_error(F_tgt_GT[timepoint].cpu(), F_fake_i[timepoint].cpu())
                        elif metric == "ns":
                            metric_LR = calculate_mae_ns(M_tgt_GT[timepoint].cpu(), M_fake_i[timepoint].cpu(), self.opts.lr_dim)
                            metric_SR = calculate_mae_ns(F_tgt_GT[timepoint].cpu(), F_fake_i[timepoint].cpu(), self.opts.sr_dim)

                        mean_metric = (metric_LR + metric_SR) / 2
                        print(f"Timepoint {timepoint} {metric} total for client {k}: {mean_metric}")

                        metrics_LR_t.append(metric_LR)
                        metrics_SR_t.append(metric_SR)

                    metrics_LR.append(metrics_LR_t)
                    metrics_SR.append(metrics_SR_t)

                metrics_LR_clients.append(metrics_LR)
                metrics_SR_clients.append(metrics_SR)

        return metrics_LR_clients, metrics_SR_clients

    def forward(self):
        """
        Forward pass - returns predicted trajectories for LR and SR for each client (only input required)
        """
        predicted_trajectory_LR_clients = []
        predicted_trajectory_SR_clients = []
        real_trajectory_LR_clients = []
        real_trajectory_SR_clients = []
        for k in range(self.num_clients):
            self.restore_model(k)
            self.Encoders[k].eval()
            self.Decoders_LR[k].eval()
            self.Decoders_SR[k].eval()

            # Input
            t0_M_encoder = next(iter(self.input_t0_clients[k]))
            t0_M_encoder = t0_M_encoder[0].to(self.device)

            tn_morph_iters = []
            for loader in self.M_tn_loaders_clients[k]:
                tn_morph_iters.append(iter(loader))
            M_tgt_GT = []
            for tn_morph_idx in range(len(tn_morph_iters)):
                M_tgt_GT_i = next(tn_morph_iters[tn_morph_idx])
                M_tgt_GT.append(M_tgt_GT_i)
            for tn_morph_idx in range(len(M_tgt_GT)):
                M_tgt_GT[tn_morph_idx] = M_tgt_GT[tn_morph_idx][0].to(self.device)

            tn_func_iters = []
            for loader in self.F_tn_loaders_clients[k]:
                tn_func_iters.append(iter(loader))
            F_tgt_GT = []
            for tn_func_idx in range(len(tn_func_iters)):
                F_tgt_GT_i = next(tn_func_iters[tn_func_idx])
                F_tgt_GT.append(F_tgt_GT_i)
            for tn_func_idx in range(len(F_tgt_GT)):
                F_tgt_GT[tn_func_idx] = F_tgt_GT[tn_func_idx][0].to(self.device)

            with torch.no_grad():
                adj = self.adjs[k]
                embedding = self.Encoders[k](t0_M_encoder, adj)
                predicted_trajectory_LR = []
                predicted_trajectory_SR = []
                real_trajectory_LR = []
                real_trajectory_SR = []
                M_fake_i = self.Decoders_LR[k](embedding, adj)
                F_fake_i = self.Decoders_SR[k](embedding, adj)
                for timepoint in range(0, self.opts.nb_timepoints):
                    # the below list is the predictions
                    predicted_trajectory_LR.append(M_fake_i[timepoint].cpu())
                    predicted_trajectory_SR.append(F_fake_i[timepoint].cpu())

                    real_trajectory_LR.append(M_tgt_GT[timepoint].cpu())
                    real_trajectory_SR.append(F_tgt_GT[timepoint].cpu())

                predicted_trajectory_LR_clients.append(predicted_trajectory_LR)
                predicted_trajectory_SR_clients.append(predicted_trajectory_SR)

                real_trajectory_LR_clients.append(real_trajectory_LR)
                real_trajectory_SR_clients.append(real_trajectory_SR)

        return predicted_trajectory_LR_clients, predicted_trajectory_SR_clients, real_trajectory_LR_clients, real_trajectory_SR_clients



