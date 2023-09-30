import numpy as np
import os
from scipy.stats import pearsonr
from plotting import *
from utils import vectorize
from data_loader import create_dirs_if_not_exist

# --------------------------------------------------------------
# SHAPE: (n_subjects, n_timepoints, n_rois, n_rois)
# --------------------------------------------------------------

# Complete incomplete dataset of lr and sr samples
def complete_dataset(full_data, n_time=3, weighted=False):
    for i, sample in enumerate(full_data):
        for t in range(1, n_time):
            sample_t = sample[t]
            if np.isnan(sample_t).any():
                val = aggregate_nearest_neighbours(full_data=full_data, timepoint=t, sample_num=i, weighted=weighted)
                full_data[i, t] = val

    return full_data

# Nearest neighbours
def aggregate_nearest_neighbours(full_data, timepoint, sample_num, k=2, weighted=False):
    idx_sorted, neighbours_dist = pcc_sort(full_data=full_data, sample_num=sample_num)
    # Check for Nan values at desired timepoint and collect neighbours
    neighbours = []
    distances = []
    for i in range(k):
        idx = idx_sorted[i]
        sample_t = full_data[idx, timepoint]
        if not np.isnan(sample_t).any():
            neighbours.append(sample_t)
            distances.append(neighbours_dist[idx])

    # If all Nan check further neighbours
    if len(neighbours) == 0:
        i = k
        while True:
            idx = idx_sorted[i]
            sample_t = full_data[idx, timepoint]
            if not np.isnan(sample_t).any():
                neighbours.append(sample_t)
                distances.append(neighbours_dist[idx])
                break
            else:
                i += 1

    neighbours = np.array(neighbours)

    # Set missing value to average of nearest neighbours
    if weighted:
        imp_value = 0
        total_weight = 0
        total_weight = sum(distances)
        for i, neighbour in enumerate(neighbours):
            imp_value += distances[i] / total_weight * neighbour
    else:
        imp_value = np.mean(neighbours, axis=0)

    return imp_value


# Return indices sorted by pcc
def pcc_sort(full_data, sample_num, t_comp=0):
    # Use time t=0 for comparison
    all_samples = full_data[:, t_comp]
    subject = full_data[sample_num, t_comp]

    neighbours_dist = []
    for i, sample in enumerate(all_samples):
        if i != sample_num:
            if subject.ndim == 1:
                pc,_ = pearsonr(subject, sample)
            else:
                pc,_ = pearsonr(vectorize(subject), vectorize(sample))
            distance = max(0,pc)
            neighbours_dist.append(distance)
        else:
            neighbours_dist.append(-np.inf)

    # Sorted indices according to pcc (highest first)
    neighbours_dist = np.array(neighbours_dist)
    idx_sorted = neighbours_dist.argsort()[::-1]
    return idx_sorted, neighbours_dist

def vectorise_data(data):
    vec_samples = []
    for sample in data:
        vec_samples_t = []
        if data.ndim == 4:
            for t_sample in sample:
                vec_sample = vectorize(t_sample)
                vec_samples_t.append(vec_sample)
            vec_samples.append(vec_samples_t)
        else:
            vec_samples.append(vectorize(sample))

    vec_samples = np.array(vec_samples)
    return vec_samples


def create_plots(data, directory, ext="LR"):
    all_time_samples_lr = []
    vec_samples = vectorise_data(data)
    labels = []
    for t in range(len(vec_samples[0])):
        all_time_samples_lr.extend(vec_samples[:, t])
        labels.extend(np.full((len(vec_samples),), f"t{t}"))
    all_time_samples_lr = np.array(all_time_samples_lr)
    labels = np.array(labels)

    # TSNE plot
    save_path = os.path.join(directory, f'{ext} TSNe')
    plot_tsne(all_time_samples_lr, labels, save_path)

    # PCA plot
    save_path = os.path.join(directory, f'{ext} PCA')
    plot_pca(all_time_samples_lr, labels, save_path)

    # SVD plot
    save_path = os.path.join(directory, f'{ext} SVD')
    plot_svd(all_time_samples_lr, labels, save_path)

    # Visualise samples
    for t in range(len(data[0])):
        save_path = os.path.join(directory, f'{ext} sample - t_{t}')
        plot_cbt(data[0, t], t, save_path)

def prepare_data(data_type="simulate_multi"):
    # Simulated data
    if data_type == "simulate_multi":
        try:
            samples_lr = np.load(f'./datasets/multivariate_simulation_data_lr.npy')
            samples_sr = np.load(f'./datasets/multivariate_simulation_data_sr.npy')
        except:
            assert False, 'No data available'
    else:
        assert False, 'Data type not implemented'
    return samples_lr, samples_sr

if __name__ == "__main__":

    plot_dir = "data_exploration/"
    multi_sim_dir = plot_dir + "multi_sim/"

    create_dirs_if_not_exist([multi_sim_dir])

    multi_sim_data_lr, multi_sim_data_sr = prepare_data(data_type="simulate_multi")
    create_plots(multi_sim_data_lr, multi_sim_dir, "LR")
    create_plots(multi_sim_data_sr, multi_sim_dir, "SR")
    print(np.shape(multi_sim_data_lr))
    print(np.shape(multi_sim_data_sr))
