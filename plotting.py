import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_cbt(img, timepoint, save_path, vmin=None, vmax=None):
    img = np.repeat(np.repeat(img, 10, axis=1), 10, axis=0)
    plt.imshow(img, vmin=vmin, vmax=vmax)
    plt.title(f"Sample - Time {timepoint}")
    plt.axis('off')
    plt.colorbar()
    plt.savefig(save_path)
    plt.clf()


def plot_loss(LR_loss, SR_loss, total_loss, method_names, save_path):
    for i, method in enumerate(method_names):
        epochs = range(1, len(total_loss[i]) + 1)
        plt.plot(epochs, LR_loss[i], label=f'LR loss ({method})')
        plt.plot(epochs, SR_loss[i], label=f'SR loss ({method})')
        plt.plot(epochs, total_loss[i], label=f'Total loss ({method})')
    plt.title('Training losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # adjust the plot size and spacing
    plt.gcf().set_size_inches(8, 6)
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.clf()

def plot_mae(maes_ts, timepoints, method_names, save_path, min_y):
    _, ax = plt.subplots()
    width = 0.2
    x_pos = np.arange(len(maes_ts[0][0])) + 1 - width
    for i, method in enumerate(method_names):
        means = np.mean(maes_ts[i], axis=0)
        stds = np.std(maes_ts[i], axis=0)
        print(f"{method}-means: {means}")
        print(f"{method}-stds: {stds}")
        # Build the plot
        ax.bar(x_pos, means, width=width, yerr=stds, align='center', alpha=0.5, ecolor='black', capsize=10, label=method)
        x_pos += width
    ax.set_ylabel('MAE')
    ax.set_ylim(bottom=min_y)
    ax.set_xticks(np.arange(len(maes_ts[0][0])) + 1)
    ax.set_xticklabels(timepoints)
    ax.set_title('MAEs across timepoints')
    ax.yaxis.grid(True)
    # Save the figure
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    # adjust the plot size and spacing
    plt.gcf().set_size_inches(8, 6)
    plt.subplots_adjust(bottom=0.2)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.clf()

def plot_tsne(X, labels, save_path, perplexity=30):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_tsne = tsne.fit_transform(X)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        # Get the indices of the data points with the current label
        indices = labels == label

        # Plot the data points with the current label
        plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=str(label))

    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.clf()

def plot_svd(X, labels, save_path):
    _, _, V_t = np.linalg.svd(X, full_matrices=False)
    projected_data = np.dot(X, V_t[:2, :].T)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        # Get the indices of the data points with the current label
        indices = labels == label

        # Plot the data points with the current label
        plt.scatter(projected_data[indices, 0], projected_data[indices, 1], label=str(label))

    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.clf()

def plot_pca(X, labels, save_path):
    # Perform PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    unique_labels = sorted(set(labels))
    for label in unique_labels:
        # Get the indices of the data points with the current label
        indices = labels == label

        # Plot the data points with the current label
        plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=str(label))

    plt.legend(loc='upper right')
    plt.savefig(save_path)
    plt.clf()
