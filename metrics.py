import numpy as np
from sklearn.metrics import mean_absolute_error
from utils import antiVectorize

def calculate_mae_ns(true, predicted, dim):
    ns_samples_true = []
    ns_samples_predicted = []
    for i in range(len(true)):
        real = true[i]
        fake = predicted[i]

        real_M = antiVectorize(real, dim)
        fake_M = antiVectorize(fake, dim)

        ns_real = np.sum(real_M, axis=1)
        ns_fake =np.sum(fake_M, axis=1)

        ns_samples_true.append(ns_real)
        ns_samples_predicted.append(ns_fake)

    ns_samples_true = np.array(ns_samples_true)
    ns_samples_predicted = np.array(ns_samples_predicted)

    return mean_absolute_error(ns_samples_true, ns_samples_predicted)
