import numpy as np


class LatencyNormalizer:
    def __init__(self, offset: int, max_time_out: int, max_value=20):
        self.offset = offset
        self.max_time_out = max_time_out
        self.max_value = max_value

    def encode(self, v):
        return int(np.log(2 + v) / np.log(self.max_time_out) * 200) / 200.

    def decode(self, v):
        return np.exp(v * np.log(self.max_time_out))

    def cost_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def cost_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost

    def latency_encode(self, v, min_latency, max_latency):
        return (v - min_latency) / (max_latency - min_latency)

    def latency_decode(self, v, min_latency, max_latency):
        return (max_latency - min_latency) * v + min_latency

    def rows_encode(self, v, min_cost, max_cost):
        return (v - min_cost) / (max_cost - min_cost)

    def rows_decode(self, v, min_cost, max_cost):
        return (max_cost - min_cost) * v + min_cost


class Normalizer:
    def __init__(self, mini=None, maxi=None):
        self.mini = mini
        self.maxi = maxi

    def normalize_labels(self, labels, reset_min_max=False):
        ## added 0.001 for numerical stability
        labels = np.array([np.log(float(l) + 0.001) for l in labels])
        if self.mini is None or reset_min_max:
            self.mini = labels.min()
            print("min log(label): {}".format(self.mini))
        if self.maxi is None or reset_min_max:
            self.maxi = labels.max()
            print("max log(label): {}".format(self.maxi))
        labels_norm = (labels - self.mini) / (self.maxi - self.mini)
        # Threshold labels <-- but why...
        labels_norm = np.minimum(labels_norm, 1)
        labels_norm = np.maximum(labels_norm, 0.001)

        return labels_norm

    def unnormalize_labels(self, labels_norm):
        labels_norm = np.array(labels_norm, dtype=np.float32)
        labels = (labels_norm * (self.maxi - self.mini)) + self.mini
        #         return np.array(np.round(np.exp(labels) - 0.001), dtype=np.int64)
        return np.array(np.exp(labels) - 0.001)


def normalize_data(val, column_name, column_min_max_vals):
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    val_norm = 0.0
    if max_val > min_val:
        val_norm = (val - min_val) / (max_val - min_val)
    return np.array(val_norm, dtype=np.float32)
