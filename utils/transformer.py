import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, w=50, sampling_rate=250, frequency_range=(16, 17), n_frequencies=50, n_samples=None):
        self.w = w
        self.sampling_rate = sampling_rate
        self.frequency_range = frequency_range
        self.n_frequencies = n_frequencies
        self.frequencies = np.linspace(*frequency_range, n_frequencies)
        self.widths = self.w * self.sampling_rate / (2 * np.pi * self.frequencies)
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_samples = self.n_samples if self.n_samples is not None else X.shape[1]
        idx_mask = X.shape[1] // n_samples * np.arange(n_samples)
        cwt_matrix = np.apply_along_axis(lambda x: np.abs(cwt(x, morlet2, widths=self.widths, w=self.w, dtype='complex128'))[:, idx_mask].flatten(), axis=1, arr=X)
        return cwt_matrix

class Subsampler(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=2500):
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        idx_mask = X.shape[1] // self.n_samples * np.arange(self.n_samples)
        return X[:, idx_mask]
