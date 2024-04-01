import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.base import BaseEstimator, TransformerMixin


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, w_list=[50], sampling_rate=250, frequency_range=(16, 17), n_frequencies=100, n_samples=None):
        self.w_list = w_list
        self.sampling_rate = sampling_rate
        self.frequency_range = frequency_range
        self.n_frequencies = n_frequencies
        self.frequencies = np.linspace(*self.frequency_range, self.n_frequencies)
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        n_samples = self.n_samples if self.n_samples is not None else X.shape[1]
        self.idx_mask = X.shape[1] // n_samples * np.arange(n_samples)
        return np.apply_along_axis(lambda x: self._multi_channel_cwt(x).flatten(), 1, X)
    
    def _multi_channel_cwt(self, x):
        cwts = [self._cwt(x, w) for w in self.w_list]
        return np.array(cwts)
    
    def _cwt(self, x, w):
        widths = w * self.sampling_rate / (2 * np.pi * self.frequencies)
        return np.abs(cwt(x, morlet2, widths=widths, w=w, dtype='complex128'))[:, self.idx_mask]


class Subsampler(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=2500):
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        idx_mask = X.shape[1] // self.n_samples * np.arange(self.n_samples)
        return X[:, idx_mask]
