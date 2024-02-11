import numpy as np
from scipy.signal import cwt, morlet2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class WaveletTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, w=50, sampling_rate=250, frequency=16.5):
        self.w = w
        self.sampling_rate = sampling_rate
        self.frequency = frequency

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        width = self.w * self.sampling_rate / (2 * np.pi * self.frequency)
        cwt_matrix = np.apply_along_axis(lambda x: np.abs(cwt(x, morlet2, widths=[width], w=self.w, dtype='complex128')).flatten(), axis=1, arr=X)
        return cwt_matrix

class Subsampler(BaseEstimator, TransformerMixin):
    def __init__(self, n_samples=100):
        self.n_samples = n_samples

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        idx_mask = X.shape[1] // self.n_samples * np.arange(self.n_samples)
        return X[:, idx_mask]
    
class PcaRankReducer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.array([self._first_principal_component(x) for x in X])
    
    def _first_principal_component(self, X):
        pca = PCA(n_components=1)
        return pca.fit_transform(X).flatten()
