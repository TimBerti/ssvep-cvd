import numpy as np
from scipy.signal import cwt, morlet2
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from . import ssvep_analysis as sa


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_jobs=-1,
        max_len=8000,
        sampling_rate=250,
        stimulus_frequency=16.5,
        ransac_iterations=100,
        ransac_min_samples=None,
        filter_threshold_factor=3,
        lowpass_cutoff=35,
        highpass_cutoff=1,
        notch_freq=50,
        notch_width=3,
        cca_components=1,
        cca_harmonics=2,
    ):
        self.n_jobs = n_jobs
        self.max_len = max_len
        self.sampling_rate = sampling_rate
        self.stimulus_frequency = stimulus_frequency
        self.ransac_iterations = ransac_iterations
        self.ransac_min_samples = ransac_min_samples
        self.filter_threshold_factor = filter_threshold_factor
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff
        self.notch_freq = notch_freq
        self.notch_width = notch_width
        self.cca_components = cca_components
        self.cca_harmonics = cca_harmonics

        sa.update_default_sampling_rate(sampling_rate)
        sa.update_default_stimulus_frequency(stimulus_frequency)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = Parallel(n_jobs=self.n_jobs)(delayed(self.preprocess)(x) for x in X)
        return np.array(X_new)

    def preprocess(self, x):
        x = sa.apply_ransac_detrending(x, ransac_iterations=self.ransac_iterations, ransac_min_samples=self.ransac_min_samples)
        x = sa.filter_extreme_values(x, threshold_factor=self.filter_threshold_factor)
        x = sa.apply_lowpass_filter(x, cutoff=self.lowpass_cutoff)
        x = sa.apply_highpass_filter(x, cutoff=self.highpass_cutoff)
        x = sa.apply_notch_filter(x, notch_freq=self.notch_freq, bandwidth=self.notch_width)
        x, _ = sa.compute_reduced_signal(x, n_components=self.cca_components, n_harmonics=self.cca_harmonics)
        x = np.concatenate((x, np.zeros(self.max_len - len(x))))
        return x

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