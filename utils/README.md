# EEG Signal Analysis Toolkit

[ssvep_analysis.py](../utils/ssvep_analysis.py) provides a set of functions to analyze electroencephalogram (EEG) data. It includes utilities for filtering, plotting, and transforming EEG signals, as well as computing Canonical Correlation Analysis (CCA) and wavelet transforms.

## Usage

An example of using the toolkit is shown in [ssvep_analysis.ipynb](../notebooks/ssvep_analysis.ipynb).

## Functions

### `update_default_sampling_rate(new_sampling_rate)`
Update the default sampling rate used by the toolkit.

#### Parameters
- **new_sampling_rate** : float  
  The new sampling rate to be set as default.

### `update_default_stimulus_frequency(new_stimulus_frequency)`
Update the default stimulus frequency used in signal generation.

#### Parameters
- **new_stimulus_frequency** : float  
  The new stimulus frequency to be set as default.

### `plot_eeg(eeg_data, sampling_rate)`
Plot EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data to be plotted.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The plot of EEG data.

### `apply_linear_detrending(eeg_data)`
Apply linear detrending to EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for detrending.

#### Returns
- **ndarray** : Detrended EEG data.

### `apply_ransac_detrending(eeg_data, ransac_iterations, ransac_min_samples)`
Apply RANSAC (RANdom SAmple Consensus) detrending to EEG data. This method fits a linear model to the data with robustness to outliers.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for detrending.
- **ransac_iterations** : int, optional  
  The number of iterations for the RANSAC algorithm. Default is 100.
- **ransac_min_samples** : int, optional  
  The minimum number of samples to be considered as a consensus set in each iteration of RANSAC. Default is None.

#### Returns
- **ndarray** : Detrended EEG data.  
  The EEG data after RANSAC detrending has been applied.

### `remove_artefacts(eeg_data, threshold=8)`
Remove artifacts from EEG data by identifying and nullifying extreme changes that exceed a specified threshold based on the interquartile range (IQR) of the data's first difference. This method helps in cleaning the EEG signal by mitigating the effects of large, non-physiological spikes or drops.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data from which artifacts are to be removed.
- **threshold** : float, optional  
  The multiplier used in conjunction with the IQR to define what constitutes an extreme change in the EEG signal. The default value is 8, meaning that any change greater than 8 times the IQR of the first difference of the data is considered an artifact.

#### Returns
- **ndarray** : Artifact-removed EEG data.


### `apply_lowpass_filter(eeg_data, cutoff, filter_order, sampling_rate)`
Apply a lowpass filter to EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data to be filtered.
- **cutoff** : float, optional  
  The cutoff frequency for the filter. Default is 35 Hz.
- **filter_order** : int, optional  
  The order of the filter. Default is 5.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray** : Filtered EEG data.

### `apply_highpass_filter(eeg_data, cutoff, filter_order, sampling_rate)`
Apply a highpass filter to EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data to be filtered.
- **cutoff** : float, optional  
  The cutoff frequency for the filter. Default is 1 Hz.
- **filter_order** : int, optional  
  The order of the filter. Default is 5.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray** : Filtered EEG data.

### `apply_bandpass_filter(eeg_data, lowcut, highcut, filter_order, sampling_rate)`
Apply a bandpass filter to EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data to be filtered.
- **lowcut** : float, optional
  The low cutoff frequency for the filter. Default is 1 Hz.
- **highcut** : float, optional
  The high cutoff frequency for the filter. Default is 35 Hz.
- **filter_order** : int, optional  
  The order of the filter. Default is 5.
- **sampling_rate** : float, optional
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray** : Filtered EEG data.

### `apply_notch_filter(eeg_data, notch_freq, bandwidth, filter_order, sampling_rate)`
Apply a notch filter to EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data to be filtered.
- **notch_freq** : float, optional  
  The center frequency for the notch filter. Default is 50 Hz.
- **bandwidth** : float, optional  
  The bandwidth of the notch filter. Default is 3 Hz.
- **filter_order** : int, optional  
  The order of the filter. Default is 5.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray** : Filtered EEG data.

### `compute_cca(eeg_data, n_components, n_harmonics, sampling_rate, stimulus_frequency)`
Compute Canonical Correlation Analysis (CCA) between EEG data and a reference signal.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for CCA computation.
- **n_components** : int, optional  
  Number of components to keep. Default is 1.
- **n_harmonics** : int, optional  
  Number of harmonics of the stimulus frequency. Default is 2.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.
- **stimulus_frequency** : float, optional  
  The stimulus frequency used in reference signal generation. Default is `DEFAULT_STIMULUS_FREQUENCY`.

#### Returns
- **CCA object, ndarray** : The fitted CCA object and the target (reference signal).

### `generate_reference_signals(n_harmonics, length, sampling_rate, stimulus_frequency)`
Generate reference signals for CCA computation.

#### Parameters
- **n_harmonics** : int  
  Number of harmonics to generate.
- **length** : int  
  The length of the generated signals.
- **sampling_rate** : float, optional  
  The sampling rate for the signals. Default is `DEFAULT_SAMPLING_RATE`.
- **stimulus_frequency** : float, optional  
  The stimulus frequency used in signal generation. Default is `DEFAULT_STIMULUS_FREQUENCY`.

#### Returns
- **ndarray** : Generated reference signals.

### `compute_reduced_signal(eeg_data, n_components, n_harmonics, sampling_rate, stimulus_frequency)`
Reduce EEG data to specified dimensions using CCA.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for dimensionality reduction.
- **n_components** : int, optional  
  Number of components to keep. Default is 1.
- **n_harmonics** : int, optional  
  Number of harmonics of the stimulus frequency. Default is 2.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.
- **stimulus_frequency** : float, optional  
  The stimulus frequency used in CCA computation. Default is `DEFAULT_STIMULUS_FREQUENCY`.

#### Returns
- **ndarray, ndarray** : Reduced signal and the coefficient matrix.

### `compute_power_spectrum(eeg_data, sampling_rate)`
Compute the power spectrum of EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for power spectrum computation.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray, ndarray** : Frequencies and the computed power spectrum.

### `plot_power_spectrum(frequencies, spectrum)`
Plot the power spectrum of EEG data.

#### Parameters
- **frequencies** : ndarray  
  The frequencies of the power spectrum.
- **spectrum** : ndarray  
  The computed power spectrum.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The plot of the power spectrum.

### `plot_coefficient_matrix(coefficient_matrix)`
Plot the coefficient matrix from CCA.

#### Parameters
- **coefficient_matrix** : ndarray  
  The coefficient matrix to be plotted.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The heatmap plot of the coefficient matrix.

### `compute_running_r_values(eeg_data, marker, n_components, n_harmonics, window_duration, step_size, sampling_rate)`
Compute running r-values for EEG data using a sliding window approach.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for r-value computation.
- **marker** : ndarray, optional  
  Marker values associated with EEG data. Default is None.
- **n_components** : int, optional  
  Number of components for CCA. Default is 1.
- **n_harmonics** : int, optional  
  Number of harmonics for CCA. Default is 2.
- **window_duration** : int, optional  
  The duration of each window in seconds. Default is 2.
- **step_size** : int, optional  
  The step size in samples. Default is 40.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray, ndarray, ndarray (optional)** : r-values, corresponding times, and marker values (if provided).

### `plot_r_values(r_values, times, marker_values)`
Plot r-values computed for EEG data.

#### Parameters
- **r_values** : ndarray  
  The computed r-values.
- **times** : ndarray  
  The corresponding times for the r-values.
- **marker_values** : ndarray, optional  
  Marker values associated with r-values. Default is None.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The plot of r-values.

### `compute_wavelet_transform(eeg_data, w, frequencies, n_times, sampling_rate)`
Compute the wavelet transform for EEG data.

#### Parameters
- **eeg_data** : ndarray  
  The EEG data for wavelet transform computation.
- **w** : int, optional  
  The wavelet parameter w. Default is 50.
- **frequencies** : ndarray, optional  
  Frequencies to compute the transform for. Default is `np.linspace(0, 35, 300)`.
- **n_times** : int, optional  
  Number of time points. Default is 300.
- **sampling_rate** : float, optional  
  The sampling rate of the EEG data. Default is `DEFAULT_SAMPLING_RATE`.

#### Returns
- **ndarray, ndarray, ndarray** : Frequencies, times, and the wavelet transform matrix.

### `plot_wavelet_transform(frequencies, times, cwt_matrix)`
Plot the wavelet transform of EEG data.

#### Parameters
- **frequencies** : ndarray  
  The frequencies of the wavelet transform.
- **times** : ndarray  
  The times of the wavelet transform.
- **cwt_matrix** : ndarray  
  The computed wavelet transform matrix.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The heatmap plot of the wavelet transform.

### `plot_time_signal(times, amplitudes)`
Plot a time signal.

#### Parameters
- **times** : ndarray  
  The times for the signal.
- **amplitudes** : ndarray  
  The amplitudes of the signal.

#### Returns
- **matplotlib.pyplot** : Plot object  
  The plot of the time signal.

## Requirements
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

# Transformer

[transformer.py](../utils/transformer.py) offers classes for applying wavelet transforms and subsampling. These classes are designed to be compatible with scikit-learn's estimator and transformer interfaces.

## Usage

An example of using the toolkit is shown in [cca_cwt_svm_classifier.ipynb](../notebooks/cca_cwt_svm_classifier.ipynb).

## Classes

### `WaveletTransformer`
A transformer for applying a wavelet transform to EEG data.

#### Parameters
- **w** : int  
  The wavelet parameter w. Default is `50`.
- **sampling_rate** : float  
  The sampling rate of the EEG data. Default is `250`.
- **frequency** : float  
  The frequency to be used for the wavelet transform. Default is `16.5`.

#### Methods
- `transform(X)`  
  Apply the wavelet transform to the provided EEG data.

### `Subsampler`
A transformer for subsampling EEG data.

#### Parameters
- **n_samples** : int  
  The number of samples to subsample the EEG data to. Default is `100`.

#### Methods
- `transform(X)`  
  Subsample the provided EEG data.

## Requirements
- numpy
- scipy
- scikit-learn

# Kernel Functions

[kernels.py](../utils/kernels.py) provides custom kernel functions. These kernels include a cosine similarity-based RBF kernel and a generalized cone kernel, which can be useful in various applications including clustering, classification, and dimensionality reduction tasks.

## Usage

A more detailed explanation can be found in [generalized_cone_kernel.ipynb](../notebooks/generalized_cone_kernel.ipynb).

## Functions

### `cosine_rbf_kernel`
A kernel function that combines the Radial Basis Function (RBF) kernel with cosine similarity.

#### Parameters
- **X** : array-like, shape (n_samples_X, n_features)  
  Input data.
- **Y** : array-like, shape (n_samples_Y, n_features), optional  
  Input data. If `None`, the kernel is calculated among the samples of `X`.
- **gamma** : float, optional  
  Kernel coefficient. If `None`, `gamma` is set to `1 / n_features`.

#### Returns
- **K** : array, shape (n_samples_X, n_samples_Y)  
  The computed kernel matrix.

### `generalized_cone_kernel`
A kernel function that generalizes the concept of cone kernels, incorporating both Euclidean distance and cosine similarity, adjusted by an alpha parameter based on the norms of data points.

#### Parameters
- **X** : array-like, shape (n_samples_X, n_features)  
  Input data.
- **Y** : array-like, shape (n_samples_Y, n_features), optional  
  Input data. If `None`, the kernel is calculated among the samples of `X`.
- **gamma** : float, optional  
  Kernel coefficient for the distance component. If `None`, `gamma` is set to `1 / n_features`.
- **beta** : float, optional  
  Coefficient for the norm-based exponential weighting. If `None`, `beta` is automatically determined based on the standard deviation of the sum of norms.

#### Returns
- **K** : array, shape (n_samples_X, n_samples_Y)  
  The computed kernel matrix.

## Requirements
- numpy
- scikit-learn