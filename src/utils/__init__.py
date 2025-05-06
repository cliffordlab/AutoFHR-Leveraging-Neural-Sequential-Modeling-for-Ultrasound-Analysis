"""Utility functions for AutoFHR"""

from .signal_processing import (
    normalize, normalize_mean_std, 
    create_scalogram,
    butter_bandpass, butter_bandpass_filter, DUS_filtering,
    signal_resample, find_closer_duplicates, grouping_beats
)

from .evaluation import (
    prediction_with_ground_truth
)

from .visualization import (
    bland_altman_plot, scatter_plot_with_regression,
    attention_plot, time_series_robustness_plot,
    plot_predictions_with_ground_truth
)

__all__ = [
    'normalize', 'normalize_mean_std', 
    'create_scalogram', 'create_scalogram_frequencies',
    'butter_bandpass', 'butter_bandpass_filter', 'DUS_filtering',
    'signal_resample', 'find_closer_duplicates', 'grouping_beats',
    'prediction_with_ground_truth',
    
    # Visualization functions
    'bland_altman_plot', 'scatter_plot_with_regression',
    'attention_plot', 'time_series_robustness_plot',
    'plot_predictions_with_ground_truth'
] 