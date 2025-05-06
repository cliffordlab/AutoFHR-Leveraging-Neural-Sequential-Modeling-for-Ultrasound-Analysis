import tensorflow as tf

def introduced_loss(y_true, y_pred, penalty_weight=0.000006):
    """
    Custom loss function combining binary cross-entropy with a spectral mismatch penalty
    to encourage rhythmic pattern generation.
    
    Parameters:
    -----------
    y_true : tensor
        Ground truth signal
    y_pred : tensor
        Estimated signal
    penalty_weight : float
        Weight for the penalty component
        
    Returns:
    --------
    tensor: Computed loss value
    """
    y_true = tf.cast(y_true, tf.float64)
    # Ensure numerical stability by clipping prediction values
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Binary Cross-entropy Component
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Periodicity Penalty Component
    def autocorrelation(signal):
        """
        Compute autocorrelation of a signal
        
        Parameters:
        -----------
        signal : tensor
            Input signal to compute autocorrelation for
            
        Returns:
        --------
        tensor: Autocorrelation of the signal
        """
        signal_mean = tf.reduce_mean(signal)
        signal_centered = signal - signal_mean
        autocorr = tf.nn.conv1d(
            tf.expand_dims(signal_centered, axis=0),
            tf.expand_dims(tf.reverse(signal_centered, axis=[0]), axis=-1),
            stride=1,
            padding='SAME'
        )
        return tf.squeeze(autocorr)
    
    true_autocorr = autocorrelation(y_true)
    pred_autocorr = autocorrelation(y_pred)
    
    # Compute the difference between true and predicted autocorrelations
    periodicity_penalty = tf.reduce_mean(tf.square(true_autocorr - pred_autocorr))
    
    # Combine the BCE loss with the periodicity penalty
    return tf.reduce_mean(bce) + penalty_weight * periodicity_penalty 