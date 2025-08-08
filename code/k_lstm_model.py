import numpy as np
from scipy.stats import kurtosis
import tensorflow as tf
from tensorflow.keras import layers, models

def reshape_sequences(X, y, seq_len=30):
    n = X.shape[0] // seq_len
    X = X[:n*seq_len].reshape(n, seq_len, X.shape[1])
    y = y[:n*seq_len].reshape(n, seq_len)[:, -1]  # target at last step
    return X, y

def append_kurtosis_feature(X):
    # compute kurtosis per sequence and repeat across time axis as extra feature
    k = np.apply_along_axis(lambda a: kurtosis(a.flatten(), fisher=True), 1, X)
    k = k.reshape(-1, 1, 1)
    k_rep = np.repeat(k, X.shape[1], axis=1)
    return np.concatenate([X, k_rep], axis=-1)

def build_model(seq_len, feat_dim):
    m = models.Sequential([
        layers.LSTM(64, input_shape=(seq_len, feat_dim), return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m
