"""ML helpers: synthetic dataset generation and simple Keras models.

This module provides small utilities to generate simulated datasets and a
compact dense Keras model for parameter estimation. Note that some functions
rely on external variables/functions (``L``, ``STAR_SIGNALS``,
``kn_fields_jit``, etc.). They are not executed during documentation builds.
"""

from typing import Any

import numpy as np
import matplotlib.pyplot as plt

try:  # TensorFlow est optionnel pour la génération de docs
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover - non essentiel pour la doc
    tf = None  # type: ignore

# Placeholders pour dépendances externes non strictement requises à l'import
try:
    kn_fields_jit  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    kn_fields_jit = None  # type: ignore
try:
    STAR_SIGNALS  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    STAR_SIGNALS = None  # type: ignore
try:
    L  # type: ignore[name-defined]
except NameError:  # pragma: no cover
    L = None  # type: ignore

def parameter_grid(N, D, a, b):
    """Generate a regular grid [a, b]^D with N steps per axis.

    Args:
        N: Resolution per axis (points per dimension).
        D: Parameter space dimensionality.
        a: Lower bound.
        b: Upper bound.

    Returns:
        np.ndarray of shape (N**D, D) listing all grid points.
    """
    return np.array([a + (b - a) * (x // N ** np.arange(D, dtype=float) % N / N) for x in range(N ** D)])

def parameter_basis(D, b=1):
    """Canonical basis augmented with the zero vector in R^D.

    Args:
        D: Dimension.
        b: Norm of each basis vector (default 1).

    Returns:
        np.ndarray of shape (D+1, D).
    """
    vectors = np.zeros((D + 1, D))
    for i in range(D):
        vectors[i + 1, i] = b
    return vectors

def parameter_basis_2p(D, b=1):
    """Two-point basis: {0, b·e_i, 2b·e_i} for i=1..D.

    Args:
        D: Dimension.
        b: Basis step.

    Returns:
        np.ndarray of shape (2D+1, D).
    """
    vectors = np.zeros((2 * D + 1, D))
    for i in range(D):
        vectors[2 * i + 1, i] = b
        vectors[2 * i + 2, i] = 2 * b
    return vectors

def get_dataset(size=1000):
    """Build a structured synthetic dataset.

    Note:
        Depends on external objects (e.g. ``L``, ``STAR_SIGNALS``, ``kn_fields_jit``).

    Args:
        size: Number of samples to generate.

    Returns:
        np.ndarray of shape (size, vector_len).
    """
    grid_points = parameter_basis_2p(14, 1.65 / 3)
    vector_len = len(grid_points) * 7 + 14
    dataset = np.empty((size, vector_len))
    for v in range(size):
        shifts_total_opd = np.random.uniform(0, 1, 14) * L / 10
        vector = np.empty(vector_len)
        for (p, point) in enumerate(grid_points):
            (_, darks, bright) = kn_fields_jit(beams=STAR_SIGNALS, shifts=point, shifts_total_opd=shifts_total_opd)
            vector[p * 7:p * 7 + 6] = np.abs(darks) ** 2
            vector[p * 7 + 6] = np.abs(bright) ** 2
        vector[-14:] = shifts_total_opd
        dataset[v] = vector
    return dataset

def get_random_dataset(size=1000):
    """Build a random-point synthetic dataset.

    Args:
        size: Number of samples to generate.

    Returns:
        np.ndarray of shape (size, vector_len).
    """
    nb_points = 100
    i_len = 7 + 14
    o_len = 14
    vector_len = nb_points * i_len + o_len
    dataset = np.empty((size, vector_len))
    pv = 0
    for v in range(size):
        if (nv := (v * 100 // size)) > pv:
            print(nv, '%', end='\r')
            pv = nv
        shifts_total_opd = np.random.uniform(0, 1, 14) * L / 10
        vector = np.empty(vector_len)
        for p in range(nb_points):
            shifts = np.random.uniform(0, L.value, size=14)
            (_, darks, bright) = kn_fields_jit(beams=STAR_SIGNALS, shifts=shifts, shifts_total_opd=shifts_total_opd)
            vector[p * i_len:(p + 1) * i_len] = np.concatenate([shifts, np.abs(darks) ** 2, [np.abs(bright) ** 2]])
        vector[-14:] = shifts_total_opd
        dataset[v] = vector
    return dataset

def get_model(input_shape):
    """Create a small dense Keras network for 14 output parameters.

    Args:
        input_shape: Input vector length.

    Returns:
        Compiled tf.keras.Model (Adam optimizer, MSE loss).
    """
    i = tf.keras.Input(shape=(input_shape,), name='Input')
    x = tf.keras.layers.Dense(128, activation='relu', name='Dense_1')(i)
    x = tf.keras.layers.Dense(64, activation='relu', name='Dense_2')(x)
    x = tf.keras.layers.Dense(32, activation='relu', name='Dense_3')(x)
    o = tf.keras.layers.Dense(14, activation='relu', name='Output')(x)
    model = tf.keras.Model(inputs=i, outputs=o)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-05)
    model.compile(optimizer=optimizer, loss='mse')
    return model

def train_model(model, dataset, plot=True):
    """Train the model and optionally plot the loss curves.

    Args:
        model: Compiled Keras model.
        dataset: Data with targets in the last 14 columns.
        plot: If True, plots loss and val_loss (log scale).

    Returns:
        tf.keras.callbacks.History
    """
    X = dataset[:, :-14]
    Y = dataset[:, -14:]
    print(dataset.shape, X.shape, Y.shape)
    history = model.fit(X, Y, epochs=10, batch_size=5, validation_split=0.2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log')
    plt.legend()
    plt.show()
    return history

def test_model(model, dataset):
    """Plot ground-truth targets vs predictions for a quick visual check.

    Args:
        model: Trained Keras model.
        dataset: Test data; uses 10 samples.

    Returns:
        None
    """
    TEST_SET = get_dataset(size=10)
    X = TEST_SET[:, :-14]
    Y = TEST_SET[:, -14:]
    PREDICTIONS = model.predict(X)
    print(X)
    print(PREDICTIONS)
    cpt = 0
    for i in range(10):
        for j in range(len(Y[i])):
            plt.scatter(Y[i][j], PREDICTIONS[i][j])
            cpt += 1
    plt.xlabel('Expectations')
    plt.ylabel('Preditions')
    plt.show()