"""Module generated docstring."""

def parameter_grid(N, D, a, b):
    """
    Generate a list of points forming a grid in a parameter space.

    Parameters
    ----------
    - N: Resolution of the grid (point per axes)
    - D: Dimension of the space parameter
    - a: Minimum value of the space parameter
    - b: Maximum value of the space parameter

    Returns
    -------
    - An array of vectors describing a point in the parameter space
    """
    return np.array([a + (b - a) * (x // N ** np.arange(D, dtype=float) % N / N) for x in range(N ** D)])

def parameter_basis(D, b=1):
    """
    Return the basis vectors of the parameter space (+ the null vector)

    Parameters
    ----------
    - D: Dimension of the space parameter
    - b: Norm of the basis vector (default=1)

    Returns
    -------
    - An array of vectors describing a point in the parameter space
    """
    vectors = np.zeros((D + 1, D))
    for i in range(D):
        vectors[i + 1, i] = b
    return vectors

def parameter_basis_2p(D, b=1):
    """
    Return the basis vectors of the parameter space (+ the null vector)

    Parameters
    ----------
    - D: Dimension of the space parameter
    - b: Norm of the basis vector (default=1)

    Returns
    -------
    - An array of vectors describing a point in the parameter space
    """
    vectors = np.zeros((2 * D + 1, D))
    for i in range(D):
        vectors[2 * i + 1, i] = b
        vectors[2 * i + 2, i] = 2 * b
    return vectors

def get_dataset(size=1000):
    """"get_dataset.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    grid_points = parameter_basis_2p(14, 1.65 / 3)
    vector_len = len(grid_points) * 7 + 14
    dataset = np.empty((size, vector_len))
    for v in range(size):
        shifts_total_opd = np.random.uniform(0, 1, 14) * L / 10
        vector = np.empty(vector_len)
        for (p, point) in enumerate(grid_points):
            (_, darks, bright) = kn_fields_njit(beams=STAR_SIGNALS, shifts=point, shifts_total_opd=shifts_total_opd)
            vector[p * 7:p * 7 + 6] = np.abs(darks) ** 2
            vector[p * 7 + 6] = np.abs(bright) ** 2
        vector[-14:] = shifts_total_opd
        dataset[v] = vector
    return dataset

def get_random_dataset(size=1000):
    """"get_random_dataset.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
            (_, darks, bright) = kn_fields_njit(beams=STAR_SIGNALS, shifts=shifts, shifts_total_opd=shifts_total_opd)
            vector[p * i_len:(p + 1) * i_len] = np.concatenate([shifts, np.abs(darks) ** 2, [np.abs(bright) ** 2]])
        vector[-14:] = shifts_total_opd
        dataset[v] = vector
    return dataset

def get_model(input_shape):
    """"get_model.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
    """"train_model.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
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
    """"test_model.

Parameters
----------
(Automatically added placeholder.)

Returns
-------
(Automatically added placeholder.)
"""
    TEST_SET = get_dataset(size=10)
    X = TEST_SET[:, :-14]
    Y = TEST_SET[:, -14:]
    PREDICTIONS = MODEL.predict(X)
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