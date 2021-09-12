import numpy as np

layers = [
    'X', 'X_Dense', 'X_MaxPool2D', 'X_Conv2D', 'X_BatchNorm',
    'g', 'g_Dense', 'g_Conv1D', 'g_BatchNorm',
    'M_Concatenate', 'M_Dense', 'M_BatchNorm', 'outputs'
]

layer_transitions = [
#    X      X_Dense X_MaxPool2D X_Conv2D    X_BatchNorm     g       g_Dense     g_Conv1D    g_BatchNorm     concat  M_Dense   M_BatchNorm   outputs
    [0,     0.1,    0.1,        0.6,        0,              0,      0,          0,          0,              0.2,    0,        0,            0],   # X
    [0,     0.1,    0,          0.2,        0.2,            0,      0,          0,          0,              0.5,    0,        0,            0],   # X_Dense
    [0,     0.3,    0,          0.4,        0,              0,      0,          0,          0,              0.3,    0,        0,            0],   # X_MaxPool2D
    [0,     0.1,    0.4,        0.2,        0.2,            0,      0,          0,          0,              0.1,    0,        0,            0],   # X_Conv2D
    [0,     0.2,    0.1,        0,          0,              0,      0,          0,          0,              0.7,    0,        0,            0],   # X_BatchNorm
    [0,     0,      0,          0,          0,              0,      0.5,        0.4,        0,              0.2,    0,        0,            0],   # g
    [0,     0,      0,          0,          0,              0,      0.1,        0.2,        0.2,            0.5,    0,        0,            0],   # g_Dense
    [0,     0,      0,          0,          0,              0,      0.3,        0.2,        0.2,            0.3,    0,        0,            0],   # g_Conv1D
    [0,     0,      0,          0,          0,              0,      0.2,        0.15,       0,              0.65,   0,        0,            0],   # g_BatchNorm
    [0,     0,      0,          0,          0,              0,      0,          0,          0,              0,      1,        0,            0],   # Concatenate
    [0,     0,      0,          0,          0,              0,      0,          0,          0,              0,      0.2,      0.2,          0.6], # M_Dense
    [0,     0,      0,          0,          0,              0,      0,          0,          0,              0,      0.1,      0,            0.9]  # M_BatchNorm
]


def kernel_size_gen():
    x = np.clip(int(np.random.randn() * 1.3 + 6), 2, 10)
    return x + (x % 2) - 1

layer_args = {
    'X_Dense': {
        'units': lambda: np.clip(int(np.random.randn() * 8 + 64), 8, 128),
        'activation': [('relu', 0.8), ('sigmoid', 0.2)]
    },
    'X_MaxPool2D': {
        'pool_size': lambda: np.clip(int(np.random.randn() * 1.1 + 2), 2, 5),
    },
    'X_Conv2D': {
        'filters': lambda: 2 ** np.clip(int(np.random.randn() + 4.5), 0, 6),
        'kernel_size': kernel_size_gen,
        'strides': [(1, 0.8), (2, 0.2)]
    },
    'X_BatchNorm': {},
    'g_Dense': {
        'units': lambda: np.clip(int(np.random.randn() * 15 + 64), 8, 128),
        'activation': [('relu', 0.8), ('sigmoid', 0.2)]
    },
    'g_Conv1D': {
        'filters': lambda: 2 ** np.clip(int(np.random.randn() + 4.5), 0, 6),
        'kernel_size': kernel_size_gen,
        'strides': [(1, 0.8), (2, 0.2)]
    },
    'g_BatchNorm': {},
    'M_Dense': {
        'units': lambda: np.clip(int(np.random.randn() * 30 + 128), 32, 256),
        'activation': [('relu', 0.8), ('sigmoid', 0.2)]
    },
    'M_BatchNorm': {}
}

layer_templates = {
    'X_Dense': '{name} = Dense(units={units}, activation=\'{activation}\')({source})',
    'X_MaxPool2D': '{name} = MaxPool2D(pool_size={pool_size})({source})',
    'X_Conv2D': '{name} = Conv2D(filters={filters}, kernel_size={kernel_size}, strides={strides})({source})',
    'X_BatchNorm': '{name} = BatchNormalization()({source})',
    'g_Dense': '{name} = Dense(units={units}, activation=\'{activation}\')({source})',
    'g_Conv1D': '{name} = Conv1D(filters={filters}, kernel_size={kernel_size}, strides={strides})({source})',
    'g_BatchNorm': '{name} = BatchNormalization()({source})',
    'M_Concatenate': '{name} = concatenate([Reshape((1, -1))({source1}), Reshape((1, -1))({source2})])',
    'M_Dense': '{name} = Dense(units={units}, activation=\'{activation}\')({source})',
    'M_BatchNorm': '{name} = BatchNormalization()({source})'
}