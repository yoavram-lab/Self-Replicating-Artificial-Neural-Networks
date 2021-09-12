import argparse
import hashlib
import os
from pathlib import Path

from global_config import config

# SeRANN are generated using multiple process, so the GPU cannot be used.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
import numpy as np
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Conv1D, concatenate, Flatten, Input, Reshape, Lambda, \
    BatchNormalization

from synthetic_serann_generator.layer_transitions import layer_transitions, layers, layer_templates, layer_args


def _sample(values, probabilities):
    order = np.argsort(probabilities)
    values, probabilities = np.array(values)[order], np.array(probabilities)[order]
    cumprobs = np.cumsum(probabilities)
    sampled_index = np.where(cumprobs > np.random.rand())[0][0]
    return values[sampled_index]


def get_num_of_parameters(hidden_layers, last_layer):

    m, n, k, l = 28, 28, 350, 10

    X_input = X_layer = Input(shape=(m, n, 1))
    g_input = g_layer = Input(shape=(k, 1))

    exec(hidden_layers)

    y_hat = Dense(l, activation='softmax')(locals()[last_layer])
    g_rep = Dense(k, activation='linear')(locals()[last_layer])

    model = Model(inputs=[X_input, g_input], outputs=[y_hat, g_rep])

    params_count = model.count_params()
    keras.backend.clear_session()
    return params_count


def eval_transitions(current_layer_type, current_layer_name, stop_layer):

    code_lines = []

    while True:

        current_layer_type = _sample(layers, layer_transitions[layers.index(current_layer_type)])

        if current_layer_type == stop_layer:
            break

        args = {'source': current_layer_name, 'name': current_layer_name}

        for key, arg in layer_args.get(current_layer_type, {}).items():
            args[key] = _sample(*zip(*arg)) if type(arg) == list else arg()

        code_lines.append(layer_templates[current_layer_type].format(**args))

    return '\n'.join(code_lines), current_layer_name


def generate_network(_):

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    x_branch, x_last = eval_transitions('X', 'X_layer', 'M_Concatenate')
    g_branch, g_last = eval_transitions('g', 'g_layer', 'M_Concatenate')
    concat_layer = layer_templates['M_Concatenate'].format(name='con', scope_name='con', source1=x_last, source2=g_last)
    merged, last_layer = eval_transitions('M_Concatenate', 'con', 'outputs')

    net = '\n\n'.join([x_branch, g_branch, concat_layer, merged])

    net_hash = hashlib.md5(net.encode()).hexdigest()

    loss_balance = np.random.rand()
    net += '\n\nloss_balance = {:.4f}'.format(loss_balance)

    try:
        params_count = get_num_of_parameters(net, last_layer)
    except ValueError as e:
        return None

    x_layers = len(x_branch.split('\n'))
    g_layers = len(g_branch.split('\n'))
    m_layers = len(merged.split('\n'))

    return {'code': net, 'parameters_count': params_count, 'last_layer': last_layer, 'net_hash': net_hash,
            'x_layers': x_layers, 'g_layers': g_layers, 'm_layers': m_layers, 'loss_balance': loss_balance}


def generate(output_file, nets=1000000):

    df = pd.DataFrame()

    # multiprocessing.Pool is only used here as a convenient way to start multiple process.
    with mp.Pool() as p:

        while len(df) < nets:

            results = []

            for result in tqdm(p.imap_unordered(generate_network, range(nets - len(df))), total=(nets - len(df)),
                               leave=False):

                if result is not None:
                    results.append(result)

            prev_len = len(df)
            df = df.append(pd.DataFrame(results)).drop_duplicates('net_hash')

            if len(df) != prev_len:
                print(f'{len(df) - prev_len} unique nets were added. {len(df)} generated in total.')

    df.to_csv(output_file, index=False, compression='zip')


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset-name", required=True, help='Output dataset name')
    parser.add_argument("-n", "--num-of-examples", default=10000, type=int, help='Number of examples to generate')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    output_path = Path(config['synthetic_datasets_dir']) / (args.dataset_name + '.csv')
    generate(output_path, args.num_of_examples)
