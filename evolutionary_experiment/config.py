from pathlib import Path
from global_config import config as global_config

config = global_config.copy()

config.update({
    'worker_pool_job_timeout': 1080,
    'max_serann_per_gpu': 112,

    'encodings_dataset_path': str(Path(global_config['encodings_datasets_dir'],
                                       'generated_27032020__sloppy-cornflower-dane_b69079.npz')),

    'vocabulary_path': str(Path(global_config['vocabularies_dir'], 'generated_27032020.csv')),
    'ribosomal_autoencoder_path': str(Path(global_config['ribosomal_autoencoders_dir'],
                                           'sloppy-cornflower-dane_b69079')),

    'random_seed': 79375,
})