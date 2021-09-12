from pathlib import Path
from global_config import config as global_config

config = global_config.copy()

config.update({
    'half_precision': True,
    'token_sequences_dataset_path': str(Path(global_config['token_sequences_dir'], 'generated_27032020.npz')),
    'vocabulary_path': str(Path(global_config['vocabularies_dir'], 'generated_27032020.csv')),
    'random_seed': 534213
})