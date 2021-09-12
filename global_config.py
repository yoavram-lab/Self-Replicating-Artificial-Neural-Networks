import tempfile
from pathlib import Path

_project_root = Path(__file__).parent.absolute()

config = {
    'workers_pool_port': 6379,
    'project_root': _project_root,
    'experiment_results_dir': str(Path(_project_root, 'data', 'experiment_results')),
    'synthetic_datasets_dir': str(Path(_project_root, 'data', 'synthetic_datasets')),
    'token_sequences_dir': str(Path(_project_root, 'data', 'token_sequences')),
    'vocabularies_dir': str(Path(_project_root, 'data', 'vocabularies')),
    'encodings_datasets_dir': str(Path(_project_root, 'data', 'encodings_datasets')),
    'evaluation_db': str(Path(_project_root, 'data', 'evaluation_db.sqlite')),
    'ribosomal_autoencoders_dir': str(Path(_project_root, 'models', 'ribosomal_autoencoder')),
    'serann_evaluations_dir': str(Path(_project_root, 'data', 'serann_evaluations')),
    'data_cache_dir': str(Path(tempfile.gettempdir()) / 'serann_cache'),
}