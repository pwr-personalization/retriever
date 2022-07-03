from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
DATA_PATH = PROJECT_PATH / 'data'
PARAMS_FILE = PROJECT_PATH / 'params.yaml'

DATASETS_PATH = DATA_PATH / 'datasets'
MODEL_DIR = DATA_PATH / 'models'
CHECKPOINTS_PATH = DATA_PATH / 'checkpoints'
EMBEDDINGS_PATH = DATA_PATH / 'embeddings'
LOGS_PATH = DATA_PATH / 'logs'
METRICS_PATH = DATA_PATH / 'metrics'
FIGURE_DIR = DATA_PATH / 'figures'
SENTILARE_DIR = DATA_PATH / 'sentiLARE'
WANDB_EXPORTS_DIR = DATA_PATH / 'wandb_exports'
