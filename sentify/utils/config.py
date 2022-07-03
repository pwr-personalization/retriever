import yaml

from sentify import PARAMS_FILE


def load_config(*stage_name):
    with PARAMS_FILE.open('r') as f:
        config = yaml.safe_load(f)
        for name in stage_name:
            config = config[name]
        return config
