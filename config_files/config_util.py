import yaml
from yaml.loader import FullLoader


def read_config(config_file_path):
    config = yaml.load(open(config_file_path), Loader=FullLoader)

    # Make checks

    return config
