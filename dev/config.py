# config.py: parse the parameters from the config.yaml file
import yaml

def load_config(config_file):
    # config_file: the path of the config.yaml file
    # open the config file and load the content as a dictionary
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    # return the config dictionary
    return config
