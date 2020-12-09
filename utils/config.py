import argparse
import datetime
import json
import os

from easydict import EasyDict

GRAPH_REP_SIZE = 200  # number of features in graph representation for SimGNN task

# TODO Intelligently pick values fo SimGNN; currently these are guesses
NUM_LABELS = {'COLLAB': 0, 'IMDBBINARY': 0, 'IMDBMULTI': 0, 'MUTAG': 7, 'NCI1': 37, 'NCI109': 38, 'PROTEINS': 3,
              'PTC': 22, 'DD': 89, 'QM9': 18, 'AIDS700nef': 29, 'LINUX': 0}
NUM_CLASSES = {'COLLAB': 3, 'IMDBBINARY': 2, 'IMDBMULTI': 3, 'MUTAG': 2, 'NCI1': 2, 'NCI109': 2, 'PROTEINS': 2,
               'PTC': 2, 'QM9': 12, 'AIDS700nef': GRAPH_REP_SIZE, 'LINUX': GRAPH_REP_SIZE}
LEARNING_RATES = {'COLLAB': 0.0001, 'IMDBBINARY': 0.00005, 'IMDBMULTI': 0.0001, 'MUTAG': 0.0001, 'NCI1': 0.0001,
                  'NCI109': 0.0001, 'PROTEINS': 0.001, 'PTC': 0.0001, 'AIDS700nef': 0.0001, 'LINUX': 0.0001}
DECAY_RATES = {'COLLAB': 0.5, 'IMDBBINARY': 0.5, 'IMDBMULTI': 0.75, 'MUTAG': 1.0, 'NCI1': 0.75, 'NCI109': 0.75,
               'PROTEINS': 0.5, 'PTC': 1.0, 'AIDS700nef': 0.75, 'LINUX': 0.75}
CHOSEN_EPOCH = {'COLLAB': 150, 'IMDBBINARY': 100, 'IMDBMULTI': 150, 'MUTAG': 500, 'NCI1': 200, 'NCI109': 250,
                'PROTEINS': 100, 'PTC': 400, 'AIDS700nef': 100, 'LINUX': 100}
TIME = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='./configs/SimGNN_config.json',
        help='The Configuration file')
    argparser.add_argument(
        '-d', '--dataset_name',
        metavar='D',
        default=None,
        help='The dataset name (overrides config file value)')
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = EasyDict(config_dict)

    return config


def process_config(args):
    config = get_config_from_json(args.config)
    if args.dataset_name is not None:
        config.dataset_name = args.dataset_name
    config.num_classes = NUM_CLASSES[config.dataset_name]
    if config.dataset_name == 'QM9' and config.target_param is not False:
        config.num_classes = 1
    config.node_labels = NUM_LABELS[config.dataset_name]
    config.timestamp = TIME
    config.parent_dir = config.exp_name + config.dataset_name + TIME
    config.summary_dir = os.path.join("../experiments", config.parent_dir, "summary/")
    config.checkpoint_dir = os.path.join("../experiments", config.parent_dir, "checkpoint/")
    if config.exp_name == "10fold_cross_validation":
        config.num_epochs = CHOSEN_EPOCH[config.dataset_name]
        config.hyperparams.learning_rate = LEARNING_RATES[config.dataset_name]
        config.hyperparams.decay_rate = DECAY_RATES[config.dataset_name]
    config.device = f'cuda:{config.gpu}'
    config.distributed_fold = None  # specific for distribute 10fold - override to use as a flag
    return config


if __name__ == '__main__':
    config = process_config('../configs/10fold_config.json')
    print(config.values())
