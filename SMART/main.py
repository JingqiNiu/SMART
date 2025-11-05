import os
import yaml
import argparse

from ray_tune import tune_model
from train import train_model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='retina classification train.')
    parser.add_argument('module_path', type=str, default='config/uwf/img_infor/module_config.yaml',
                        help='the path to module_config.yaml')
    parser.add_argument('-tune', '--tune_path', required=False,
                        type=str, help='the path to tune_config.yaml')

    args = vars(parser.parse_args())
    with open(args['module_path'], 'r') as f:
        module_hparams = yaml.load(f, Loader=yaml.FullLoader)

    if args['tune_path']:
        with open(args['tune_path'], 'r') as f:
            tune_hparams = yaml.load(f, Loader=yaml.FullLoader)
        tune_model(tune_hparams, module_hparams)
    else:
        train_model(search_hp={}, module_hparams=module_hparams)
