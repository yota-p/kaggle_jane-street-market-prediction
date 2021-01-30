import os
import sys
import torch
import hydra
from argparse import ArgumentParser


def get_exec_env() -> (str, str):
    kaggle_kernel_run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')
    if kaggle_kernel_run_type == 'Interactive':
        return 'kaggle-Interactive'
    elif kaggle_kernel_run_type == 'Batch':
        return 'kaggle-Batch'
    elif 'google.colab' in sys.modules:
        return 'colab'
    else:
        return 'local'


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--small', '-s',
                           default=False,
                           action='store_true',
                           help='Use small data set for debug')
    argparser.add_argument('--nocache', '-nc',
                           default=False,
                           action='store_true',
                           help='Ignore caches')
    argparser.add_argument('--notrain', '-nt',
                           default=False,
                           action='store_true',
                           help='Skip training')
    argparser.add_argument('--nopredict', '-np',
                           default=False,
                           action='store_true',
                           help='Skip prediction')
    args = argparser.parse_args()
    return args


def parse_env_var_bool(ENV_VAR) -> bool:
    try:
        str_env = os.environ[ENV_VAR]
        if str_env == 'True':
            return True
        elif str_env == 'False':
            return False
        else:
            raise ValueError
    except KeyError:
        # if not specified in the Notebook, return default value: False
        return False


def get_option():
    # This function will get execution options
    # Default values for production: All False
    option = {
        'small': False,  # use small data/model/training
        'nocache': False,  # use cache
        'notrain': False,  # skip training
        'nopredict': False  # skip prediction
    }
    # To use dev options in ipykernel(approx. Jupyter NB),
    # you need to specify environment variables.
    if is_ipykernel():
        option['small'] = parse_env_var_bool('EXP_SMALL')
        option['nocache'] = parse_env_var_bool('EXP_NOCACHE')
        option['notrain'] = parse_env_var_bool('EXP_NOTRAIN')
        option['nopredict'] = parse_env_var_bool('EXP_NOPREDICT')
    # To use dev options in cli,
    # you need to specify commandline arguments.
    else:
        args = parse_args()
        option['small'] = args.small
        option['nocache'] = args.nocache
        option['notrain'] = args.notrain
        option['nopredict'] = args.nopredict
    return option


def is_gpu() -> bool:
    return torch.cuda.is_available()


def is_ipykernel() -> bool:
    if 'ipykernel' in sys.modules:
        # Kaggle Notebook interactive, Kaggle Notebook Batch, Kaggle script Interactive, Jupyter notebook
        return True
    else:
        # ipython, python script, Kaggle script Batch
        return False


def get_original_cwd() -> str:
    '''
    Returns original working directory for execution.
    In CLI, hydra changes cwd to outputs/xxx.
    In Jupyter Notebook, hydra doesn't change cwd.
    This is due to that you need to initialize & compose hydra config using compose API in Jupyter.
    Under compose API, hydra.core.hydra_config.HydraConfig is not initialized.
    Thus, in Jupyter Notebook, you need to avoid calling hydra.utils.get_original_cwd().
    Refer:
    https://github.com/facebookresearch/hydra/issues/828
    https://github.com/facebookresearch/hydra/blob/master/hydra/core/hydra_config.py
    https://github.com/facebookresearch/hydra/blob/master/hydra/utils.py
    '''
    if hydra.core.hydra_config.HydraConfig.initialized():
        return hydra.utils.get_original_cwd()
    else:
        return os.getcwd()


def get_datadir() -> str:
    '''
    Returns absolute path to data store dir
    TODO: Add config for colab
    '''
    env = get_exec_env()
    if env in ['kaggle-Interactive', 'kaggle-Batch']:
        return '/kaggle/working/data'
    elif env == 'colab':
        return None
    elif env == 'local':
        return get_original_cwd() + '/data'
    else:
        raise ValueError


if __name__ == '__main__':
    print(f'Execution environment: {get_exec_env()}')
    print(f'Get option: {get_option()}')
    print(f'Is Jupyter Notebook: {is_ipykernel()}')
    print(f'DATA_DIR: {get_datadir()}')
