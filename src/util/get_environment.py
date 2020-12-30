import os
import sys
import torch


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


def is_gpu() -> bool:
    return torch.cuda.is_available()


def is_jupyter() -> bool:
    if 'ipykernel' in sys.modules:
        # Kaggle Notebook interactive, Kaggle Notebook Batch, Kaggle script Interactive, Jupyter notebook
        return True
    else:
        # ipython, python script, Kaggle script Batch
        return False


def get_datadir() -> str:
    env = get_exec_env()
    if env in ['kaggle-Interactive', 'kaggle-Batch']:
        return '/kaggle/working/data'
    elif env == 'colab':
        return None
    elif env == 'local':
        return './data'
    else:
        raise ValueError


if __name__ == '__main__':
    print(f'Execution environment: {get_exec_env()}')
    print(f'Is Jupyter Notebook: {is_jupyter()}')
    print(f'DATA_DIR: {get_datadir()}')
