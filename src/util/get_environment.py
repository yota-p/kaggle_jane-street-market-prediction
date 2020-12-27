import os
import sys


def is_jupyter():
    if 'ipykernel' in sys.modules:
        # Jupyter notebook
        return True
    else:
        # ipython, python script, ...
        return False


def get_environment() -> (str, str):
    '''
    Function:
    - Detect running environment from environment variables and returns str:environment, str:data_dir.
    - environment:
        - 'kaggle-interactive'
        - 'kaggle-batch'
        - 'colab-notebook'
        - 'local-notebook'
        - 'local-batch'
    - data_dir:
        - '/kaggle/working'
        - '../data'
    '''
    kaggle_kernel_run_type = os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '')

    if kaggle_kernel_run_type == 'Interactive':
        # Notebook or Script
        return('kaggle-interactive', '/kaggle/working/data')
    elif kaggle_kernel_run_type == 'Batch':
        # Save version run
        return('kaggle-batch', '/kaggle/working/data')
    elif kaggle_kernel_run_type == '':
        pass
    else:
        raise Exception()

    if 'google.colab' in sys.modules:
        # TODO: determine data path for colab
        return ('colab-notebook', None)

    if is_jupyter():
        return('local-notebook', '../data')
    else:
        return('local-batch', '../data')
