import os
from pathlib import Path
import shutil
import glob
from util.get_environment import get_datadir, get_exec_env


def main():
    EXNO = '001'
    DATA_DIR = get_datadir()
    ENV = get_exec_env()
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    if ENV in ['kaggle-Interactive', 'kaggle-Batch']:
        for f in glob.glob('/kaggle/input/jane-street-market-prediction/*.csv'):
            shutil.copy2(f, OUT_DIR)
    else:
        cmd = [f'kaggle competitions download -c jane-street-market-prediction -p {OUT_DIR}',
               f'unzip {OUT_DIR}/jane-street-market-prediction.zip -d {OUT_DIR}',
               f'rm {OUT_DIR}/jane-street-market-prediction.zip'
               ]
        for c in cmd:
            os.system(c)


if __name__ == '__main__':
    main()
