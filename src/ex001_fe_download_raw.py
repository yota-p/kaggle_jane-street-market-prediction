import os
from pathlib import Path
from util.get_environment import get_environment


def main():
    EXNO = '001'
    ENV, DATA_DIR = get_environment()
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    cmd = [f'kaggle competitions download -c jane-street-market-prediction -p {OUT_DIR}',
           f'unzip {OUT_DIR}/jane-street-market-prediction.zip -d {OUT_DIR}',
           f'rm {OUT_DIR}/jane-street-market-prediction.zip'
           ]
    for c in cmd:
        os.system(c)


if __name__ == '__main__':
    main()
