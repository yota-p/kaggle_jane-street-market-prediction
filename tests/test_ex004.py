from pathlib import Path
from src import ex004_tr_xgb
import pandas as pd


class TestEx004:
    def test_case1(self, mocker, tmpdir):
        EXNO = '004'
        option = {
            'small': False,
            'predict': False,
            'nocache': True,
            'gpu': False
        }
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/002'
        OUT_DIR = f'{DATA_DIR}/{EXNO}'
        Path(IN_DIR).mkdir(exist_ok=True, parents=True)
        Path(OUT_DIR).mkdir(exist_ok=True, parents=True)
        ENV = 'local'

        # create input
        df = pd.DataFrame({'weight': [1, 2, 3], 'resp': [10, 20, 30],
                           'feature1': [100, 200, 300], 'feature2': [1000, 2000, 3000]})
        df.to_pickle(f'{IN_DIR}/train.pkl')

        mocker.patch('src.ex004_tr_xgb.get_datadir', return_value=DATA_DIR)
        mocker.patch('src.ex004_tr_xgb.get_exec_env', return_value=ENV)
        mocker.patch('src.ex004_tr_xgb.get_option', return_value=option)
        ex004_tr_xgb.main()

        # Check if this runs till end
        assert True
