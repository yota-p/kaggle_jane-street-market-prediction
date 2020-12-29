import pandas as pd
import numpy as np
import hashlib
from pathlib import Path
from src import ex003_fe_pickle_small


class TestEx003:
    def test_ex003(self, mocker, tmpdir):
        EXNO = '003'
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/002'
        OUT_DIR = f'{DATA_DIR}/{EXNO}'
        Path(OUT_DIR).mkdir(exist_ok=True)

        # create input
        Path(IN_DIR).mkdir(exist_ok=True)
        df_in = pd.DataFrame({'key': [1, 2, 3, 4], 'date': [np.int16(0), np.int16(10), np.int16(100), np.int16(101)]})
        df_in.to_pickle(f'{IN_DIR}/train.pkl')
        for file in ['example_sample_submission', 'example_test', 'features']:
            f = open(f'{IN_DIR}/{file}.pkl', 'wb')
            f.write(b'This is {file}')

        mocker.patch('src.ex003_fe_pickle_small.get_datadir', return_value=DATA_DIR)
        ex003_fe_pickle_small.main()

        df_actual = pd.read_pickle(f'{OUT_DIR}/train.pkl')
        df_expected = df_in[[True, False, True, False]]

        assert df_actual.equals(df_expected)

        # assert copied files
        for file in ['example_sample_submission', 'example_test', 'features']:
            with open(f'{OUT_DIR}/{file}.pkl', 'rb') as f:
                in_hash = hashlib.sha256(f.read()).hexdigest()
            with open(f'{OUT_DIR}/{file}.pkl', 'rb') as f:
                out_hash = hashlib.sha256(f.read()).hexdigest()
            assert(in_hash == out_hash)
