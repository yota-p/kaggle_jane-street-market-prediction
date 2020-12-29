import pandas as pd
import numpy as np
import os
from pathlib import Path
from src import ex002_fe_pickle_reduce


class TestEx002:
    def test_reduce(self, mocker, tmpdir):
        EXNO = '002'
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/001'
        OUT_DIR = f'{DATA_DIR}/{EXNO}'
        Path(IN_DIR).mkdir(exist_ok=True)
        Path(OUT_DIR).mkdir(exist_ok=True)

        # create input
        # Declare dtype with larger than needed
        df_in = pd.DataFrame({'i8': np.array([-127, 0, 126]).astype('int16'),
                              'i16': np.array([-32767, 0, 32766]).astype('int32'),
                              'i32': np.array([-2147483647, 0, 2147483646]).astype('int64'),
                              'f16': np.array([-65400, 0, 65400]).astype('float32'),
                              'f32': np.array([-3.4028100e+38, 0, 3.4028100e+38]).astype('float64')})
        df_in.to_csv(f'{IN_DIR}/train.csv')

        pklfiles = ['example_sample_submission', 'example_test', 'features']
        for file in pklfiles:
            data = pd.DataFrame({'filename': [file]})
            data.to_csv(f'{IN_DIR}/{file}.csv')

        # call target
        mocker.patch('src.ex002_fe_pickle_reduce.get_datadir', return_value=DATA_DIR)
        ex002_fe_pickle_reduce.main()

        # assert memory size
        df_out = pd.read_pickle(f'{OUT_DIR}/train.pkl')
        assert df_out.memory_usage(deep=True).sum() <= df_in.memory_usage(deep=True).sum()

        # assert csv-> pickled files
        for file in pklfiles:
            assert(os.path.exists(f'{OUT_DIR}/{file}.pkl'))

        # assert train_dtypes csv exist
        assert(os.path.exists(f'{OUT_DIR}/train_dtypes.csv'))

    def test_noreduce(self, mocker, tmpdir):
        EXNO = '002'
        DATA_DIR = str(tmpdir)
        IN_DIR = f'{DATA_DIR}/001'
        OUT_DIR = f'{DATA_DIR}/{EXNO}'
        Path(IN_DIR).mkdir(exist_ok=True)
        Path(OUT_DIR).mkdir(exist_ok=True)

        # create input
        pklfiles = ['train', 'example_sample_submission', 'example_test', 'features']
        for file in pklfiles:
            data = pd.DataFrame({'filename': [file]})
            data.to_csv(f'{IN_DIR}/{file}.csv')

        # call target
        mocker.patch('src.ex002_fe_pickle_reduce.get_datadir', return_value=DATA_DIR)
        ex002_fe_pickle_reduce.main()

        # assert csv-> pickled files
        for file in pklfiles:
            assert(os.path.exists(f'{OUT_DIR}/{file}.pkl'))
