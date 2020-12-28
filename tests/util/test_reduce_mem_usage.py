import pandas as pd
import numpy as np
from src.util.reduce_mem_usage import reduce_mem_usage

'''
Ref:
- https://qiita.com/siruku6/items/27dd9fb84aa84186eda5
Note:
- Mixing None, np.nan, 0.0 in integer columns will result in float.
- Float dtypes can't contain much digits (look below).
'''


class TestReduceMemoryUsage:
    def test_int8(self):
        int8min = np.iinfo(np.int8).min
        int8max = np.iinfo(np.int8).max
        df = pd.DataFrame({'col1': [int8min, int8max-1],
                           'col2': [int8min+1, int8max-1],
                           'col3': [int8min+1, int8max]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'int16')
        assert(df_out.col2.dtype == 'int8')
        assert(df_out.col3.dtype == 'int16')

    def test_int16(self):
        int16min = np.iinfo(np.int16).min
        int16max = np.iinfo(np.int16).max
        df = pd.DataFrame({'col1': [int16min, int16max-1],
                           'col2': [int16min+1, int16max-1],
                           'col3': [int16min+1, int16max]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'int32')
        assert(df_out.col2.dtype == 'int16')
        assert(df_out.col3.dtype == 'int32')

    def test_int32(self):
        int32min = np.iinfo(np.int32).min
        int32max = np.iinfo(np.int32).max
        df = pd.DataFrame({'col1': [int32min, int32max-1],
                           'col2': [int32min+1, int32max-1],
                           'col3': [int32min+1, int32max]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'int64')
        assert(df_out.col2.dtype == 'int32')
        assert(df_out.col3.dtype == 'int64')

    def test_int64(self):
        int64min = np.iinfo(np.int64).min
        int64max = np.iinfo(np.int64).max
        df = pd.DataFrame({'col1': [int64min, int64max-1],
                           'col2': [int64min+1, int64max-1],
                           'col3': [int64min+1, int64max]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'int64')
        assert(df_out.col2.dtype == 'int64')
        assert(df_out.col3.dtype == 'int64')

    def test_float16(self):
        # float16 keeps 3 digits
        float16min = np.finfo(np.float16).min
        float16max = np.finfo(np.float16).max
        df = pd.DataFrame({'col1': [float16min, float16max*(1-1e-3), np.nan],
                           'col2': [float16min*(1-1e-3), float16max*(1-1e-3), np.nan],
                           'col3': [float16min*(1-1e-3), float16max, np.nan]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'float32')
        assert(df_out.col2.dtype == 'float16')
        assert(df_out.col3.dtype == 'float32')

    def test_float32(self):
        # float32 keeps 6 digits
        float32min = np.finfo(np.float32).min
        float32max = np.finfo(np.float32).max
        df = pd.DataFrame({'col1': [float32min, float32max*(1-1e-6), np.nan],
                           'col2': [float32min*(1-1e-6), float32max*(1-1e-6), np.nan],
                           'col3': [float32min*(1-1e-6), float32max, np.nan]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'float64')
        assert(df_out.col2.dtype == 'float32')
        assert(df_out.col3.dtype == 'float64')

    def test_float64(self):
        # float64 keeps 15 digits
        float64min = np.finfo(np.float64).min
        float64max = np.finfo(np.float64).max
        df = pd.DataFrame({'col1': [float64min, float64max*(1-1e-15), np.nan],
                           'col2': [float64min*(1-1e-15), float64max*(1-1e-15), np.nan],
                           'col3': [float64min*(1-1e-15), float64max, np.nan]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'float64')
        assert(df_out.col2.dtype == 'float64')
        assert(df_out.col3.dtype == 'float64')

    def test_category(self):
        df = pd.DataFrame({'col1': ['A', None]})
        df_out = reduce_mem_usage(df)
        assert(df_out.col1.dtype == 'category')
