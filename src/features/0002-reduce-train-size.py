# Machine requirement: Memory size > 16GB to execute this script
import os
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)

NB = '0002'
IN_DIR = '../../data/raw'
OUT_DIR = f'../../data/processed/{NB}'
assert(os.path.exists(IN_DIR))
assert(os.path.exists(OUT_DIR))


def reduce_mem_usage(df):
    start_mem = compute_df_total_mem(df)
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = compute_df_total_mem(df)
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def compute_df_total_mem(df):
    """Returns a dataframe's total memory usage in MB."""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def main():
    '''
    Function:
    - Reduce size of train.csv from 2.5GB to 600MB (On memory)
    Output:
    - train_dtypes.csv
    - train.pkl
    '''
    # load data
    df = pd.read_csv(f'{IN_DIR}/train.csv')
    df.info()  # Size of the dataframe is about 2.5 GB

    # reduce size
    dfnew = reduce_mem_usage(df)
    dfnew.memory_usage(deep=True)

    df.info()  # The dataframe size has decreased to 630MB (75% less).

    # Save reduced data
    print(dfnew.dtypes)
    dfnew.dtypes.to_csv(f'{OUT_DIR}/train_dtypes.csv', header=False)
    dfnew.to_pickle(f'{OUT_DIR}/train.pkl')


if __name__ == '__main__':
    main()
