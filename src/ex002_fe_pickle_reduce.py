# Machine requirement: Memory size > 16GB to execute this script
import os
import pandas as pd
from util.reduce_mem_usage import reduce_mem_usage
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


def main(EXNO, IN_DIR, OUT_DIR):
    '''
    Function:
    - Reduce size of train.csv from 2.5GB to 600MB (On memory)
      Note: Files except train.csv aren't reduced (small enough)
    Output:
    - train.pkl
    - example_sample_submission.pkl
    - example_test.pkl
    - features.pkl
    - train_dtypes.csv
    '''
    assert(not os.path.exists(f'{OUT_DIR}/train.pkl'))

    df = pd.read_csv(f'{IN_DIR}/train.csv')
    print(df.info())  # Size of the dataframe is about 2.5 GB
    dfnew = reduce_mem_usage(df)
    dfnew.memory_usage(deep=True)
    print(df.info())  # The dataframe size has decreased to 630MB (75% less).

    assert(len(df) == 2390491)

    # Save reduced data
    print(dfnew.dtypes)
    dfnew.dtypes.to_csv(f'{OUT_DIR}/train_dtypes.csv', header=False)
    dfnew.to_pickle(f'{OUT_DIR}/train.pkl')
    del df, dfnew

    for file in ['example_sample_submission', 'example_test', 'features']:
        df = pd.read_csv(f'{IN_DIR}/{file}.csv')
        df.to_pickle(f'{OUT_DIR}/{file}.pkl')


if __name__ == '__main__':
    EXNO = '002'
    IN_DIR = '../data/001'
    OUT_DIR = f'../data/{EXNO}'
    assert(os.path.exists(IN_DIR))
    assert(os.path.exists(OUT_DIR))
    main(EXNO, IN_DIR, OUT_DIR)
