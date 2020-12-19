import os
import sys
import warnings
from argparse import ArgumentParser
import pandas as pd
import xgboost as xgb
import pickle
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '../data/001'))
import janestreet

warnings.filterwarnings("ignore")

env = janestreet.make_env()  # initialize the environment
iter_test = env.iter_test()  # an iterator which loops over the test set


def main(EXNO, IN_DIR, OUT_DIR, XGB_PARAM, need_pred):
    train = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print(f'Input train shape: {train.shape}')

    # Is the data balanced or not?
    train = train[train['weight'] != 0]
    train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

    X_train = train.loc[:, train.columns.str.contains('feature')]
    y_train = train.loc[:, 'action']
    X_train = X_train.fillna(-999)
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    del train

    # Training
    # The training part taked from here https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s
    model = xgb.XGBClassifier(**XGB_PARAM)

    print('Start training')
    model.fit(X_train, y_train)
    print('End training')
    pickle.dump(model, open(f'{OUT_DIR}/model.pkl', 'wb'))

    # Predict
    if need_pred:
        print('Start predicting')
        for (test_df, sample_prediction_df) in iter_test:
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test.fillna(-999)
            y_preds = model.predict(X_test)
            sample_prediction_df.action = y_preds
            env.predict(sample_prediction_df)
        print('End predicting')

        # move submission file generated by env into experiment directory
        shutil.move('submission.csv', f'{OUT_DIR}/submission.csv')


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--small',
                           default=False,
                           action='store_true',
                           help='Use small data set for debug')
    argparser.add_argument('--predict',
                           default=False,
                           action='store_true',
                           help='Process predictions')
    args = argparser.parse_args()
    if args.small:
        print('Using small dataset & model')
        IN_DIR = '../data/003'
        XGB_PARAM = {
            'n_estimators': 500,
            'max_depth': 11,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'missing': -999,
            'random_state': 2020
            # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
        }
    else:
        IN_DIR = '../data/002'  # full dataset
        XGB_PARAM = {
            'n_estimators': 20,
            'max_depth': 11,
            'learning_rate': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'missing': -999,
            'random_state': 2020
            # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
        }

    EXNO = '004'
    OUT_DIR = f'../data/{EXNO}'
    assert(os.path.exists(IN_DIR))
    assert(os.path.exists(OUT_DIR))

    main(EXNO, IN_DIR, OUT_DIR, XGB_PARAM, args.predict)
