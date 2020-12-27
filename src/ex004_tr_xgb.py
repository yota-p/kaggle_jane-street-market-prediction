import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import xgboost as xgb
import pickle
import shutil
import warnings
from util.get_environment import get_environment
warnings.filterwarnings("ignore")


def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('--small',
                           default=False,
                           action='store_true',
                           help='Use small data set for debug')
    argparser.add_argument('--predict',
                           default=False,
                           action='store_true',
                           help='Process predictions')
    argparser.add_argument('--nocache',
                           default=False,
                           action='store_true',
                           help='Do not use cache for run')
    args = argparser.parse_args()
    return args


def xgb_param(debug):
    if debug:
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
    return XGB_PARAM


def main():
    # Setup
    EXNO = '004'
    args = parse_args()
    ENV, DATA_DIR = get_environment()
    if args.small:
        print('Using small dataset & model')
        IN_DIR = f'{DATA_DIR}/003'  # small dataset
        XGB_PARAM = xgb_param(debug=True)
    else:
        IN_DIR = f'{DATA_DIR}/002'  # full dataset
        XGB_PARAM = xgb_param(debug=False)
    assert(os.path.exists(IN_DIR))
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    train = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print(f'Input train shape: {train.shape}')

    # FE
    # Is the data balanced or not?
    train = train[train['weight'] != 0]
    train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

    X_train = train.loc[:, train.columns.str.contains('feature')]
    y_train = train.loc[:, 'action']
    X_train = X_train.fillna(-999)
    print(f'X_train.shape: {X_train.shape}')
    print(f'y_train.shape: {y_train.shape}')
    del train

    # Train
    if os.path.exists(f'{OUT_DIR}/model.pkl') and not args.nocache:
        print('Using existing model')
        model = pickle.load(open(f'{OUT_DIR}/model.pkl', 'rb'))
    else:
        model = xgb.XGBClassifier(**XGB_PARAM)
        print('Start training')
        model.fit(X_train, y_train)
        print('End training')
        pickle.dump(model, open(f'{OUT_DIR}/model.pkl', 'wb'))

    # Predict
    if args.predict:
        if ENV not in ['kaggle-interactive', 'kaggle-batch']:
            sys.path.append(os.path.join(os.path.dirname(__file__), f'{DATA_DIR}/001'))
        import janestreet
        env = janestreet.make_env()  # initialize the environment
        iter_test = env.iter_test()  # an iterator which loops over the test set

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
    main()
