import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import shutil
import warnings
from sklearn.metrics import roc_auc_score
from src.util.get_environment import get_datadir, get_exec_env, is_jupyter, is_gpu
from src.models.PurgedGroupTimeSeriesSplit import PurgedGroupTimeSeriesSplit
from src.util.calc_utility_score import utility_score_pd
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
                           help='Ignore cache')
    args = argparser.parse_args()
    return args


def get_option():
    # For production
    option = {
        'small': False,
        'predict': True,
        'nocache': True,
    }
    # for local develop
    if not is_jupyter():
        args = parse_args()
        option['small'] = args.small
        option['predict'] = args.predict
        option['nocache'] = args.nocache
    return option


def main():
    # Setup
    EXNO = '004'
    option = get_option()
    DATA_DIR = get_datadir()
    IN_DIR = f'{DATA_DIR}/002'
    XGB_PARAM = {
        'n_estimators': 500,
        'max_depth': 11,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.7,
        'missing': -999,
        'random_state': 2020
    }
    if option['small']:
        XGB_PARAM.update({'n_estimators': 2})
        XGB_PARAM.update({'max_depth': 2})
    if is_gpu():
        XGB_PARAM.update({'tree_method': 'gpu_hist'})

    assert(os.path.exists(IN_DIR))
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    train = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print(f'Input train shape: {train.shape}')
    features = [c for c in train.columns if 'feature' in c]

    # FE
    train = train.query('weight > 0').reset_index(drop=True)
    # train[features = train[features.fillna(method='ffill').fillna(0)
    train[features] = train[features].fillna(-999)
    train['action'] = (train['resp'] > 0).astype('int')
    # train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

    # Train
    scores = pd.DataFrame(index=[], columns=['fold', 'auc', 'utility_val', 'utility_pred'])

    n_splits = 3
    kf = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits,
        max_train_group_size=150,
        group_gap=20,
        max_test_group_size=60
    )

    for fold, (tr, te) in enumerate(kf.split(train['action'].values, train['action'].values, train['date'].values)):
        print(f'Starting Fold {fold}:')
        X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
        y_tr, y_val = train.loc[tr, 'action'].values, train.loc[te, 'action'].values
        model = xgb.XGBClassifier(**XGB_PARAM)
        model.fit(X_tr, y_tr,
                  eval_metric='logloss',
                  eval_set=[(X_tr, y_tr), (X_val, y_val)])
        val_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, val_pred)

        date = train.loc[te, 'date'].values
        weight = train.loc[te, 'weight'].values
        resp = train.loc[te, 'resp'].values
        action = train.loc[te, 'action'].values
        utility_val = utility_score_pd(date, weight, resp, action)
        utility_pred = utility_score_pd(date, weight, resp, val_pred)
        record = pd.Series([fold, auc, utility_val, utility_pred], index=scores.columns)
        scores = scores.append(record, ignore_index=True)
        print(f'Fold {fold} auc: {auc}, utility_val: {utility_val}, utility_pred: {utility_pred}')

        pickle.dump(model, open(f'{OUT_DIR}/model_{fold}.pkl', 'wb'))
        del model, val_pred, X_tr, X_val, y_tr, y_val

        scores.to_csv(f'{OUT_DIR}/scores.csv', index=False)

    # Predict
    if option['predict']:
        if get_exec_env() not in ['kaggle-Interactive', 'kaggle-Batch']:
            sys.path.append(f'{DATA_DIR}/001')
        import janestreet
        env = janestreet.make_env()  # initialize the environment
        iter_test = env.iter_test()  # an iterator which loops over the test set

        print('Start predicting')
        for (test_df, sample_prediction_df) in iter_test:
            X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            X_test.fillna(-999)
            X_test = X_test[features].values  # convert into np.ndarray

            y_pred = np.zeros(len(X_test))
            for i in range(n_splits):
                model = pd.read_pickle(open(f'{OUT_DIR}/model_{fold}.pkl', 'rb'))
                y_pred += model.predict(X_test) / n_splits
            y_pred = y_pred > 0
            sample_prediction_df.action = y_pred.astype(int)
            env.predict(sample_prediction_df)
        print('End predicting')

        # move submission file generated by env into experiment directory
        shutil.move('submission.csv', f'{OUT_DIR}/submission.csv')


if __name__ == '__main__':
    main()
