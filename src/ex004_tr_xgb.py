import os
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import shutil
import warnings
from sklearn.metrics import roc_auc_score
from src.util.get_environment import get_datadir, get_option, get_exec_env, is_gpu
from src.util.fast_fillna import fast_fillna
from src.models.PurgedGroupTimeSeriesSplit import PurgedGroupTimeSeriesSplit
# from src.util.calc_utility_score import utility_score_pd
warnings.filterwarnings("ignore")


def train_xgb(train, features, target, XGB_PARAM, n_splits, OUT_DIR):
    kf = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits,
        max_train_group_size=150,
        group_gap=20,
        max_test_group_size=60
    )
    # scores = pd.DataFrame(index=[], columns=['fold', 'auc', 'utility_val', 'utility_pred'])
    scores = pd.DataFrame(index=[], columns=['fold', 'auc'])
    for fold, (tr, te) in enumerate(kf.split(train[target].values, train[target].values, train['date'].values)):
        print(f'Starting Fold {fold}:')
        X_tr, X_val = train.loc[tr, features].values, train.loc[te, features].values
        y_tr, y_val = train.loc[tr, target].values, train.loc[te, target].values
        model = xgb.XGBClassifier(**XGB_PARAM)
        model.fit(X_tr, y_tr,
                  eval_metric='logloss',
                  eval_set=[(X_tr, y_tr), (X_val, y_val)])
        val_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, val_pred)

        '''
        date = train.loc[te, 'date'].values
        weight = train.loc[te, 'weight'].values
        resp = train.loc[te, 'resp'].values
        action = train.loc[te, 'action'].values
        utility_val = utility_score_pd(date, weight, resp, action)
        utility_pred = utility_score_pd(date, weight, resp, val_pred)
        record = pd.Series([fold, auc, utility_val, utility_pred], index=scores.columns)
        '''
        record = pd.Series([fold, auc], index=scores.columns)
        scores = scores.append(record, ignore_index=True)
        # print(f'Fold {fold} auc: {auc}, utility_val: {utility_val}, utility_pred: {utility_pred}')
        print(f'Fold {fold} auc: {auc}')

        pickle.dump(model, open(f'{OUT_DIR}/model_{fold}.pkl', 'wb'))
        del model, val_pred, X_tr, X_val, y_tr, y_val

    scores.to_csv(f'{OUT_DIR}/scores.csv', index=False)


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
    n_splits = 3

    assert(os.path.exists(IN_DIR))
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    # FE
    train = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print(f'Input train shape: {train.shape}')
    target = 'action'
    features = [c for c in train.columns if 'feature' in c]

    train = train.query('weight > 0').reset_index(drop=True)
    train[target] = (train['resp'] > 0).astype('int')

    # Fill missing values
    train[features] = train[features].fillna(method='ffill').fillna(0)

    # Train
    if not option['notrain']:
        train_xgb(train, features, target, XGB_PARAM, n_splits, OUT_DIR)

    # Predict
    if not option['nopredict']:
        if get_exec_env() not in ['kaggle-Interactive', 'kaggle-Batch']:
            sys.path.append(f'{DATA_DIR}/001')
        import janestreet
        env = janestreet.make_env()  # initialize the environment
        iter_test = env.iter_test()  # an iterator which loops over the test set

        models = []
        for i in range(n_splits):
            model = pd.read_pickle(open(f'{OUT_DIR}/model_{i}.pkl', 'rb'))
            models.append(model)

        print('Start predicting')
        time_start = time.time()
        '''
        Note: Be aware of the performance in the 'for' loop!
        Prediction API generates 1*130 DataFrame per iteration.
        Which means you need to process every record separately. (no vectorization)
        If the performance in 'for' loop is poor, it'll result in submission timeout.
        '''
        # Using high-performance nan forward-filling logic by Yirun Zhang
        tmp = np.zeros(len(features))  # this np.ndarray will contain last seen values for features
        for (test_df, sample_prediction_df) in iter_test:  # iter_test generates test_df(1*130)
            x_tt = test_df.loc[:, features].values  # this is 1*130 array([[values...]])
            x_tt[0, :] = fast_fillna(x_tt[0, :], tmp)  # use values in tmp to replace nan
            tmp = x_tt[0, :]  # save last seen values to tmp

            y_pred = 0.
            for model in models:
                y_pred += model.predict(x_tt) / n_splits
            y_pred = y_pred > 0
            sample_prediction_df.action = y_pred.astype(int)
            env.predict(sample_prediction_df)

        elapsed_time = time.time() - time_start
        print('End predicting')
        test_len = 15219  # length of test data (for developing API)
        print(f'Elapsed time: {elapsed_time}, Prediction speed: {test_len / elapsed_time}')

        # move submission file generated by env into experiment directory
        shutil.move('submission.csv', f'{OUT_DIR}/submission.csv')


if __name__ == '__main__':
    main()
