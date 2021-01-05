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


def create_janeapi():
    DATA_DIR = get_datadir()
    if get_exec_env() not in ['kaggle-Interactive', 'kaggle-Batch']:
        sys.path.append(f'{DATA_DIR}/001')
    import janestreet
    env = janestreet.make_env()  # initialize the environment
    iter_test = env.iter_test()  # an iterator which loops over the test set
    return env, iter_test


def predict_fillna_forward(models, features, target, OUT_DIR):
    '''
    Using high-performance nan forward-filling logic by Yirun Zhang
    Note: Be aware of the performance in the 'for' loop!
    Prediction API generates 1*130 DataFrame per iteration.
    Which means you need to process every record separately. (no vectorization)
    If the performance in 'for' loop is poor, it'll result in submission timeout.
    '''
    env, iter_test = create_janeapi()
    print('Start predicting')
    time_start = time.time()
    tmp = np.zeros(len(features))  # this np.ndarray will contain last seen values for features
    for (test_df, sample_prediction_df) in iter_test:  # iter_test generates test_df(1,130)
        x_tt = test_df.loc[:, features].values  # this is (1,130) ndarray([[values...]])
        x_tt[0, :] = fast_fillna(x_tt[0, :], tmp)  # use values in tmp to replace nan
        tmp = x_tt[0, :]  # save last seen values to tmp
        y_pred = 0.
        for model in models:
            y_pred += model.predict(x_tt) / len(models)
        y_pred = y_pred > 0
        sample_prediction_df[target] = y_pred.astype(int)
        env.predict(sample_prediction_df)

    elapsed_time = time.time() - time_start
    test_len = 15219  # length of test data (for developing API)
    print('End predicting')
    print(f'Prediction time: {elapsed_time} s, Prediction speed: {test_len / elapsed_time} iter/s')

    # move submission file generated by env into experiment directory
    shutil.move('submission.csv', f'{OUT_DIR}/submission.csv')


def predict_fillna_999(models, features, target, OUT_DIR):
    env, iter_test = create_janeapi()

    print('Start predicting')
    time_start = time.time()
    for (test_df, sample_prediction_df) in iter_test:
        X_test = test_df.loc[:, features]
        X_test.fillna(-999)

        y_pred = 0.
        for model in models:
            y_pred += model.predict(X_test.values) / len(models)
        y_pred = y_pred > 0
        sample_prediction_df[target] = y_pred.astype(int)
        env.predict(sample_prediction_df)

    elapsed_time = time.time() - time_start
    test_len = 15219  # length of test data (for developing API)
    print('End predicting')
    print(f'Prediction time: {elapsed_time} s, Prediction speed: {test_len / elapsed_time} iter/s')

    # move submission file generated by env into experiment directory
    shutil.move('submission.csv', f'{OUT_DIR}/submission.csv')


def train_xgb_cv(train, features, target, XGB_PARAM, n_splits, OUT_DIR):
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


def train_xgb(train, features, target, XGB_PARAM, OUT_DIR):
    print('Start training')
    X_train = train.loc[:, features].values
    y_train = train.loc[:, target].values
    model = xgb.XGBClassifier(**XGB_PARAM)
    model.fit(X_train, y_train)
    pickle.dump(model, open(f'{OUT_DIR}/model_0.pkl', 'wb'))
    pd.DataFrame(features).to_csv(f'{OUT_DIR}/features.csv', header=False)
    print('End training')


def main():
    # Setup
    EXNO = '005'
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
    cv = None
    # cv = 'PurgedGroupTimeSeriesSplit'
    method_fillna = '-999'
    # method_fillna = 'forward'

    assert(os.path.exists(IN_DIR))
    OUT_DIR = f'{DATA_DIR}/{EXNO}'
    Path(OUT_DIR).mkdir(exist_ok=True)

    # FE
    train = pd.read_pickle(f'{IN_DIR}/train.pkl')
    print(f'Input train shape: {train.shape}')
    target = 'action'
    features = [c for c in train.columns if 'feature' in c]
    features

    train = train.query('weight > 0').reset_index(drop=True)
    train[target] = (train['resp'] > 0).astype('int')

    # Fill missing values
    if method_fillna == '-999':
        train[features] = train[features].fillna(-999)
    elif method_fillna == 'forward':
        train[features] = train[features].fillna(method='ffill').fillna(0)

    # Train
    if not option['notrain']:
        if cv is None:
            train_xgb(train, features, target, XGB_PARAM, OUT_DIR)
        elif cv == 'PurgedGroupTimeSeriesSplit':
            train_xgb_cv(train, features, target, XGB_PARAM, n_splits, OUT_DIR)

    # Predict
    if cv is None:
        n_splits = 1

    if not option['nopredict']:
        models = []
        for i in range(n_splits):
            model = pd.read_pickle(open(f'{OUT_DIR}/model_{i}.pkl', 'rb'))
            models.append(model)

        if method_fillna == '-999':
            predict_fillna_999(models, features, target, OUT_DIR)
        elif method_fillna == 'forward':
            predict_fillna_forward(models, features, target, OUT_DIR)


if __name__ == '__main__':
    main()
