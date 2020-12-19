import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

import janestreet
env = janestreet.make_env()  # initialize the environment
iter_test = env.iter_test()  # an iterator which loops over the test set

EXNO = '003'
IN_DIR = '../data/002'
OUT_DIR = f'../data/{EXNO}'

train = pd.read_pickle(f'{IN_DIR}/train.pkl')

debug_mode = True
if debug_mode:
    frac = 0.001
    train = train.sample(frac=frac, random_state=42)
    print(train.shape)

sample_prediction_df = pd.read_pickle(f'{IN_DIR}/example_sample_submission.pkl')

print('train shape is {}'.format(train.shape))
print('sample_prediction_df shape is {}'.format(sample_prediction_df.shape))

# Is the data balanced or not?
train = train[train['weight'] != 0]
train['action'] = ((train['weight'].values * train['resp'].values) > 0).astype('int')

X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']
X_train = X_train.fillna(-999)
del train

# Training
# The training part taked from here https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    missing=-999,
    random_state=2020
    # tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)

clf.fit(X_train, y_train)

# Predict
for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    X_test.fillna(-999)
    y_preds = clf.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)
