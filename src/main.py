import os
import sys

from data.make_dataset import read_data
from data.data_split import train_test_split
from features.build_features import one_hot_encod
from training.train_models import train_RF, train_XGB, train_dnn
from evaluate.evaluation import evaluate


if __name__ == '__main__':
    DATA_PATH = sys.argv[1]
    GRID_SEARCH = sys.argv[2]

    TRAIN_FILE_PATH = os.path.join(DATA_PATH, 'training', 'leaf.csv.lz4')
    TEST_FILE_PATH = os.path.join(DATA_PATH, 'testing', 'leaf.csv.lz4')

    print('(1/5): Reading the data...')
    train_df = read_data(TRAIN_FILE_PATH)
    test_df = read_data(TEST_FILE_PATH)

    print('(2/5): Preprocessing the data...')
    train_df = one_hot_encod(train_df)
    test_df = one_hot_encod(test_df)

    print('(3/5): Splitting the data...')
    X_train, y_train = train_test_split(train_df)
    X_test, y_test = train_test_split(test_df)

    print('(4/5): Training the models...')
    print('\t(1/3): Training Random Forest model')
    y_pred_rf = train_RF(X_train, X_test, y_train, grid_search=GRID_SEARCH)
    print('\t(2/3): Training XGBoost model')
    y_pred_xgb = train_XGB(X_train, X_test, y_train, grid_search=GRID_SEARCH)
    print('\t(3/3): Training DNN model')
    y_pred_dnn = train_dnn(X_train, X_test, y_train, y_test, grid_search=GRID_SEARCH)

    print('(5/5): Evaluating the models...')
    print('\t(1/3): Evaluating Random Forest model')
    evaluate(test_df, y_test, y_pred_rf, title='Random forest model')
    print('\t(1/3): Evaluating XGBoost model')
    evaluate(test_df, y_test, y_pred_xgb, title='XGBoost model')
    print('\t(1/3): Evaluating DNN model')
    evaluate(test_df, y_test, y_pred_dnn, title='DNN model')