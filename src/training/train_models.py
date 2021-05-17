import itertools
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from .nn_architectures import DNNModel

from sklearn.model_selection import GridSearchCV

import tensorflow as tf

from sklearn.metrics import r2_score


def train_RF(X_train, X_test, y_train, grid_search=False):
    """
    Train the random forest model.

    Parameters
    ----------
    X_train : array
        The input matrix to train
    X_test : array
        The test matrix to use to generate predictions
    y_train : array
        The output array to train
    grid_search : str
        Whether to apply grid search or not

    Returns
    -------
    The predictions for test set
    """
    if grid_search == "True":
        # Set the possible values for grid search
        param_grid = {
            'min_samples_split': [1., 2, 3],
            'n_estimators': [90, 100, 110]
        }
        # Create a based model
        regr = RandomForestRegressor()

        # Instantiate the grid search model
        grid_searcher = GridSearchCV(estimator=regr,
                                     param_grid=param_grid,
                                     cv=5, n_jobs=-1,
                                     verbose=2)
        # Fit the grid search
        grid_searcher.fit(X_train, y_train)
        # Select the best model
        best_regr = grid_searcher.best_estimator_
        # Compute the predictions
        y_pred = best_regr.predict(X_test)
        return y_pred
    else:
        # Instantiate model
        regr = RandomForestRegressor()
        # Fit the model
        regr.fit(X_train, y_train)
        # Generate the predictions
        y_pred = regr.predict(X_test)
        return y_pred


def train_XGB(X_train, X_test, y_train, grid_search=False):
    """
    Train the XGBoost model.

    Parameters
    ----------
    X_train : array
        The input matrix to train
    X_test : array
        The test matrix to use to generate predictions
    y_train : array
        The output array to train
    grid_search : str
        Whether to apply grid search or not

    Returns
    -------
    The predictions for test set
    """
    if grid_search == "True":
        param_grid = {
            'learning_rate': [.25, .3, .35],
            'max_depth': [5, 6, 7],
        }
        # Create a based model
        xg_regr = xgb.XGBRegressor()

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=xg_regr,
                                   param_grid=param_grid,
                                   cv=5,
                                   n_jobs=-1,
                                   verbose=2
                                   )
        # Fit the grid search
        grid_search.fit(X_train, y_train)
        # Select the best model
        best_regr = grid_search.best_estimator_
        # Generate the predictions
        y_pred = best_regr.predict(X_test)
        return y_pred
    else:
        # Instantiate model
        xg_reg = xgb.XGBRegressor()
        # Fit the model
        xg_reg.fit(X_train, y_train)
        # Compute the model predictions
        y_pred = xg_reg.predict(X_test)
        return y_pred


def train_dnn(X_train, X_test, y_train, y_test, grid_search=False):
    """
    Train the DNN model.

    Parameters
    ----------
    X_train : array
        The input matrix to train
    X_test : array
        The test matrix to use to generate predictions
    y_train : array
        The output array to train
    y_test : array
        The output array to evaluate
    grid_search : str
        Whether to apply grid search or not

    Returns
    -------
    The predictions for test set
    """
    if grid_search == "True":
        # Create lists of parameters to test
        lrs = [0.001, 0.0001]
        bs_list = [16, 32, 64, 128]
        epochs = [5, 10, 15]
        # Create a set of all possible combinations
        param_list = list(itertools.product(lrs, bs_list, epochs))
        # Start a dataframe to keep track of scores
        scores_df = pd.DataFrame(columns=['lr', 'bs', 'n_epochs', 'R2'])
        # Run the multiple experiences
        for lr, bs, n_epochs in param_list:
            print("############################################")
            print("lr: ", lr, ", bs: ", bs, ", epochs: ", n_epochs)
            print("############################################")

            dnn_model = DNNModel(X_train)

            dnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=lr), loss='mse',
                              metrics=['mean_absolute_error'])

            history = dnn_model.fit(X_train, y_train, epochs=n_epochs, batch_size=bs)

            # Plot the loss and the MAE for each epoch
            plt.plot(history.history['mean_absolute_error'])
            plt.plot(history.history['loss'])
            plt.legend(['mae', 'loss'], loc='upper left')
            plt.show()

            # Compute predictions
            y_pred = np.concatenate(dnn_model.predict(X_test))
            # Add score to dataframe
            scores_df = pd.concat(
                [scores_df,
                 pd.DataFrame({'lr': lr, 'bs': bs, 'n_epochs': n_epochs, 'R2': r2_score(y_pred, y_test)})]
            )
        scores_df = scores_df.set_index(['lr', 'bs', 'epochs'])
        best_params = scores_df.idxmax()

        # Instantiate new model
        dnn_model = DNNModel(X_train)

        # compile the model
        dnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=best_params['lr']), loss='mse',
                          metrics=['mean_absolute_error'])

        # Fit the model and store the evolution of metrics and loss
        dnn_model.fit(X_train, y_train, epochs=best_params['n_epochs'], batch_size=best_params['bs'])
        # Compute predictions
        y_pred = np.concatenate(dnn_model.predict(X_test))
        return y_pred
    else:
        # Fix the number of epochs
        n_epochs = 10
        # Instantiate the model
        dnn_model = DNNModel(X_train)

        # compile the model
        dnn_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='mse',
                          metrics=['mean_absolute_error'])

        # Fit the model and store the evolution of metrics and loss
        dnn_model.fit(X_train, y_train, epochs=n_epochs)
        # Compute predictions
        y_pred = np.concatenate(dnn_model.predict(X_test))
        return y_pred
