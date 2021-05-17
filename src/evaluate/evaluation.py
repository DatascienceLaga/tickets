import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.util_functions import relative_error
from features.build_features import day_x_interval

import plotly.graph_objects as go
import plotly.io as pio

def evaluate(test_df, y_real, y_hat, title):
    """
    Shows different scores and metrics between predictions and real values.

    The results are computed based on the full data and aggregated data.

    Parameters
    ----------
    y_real : array
        The values of the real target
    y_hat : array
        The values of the predicted target

    Returns
    -------
    None
    """
    # Delete negative values & round up float numbers
    y_hat = np.around(np.maximum(0, y_hat))
    # List columns for evaluation
    eval_cols = ['origin_station_name', 'destination_station_name',
                 'sale_day_x_interval', 'demand', 'predictions']
    # Copy evaluation data set and add predictions
    X_test_xgb = test_df.copy()
    X_test_xgb['predictions'] = y_hat
    # Compute sale_day_x interval
    X_test_xgb['sale_day_x_interval'] = X_test_xgb.sale_day_x.apply(day_x_interval)
    # Keep only columns needed for evaluation
    X_test_xgb = X_test_xgb[eval_cols]
    # Sum demands by origin and destination
    global_pred = X_test_xgb.groupby(['origin_station_name', 'destination_station_name']).agg('sum')
    # Compute relative error for aggregated demands and predictions
    global_pred['RE'] = relative_error(global_pred.demand, global_pred.predictions)
    # Compute MAE for aggregated demands and predictions
    global_pred['MAE'] = mean_absolute_error(global_pred.demand, global_pred.predictions)
    # Set MAE score as index as well
    global_pred = global_pred.reset_index().set_index(['MAE', 'origin_station_name', 'destination_station_name'])
    # Create new aggregation with sale_day_x as well
    per_day_pred = (X_test_xgb
                    .groupby(['origin_station_name', 'destination_station_name'
                                 , 'sale_day_x_interval'])
                    .agg('sum')
                    )
    # Compute relative error for new aggregation
    per_day_pred['RE'] = relative_error(per_day_pred.demand, per_day_pred.predictions)
    # Compute MAE per origin - destination
    mae_per_day = (per_day_pred
                   .groupby(level=[0, 1], axis=0)
                   .apply(lambda sub: mean_absolute_error(sub.demand, sub.predictions))
                   )
    # Show scores and results
    ##Show scores per day
    print("################ Scores by origin-destination-sale_day ################")
    print(per_day_pred)
    print(pd.DataFrame(mae_per_day, columns=['MAE']))

    ##Show scores per origin-destination
    print("################ Scores by origin-destination ################")
    print(global_pred)

    ##Show global scores
    print("################ Scores global scores ################")
    print("MAE: ", mean_absolute_error(y_real, y_hat))
    print("MSE: ", mean_squared_error(y_real, y_hat))
    print("R2 score: ", r2_score(y_real, y_hat))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_real, name="Real"
                             , line_shape='linear'))
    fig.add_trace(go.Scatter(y=y_hat, name="Prediction"
                             , line_shape='linear'))
    fig.update_layout(title=title)
    pio.show(fig)