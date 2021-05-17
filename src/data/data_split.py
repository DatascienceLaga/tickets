def train_test_split(df):
    """
    Creates train-test input-output matrices.

    Parameters
    ----------
    df : dataframe
        The dataframe from which the matrices are extracted

    Returns
    -------
    The matrices for training and testing
    """
    #Name of target and input columns
    target_col = 'demand'
    train_cols = [col for col in df.columns if
                  col not in ['departure_date', 'sale_date', 'dataset_type', 'demand', 'destination_station_name',
                              'origin_station_name']
                  ]
    # Create training matrices
    X = df[train_cols]
    y = df[target_col]
    return X, y