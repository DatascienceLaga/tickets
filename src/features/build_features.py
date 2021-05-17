import pandas as pd

def one_hot_encod(df):
    """
    Creates one hot encoded columns for categorical variables.

    Parameters
    ----------
    df : dataframe
        The dataframe to which should be added the one hot encoded columns

    Returns
    -------
    The dataframe containing the additional one hot encoded vectors
    """
    df = pd.concat(
        [
            df,
            pd.get_dummies(df.destination_station_name, prefix='destination_station'),
            pd.get_dummies(df.origin_station_name, prefix='origin_station')
        ], axis=1
    )
    return df


def day_x_interval(value):
    """
    Replace a value by the interval that contains it.

    Parameters
    ----------
    value : int
        The value that shall be replaced by the corresponding interval.

    Returns
    -------
    interval : str
        The corresponding interval to the input value.
    """
    # For each value return the corresponding interval
    if value == -1:
        return "[-1]"
    if value == -2:
        return "[-2]"
    elif value == -3:
        return "[-3]"
    elif value == -4:
        return "[-4]"
    elif value == -5:
        return "[-5]"
    elif value == -6:
        return "[-6]"
    elif value == -7:
        return "[-7]"
    elif value < -7 and value >= -10:
        return "[-10,-8]"
    elif value < -10 and value >= -15:
        return "[-15,-11]"
    elif value < -15 and value >= -20:
        return "[-20,-16]"
    elif value < -20 and value >= -30:
        return "[-30,-21]"
    elif value < -30 and value >= -60:
        return "[-60,-31]"
    elif value < -60 and value >= -90:
        return "[-90,-61]"
    else:
        return None