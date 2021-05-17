def relative_error(col1, col2):
    """
    Computes relative error between real values and predictions.

    Parameters
    ----------
    y_real : array
        The values of the real target
    y_hat : array
        The values of the predicted target

    Returns
    -------
    Relative error of two values
    """
    return ((col1 - col2) / col2) * 100