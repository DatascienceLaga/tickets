import lz4
import pandas as pd
from io import StringIO

def read_data(file_path):
    """
    Reads the compressed csv file.

    Parameters
    ----------
    file_path : str
        The pat to the directory where the compressed csv file is stored

    Returns
    -------
    The dataframe containing the data from the compressed csv file
    """
    # Open the file
    with lz4.frame.open(file_path, mode='r') as fp:
        output_data = fp.read()
    # Transform file from bytes to str
    s = str(output_data ,'utf-8')
    data = StringIO(s)
    # Format data into a dataframe
    df = pd.read_csv(data)

    return df

