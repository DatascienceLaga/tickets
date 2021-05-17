import tensorflow as tf

class DNNModel(tf.keras.Model):
    """
    A class to create the arcitecture of the DNN model

    ...

    Attributes
    ----------
    inputs : array
       the array of inputs that the model would train on
    """

    def __init__(self, X_train):
        """
        Initialize the layers of the model
        """
        super(DNNModel, self).__init__()

        self.dnn1 = tf.keras.layers.Dense(64, input_shape=(X_train.shape[0],), activation='relu')
        self.dnn2 = tf.keras.layers.Dense(32, activation='relu')
        self.dnn3 = tf.keras.layers.Dense(32, activation='relu')
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """Forwad propagates the inputs into the model

        Parameters
        ----------
        inputs : array
           the array of inputs that the model would train on

        Returns
        -------
        x : tensor
            the output of the model
        """
        x = self.dnn1(inputs)
        x = self.dnn2(x)
        x = self.dnn3(x)
        x = self.fc1(x)
        return x

