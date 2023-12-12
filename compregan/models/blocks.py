import tensorflow.keras as keras


def dense_block(x, num_units: int, dropout_rate: float = 0.5):
    x = keras.layers.Dense(num_units)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    return x
