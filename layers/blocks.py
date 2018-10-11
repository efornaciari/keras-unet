from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.merge import Concatenate


def build_encoder_block(
        filters,
        kernel_size,
        encoded_input,
        block_depth=2,
        pool_size=(2, 2),
        dropout=None,
        activation='elu',
        **kwargs
):
    """
    TODO: fill out
    """
    main_output = encoded_input
    for block_i in range(block_depth):
        main_output = Conv2D(filters, kernel_size, activation=activation, **kwargs)(main_output)
        if dropout and dropout > 0:
            main_output = Dropout(dropout)(main_output)
    side_output = MaxPooling2D(pool_size=pool_size, **kwargs)(main_output)
    return main_output, side_output


def build_decoder_block(
        filters,
        kernel_size,
        main_input,
        side_input,
        block_depth=2,
        up_conv_size=(2, 2),
        dropout=None,
        activation='elu',
        **kwargs
):
    """
    TODO: fill out
    """
    side_input = Conv2DTranspose(filters, up_conv_size, strides=up_conv_size, activation=activation, **kwargs)(side_input)
    main_output = Concatenate(axis=-1)([side_input, main_input])
    for block_i in range(block_depth):
        main_output = Conv2D(filters, kernel_size, activation=activation, **kwargs)(main_output)
        if dropout and dropout > 0:
            main_output = Dropout(dropout)(main_output)
    return main_output


def build_bridge(filters, kernel_size, encoded_input, activation='elu', **kwargs):
    bridge = Conv2D(filters, kernel_size, activation=activation, **kwargs)(encoded_input)
    bridge = Conv2D(filters, kernel_size, activation=activation, **kwargs)(bridge)
    return bridge
