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
    """ Creates a encoder block.

    Args:
        filters         : The depth of convolution kernel to use throughout the block
        kernel_size     : The size of the convolution kernel to use throughout the block.
        encoded_input   : The keras.layer from the previous encoder block.
        block_depth     : The number of convolutions within the decoder block.
        pool_size       : The size of the convolution kernel to use in the up max pooling operation.
        dropout         : Enables dropout if not `None`.
        activation      : The activation function to use throughout the block.

    Returns:
        The output keras.layer of the decoder block.
    """

    main_output = encoded_input
    # Add a convolution layer for range of block_depth
    for block_i in range(block_depth):
        main_output = Conv2D(filters, kernel_size, activation=activation, **kwargs)(main_output)
        # Add dropout layers if applicable.
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
    """ Creates a decoder block.

    Args:
        filters         : The depth of convolution kernel to use throughout the block
        kernel_size     : The size of the convolution kernel to use throughout the block.
        main_input      : The keras.layer from the analogous encoder block at the same depth.
        side_input      : The keras.layer from the previous decoder block.
        block_depth     : The number of convolutions within the decoder block.
        up_conv_size    : The size of the convolution kernel to use in the up sampling operation.
        dropout         : Enables dropout if not `None`.
        activation      : The activation function to use throughout the block.

    Returns:
        The output keras.layer of the decoder block.
    """

    # Up sample the input from the previous decoder block.
    side_input = Conv2DTranspose(filters, up_conv_size, strides=up_conv_size, activation=activation, **kwargs)(
        side_input)
    # Concatenate the output from the analogous encoder block at the same depth with the up sampled block.
    main_output = Concatenate(axis=-1)([side_input, main_input])
    # Add a convolution layer for range of block_depth
    for block_i in range(block_depth):
        main_output = Conv2D(filters, kernel_size, activation=activation, **kwargs)(main_output)
        # Add dropout layers if applicable.
        if dropout and dropout > 0:
            main_output = Dropout(dropout)(main_output)
    return main_output


def build_bridge_block(
        filters,
        kernel_size,
        encoded_input,
        activation='elu',
        **kwargs
):
    """ Creates a bridge block.

    Args:
        filters         : The depth of convolutional kerne to use throughout the block
        kernel_size     : The size of the convolution kernel to use throughout the block.
        encoded_input   : The input keras.layer to the bridge block.
        activation      : The activation function to use throughout the block.

    Returns:
        The output keras.layer of the bridge block.
    """
    bridge = Conv2D(filters, kernel_size, activation=activation, **kwargs)(encoded_input)
    bridge = Conv2D(filters, kernel_size, activation=activation, **kwargs)(bridge)
    return bridge
