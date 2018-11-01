from keras.models import Model

from keras_unet.layers.blocks import build_encoder_block, build_decoder_block, build_bridge_block

from keras.layers import Conv2D
from keras.layers.core import Lambda


def build_model(
        inputs,
        kernel_size,
        block_sizes,
        dropout=None,
        normalize_input=False,
        target_classes=1,
        target_activation='softmax',
        dense=False,
        **kargs
):
    """ Creates a U-Net model

    Args:
        inputs          : keras.layers.Input for the input to the model.
        kernel_size     : The size of the convolution kernel to use throughout the network.
        block_sizes     : A list of integers to specify the filters at different block depths within the model. 
        dropout         : Enables dropout if not `None`.
        normalize_input : 
        target_classes  : Number of target classes.

    Returns:
        A keras.models.Model which takes an image as input and outputs a segmentation of the input image.
    """

    # Create list to store the main output of each encoder block. These will be used as main inputs to each decoder
    # block.
    block_inputs = [None for _ in range(len(block_sizes))]

    encoded_input = Lambda(lambda x: x / 255)(inputs) if normalize_input else inputs
    # Iterate through each block size and construct an encoder block.
    for block_i in range(len(block_sizes)):
        block_size = block_sizes[block_i]
        block, encoded_input = build_encoder_block(block_size, kernel_size, encoded_input, dropout=dropout, dense=dense, **kargs)
        block_inputs[block_i] = block

    # Add the bridge blocks.
    decoded_output = build_bridge_block(block_sizes[-1] * 2, kernel_size, encoded_input, **kargs)
    # Both block size & block inputs are reversed as the decoder blocks are now added
    block_sizes.reverse()
    block_inputs.reverse()

    # Iterate through each block size and construct a decoder block.
    for block_i in range(len(block_sizes)):
        block_size = block_sizes[block_i]
        block_input = block_inputs[block_i]
        decoded_output = build_decoder_block(block_size, kernel_size, block_input, decoded_output, dropout=dropout, dense=dense, **kargs)

    # Convolve with a 1x1 kernel to yield the final output.
    # TODO: make filter a variable for different number of output classes
    model_output = Conv2D(target_classes, (1, 1), activation=target_activation)(decoded_output)

    # Return the keras.models.Model with the provided input and the constructed output.
    return Model(inputs=[inputs], outputs=[model_output])
