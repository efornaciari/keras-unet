from keras.models import Model
from keras.layers import Input

from layers.blocks import build_encoder_block, build_decoder_block, build_bridge
from keras.layers import Conv2D


def build_model(kernel_size, block_sizes, dropout=None, **kargs):
    """
    
    """

    # create input layer
    model_input = Input((512, 512, 3))
    encoded_input = model_input
    block_inputs = [None for _ in range(len(block_sizes))]

    for block_i in range(len(block_sizes)):
        block_size = block_sizes[block_i]
        block, encoded_input = build_encoder_block(block_size, kernel_size, encoded_input, dropout=dropout, **kargs)
        block_inputs[block_i] = block

    decoded_output = build_bridge(block_sizes[-1] * 2, kernel_size, encoded_input, **kargs)
    block_sizes.reverse()
    block_inputs.reverse()

    for block_i in range(len(block_sizes)):
        block_size = block_sizes[block_i]
        block_input = block_inputs[block_i]
        decoded_output = build_decoder_block(block_size, kernel_size, block_input, decoded_output, dropout=dropout,
                                             **kargs)

    model_output = Conv2D(1, (1, 1), activation='sigmoid')(decoded_output)

    return Model(inputs=[model_input], outputs=[model_output])
