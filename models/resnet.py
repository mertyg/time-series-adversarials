#conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
#conv_x = keras.layers.BatchNormalization()(conv_x)
#conv_x = keras.layers.Activation('relu')(conv_x)
#
#conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
#conv_y = keras.layers.BatchNormalization()(conv_y)
#conv_y = keras.layers.Activation('relu')(conv_y)
#
#conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
#conv_z = keras.layers.BatchNormalization()(conv_z)
#
## expand channels for the sum
#shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
#shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
#
#output_block_1 = keras.layers.add([shortcut_y, conv_z])
#output_block_1 = keras.layers.Activation('relu')(output_block_1)
import torch.nn as nn

class TSResNetBlock(nn.Module):
    def __init__(self):
        super(TSResNetBlock, self).__init__()