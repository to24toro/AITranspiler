import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu

class ResNet(tf.keras.Model):

    def __init__(self, action_space, n_blocks=20, filters=256, use_bias=False):
        super().__init__()

        self.action_space = action_space
        self.n_blocks = n_blocks
        self.filters =filters
        self.use_bias = use_bias

        self.conv1 = kl.Conv2D(filters, kernel_size=3, padding="same",
                               use_bias=use_bias, kernel_regularizer=l2(0.001),
                               kernel_initializer="he_normal")
        self.bn1 = kl.BatchNormalization()

        self.policy_layers = self._build_head(2, action_space)
        self.value_layers = self._build_head(1,1,activation="tanh")

        self.res_blocks = [ResBlock(filters=self.filters, use_bias=use_bias) for _ in range(self.n_blocks)]


    def _build_head(self, num_filters, output_dim, activation=None):
        return [
            kl.Conv2D(num_filters, kernel_size=1, use_bias=self.use_bias,
                      kernel_regularizer=l2(0.001), kernel_initializer="he_normal"),
            kl.BatchNormalization(),
            kl.Flatten(),
            kl.Dense(output_dim, activation=activation,
                     kernel_regularizer=l2(0.001), kernel_initializer="he_normal")
        ]
    
    def call(self, inputs, training=False):
        x = relu(self.bn1(self.conv1(inputs), training=training))

        for res_block in self.res_blocks:
            x = res_block(x, training=training)
        
        policy_output = self._apply_head(self.policy_layers,training)
        policy_output = tf.nn.softmax(policy_output)
        
        value_output = self._apply_head(self.value_layers,training)

        return policy_output, value_output
    
    def _apply_head(self, layers, training):
        for layer in layers:
            x = layer(x, training=training) if isinstance(layer, kl.BatchNormalization) else layer(x)
        return x


    def predict(self, state):
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]

        policy, value = self(state)

        return policy, value


class ResBlock(tf.keras.layers.Layer):

    def __init__(self, filters, use_bias):
        super().__init__()
        self.filters = filters
        self.use_bias = use_bias

        self.conv_layers = self._build_conv_block()


    def _build_conv_block(self):
        return [
            kl.Conv2D(self.filters, kernel_size=3, padding="same",
                      use_bias=self.use_bias, kernel_regularizer=l2(0.001),
                      kernel_initializer="he_normal"),
            kl.BatchNormalization(),
            kl.Conv2D(self.filters, kernel_size=3, padding="same",
                      use_bias=self.use_bias, kernel_regularizer=l2(0.001),
                      kernel_initializer="he_normal"),
            kl.BatchNormalization()
        ]


    def call(self, x, training=False):

        inputs = x

        x = relu(self.conv_layers[1](self.conv_layers[0](x), training=training))
        x = self.conv_layers[3](self.conv_layers[2](x), training=training)

        x = relu(x + inputs)

        return x