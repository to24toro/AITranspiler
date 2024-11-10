import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import yaml
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2

# Load network settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

network_settings = config["network_settings"]


class ResNet(tf.keras.Model):
    def __init__(self, action_space):
        super().__init__()

        self.action_space = action_space
        self.n_blocks = network_settings["n_blocks"]
        self.filters = network_settings["filters"]
        self.use_bias = network_settings["use_bias"]

        self.conv1 = kl.Conv2D(
            self.filters,
            kernel_size=3,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()

        self.policy_layers = self._build_head(2, self.action_space)
        self.value_layers = self._build_head(1, 1, activation='tanh')

        self.res_blocks = [
            ResBlock(filters=self.filters, use_bias=self.use_bias)
            for _ in range(self.n_blocks)
        ]

    def _build_head(self, num_filters, output_dim, activation=None):
        return [
            kl.Conv2D(
                num_filters,
                kernel_size=1,
                use_bias=self.use_bias,
                kernel_regularizer=l2(0.001),
                kernel_initializer="he_normal",
            ),
            kl.BatchNormalization(),
            kl.Flatten(),
            kl.Dense(
                output_dim,
                activation=activation,
                kernel_regularizer=l2(0.001),
                kernel_initializer="he_normal",
            ),
        ]

    def call(self, inputs, training=False):
        x = relu(self.bn1(self.conv1(inputs), training=training))

        for res_block in self.res_blocks:
            x = res_block(x, training=training)

        # Apply policy head
        policy_x = x
        for layer in self.policy_layers:
            if isinstance(layer, kl.BatchNormalization):
                policy_x = layer(policy_x, training=training)
            else:
                policy_x = layer(policy_x)
        policy_output = tf.nn.softmax(policy_x)

        # Apply value head
        value_x = x
        for layer in self.value_layers:
            if isinstance(layer, kl.BatchNormalization):
                value_x = layer(value_x, training=training)
            else:
                value_x = layer(value_x)
        value_output = value_x

        return policy_output, value_output

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
            kl.Conv2D(
                self.filters,
                kernel_size=3,
                padding="same",
                use_bias=self.use_bias,
                kernel_regularizer=l2(0.001),
                kernel_initializer="he_normal",
            ),
            kl.BatchNormalization(),
            kl.Conv2D(
                self.filters,
                kernel_size=3,
                padding="same",
                use_bias=self.use_bias,
                kernel_regularizer=l2(0.001),
                kernel_initializer="he_normal",
            ),
            kl.BatchNormalization(),
        ]

    def call(self, x, training=False):
        inputs = x

        x = relu(self.conv_layers[1](self.conv_layers[0](x), training=training))
        x = self.conv_layers[3](self.conv_layers[2](x), training=training)

        x = relu(x + inputs)

        return x
