import os
import numpy as np
import tensorflow as tf

import tf_keras as keras
import tf_keras.layers as kl
from tf_keras.activations import relu
from tf_keras.regularizers import l2


class ResNet(keras.Model):
    def __init__(self, action_space: int, config: dict):
        """
        Initialize the ResNet model for policy and value prediction.

        :param action_space: Number of possible actions in the game (output size of the policy head).
        :param config: Configuration dictionary containing network settings.
        """
        super().__init__()

        self.action_space = action_space

        network_settings = config["network_settings"]
        self.n_blocks: int = network_settings["n_blocks"]  # Number of residual blocks.
        self.filters: int = network_settings[
            "filters"
        ]  # Number of filters in each convolutional layer.
        self.use_bias: bool = network_settings[
            "use_bias"
        ]  # Whether to use bias in convolutional layers.

        # Initial convolutional layer and batch normalization.
        self.conv1 = kl.Conv2D(
            self.filters,
            kernel_size=3,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),  # L2 regularization for weight decay.
            kernel_initializer="he_normal",  # He initialization for better convergence.
        )
        self.bn1 = kl.BatchNormalization()

        # Policy and value heads.
        self.policy_layers = self._build_head(2, self.action_space)
        self.value_layers = self._build_head(1, 1, activation="tanh")

        # Residual blocks.
        self.res_blocks = [
            ResBlock(filters=self.filters, use_bias=self.use_bias)
            for _ in range(self.n_blocks)
        ]

    def _build_head(self, num_filters, output_dim, activation=None):
        """
        Build the head (policy or value) of the network.

        :param num_filters: Number of filters in the first convolutional layer.
        :param output_dim: Number of output units in the dense layer.
        :param activation: Activation function for the dense layer (optional).
        :return: A list of layers defining the head.
        """
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
        """
        Forward pass through the ResNet model.

        :param inputs: Input tensor (state representation).
        :param training: Whether the model is in training mode (for batch normalization).
        :return: Policy output (probabilities over actions) and value output (state value).
        """
        # Apply initial convolution and residual blocks.
        x = relu(self.bn1(self.conv1(inputs), training=training))

        for res_block in self.res_blocks:
            x = res_block(x, training=training)

        # Process the policy head.
        policy_x = x
        for layer in self.policy_layers:
            if isinstance(layer, kl.BatchNormalization):
                policy_x = layer(policy_x, training=training)
            else:
                policy_x = layer(policy_x)
        policy_output = tf.nn.softmax(policy_x)  # Apply softmax to get probabilities.

        # Process the value head.
        value_x = x
        for layer in self.value_layers:
            if isinstance(layer, kl.BatchNormalization):
                value_x = layer(value_x, training=training)
            else:
                value_x = layer(value_x)
        value_output = value_x

        return policy_output, value_output

    def predict(self, state):
        """
        Predict the policy and value for a given state.

        :param state: Input state tensor. Can be a single state or a batch of states.
        :return: Tuple containing policy (action probabilities) and value (state value).
        """
        if len(state.shape) == 3:
            # Add batch dimension if the input is a single state.
            state = state[np.newaxis, ...]

        policy, value = self(state)
        return policy, value


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, use_bias):
        """
        Initialize a residual block.

        :param filters: Number of filters in the convolutional layers.
        :param use_bias: Whether to use bias in the convolutional layers.
        """
        super().__init__()
        self.filters = filters
        self.use_bias = use_bias

        # Two convolutional layers with batch normalization.
        self.conv_layers = self._build_conv_block()

    def _build_conv_block(self):
        """
        Build the convolutional layers of the residual block.

        :return: A list of layers defining the residual block.
        """
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
        """
        Forward pass through the residual block.

        :param x: Input tensor.
        :param training: Whether the model is in training mode (for batch normalization).
        :return: Output tensor after applying the residual block.
        """
        inputs = x

        # First convolution and batch normalization with ReLU activation.
        x = relu(self.conv_layers[1](self.conv_layers[0](x), training=training))

        # Second convolution and batch normalization.
        x = self.conv_layers[3](self.conv_layers[2](x), training=training)

        # Add the input (residual connection) and apply ReLU.
        x = relu(x + inputs)

        return x
