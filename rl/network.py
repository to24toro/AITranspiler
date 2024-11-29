import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import l2
from typing import List, Tuple, Any


class ResNet(tf.keras.Model):
    def __init__(self, action_space: int, config: dict):
        """
        Residual Network for policy and value predictions.

        Args:
            action_space (int): The size of the action space.
            config (dict): Configuration dictionary containing network settings.
        """
        super().__init__()

        self.action_space = action_space

        network_settings = config["network_settings"]
        self.n_blocks: int = network_settings["n_blocks"]
        self.filters: int = network_settings["filters"]
        self.use_bias: bool = network_settings["use_bias"]

        self.conv1 = kl.Conv2D(
            self.filters,
            kernel_size=3,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()

        self.policy_layers: List[Any] = self._build_head(2, self.action_space)
        self.value_layers: List[Any] = self._build_head(1, 1, activation="tanh")

        self.res_blocks: List[ResBlock] = [
            ResBlock(filters=self.filters, use_bias=self.use_bias)
            for _ in range(self.n_blocks)
        ]

    def _build_head(self, num_filters: int, output_dim: int, activation: str = None) -> List[Any]:
        """
        Build the policy or value head.

        Args:
            num_filters (int): Number of filters for the first convolutional layer.
            output_dim (int): Dimension of the output layer.
            activation (str, optional): Activation function for the output layer.

        Returns:
            List[Any]: List of layers comprising the head.
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

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Forward pass for the ResNet.

        Args:
            inputs (tf.Tensor): Input tensor with shape (batch_size, height, width, channels).
            training (bool): Whether the network is in training mode.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: 
                - Policy output (probabilities over actions).
                - Value output (scalar value prediction).
        """
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

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make a prediction using the ResNet model.

        Args:
            state (np.ndarray): Input state matrix of shape (height, width, channels).

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - Policy output as probabilities over actions.
                - Value output as a scalar.
        """
        if len(state.shape) == 3:  # Add batch dimension if missing
            state = state[np.newaxis, ...]

        policy, value = self(state)

        return policy.numpy(), value.numpy()


class ResBlock(kl.Layer):
    def __init__(self, filters: int, use_bias: bool):
        """
        Residual block with two convolutional layers.

        Args:
            filters (int): Number of filters for the convolutional layers.
            use_bias (bool): Whether to use bias in the convolutional layers.
        """
        super().__init__()
        self.filters: int = filters
        self.use_bias: bool = use_bias

        self.conv_layers: List[kl.Layer] = self._build_conv_block()

    def _build_conv_block(self) -> List[kl.Layer]:
        """
        Build the convolutional block for the residual layer.

        Returns:
            List[kl.Layer]: List of convolutional and batch normalization layers.
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

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass for the residual block.

        Args:
            x (tf.Tensor): Input tensor.
            training (bool): Whether the network is in training mode.

        Returns:
            tf.Tensor: Output tensor after applying the residual connection.
        """
        inputs = x

        x = relu(self.conv_layers[1](self.conv_layers[0](x), training=training))
        x = self.conv_layers[3](self.conv_layers[2](x), training=training)

        x = relu(x + inputs)  # Residual connection

        return x