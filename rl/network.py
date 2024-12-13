import os
import numpy as np
import tensorflow as tf

import tf_keras as keras
import tf_keras.layers as kl
from tf_keras.activations import relu
from tf_keras.regularizers import l2
os.environ["TF_USE_LEGACY_KERAS"]="1"


class SqueezeExciteBlock(kl.Layer):
    """
    Squeeze and Excitation block to enhance representational power.
    SEブロックはResidual Block内でオプションとして使用。
    """
    def __init__(self, filters, reduction=4):
        super().__init__()
        self.filters = filters
        self.reduction = reduction
        self.global_pool = kl.GlobalAveragePooling2D()
        self.fc1 = kl.Dense(filters // reduction, activation='relu', kernel_initializer='he_normal')
        self.fc2 = kl.Dense(filters, activation='sigmoid', kernel_initializer='he_normal')

    def call(self, x, training=False):
        w = self.global_pool(x)
        w = self.fc1(w)
        w = self.fc2(w)
        w = tf.reshape(w, [-1, 1, 1, self.filters])
        return x * w


class ResBlock(keras.layers.Layer):
    def __init__(self, filters, use_bias, use_se=False, reduction=4):
        """
        Residual block with optional Squeeze-and-Excitation.
        Each block: Conv->BN->ReLU->Conv->BN (+SE) and add skip connection.
        """
        super().__init__()
        self.filters = filters
        self.use_bias = use_bias
        self.use_se = use_se
        self.reduction = reduction

        self.conv1 = kl.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()

        self.conv2 = kl.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal",
        )
        self.bn2 = kl.BatchNormalization()

        if use_se:
            self.se_block = SqueezeExciteBlock(filters, reduction=reduction)

    def call(self, x, training=False):
        inputs = x
        x = relu(self.bn1(self.conv1(x), training=training))
        x = self.bn2(self.conv2(x), training=training)

        if self.use_se:
            x = self.se_block(x, training=training)

        x = relu(x + inputs)
        return x


class ResNet(keras.Model):
    def __init__(self, action_space: int, config: dict):
        """
        ResNet model for policy and value prediction with possible enhancements:
        - Larger n_blocks, filters for more capacity
        - Optional Squeeze-and-Excitation blocks
        - AlphaZero-inspired head architectures
        """
        super().__init__()
        self.action_space = action_space

        network_settings = config["network_settings"]
        self.n_blocks: int = network_settings.get("n_blocks", 10)   # Increase as needed
        self.filters: int = network_settings.get("filters", 128)    # Increase as needed
        self.use_bias: bool = network_settings.get("use_bias", False)
        self.use_se: bool = network_settings.get("use_se", False)
        self.dropout_rate = network_settings.get("dropout_rate", 0.0)  # Set dropout if needed

        # Initial convolution + BN
        self.conv1 = kl.Conv2D(
            self.filters,
            kernel_size=3,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()

        # Residual blocks
        self.res_blocks = [
            ResBlock(filters=self.filters, use_bias=self.use_bias, use_se=self.use_se)
            for _ in range(self.n_blocks)
        ]

        # Policy head
        # AlphaZero style: Conv(2 filters, 1x1) -> BN -> ReLU -> Flatten -> Dense(action_space)
        self.policy_conv = kl.Conv2D(
            2, kernel_size=1,
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal"
        )
        self.policy_bn = kl.BatchNormalization()
        self.policy_flat = kl.Flatten()
        self.policy_dense = kl.Dense(
            self.action_space,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal"
        )

        # Value head
        # AlphaZero style: Conv(1 filter, 1x1) -> BN -> ReLU -> Flatten -> Dense(256, ReLU) -> Dense(1, tanh)
        self.value_conv = kl.Conv2D(
            1, kernel_size=1,
            use_bias=self.use_bias,
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal"
        )
        self.value_bn = kl.BatchNormalization()
        self.value_flat = kl.Flatten()
        self.value_fc1 = kl.Dense(
            256,
            activation='relu',
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal"
        )
        self.value_fc2 = kl.Dense(
            1,
            activation='tanh',
            kernel_regularizer=l2(0.001),
            kernel_initializer="he_normal"
        )

        # Optional dropout layers in heads if needed
        if self.dropout_rate > 0:
            self.policy_dropout = kl.Dropout(self.dropout_rate)
            self.value_dropout = kl.Dropout(self.dropout_rate)
        else:
            self.policy_dropout = None
            self.value_dropout = None

    def call(self, inputs, training=False):
        # Initial layers
        x = relu(self.bn1(self.conv1(inputs), training=training))
        # Residual layers
        for res_block in self.res_blocks:
            x = res_block(x, training=training)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p, training=training)
        p = relu(p)
        if self.policy_dropout is not None:
            p = self.policy_dropout(p, training=training)
        p = self.policy_flat(p)
        p = self.policy_dense(p)
        # softmaxは出力時にかける
        policy_output = tf.nn.softmax(p)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v, training=training)
        v = relu(v)
        if self.value_dropout is not None:
            v = self.value_dropout(v, training=training)
        v = self.value_flat(v)
        v = self.value_fc1(v)
        v = self.value_fc2(v)  # tanh出力

        return policy_output, v

    def predict(self, state):
        # Single state support
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]
        policy, value = self(state, training=False)
        return policy, value