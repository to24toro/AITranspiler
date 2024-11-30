import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.regularizers import l2
from typing import List, Tuple, Any


class ResNet(tf.keras.Model):
    def __init__(self, action_space: int, config: dict):
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
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )
        self.bn1 = kl.BatchNormalization()

        self.res_blocks: List[ResBlock] = [
            ResBlock(filters=self.filters, use_bias=self.use_bias)
            for _ in range(self.n_blocks)
        ]

        # Policy head
        self.policy_conv = kl.Conv2D(
            filters=2,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )
        self.policy_bn = kl.BatchNormalization()
        self.policy_flatten = kl.Flatten()
        self.policy_dense = kl.Dense(
            self.action_space,
            activation='linear',
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )

        # Value head
        self.value_conv = kl.Conv2D(
            filters=1,
            kernel_size=1,
            padding="same",
            use_bias=self.use_bias,
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )
        self.value_bn = kl.BatchNormalization()
        self.value_flatten = kl.Flatten()
        self.value_dense1 = kl.Dense(
            units=self.filters,
            activation='relu',
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )
        self.value_dense2 = kl.Dense(
            units=1,
            activation='linear',  # 報酬の範囲に合わせて活性化関数を設定
            kernel_regularizer=l2(1e-4),
            kernel_initializer="he_normal",
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        for res_block in self.res_blocks:
            x = res_block(x, training=training)

        # Policy head
        p = self.policy_conv(x)
        p = self.policy_bn(p, training=training)
        p = tf.nn.relu(p)
        p = self.policy_flatten(p)
        p = self.policy_dense(p)
        # ソフトマックスは損失関数で適用

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v, training=training)
        v = tf.nn.relu(v)
        v = self.value_flatten(v)
        v = self.value_dense1(v)
        v = self.value_dense2(v)

        return p, v

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(state.shape) == 3:
            state = state[np.newaxis, ...]
        policy_logits, value = self(state)
        policy = tf.nn.softmax(policy_logits).numpy()
        return policy, value.numpy()


class ResBlock(kl.Layer):
    def __init__(self, filters: int, use_bias: bool):
        super().__init__()
        self.conv1 = kl.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            use_bias=use_bias,
            kernel_regularizer=l2(1e-4),
            kernel_initializer='he_normal'
        )
        self.bn1 = kl.BatchNormalization()
        self.conv2 = kl.Conv2D(
            filters,
            kernel_size=3,
            padding='same',
            use_bias=use_bias,
            kernel_regularizer=l2(1e-4),
            kernel_initializer='he_normal'
        )
        self.bn2 = kl.BatchNormalization()

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = x + inputs
        x = tf.nn.relu(x)
        return x