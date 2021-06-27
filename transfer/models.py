
from collections.abc import Callable
from typing import Tuple, Union

import tensorflow as tf


class TransferedModel(tf.keras.Model):
    def __init__(
        self,
        base_model: str,
        n_classes: int,
        dropout_rate: float=0.,
        preprocess_fn: Union[Callable[[tf.Tensor], None], None]=None,
        input_shape: Union[Tuple[Union[int, None], Union[int, None], int], None]=None,
        weights: Union[str, None]='imagenet', 
        **kwargs
    ):
        super().__init__(*kwargs)
        self.preprocess = tf.keras.layers.Lambda(preprocess_fn)
        self.base_model = getattr(tf.keras.applications, base_model)(
            input_shape=input_shape,
            include_top=False,
            weights=weights
        )
        self.pooling = tf.keras.layers.GlobalAveragePooling2D()
        if dropout_rate > 0:
            self.dropout = tf.keras.layers.Dropout(dropout_rate, name='top_dropout')
        else:
            self.dropout = None
        self.dense = tf.keras.layers.Dense(n_classes, dtype=tf.float32, name='output')

    def call(self, x):
        x = self.preprocess(x)
        x = self.base_model(x)
        x = self.pooling(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.dense(x)
        return x
