from typing import Callable, Union, Tuple
from collections.abc import Callable

import click
import mlflow
import tensorflow as tf

from .models import TransferedModel
from . import utils


BASE_MODELS = [
    'MobileNetV2',
    'ResNet50',
    'EfficientNetB0',
    'EfficientNetB4',
    'EfficientNetB7'
]
PREPROCESS_INPUT_FN = {
    'MobileNetV2': tf.keras.applications.mobilenet.preprocess_input,
    'ResNet50': tf.keras.applications.resnet.preprocess_input,
    'EfficientNetB0': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB4': tf.keras.applications.efficientnet.preprocess_input,
    'EfficientNetB7': tf.keras.applications.efficientnet.preprocess_input
}


def train(
    base_model: str,
    train_data: tf.data.Dataset,
    valid_data: tf.data.Dataset,
    n_classes: int,
    steps_per_epoch: int=100,
    input_shape: Union[Tuple[Union[int, None], Union[int, None]], None]=(224, 224, 3),
    learning_rate: float=1e-5,
    preprocess_input: Union[Callable[[tf.Tensor], tf.Tensor], None]=None,
    max_epochs: int=100
) -> None:
    tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        callbacks =[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5
            )
        ]

        model = TransferedModel(
            base_model,
            n_classes,
            input_shape=input_shape,
            preprocess_fn=preprocess_input
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

        model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=['accuracy']
        )

    model.fit(
        train_data,
        validation_data=valid_data,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        callbacks=callbacks
    )
    return model


@click.command()
@click.option(
    '--base_model',
    default='EfficientNetB0',
    type=click.Choice(BASE_MODELS, case_sensitive=False)
)
@click.option(
    '--dataset',
    default='food101',
    help='tfds dataset name or csv file path.'
)
@click.option(
    '--learning_rate',
    default=1e-3,
    type=float,
    help='Initial learning rate. Deafults to 1e-3.'
)
@click.option(
    '--max_epochs',
    default=100,
    type=int,
    help='Maximum training epochs. Deafults to 100.'
)
@click.option(
    '--experiment',
    default=None,
    help='Name of MLflow experiment.'
)
def main(
    base_model: str,
    dataset: str,
    learning_rate: float=1e-3,
    max_epochs: int=100,
    experiment: str=None
):
    if experiment is not None:
        mlflow.set_experiment(experiment)
    mlflow.tensorflow.autolog()
    (train_data, valid_data, test_data), info = utils.load_dataset(dataset)
    with mlflow.start_run():
        mlflow.set_tag('dataset', dataset)
        mlflow.log_params({
            'base_model': base_model
        })
        model = train(
            base_model,
            train_data,
            valid_data,
            n_classes=info['n_classes'],
            steps_per_epoch=int(info['n_train']/info['batch_size']),
            input_shape=info['image_size'] + (3,),
            learning_rate=learning_rate,
            preprocess_input=PREPROCESS_INPUT_FN[base_model],
            max_epochs=max_epochs
        )
        model.evaluate(test_data)


if __name__ == '__main__':
    main()


