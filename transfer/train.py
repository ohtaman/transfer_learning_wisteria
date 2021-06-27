from typing import Callable, Union, Tuple
from collections.abc import Callable
import os
import json
import tempfile

import click
import mlflow
import tensorflow as tf
try:
    from mpi4py import MPI
except:
    pass

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


def get_distribute_strategy(use_mpi):
    if use_mpi:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        host = os.environ['PMIX_HOSTNAME']
        workers = comm.gather(host, root=0)
        workers = comm.bcast(workers, root=0)
        tf_config = {
            'cluster': {
                'worker': [
                    f'{worker}:20000'
                    for worker in workers
                ]
            },
            'task': {'type': 'worker', 'index': rank} 
        }
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
        
        strategy = tf.distribute.MultiWorkerMirroredStrategy(
            communication_options=tf.distribute.experimental.CommunicationOptions(
                # implementation=tf.distribute.experimental.CollectiveCommunication.NCCL
            )
        )
        if rank == 0:
            print('Use MultiWorkerMirroedStrategy')
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    else:
        print('Use MirroredStrategy')
        strategy = tf.distribute.MirroredStrategy()
    return strategy


def is_chief(distribute_strategy):
    task_type = distribute_strategy.cluster_resolver.task_type
    task_id = distribute_strategy.cluster_resolver.task_id
    return task_type is None or (task_type == 'worker' and task_id == 0)


def train(
    base_model: str,
    train_data: tf.data.Dataset,
    valid_data: tf.data.Dataset,
    n_classes: int,
    steps_per_epoch: int=100,
    input_shape: Union[Tuple[Union[int, None], Union[int, None]], None]=(224, 224, 3),
    learning_rate: float=1e-5,
    preprocess_input: Union[Callable[[tf.Tensor], tf.Tensor], None]=None,
    max_epochs: int=100,
    is_chief: bool=True
) -> None:
    # tf.config.optimizer.set_jit(True)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    callbacks = []
    # if is_chief:
    #     callbacks.append(
    #         tf.keras.callbacks.EarlyStopping(
    #             monitor='val_loss',
    #             patience=5
    #         )
    #     )

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
        callbacks=callbacks,
        verbose=(1 if is_chief else 0)
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
    help='Initial learning rate. Defaults to 1e-3.'
)
@click.option(
    '--batch_size',
    default=32,
    type=int,
    help='Batch size. Defaults to 32.'
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
@click.option(
    '--mpi',
    is_flag=True,
    help='Distribute with MPI'
)
@click.option(
    '--model_dir',
    default=None,
    help='Model output directory.'
)
def main(
    base_model: str,
    dataset: str,
    learning_rate: float=1e-3,
    batch_size: int=32,
    max_epochs: int=100,
    experiment: str=None,
    mpi: bool=False,
    model_dir: Union[str, None]=None
):
    strategy = get_distribute_strategy(mpi)
    if is_chief(strategy):
        if experiment is not None:
            mlflow.set_experiment(experiment)
            mlflow.tensorflow.autolog()
            mlflow.set_tag('dataset', dataset)
            mlflow.log_params({
                'base_model': base_model
            })
    
    (train_data, valid_data, test_data), info = utils.load_dataset(dataset, batch_size=batch_size, distribute=True)
    with strategy.scope():
        model = train(
            base_model,
            train_data,
            valid_data,
            n_classes=info['n_classes'],
            steps_per_epoch=int(info['n_train']/info['batch_size']),
            input_shape=info['image_size'] + (3,),
            learning_rate=learning_rate,
            preprocess_input=PREPROCESS_INPUT_FN[base_model],
            max_epochs=max_epochs,
            is_chief=is_chief(strategy)
        )

        model.evaluate(
            test_data,
            verbose=(1 if is_chief(strategy) else 0)
        )

    if model_dir is not None:
        if is_chief(strategy):
            model.save(model_dir)
        else:
            with tempfile.TemporaryDirectory() as dest:
                model.save(os.path.join(dest, 'model'))
    print('Model Saved.')


if __name__ == '__main__':
    main()


