import os
from typing import List, Tuple, Union, Dict

import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def build_dataset(
    paths: List[str],
    labels: list[int]
) -> tf.data.Dataset:
    def load_image(path):
        jpeg = tf.io.read_file(path)
        image = tf.image.decode_jpeg(jpeg)
        return image

    images = tf.data.Dataset.from_tensor_slices(paths).map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    labels =tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((images, labels))


def load_csv(
    path: str,
    shuffle: bool=True
)-> Tuple[Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset], List[str]]:
    df = pd.read_csv(
        path,
        header=None,
        names=('phase', 'path', 'label')
    )
    label_names = df[df['phase'] == 'TRAIN']['label'].unique().tolist()
    label_index = {label:index for index, label in enumerate(label_names)}
    df['label'] = df['label'].map(lambda x: label_index[x])

    if shuffle:
        df = df.sample(frac=1)
    train_df = df[df['phase'] == 'TRAIN']
    valid_df = df[df['phase'] == 'VALIDATION']
    test_df = df[df['phase'] == 'TEST']

    ds_train = build_dataset(train_df['path'], train_df['path'])
    ds_valid = build_dataset(valid_df['path'], valid_df['path'])
    ds_test = build_dataset(test_df['path'], test_df['path'])
    return (ds_train, ds_valid, ds_test), label_names


def preprocess_dataset(
    dataset: tf.data.Dataset,
    n_classes: int,
    batch_size: int=32,
    buffer_size: int=1000,
    target_size: Union[Tuple[Union[int, None], Union[int, None]]]=(224, 224),
    repeat: bool=False,
    shuffle: bool=False
) -> tf.data.Dataset:
    dataset = dataset.map(
        lambda x, y: (
            tf.image.resize(x, target_size),
            tf.one_hot(y, n_classes)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_dataset(
    dataset_name: str,
    batch_size: int=32,
    valid_batch_size: int=32,
    buffer_size: int=1000,
    image_size: Tuple[int, int]=(224, 224)
)->Tuple[Tuple[tf.data.Dataset, tf.data.Dataset], Dict]:
    if dataset_name.endswith('.csv'):
        (ds_train, ds_valid, ds_test), label_names = load_csv(dataset_name)
        n_classes = len(label_names)
    else:
        if dataset_name == 'food101':
            (ds_train, ds_valid, ds_test), ds_info = tfds.load(
                dataset_name,
                split=['train[:80%]', 'train[80%:]', 'validation'],
                as_supervised=True,
                with_info=True
            )
        else:
            (ds_train, ds_valid, ds_test), ds_info = tfds.load(
                dataset_name,
                split=['train', 'validation', 'test'],
                as_supervised=True,
                with_info=True
            )
        label_names = ds_info.features['label'].names
        n_classes = len(label_names)
        
    info = {
        'n_train': len(ds_train),
        'n_valid': len(ds_valid),
        'n_test': len(ds_test),
        'n_classes': n_classes,
        'batch_size': batch_size,
        'labels': label_names,
        'image_size': image_size
    }

    ds_train = preprocess_dataset(
        dataset=ds_train,
        n_classes=n_classes,
        batch_size=batch_size,
        target_size=image_size,
        buffer_size=buffer_size,
        repeat=True,
        shuffle=True
    )
    ds_valid = preprocess_dataset(
        dataset=ds_valid,
        n_classes=n_classes,
        batch_size=valid_batch_size,
        target_size=image_size,
        buffer_size=buffer_size,
        repeat=False,
        shuffle=False
    )
    ds_test = preprocess_dataset(
        dataset=ds_test,
        n_classes=n_classes,
        batch_size=valid_batch_size,
        target_size=image_size,
        buffer_size=buffer_size,
        repeat=False,
        shuffle=False
    )

    return (ds_train, ds_valid, ds_test), info