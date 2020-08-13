# Lint as: python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX template iris model.

A DNN keras model which uses features defined in features.py and network
parameters defined in constants.py.
"""

import os
from typing import List, Text, Mapping, Any
from absl import logging
import kerastuner
import tensorflow as tf
from tensorflow import keras
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.executor import TrainerFnArgs

from tfx.experimental.templates.iris.models import constants
from tfx.experimental.templates.iris.models import features

from tfx.utils import io_utils

from tensorflow_metadata.proto.v0 import schema_pb2


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, training_feature_spec):
  """Returns a function that parses a serialized tf.Example."""

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = dict(training_feature_spec)
    feature_spec.pop(features.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    return model(parsed_features)

  return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              feature_spec: Mapping[Text, Any],
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    feature_spec: Feature spec of examples.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  return tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=feature_spec,
      reader=_gzip_reader_fn,
      label_key=features.LABEL_KEY)


def _get_feature_spec(schema_file: Text) -> Mapping[Text, Any]:
  schema = io_utils.parse_pbtxt_file(schema_file, schema_pb2.Schema())
  return schema_utils.schema_as_feature_spec(schema).feature_spec


def _get_fixed_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kerastuner.HyperParameters()
  hp.Fixed('learning_rate', constants.LEARNING_RATE)
  hp.Fixed('num_layers', constants.NUM_LAYERS)
  return hp


def _build_keras_model(hparams: kerastuner.HyperParameters) -> tf.keras.Model:
  """Creates a DNN Keras model for classifying iris data.

  Args:
    hparams: Holds HyperParameters for tuning.

  Returns:
    A Keras Model.
  """
  # The model below is built with Functional API, please refer to
  # https://www.tensorflow.org/guide/keras/overview for all API options.
  inputs = [
      keras.layers.Input(shape=(1,), name=f) for f in features.FEATURE_KEYS
  ]
  d = keras.layers.concatenate(inputs)
  for _ in range(int(hparams.get('num_layers'))):
    d = keras.layers.Dense(constants.HIDDEN_LAYER_UNITS, activation='relu')(d)
  outputs = keras.layers.Dense(
      constants.OUTPUT_LAYER_UNITS, activation='softmax')(
          d)

  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(hparams.get('learning_rate')),
      loss='sparse_categorical_crossentropy',
      metrics=[keras.metrics.SparseCategoricalAccuracy()])

  model.summary(print_fn=logging.info)
  return model


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.

  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """
  feature_spec = _get_feature_spec(fn_args.schema_file)
  train_dataset = _input_fn(
      fn_args.train_files, feature_spec, batch_size=constants.TRAIN_BATCH_SIZE)
  eval_dataset = _input_fn(
      fn_args.eval_files, feature_spec, batch_size=constants.EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_fixed_hyperparameters()
  logging.info('HyperParameters for training: %s', hparams.get_config())

  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = _build_keras_model(hparams)

  steps_per_epoch = constants.TRAIN_DATA_SIZE / constants.TRAIN_BATCH_SIZE

  try:
    log_dir = fn_args.model_run_dir
  except KeyError:
    # TODO(b/158106209): use ModelRun instead of Model artifact for logging.
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

  # Write logs to path
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir, update_freq='batch')

  model.fit(
      train_dataset,
      epochs=int(fn_args.train_steps / steps_per_epoch),
      steps_per_epoch=steps_per_epoch,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model, feature_spec).get_concrete_function(
              tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')),
  }
  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
