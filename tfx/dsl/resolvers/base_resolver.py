# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Base class for TFX resolvers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import inspect
import sys
from typing import Dict, List, Text, Type, Optional, cast

from six import with_metaclass
from tfx import types
from tfx.orchestration import data_types
from tfx.orchestration import metadata
from tfx.proto.orchestration import pipeline_pb2
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact


class ResolveResult(object):
  """The data structure to hold results from Resolver.

  Attributes:
    per_key_resolve_result: a key -> List[Artifact] dict containing the resolved
      artifacts for each source channel with the key as tag.
    per_key_resolve_state: a key -> bool dict containing whether or not the
      resolved artifacts for the channel are considered complete.
    has_complete_result: bool value indicating whether all desired artifacts
      have been resolved.
  """

  def __init__(self, per_key_resolve_result: Dict[Text, List[types.Artifact]],
               per_key_resolve_state: Dict[Text, bool]):
    self.per_key_resolve_result = per_key_resolve_result
    self.per_key_resolve_state = per_key_resolve_state
    self.has_complete_result = all([s for s in per_key_resolve_state.values()])


class BaseResolver(with_metaclass(abc.ABCMeta, object)):
  """Base class for resolver.

  Resolver is the logical unit that will be used optionally for input selection.
  A resolver subclass must override the resolve() function which takes a
  read-only MLMD handler and a dict of <key, Channel> as parameters and produces
  a ResolveResult instance.
  """

  @abc.abstractmethod
  def resolve(
      self,
      pipeline_info: data_types.PipelineInfo,
      metadata_handler: metadata.Metadata,
      source_channels: Dict[Text, types.Channel],
  ) -> ResolveResult:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      pipeline_info: PipelineInfo of the current pipeline. We do not want to
        query artifacts across pipeline boundary.
      metadata_handler: a read-only handler to query MLMD.
      source_channels: a key -> channel dict which contains the info of the
        source channels.

    Returns:
      a ResolveResult instance.

    """
    raise NotImplementedError

  # Following methods make sure all its subclasses are forward compatible with
  # IR.
  def _to_pipeline_into(
      self,
      pipeline_info: pipeline_pb2.PipelineInfo,
      pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
  ) -> data_types.PipelineInfo:
    return data_types.PipelineInfo(
        pipeline_info.id,
        pipeline_runtime_spec.pipeline_root.field_value.string_value,
        pipeline_runtime_spec.pipeline_run_id.field_value.string_value)

  def _to_channels(self,
                   channel: pipeline_pb2.InputSpec.Channel) -> types.Channel:
    return types.Channel(
        type=self._get_artifact_type(channel.artifact_query.type.name),
        producer_component_id=channel.producer_node_query.id,
        output_key=channel.output_key)

  def _get_artifact_type(self, type_name: Text) -> Type[Artifact]:
    standard_artifact_classe_infos = inspect.getmembers(
        sys.modules[standard_artifacts.__name__], inspect.isclass)
    for artifact_class_info in standard_artifact_classe_infos:
      artifact_class = artifact_class_info[1]
      if artifact_class.TYPE_NAME == type_name:
        return cast(Type[Artifact], artifact_class)
    raise ValueError(
        "Type {} is not defined in tfx.types.standard_artifacts".format(
            type_name))

  def Resolve(
      self, metadata_handler: metadata.Metadata,
      pipeline_info: pipeline_pb2.PipelineInfo,
      pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
      node_inputs: pipeline_pb2.NodeInputs
  ) -> Optional[Dict[Text, List[types.Artifact]]]:
    """Resolves artifact given IR.

    Args:
      metadata_handler: A metadata handler to access MLMD store.
      pipeline_info: The information of the pipeline that this node runs in.
      pipeline_runtime_spec: The runtime information of the pipeline that this
        node runs in.
      node_inputs: A pipeline_pb2.NodeInputs message that instructs artifact
        resolution for a pipeline node.

    Returns:
      If `min_count` for every input is met, returns a
      Dict[str, List[Artifact]]. Otherwise, return None.
    """
    legact_pipeline_info = self._to_pipeline_into(pipeline_info,
                                                  pipeline_runtime_spec)
    source_channels = {}
    for k, input_spec in node_inputs.inputs.items():
      if len(input_spec.channels) != 1:
        raise ValueError(
            "input_spec of input {} doesn't have exact one channel: {}".format(
                k, input_spec))
      source_channels[k] = self._to_channels(input_spec.channels[0])
    resolve_result = self.resolve(legact_pipeline_info, metadata_handler,
                                  source_channels)
    return (resolve_result.per_key_resolve_result
            if resolve_result.has_complete_result else None)
