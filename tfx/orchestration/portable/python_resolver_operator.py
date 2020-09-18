# Copyright 2020 Google LLC. All Rights Reserved.
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
"""In process inplementation of Resolvers."""

from typing import Dict, List, Optional

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import base_resolver_operator
from tfx.orchestration.portable.resolvers import default_resolver
from tfx.proto.orchestration import pipeline_pb2


class PythonResolverOperator(base_resolver_operator.BaseResolverOperator):
  """PythonResolverOperator resolves artifacts in process."""

  _RESOLVER_POLICY_TO_RESOLVER_CLASS = {
      pipeline_pb2.ResolverConfig.RESOLVER_POLICY_UNSPECIFIED:
          default_resolver.DefaultResolver
  }
  """Base class for resolver operators.

  Resolver is the logical unit that will be used optionally for input selection.
  A resolver subclass must override the resolve() function which takes a
  read-only MLMD handler and a dict of <key, Channel> as parameters and produces
  a ResolveResult instance.
  """

  def __init__(self, pipeline_info: pipeline_pb2.PipelineInfo,
               pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
               node_inputs: pipeline_pb2.NodeInputs):
    super().__init__(pipeline_info, pipeline_runtime_spec, node_inputs)
    self._resolver = self._RESOLVER_POLICY_TO_RESOLVER_CLASS.get(
        self._node_inputs.resolver_config.resolver_policy)()

  def ResolveInputs(
      self, metadata_handler: metadata.Metadata
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves artifacts from channels by querying MLMD.

    Args:
      metadata_handler: A metadata handler to access MLMD store.

    Returns:
      If `min_count` for every input is met, returns a
      Dict[str, List[Artifact]]. Otherwise, return None.
    """
    return self._resolver.Resolve(metadata_handler, self._pipeline_info,
                                  self._pipeline_runtime_spec,
                                  self._node_inputs)
