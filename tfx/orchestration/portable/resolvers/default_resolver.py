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
"""The Resolver when Resolver Policy is not specified."""

from typing import Dict, List, Optional

from tfx import types
from tfx.orchestration import metadata
from tfx.orchestration.portable import inputs_utils
from tfx.orchestration.portable.resolvers import base_resolver
from tfx.proto.orchestration import pipeline_pb2


class DefaultResolver(base_resolver.BaseResolver):
  """Default resolver resolves all input from upstream nodes with in the same pipeline run."""

  def Resolve(
      self, metadata_handler: metadata.Metadata,
      pipeline_info: pipeline_pb2.PipelineInfo,
      pipeline_runtime_spec: pipeline_pb2.PipelineRuntimeSpec,
      node_inputs: pipeline_pb2.NodeInputs
  ) -> Optional[Dict[str, List[types.Artifact]]]:
    """Resolves all input from upstream nodes with in the same pipeline run.

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
    del pipeline_info, pipeline_runtime_spec  # unused
    return inputs_utils.resolve_input_artifacts(metadata_handler, node_inputs)
