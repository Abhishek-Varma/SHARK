# Copyright 2020 The Nod Team. All rights reserved.
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

from torch_mlir_utils import get_torch_mlir_module
from iree_utils import get_results, get_iree_compiled_module


class SharkRunner:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool,
        device: str,
        tracing_required: bool,
        from_aot: bool,
    ):
        self.torch_module = model
        self.input = input
        self.torch_mlir_module = get_torch_mlir_module(
            model, input, dynamic, tracing_required, from_aot
        )
        self.iree_compilation_module = get_iree_compiled_module(
            self.torch_mlir_module, device
        )

    def forward(self, input):
        return get_results(self.iree_compilation_module, input)


class SharkInference:
    """TODO: Write the description"""

    def __init__(
        self,
        model,
        input: tuple,
        dynamic: bool = False,
        device: str = "cpu",
        jit_trace: bool = False,
        from_aot: bool = False,
    ):
        self.shark_runner = SharkRunner(
            model, input, dynamic, device, jit_trace, from_aot
        )

    def forward(self, input):
        return self.shark_runner.forward(input)


class SharkTrainer:
    """TODO: Write the description"""

    def __init__(self):
        pass