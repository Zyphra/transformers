# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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


from ...configuration_utils import PretrainedConfig
from ...utils import (
    logging,
)
from ..auto import CONFIG_MAPPING, AutoConfig
from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


logger = logging.get_logger(__name__)


class Zamba2_VLConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Zamba2_VLForConditionalGeneration`]. It is used
    to instantiate a Zamba2-VL model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[Qwen2_5_VLVisionConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone. Reuses the Qwen2.5-VL vision config.
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object or dictionary of the text backbone.
        image_token_index (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.

    Example:

    ```python
    >>> from transformers import Zamba2_VLForConditionalGeneration, Zamba2_VLConfig

    >>> configuration = Zamba2_VLConfig()
    >>> model = Zamba2_VLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "zamba2_vl"
    sub_configs = {"text_config": AutoConfig, "vision_config": Qwen2_5_VLVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=151646,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "qwen2"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["qwen2"]()

        self.text_config = text_config

        super().__init__(**kwargs)


__all__ = ["Zamba2_VLConfig"]
