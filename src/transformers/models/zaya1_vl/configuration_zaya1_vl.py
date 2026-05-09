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


logger = logging.get_logger(__name__)


from ..qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig


class Zaya1VLConfig(PretrainedConfig):
    r"""
    This is the configuration class for [`Zaya1VLForConditionalGeneration`]. It stores the Zaya text backbone
    parameters directly and the vision backbone configuration used to instantiate a Zaya1-VL model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[Qwen2_5_VLVisionConfig, dict]`, *optional*):
            The config object or dictionary of the vision backbone.
        image_token_id (`int`, *optional*, defaults to 151646):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.

    Example:

    ```python
    >>> from transformers import Zaya1VLConfig, Zaya1VLForConditionalGeneration

    >>> configuration = Zaya1VLConfig()
    >>> model = Zaya1VLForConditionalGeneration(configuration)
    >>> configuration = model.config
    ```"""

    model_type = "zaya1_vl"
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        cca=True,
        num_query_groups=2,
        use_cache=True,
        attention_bias=False,
        lm_head_bias=False,
        vocab_size=262272,
        hidden_size=2048,
        ffn_hidden_size=4096,
        num_hidden_layers=80,
        num_experts=16,
        num_attention_heads=8,
        head_dim=128,
        activation_func="swiglu",
        max_position_embeddings=131072,
        norm_epsilon=1e-05,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        rope_theta=5000000,
        attention_dropout=0.0,
        moe_router_topk=1,
        normalization="RMSNorm",
        zaya_mlp_expansion=256,
        zaya_use_mod=True,
        zaya_high_prec=True,
        zaya_use_eda=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        scale_residual_merge=True,
        fused_add_norm=False,
        residual_in_fp32=True,
        apply_rope_fusion=True,
        bias_activation_fusion=True,
        activation_func_fp8_input_store=False,
        sliding_window=None,
        rope_scaling=None,
        rope_parameters=None,
        partial_rotary_factor=0.5,
        num_key_value_heads=2,
        clamp_temp=False,
        cca_time0=2,
        cca_time1=2,
        swa_layers=None,
        swa_rotary_base=None,
        _attn_implementation="eager",

        vision_config=None,
        initializer_range=0.02,
        vision_lora=False,
        vision_lora_rank_attn=None,
        vision_lora_rank_mlp=None,
        use_lora_att=False,
        lora_rank=0,
        image_token_id=151646,
        vision_start_token_id=None,
        vision_end_token_id=None,
        projector_hidden_act="gelu",
        **kwargs,
    ):
        self.cca = cca
        self.num_query_groups = num_query_groups
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.lm_head_bias = lm_head_bias
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert self.head_dim is not None
        assert self.num_query_groups == num_key_value_heads
        self.num_key_value_heads = num_key_value_heads
        self.activation_func = activation_func
        self.max_position_embeddings = max_position_embeddings
        self.norm_epsilon = norm_epsilon
        self.normalization = normalization
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_dropout = attention_dropout
        self.moe_router_topk = moe_router_topk
        self.zaya_mlp_expansion = zaya_mlp_expansion
        self.zaya_use_mod = zaya_use_mod
        self.zaya_high_prec = zaya_high_prec
        self.zaya_use_eda = zaya_use_eda
        self.add_bias_linear = add_bias_linear
        self.gated_linear_unit = gated_linear_unit
        self.scale_residual_merge = scale_residual_merge
        self.residual_in_fp32 = residual_in_fp32
        self.bias_activation_fusion = bias_activation_fusion
        self.activation_func_fp8_input_store = activation_func_fp8_input_store
        self.sliding_window = sliding_window
        self.partial_rotary_factor = partial_rotary_factor
        if "rotary_base" in kwargs:
            rope_theta = kwargs.pop("rotary_base")
        self.rope_theta = rope_theta
        if isinstance(rope_parameters, dict):
            rope_parameters = dict(rope_parameters)
        elif isinstance(rope_scaling, dict):
            rope_parameters = dict(rope_scaling)
        else:
            rope_parameters = {"rope_type": "default"}
        if "type" in rope_parameters:
            rope_parameters.setdefault("rope_type", rope_parameters.pop("type"))
        rope_parameters.setdefault("rope_theta", rope_theta)
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.rope_parameters = rope_parameters
        self.num_key_value_heads = num_key_value_heads
        self.clamp_temp = clamp_temp
        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.swa_layers = swa_layers
        self.swa_rotary_base = swa_rotary_base
        self._attn_implementation = _attn_implementation

        self.fused_add_norm = fused_add_norm
        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config
        self.initializer_range = initializer_range
        self.vision_lora = vision_lora
        if self.vision_lora:
            self.vision_lora_rank_attn = vision_lora_rank_attn
            self.vision_lora_rank_mlp = vision_lora_rank_mlp
        self.use_lora_att = use_lora_att
        self.lora_rank = lora_rank

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            **kwargs,
        )


__all__ = ["Zaya1VLConfig"]
