# SPDX-License-Identifier: Apache-2.0
"""HeartCodec checkpoint configuration."""
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from sglang_omni.models.heartmoe.codec_flow import FlowMatching
from sglang_omni.models.heartmoe.codec_vocoder import ScalarModel


class HeartCodecConfig(PretrainedConfig):
    model_type = "heartcodec"

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        num_bands: int = 1,
        sample_rate: int = 48000,
        causal: bool = True,
        num_samples: int = 2,
        downsample_factors: list[int] | None = None,
        downsample_kernel_sizes: list[int] | None = None,
        upsample_factors: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        latent_hidden_dim: int = 128,
        default_kernel_size: int = 7,
        delay_kernel_size: int = 5,
        init_channel: int = 64,
        res_kernel_size: int = 7,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.use_cosine_sim = use_cosine_sim
        self.codebook_dim = codebook_dim
        self.num_quantizers = num_quantizers
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.norm_type = norm_type
        self.num_attention_heads = num_attention_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers_2
        self.out_channels = out_channels
        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.causal = causal
        self.num_samples = num_samples
        self.downsample_factors = (
            [3, 4, 4, 4, 5] if downsample_factors is None else downsample_factors
        )
        self.downsample_kernel_sizes = (
            [6, 8, 8, 8, 10]
            if downsample_kernel_sizes is None
            else downsample_kernel_sizes
        )
        self.upsample_factors = (
            [5, 4, 4, 4, 3] if upsample_factors is None else upsample_factors
        )
        self.upsample_kernel_sizes = (
            [10, 8, 8, 8, 6] if upsample_kernel_sizes is None else upsample_kernel_sizes
        )
        self.latent_hidden_dim = latent_hidden_dim
        self.default_kernel_size = default_kernel_size
        self.delay_kernel_size = delay_kernel_size
        self.init_channel = init_channel
        self.res_kernel_size = res_kernel_size


class HeartCodec(nn.Module):
    """Own the shared RVQ, latent generation, and stereo waveform assembly."""

    def __init__(self, config: HeartCodecConfig) -> None:
        super().__init__()
        self.sample_rate = config.sample_rate
        self.flow_matching = FlowMatching(
            dim=config.dim,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            norm_type=config.norm_type,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
        )
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )
