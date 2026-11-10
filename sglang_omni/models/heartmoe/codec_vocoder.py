# SPDX-License-Identifier: Apache-2.0
"""HeartCodec sq codec."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class Conv1d(nn.Conv1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        self.causal = causal
        self.left_padding = dilation * (kernel_size - 1)
        padding = 0 if causal else int((kernel_size * dilation - dilation) / 2)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        else:
            pass
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
    ) -> None:
        padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert kernel_size == 2 * stride
        else:
            pass
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        else:
            pass
        return x


class PreProcessor(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_samples: int,
        kernel_size: int = 7,
        causal: bool = False,
    ) -> None:
        super(PreProcessor, self).__init__()
        self.pooling = torch.nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(
            input_channels, output_channels, kernel_size=kernel_size, causal=causal
        )
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.activation(self.conv(x))
        output = self.pooling(output)
        return output


class PostProcessor(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_samples: int,
        kernel_size: int = 7,
        causal: bool = False,
    ) -> None:
        super(PostProcessor, self).__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(
            input_channels, output_channels, kernel_size=kernel_size, causal=causal
        )
        self.activation = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, 1, 2)
        batch_size, sample_count, channels = x.size()
        x = x.repeat(1, 1, self.num_samples).view(batch_size, -1, channels)
        x = torch.transpose(x, 1, 2)
        output = self.activation(self.conv(x))
        return output


class ResidualUnit(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilation: int,
        res_kernel_size: int = 7,
        causal: bool = False,
    ) -> None:
        super(ResidualUnit, self).__init__()
        self.conv1 = weight_norm(
            Conv1d(
                input_channels,
                output_channels,
                kernel_size=res_kernel_size,
                dilation=dilation,
                causal=causal,
            )
        )
        self.conv2 = weight_norm(
            Conv1d(input_channels, output_channels, kernel_size=1, causal=causal)
        )
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.activation1(self.conv1(x))
        output = self.activation2(self.conv2(output))
        return output + x


class ResEncoderBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        down_kernel_size: int,
        res_kernel_size: int = 7,
        causal: bool = False,
    ) -> None:
        super(ResEncoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    input_channels if i == 0 else output_channels // 2,
                    output_channels // 2,
                    dilation=dilation,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                )
                for i, dilation in enumerate((1, 3, 5, 7, 9))
            ]
        )
        self.down_conv = DownsampleLayer(
            input_channels,
            output_channels,
            down_kernel_size,
            stride=stride,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        x = self.down_conv(x)
        return x


class ResDecoderBlock(nn.Module):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        stride: int,
        up_kernel_size: int,
        res_kernel_size: int = 7,
        causal: bool = False,
    ) -> None:
        super(ResDecoderBlock, self).__init__()
        self.up_conv = UpsampleLayer(
            input_channels,
            output_channels,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
        )
        self.convs = nn.ModuleList(
            [
                ResidualUnit(
                    output_channels,
                    output_channels,
                    dilation=dilation,
                    res_kernel_size=res_kernel_size,
                    causal=causal,
                )
                for dilation in (1, 3, 5, 7, 9)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x


class DownsampleLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.activation = nn.PReLU()
        self.layer = weight_norm(
            Conv1d(in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer(x))


class UpsampleLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.layer = weight_norm(
            ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class ScalarModel(nn.Module):

    def __init__(
        self,
        num_bands: int,
        causal: bool,
        num_samples: int,
        downsample_factors: list[int],
        downsample_kernel_sizes: list[int],
        upsample_factors: list[int],
        upsample_kernel_sizes: list[int],
        latent_hidden_dim: int,
        default_kernel_size: int,
        delay_kernel_size: int,
        init_channel: int,
        res_kernel_size: int,
    ) -> None:
        super(ScalarModel, self).__init__()
        self.encoder = []
        self.decoder = []
        self.encoder.append(
            weight_norm(
                Conv1d(
                    num_bands,
                    init_channel,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        if num_samples > 1:
            self.encoder.append(
                PreProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        else:
            pass
        for i, down_factor in enumerate(downsample_factors):
            self.encoder.append(
                ResEncoderBlock(
                    init_channel * np.power(2, i),
                    init_channel * np.power(2, i + 1),
                    down_factor,
                    downsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        self.encoder.append(
            weight_norm(
                Conv1d(
                    init_channel * np.power(2, len(downsample_factors)),
                    latent_hidden_dim,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        self.decoder.append(
            weight_norm(
                Conv1d(
                    latent_hidden_dim,
                    init_channel * np.power(2, len(upsample_factors)),
                    kernel_size=delay_kernel_size,
                )
            )
        )
        for i, upsample_factor in enumerate(upsample_factors):
            self.decoder.append(
                ResDecoderBlock(
                    init_channel * np.power(2, len(upsample_factors) - i),
                    init_channel * np.power(2, len(upsample_factors) - i - 1),
                    upsample_factor,
                    upsample_kernel_sizes[i],
                    res_kernel_size,
                    causal=causal,
                )
            )
        if num_samples > 1:
            self.decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        else:
            pass
        self.decoder.append(
            weight_norm(
                Conv1d(
                    init_channel,
                    num_bands,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.round(9 * x) / 9
        for layer in self.decoder:
            x = layer(x)
        return x
