from __future__ import annotations

import math
import os
from collections import OrderedDict
from itertools import product
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models._utils as _utils
from torch import Tensor, nn


_RGB_MEAN = (104, 117, 123)
_MODEL_CACHE: dict[tuple[str, str, str], "RetinaFaceDetector"] = {}


def get_config(network: str) -> dict[str, Any]:
    configs = {
        "mobilenetv1": {
            "name": "mobilenet_v1",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
            "in_channel": 128,
            "out_channel": 128,
        },
        "mobilenetv1_0.25": {
            "name": "mobilenet0.25",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
            "in_channel": 32,
            "out_channel": 64,
        },
        "mobilenetv1_0.50": {
            "name": "mobilenet0.50",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
            "in_channel": 64,
            "out_channel": 128,
        },
        "mobilenetv2": {
            "name": "mobilenet_v2",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": [6, 13, 18],
            "in_channel": 32,
            "out_channel": 128,
        },
        "resnet18": {
            "name": "resnet18",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
            "in_channel": 64,
            "out_channel": 128,
        },
        "resnet34": {
            "name": "resnet34",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
            "in_channel": 64,
            "out_channel": 128,
        },
        "resnet50": {
            "name": "resnet50",
            "min_sizes": [[16, 32], [64, 128], [256, 512]],
            "steps": [8, 16, 32],
            "variance": [0.1, 0.2],
            "clip": False,
            "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
            "in_channel": 256,
            "out_channel": 256,
        },
    }
    cfg = configs.get(network)
    if cfg is None:
        raise KeyError(f"Unsupported RetinaFace network: {network}")
    return dict(cfg)


def _make_divisible(value: float, divisor: int = 8) -> int:
    new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


class Conv2dNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        norm_layer: type[nn.Module] | None = nn.BatchNorm2d,
        activation_layer: type[nn.Module] | None = nn.LeakyReLU,
        dilation: int = 1,
        inplace: bool | None = True,
        negative_slope: float | None = None,
        bias: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            params: dict[str, Any] = {} if inplace is None else {"inplace": inplace}
            if negative_slope is not None:
                params["negative_slope"] = negative_slope
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class DepthWiseSeparableConv2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        super().__init__(
            Conv2dNormActivation(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                groups=in_channels,
                negative_slope=0.1,
            ),
            Conv2dNormActivation(in_channels, out_channels, kernel_size=1, negative_slope=0.1),
        )


class SSH(nn.Module):
    def __init__(self, in_channel: int, out_channels: int) -> None:
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("Output channel must be divisible by 4.")
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.conv3X3 = Conv2dNormActivation(in_channel, out_channels // 2, kernel_size=3, activation_layer=None)
        self.conv5X5_1 = Conv2dNormActivation(in_channel, out_channels // 4, kernel_size=3, negative_slope=leaky)
        self.conv5X5_2 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None,
        )
        self.conv7X7_2 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            negative_slope=leaky,
        )
        self.conv7x7_3 = Conv2dNormActivation(
            out_channels // 4,
            out_channels // 4,
            kernel_size=3,
            activation_layer=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        conv3x3 = self.conv3X3(x)
        conv5x5 = self.conv5X5_2(self.conv5X5_1(x))
        conv7x7 = self.conv7x7_3(self.conv7X7_2(self.conv5X5_1(x)))
        out = torch.cat([conv3x3, conv5x5, conv7x7], dim=1)
        return F.relu(out, inplace=True)


class FPN(nn.Module):
    def __init__(self, in_channels_list: list[int], out_channels: int) -> None:
        super().__init__()
        leaky = 0.1 if out_channels <= 64 else 0.0
        self.output1 = Conv2dNormActivation(in_channels_list[0], out_channels, kernel_size=1, negative_slope=leaky)
        self.output2 = Conv2dNormActivation(in_channels_list[1], out_channels, kernel_size=1, negative_slope=leaky)
        self.output3 = Conv2dNormActivation(in_channels_list[2], out_channels, kernel_size=1, negative_slope=leaky)
        self.merge1 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, negative_slope=leaky)
        self.merge2 = Conv2dNormActivation(out_channels, out_channels, kernel_size=3, negative_slope=leaky)

    def forward(self, inputs: OrderedDict[str, Tensor] | dict[str, Tensor]) -> list[Tensor]:
        features = list(inputs.values())
        output1 = self.output1(features[0])
        output2 = self.output2(features[1])
        output3 = self.output3(features[2])
        output2 = self.merge2(output2 + F.interpolate(output3, size=output2.shape[2:], mode="nearest"))
        output1 = self.merge1(output1 + F.interpolate(output2, size=output1.shape[2:], mode="nearest"))
        return [output1, output2, output3]


class IntermediateLayerGetterByIndex(nn.Module):
    def __init__(self, model: nn.Module, indexes: list[int]) -> None:
        super().__init__()
        self.features = model.features
        self.indexes = indexes

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        outputs: OrderedDict[str, Tensor] = OrderedDict()
        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in self.indexes:
                outputs[f"layer_{idx}"] = x
        return outputs


class MobileNetV1(nn.Module):
    def __init__(self, width_mult: float) -> None:
        super().__init__()
        filters = [_make_divisible(value * width_mult) for value in [32, 64, 128, 256, 512, 1024]]
        self.stage1 = nn.Sequential(
            Conv2dNormActivation(3, filters[0], kernel_size=3, stride=2, negative_slope=0.1),
            DepthWiseSeparableConv2d(filters[0], filters[1], stride=1),
            DepthWiseSeparableConv2d(filters[1], filters[2], stride=2),
            DepthWiseSeparableConv2d(filters[2], filters[2], stride=1),
            DepthWiseSeparableConv2d(filters[2], filters[3], stride=2),
            DepthWiseSeparableConv2d(filters[3], filters[3], stride=1),
        )
        self.stage2 = nn.Sequential(
            DepthWiseSeparableConv2d(filters[3], filters[4], stride=2),
            DepthWiseSeparableConv2d(filters[4], filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], filters[4], stride=1),
            DepthWiseSeparableConv2d(filters[4], filters[4], stride=1),
        )
        self.stage3 = nn.Sequential(
            DepthWiseSeparableConv2d(filters[4], filters[5], stride=2),
            DepthWiseSeparableConv2d(filters[5], filters[5], stride=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


def mobilenet_v1_025() -> MobileNetV1:
    return MobileNetV1(width_mult=0.25)


def mobilenet_v1_050() -> MobileNetV1:
    return MobileNetV1(width_mult=0.50)


def mobilenet_v1() -> MobileNetV1:
    return MobileNetV1(width_mult=1.0)


class InvertedResidual(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")
        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = stride == 1 and in_planes == out_planes
        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(in_planes, hidden_dim, kernel_size=1, activation_layer=nn.ReLU6))
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(input_channel, 8)
        self.last_channel = _make_divisible(last_channel, 8)
        features: list[nn.Module] = [
            Conv2dNormActivation(3, input_channel, stride=2, activation_layer=nn.ReLU6)
        ]
        for expand_ratio, channels, num_blocks, stride in inverted_residual_setting:
            output_channel = _make_divisible(channels, 8)
            for index in range(num_blocks):
                block_stride = stride if index == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, block_stride, expand_ratio))
                input_channel = output_channel
        features.append(Conv2dNormActivation(input_channel, self.last_channel, kernel_size=1, activation_layer=nn.ReLU6))
        self.features = nn.Sequential(*features)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


def mobilenet_v2() -> MobileNetV2:
    return MobileNetV2()


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        width = out_channels
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock] | type[Bottleneck], layers: list[int]) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers: list[nn.Module] = [block(self.in_channels, planes, stride, downsample)]
        self.in_channels = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18() -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2])


def resnet34() -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3])


def resnet50() -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3])


def get_layer_extractor(cfg: dict[str, Any], backbone: nn.Module) -> nn.Module:
    if cfg["name"] == "mobilenet_v2":
        return IntermediateLayerGetterByIndex(backbone, [6, 13, 18])
    return _utils.IntermediateLayerGetter(backbone, cfg["return_layers"])


def build_backbone(name: str) -> nn.Module:
    backbone_map = {
        "mobilenet0.25": mobilenet_v1_025,
        "mobilenet0.50": mobilenet_v1_050,
        "mobilenet_v1": mobilenet_v1,
        "mobilenet_v2": mobilenet_v2,
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
    }
    try:
        return backbone_map[name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported backbone name: {name}") from exc


class ClassHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.class_head = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 2, kernel_size=1) for _ in range(fpn_num)]
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        outputs = [layer(feature).permute(0, 2, 3, 1).contiguous() for feature, layer in zip(features, self.class_head)]
        return torch.cat([output.view(output.shape[0], -1, 2) for output in outputs], dim=1)


class BboxHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.bbox_head = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=1) for _ in range(fpn_num)]
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        outputs = [layer(feature).permute(0, 2, 3, 1).contiguous() for feature, layer in zip(features, self.bbox_head)]
        return torch.cat([output.view(output.shape[0], -1, 4) for output in outputs], dim=1)


class LandmarkHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int = 2, fpn_num: int = 3) -> None:
        super().__init__()
        self.landmark_head = nn.ModuleList(
            [nn.Conv2d(in_channels=in_channels, out_channels=num_anchors * 10, kernel_size=1) for _ in range(fpn_num)]
        )

    def forward(self, features: list[Tensor]) -> Tensor:
        outputs = [layer(feature).permute(0, 2, 3, 1).contiguous() for feature, layer in zip(features, self.landmark_head)]
        return torch.cat([output.view(output.shape[0], -1, 10) for output in outputs], dim=1)


class RetinaFace(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        backbone = build_backbone(cfg["name"])
        self.fx = get_layer_extractor(cfg, backbone)
        if cfg["name"] == "mobilenet_v2":
            fpn_in_channels = [32, 96, 1280]
        else:
            base_in_channels = cfg["in_channel"]
            fpn_in_channels = [base_in_channels * 2, base_in_channels * 4, base_in_channels * 8]
        out_channels = cfg["out_channel"]
        self.fpn = FPN(fpn_in_channels, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.class_head = ClassHead(in_channels=out_channels)
        self.bbox_head = BboxHead(in_channels=out_channels)
        self.landmark_head = LandmarkHead(in_channels=out_channels)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.fx(x)
        fpn = self.fpn(out)
        features = [self.ssh1(fpn[0]), self.ssh2(fpn[1]), self.ssh3(fpn[2])]
        classifications = self.class_head(features)
        bbox_regressions = self.bbox_head(features)
        landmark_regressions = self.landmark_head(features)
        if self.training:
            return bbox_regressions, classifications, landmark_regressions
        return bbox_regressions, F.softmax(classifications, dim=-1), landmark_regressions


class PriorBox:
    def __init__(self, cfg: dict[str, Any], image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        self.clip = cfg["clip"]
        self.steps = cfg["steps"]
        self.min_sizes = cfg["min_sizes"]
        self.feature_maps = [[math.ceil(image_size[0] / step), math.ceil(image_size[1] / step)] for step in self.steps]

    def generate_anchors(self) -> torch.Tensor:
        anchors: list[float] = []
        for level, (map_height, map_width) in enumerate(self.feature_maps):
            step = self.steps[level]
            for row, col in product(range(map_height), range(map_width)):
                for min_size in self.min_sizes[level]:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [value * step / self.image_size[1] for value in [col + 0.5]]
                    dense_cy = [value * step / self.image_size[0] for value in [row + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors.extend([cx, cy, s_kx, s_ky])
        output = torch.tensor(anchors, dtype=torch.float32).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def decode(loc: Tensor, priors: Tensor, variances: list[float]) -> Tensor:
    centers = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
    wh = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    boxes = torch.empty_like(loc)
    boxes[:, :2] = centers - wh / 2
    boxes[:, 2:] = centers + wh / 2
    return boxes


def decode_landmarks(predictions: Tensor, priors: Tensor, variances: list[float]) -> Tensor:
    predictions = predictions.view(predictions.size(0), 5, 2)
    landmarks = priors[:, :2].unsqueeze(1) + predictions * variances[0] * priors[:, 2:].unsqueeze(1)
    return landmarks.view(landmarks.size(0), -1)


def nms(dets: np.ndarray, threshold: float) -> list[int]:
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width * height
        union = areas[index] + areas[order[1:]] - intersection
        overlap = intersection / np.maximum(union, 1e-12)
        remaining = np.where(overlap <= threshold)[0]
        order = order[remaining + 1]
    return keep


def _default_device() -> torch.device:
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def resolve_device(device_name: str | None) -> torch.device:
    name = (device_name or "auto").strip().lower()
    if name == "auto":
        return _default_device()
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for RetinaFace, but CUDA is not available.")
    if name == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise ValueError("MPS was requested for RetinaFace, but MPS is not available.")
    return torch.device(name)


def _device_key(device: torch.device) -> str:
    if device.index is None:
        return device.type
    return f"{device.type}:{device.index}"


def _load_checkpoint(path: str, device: torch.device) -> dict[str, Any]:
    if path.lower().endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path, device="cpu")
    return torch.load(path, map_location=device)


def _extract_state_dict(raw_state: Any) -> dict[str, Tensor]:
    if not isinstance(raw_state, dict):
        raise TypeError("Unsupported RetinaFace checkpoint format.")
    state_dict = raw_state
    for key in ("state_dict", "model_state_dict", "model"):
        maybe_state = state_dict.get(key)
        if isinstance(maybe_state, dict):
            state_dict = maybe_state
            break
    cleaned: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        cleaned_key = key
        for prefix in ("module.", "model."):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[len(prefix) :]
        cleaned[cleaned_key] = value
    if not cleaned:
        raise ValueError("RetinaFace checkpoint does not contain tensor weights.")
    return cleaned


class RetinaFaceDetector:
    def __init__(self, *, weights_path: str, network: str, device: torch.device) -> None:
        self.weights_path = weights_path
        self.network = network
        self.device = device
        self.cfg = get_config(network)
        self.model = RetinaFace(self.cfg).to(device)
        checkpoint = _load_checkpoint(weights_path, device)
        state_dict = _extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        image_rgb: np.ndarray,
        *,
        conf_threshold: float = 0.6,
        pre_nms_topk: int = 5000,
        nms_threshold: float = 0.4,
        post_nms_topk: int = 750,
        resize_short_edge: int = 512,
    ) -> np.ndarray:
        original_height, original_width = image_rgb.shape[:2]
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        scale = 1.0
        if resize_short_edge > 0 and min(original_height, original_width) < resize_short_edge:
            scale = resize_short_edge / float(min(original_height, original_width))
            new_width = max(1, int(round(original_width * scale)))
            new_height = max(1, int(round(original_height * scale)))
            image_bgr = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        image = np.float32(image_bgr)
        image_height, image_width, _ = image.shape
        image -= np.array(_RGB_MEAN, dtype=np.float32)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        loc, conf, landmarks = self.model(image_tensor)
        loc = loc.squeeze(0)
        conf = conf.squeeze(0)
        landmarks = landmarks.squeeze(0)

        priors = PriorBox(self.cfg, image_size=(image_height, image_width)).generate_anchors().to(self.device)
        boxes = decode(loc, priors, self.cfg["variance"])
        landmarks = decode_landmarks(landmarks, priors, self.cfg["variance"])

        bbox_scale = torch.tensor([image_width, image_height, image_width, image_height], device=self.device)
        boxes = (boxes * bbox_scale).cpu().numpy()
        landmark_scale = torch.tensor([image_width, image_height] * 5, device=self.device)
        landmarks = (landmarks * landmark_scale).cpu().numpy()
        scores = conf[:, 1].cpu().numpy()

        keep = scores > conf_threshold
        boxes = boxes[keep]
        landmarks = landmarks[keep]
        scores = scores[keep]
        if boxes.shape[0] == 0:
            return np.zeros((0, 15), dtype=np.float32)

        order = scores.argsort()[::-1][:pre_nms_topk]
        boxes = boxes[order]
        landmarks = landmarks[order]
        scores = scores[order]

        detections = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(detections, nms_threshold)
        detections = detections[keep][:post_nms_topk]
        landmarks = landmarks[keep][:post_nms_topk]

        if scale != 1.0:
            detections[:, :4] /= scale
            landmarks /= scale

        return np.concatenate((detections, landmarks), axis=1).astype(np.float32, copy=False)


def get_retinaface_detector(*, weights_path: str, network: str, device_name: str | None = None) -> RetinaFaceDetector:
    resolved_path = os.path.abspath(weights_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"RetinaFace weights not found: {resolved_path}")
    device = resolve_device(device_name)
    cache_key = (resolved_path, network, _device_key(device))
    detector = _MODEL_CACHE.get(cache_key)
    if detector is None:
        detector = RetinaFaceDetector(weights_path=resolved_path, network=network, device=device)
        _MODEL_CACHE[cache_key] = detector
    return detector


def prepare_retinaface_model(*, weights_path: str, network: str, device_name: str | None = None) -> dict[str, Any]:
    detector = get_retinaface_detector(weights_path=weights_path, network=network, device_name=device_name)
    return {
        "detector": detector,
        "weights_path": detector.weights_path,
        "network": detector.network,
        "device": _device_key(detector.device),
    }
