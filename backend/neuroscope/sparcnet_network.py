"""SPaRCNet architecture from bdsp-core/IIIC-SPaRCNet.
Jing et al., Neurology 2023. Upstream d374d5137fe30e595d686ecc1b584a2e551f1aa0.
Class definitions retained for exact checkpoint compatibility; see THIRD_PARTY_NOTICES.md.
"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, conv_bias, batch_norm):
        super(_DenseLayer, self).__init__()
        if batch_norm:
            (self.add_module('norm1', nn.BatchNorm1d(num_input_features)),)
        (self.add_module('elu1', nn.ELU()),)
        (self.add_module('conv1', nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=conv_bias)),)
        if batch_norm:
            (self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),)
        (self.add_module('elu2', nn.ELU()),)
        (self.add_module('conv2', nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=conv_bias)),)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, conv_bias, batch_norm):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, conv_bias, batch_norm)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, conv_bias, batch_norm):
        super(_Transition, self).__init__()
        if batch_norm:
            self.add_module('norm', nn.BatchNorm1d(num_input_features))
        self.add_module('elu', nn.ELU())
        self.add_module('conv', nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=conv_bias))
        self.add_module('pool', nn.AvgPool1d(kernel_size=2, stride=2))

class DenseNetEnconder(nn.Module):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False):
        super(DenseNetEnconder, self).__init__()
        first_conv = OrderedDict([('conv0', nn.Conv1d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=conv_bias))])
        if batch_norm:
            first_conv['norm0'] = nn.BatchNorm1d(num_init_features)
        first_conv['elu0'] = nn.ELU()
        first_conv['pool0'] = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.densenet = nn.Sequential(first_conv)
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, conv_bias=conv_bias, batch_norm=batch_norm)
            self.densenet.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, conv_bias=conv_bias, batch_norm=batch_norm)
                self.densenet.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        if batch_norm:
            self.densenet.add_module('norm{}'.format(len(block_config) + 1), nn.BatchNorm1d(num_features))
        self.densenet.add_module('relu{}'.format(len(block_config) + 1), nn.ReLU())
        self.densenet.add_module('pool{}'.format(len(block_config) + 1), nn.AvgPool1d(kernel_size=7, stride=3))
        self.num_features = num_features
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.densenet(x)
        return features.view(features.size(0), -1)

class DenseNetClassifier(nn.Module):

    def __init__(self, growth_rate=32, block_config=(4, 4, 4, 4, 4, 4, 4), in_channels=16, num_init_features=64, bn_size=4, drop_rate=0.2, conv_bias=True, batch_norm=False, drop_fc=0.5, num_classes=6):
        super(DenseNetClassifier, self).__init__()
        self.features = DenseNetEnconder(growth_rate=growth_rate, block_config=block_config, in_channels=in_channels, num_init_features=num_init_features, bn_size=bn_size, drop_rate=drop_rate, conv_bias=conv_bias, batch_norm=batch_norm)
        self.classifier = nn.Sequential(nn.Dropout(p=drop_fc), nn.Linear(self.features.num_features, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = self.classifier(features)
        return (out, features)
