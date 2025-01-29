import torch
from torch import nn
from typing import List, Union
from torch.nn import functional as F
from torch import distributions as D


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, num_layers: int, dropout_val: float):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.ReLU
        if num_layers == 0:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True), nn.ReLU(), nn.Dropout(dropout_val)])
            for i in range(self.num_layers-1):
                self.layers.extend([nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU(), nn.Dropout(dropout_val)])
            self.layers.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LocalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_val):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ouput_size = output_size
        self.bigru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=True)
        self.bigru_h = nn.Parameter(torch.randn(2, 1, hidden_size))
        self.gru = nn.GRU(input_size=2 * hidden_size, hidden_size=2 * hidden_size)
        self.gru_h = nn.Parameter(torch.randn(1, 1, 2 * hidden_size))
        self.local_mu = nn.Linear(2 * hidden_size, output_size)
        self.local_lv = nn.Linear(2 * hidden_size, output_size)
        self.prior_local = nn.GRU(input_size=1, hidden_size=hidden_size)
        self.prior_mu = nn.Linear(hidden_size, output_size)
        self.prior_lv = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x, h_prior):
        num_timesteps, batch_size, _ = x.size()
        h_bigru = self.bigru_h.repeat(1, batch_size, 1)
        h_t, _ = self.bigru(x, h_bigru)
        # Append both forward and backward so that
        # they align on the same timesteps (flip the backward across time)
        local_features = torch.cat((
            h_t[..., :self.hidden_size],
            torch.flip(h_t[..., self.hidden_size:], dims=(0, ))), dim=-1)
        # Causal (time-wise, not as in causality) generation of the local
        # features
        local_features, _ = self.gru(local_features, self.gru_h.repeat(1, batch_size, 1))
        local_features = self.dropout(local_features)
        # Map to mean and standard deviation for normal distribution
        local_mu = self.local_mu(local_features)
        local_sd = F.softplus(self.local_lv(local_features))
        local_dist = D.Normal(local_mu, local_sd)
        i_zeros = torch.zeros((num_timesteps, batch_size, 1), device=x.device)
        # Causal (time-wise, not as in causality) generation of the prior
        # the prior is generally learned for the local features
        ht_p, _ = self.prior_local(i_zeros, h_prior)
        ht_p = self.dropout(ht_p)
        prior_mu = self.prior_mu(ht_p)
        prior_sd = F.softplus(self.prior_lv(ht_p)).clamp(1E-5, 5)
        prior_dist = D.Normal(prior_mu, prior_sd)
        return local_dist, prior_dist

class ContextGRU(nn.Module):
    def __init__(self, input_size, spatial_hidden_size, temporal_hidden_size, context_size, num_layers, dropout_val):
        super().__init__()
        self.input_size = input_size
        self.spatial_hidden_size = spatial_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self.context_size = context_size
        self.mlp = MLP(input_size, spatial_hidden_size, input_size, num_layers=num_layers, dropout_val=dropout_val)
        self.gru = nn.GRU(input_size=input_size, hidden_size=temporal_hidden_size, bidirectional=True)
        self.context_mu = nn.Linear(2 * temporal_hidden_size, context_size)
        self.context_lv = nn.Linear(2 * temporal_hidden_size, context_size)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        x = self.mlp(x)
        # Find context representations
        _, hT_g = self.gru(x)
        hT_g = torch.reshape(
            hT_g.permute(1, 0, 2), (batch_size, 2 * self.temporal_hidden_size))
        hT_g = self.dropout(hT_g)
        context_mu = self.context_mu(hT_g)
        context_sd = F.softplus(self.context_lv(hT_g)).clamp(1E-5, 5.0)
        context_dist = D.Normal(context_mu, context_sd)
        if self.training:
            context_z = context_dist.rsample()
        else:
            context_z = context_dist.mean
        return context_dist, context_z

class ContextConv(nn.Module):
    def __init__(self, input_size, spatial_hidden_size, temporal_hidden_size, context_size, num_layers, dropout_val):
        super().__init__()
        self.input_size = input_size
        self.spatial_hidden_size = spatial_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self.context_size = context_size
        # GRU uses bi-directional, so also has 2 * temporal_hidden_size features
        self.channels = [input_size] + [int((2 * temporal_hidden_size) / (2 ** i)) for i in range(4, -1, -1)]
        self.conv = ResidualCNN(num_channels=self.channels, dropout=dropout_val, conv_type=nn.Conv1d)
        self.context_mu = nn.Linear(2 * temporal_hidden_size, context_size)
        self.context_lv = nn.Linear(2 * temporal_hidden_size, context_size)
        self.dropout = nn.Dropout(dropout_val)

    def forward(self, x):
        num_timesteps, batch_size, input_size = x.size()
        x = x.permute(1, 2, 0)
        x = self.conv(x).squeeze(-1)
        x = self.dropout(x)
        context_mu = self.context_mu(x)
        context_sd = F.softplus(self.context_lv(x)).clamp(1E-5, 5.0)
        context_dist = D.Normal(context_mu, context_sd)
        if self.training:
            context_z = context_dist.rsample()
        else:
            context_z = context_dist.mean
        return context_dist, context_z


class TemporalEncoder(nn.Module):
    def __init__(self, input_size, local_size,
                 context_size, dropout_val, num_layers,
                 temporal_hidden_size=256, spatial_hidden_size=256,
                 independence=False, context_model='gru'):
        super().__init__()
        self.input_size = input_size
        self.local_size = local_size
        self.temporal_hidden_size = temporal_hidden_size
        self.spatial_hidden_size = spatial_hidden_size
        self.context_size = context_size
        self.independence = independence
        self.dropout = nn.Dropout(dropout_val)
        if context_model == 'gru':
            self.context_network = ContextGRU(
                input_size=input_size, spatial_hidden_size=spatial_hidden_size,
                temporal_hidden_size=temporal_hidden_size, context_size=context_size,
                num_layers=num_layers, dropout_val=dropout_val)
        elif context_model == 'conv':
            self.context_network = ContextConv(
                input_size=input_size, spatial_hidden_size=spatial_hidden_size,
                temporal_hidden_size=temporal_hidden_size, context_size=context_size,
                num_layers=num_layers, dropout_val=dropout_val)
        self.local_encoder = LocalEncoder(
            input_size if independence else input_size + context_size,
            temporal_hidden_size, local_size, dropout_val=dropout_val)
        
        self.mlp_prior = MLP(context_size, 2 * temporal_hidden_size, temporal_hidden_size, num_layers=num_layers, dropout_val=dropout_val)

    def get_context(self, x):
        return self.context_network(x)

    def forward(self, x):
        num_timesteps, batch_size, _ = x.size()
        # Get context representations
        context_dist, context_z = self.get_context(x)
        # Use the original input and context representation to infer
        # each local representation
        context_z_repeated = context_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        if self.independence:
            context_input = x
            h_prior = torch.zeros((1, batch_size, self.temporal_hidden_size), device=x.device)
        else:
            h_prior = self.mlp_prior(context_z).unsqueeze(0)
            context_input = torch.cat((x, context_z_repeated), dim=-1)
        local_dist, prior_dist = self.local_encoder(context_input, h_prior)
        if self.training:
            local_z = local_dist.rsample()
        else:
            local_z = local_dist.mean
        # The final latent representation is a concatenation of the
        # local and global representation
        z = torch.cat((local_z, context_z_repeated), dim=-1)
        return context_dist, local_dist, prior_dist, z

    def generate_local_context(self, x, context_z):
        num_timesteps, batch_size, _ = x.size()
        context_z_repeated = context_z.unsqueeze(0).repeat(num_timesteps, 1, 1)
        if self.independence:
            context_input = x
            h_prior = torch.zeros((1, batch_size, self.temporal_hidden_size), device=x.device)
        else:
            h_prior = self.mlp_prior(context_z).unsqueeze(0)
            context_input = torch.cat((x, context_z_repeated), dim=-1)
        local_dist, _ = self.local_encoder(context_input, h_prior)
        if self.training:
            local_z = local_dist.rsample()
        else:
            local_z = local_dist.mean
        # The final latent representation is a concatenation of the
        # local and global representation
        z = torch.cat((local_z, context_z_repeated), dim=-1)
        return z

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, padding: int, dropout: float, conv_type: nn.Module, output_size: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_size = output_size

        self.conv1 = conv_type(in_channels, out_channels, kernel_size, stride=1, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.)
        self.conv2 = conv_type(out_channels, out_channels, kernel_size, stride=stride, padding=padding)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        if self.output_size is not None:
            x = self.conv2(x, output_size=(self.output_size, ))
        else:
            x = self.conv2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels :int, kernel_size: int,
                 stride: int, padding: int, dropout: float, conv_type: nn.Module, output_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_size = output_size

        self.conv1 = ConvolutionBlock(
            in_channels, out_channels, kernel_size,
            stride, padding, dropout, conv_type, output_size)
        self.conv2 = conv_type(in_channels, out_channels, kernel_size, stride, padding=padding)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.output_size is not None:
            x = self.conv1(x) + self.conv2(x, output_size=(self.output_size, ))
        else:
            x = self.conv1(x) + self.conv2(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class ResidualCNN(nn.Module):
    def __init__(self, num_channels: List[int], dropout: float, conv_type=nn.Conv1d, output_sizes: Union[List, None]=None):
        super().__init__()
        self.num_layers = len(num_channels) - 1
        self.num_channels = num_channels
        self.dropout = dropout
        if output_sizes is None:
            self.output_sizes = [None] * self.num_layers
        else:
            self.output_sizes = output_sizes

        self.layers = nn.ModuleList()
        in_channels = num_channels[0]
        for i, out_channels in enumerate(num_channels[1:]):
            self.layers.append(
                ResidualBlock(in_channels, out_channels, 3, 2, 1, dropout, conv_type, self.output_sizes[i])
            )
            in_channels = out_channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# From: https://github.com/ts-kim/RevIN/blob/master/RevIN.py
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False, skip=False):
        """
        :param num_features: the number of channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

        self.skip = skip

    def forward(self, x, mode:str):
        if self.skip:
            return x
        
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _renew_params(self, new_channel_nub):
        previous_weights = self.affine_weight.data
        previous_bias = self.affine_bias.data
        if new_channel_nub <= self.num_features:
            self.affine_weight = nn.Parameter(previous_weights[:new_channel_nub])
            self.affine_bias = nn.Parameter(previous_bias[:new_channel_nub])
        else:
            diff = new_channel_nub - self.num_features
            self.affine_weight = nn.Parameter(torch.cat((previous_weights, torch.ones(diff, device=previous_weights.device))))
            self.affine_bias = nn.Parameter(torch.cat((previous_bias, torch.zeros(diff, device=previous_bias.device))))


    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)).detach() + self.eps

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x