import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as D
from base import BaseModel
from torch import nn
from torch import distributions as D
from .modules import (
    TemporalEncoder, MLP, RevIN, ResidualCNN)


class BaseModel(nn.Module):
    def __init__(self, input_size: int, local_size: int, context_size: int, window_size: int, num_layers: int,
                 spatial_hidden_size: int, temporal_hidden_size: int, dropout: float):
        super().__init__()
        self.local_size = local_size
        self.context_size = context_size
        self.input_size = input_size
        self.window_size = window_size
        self.num_layers = num_layers
        self.spatial_hidden_size = spatial_hidden_size
        self.temporal_hidden_size = temporal_hidden_size
        self.dropout = dropout
        self.spatial_decoder = MLP(
            local_size + context_size,
            spatial_hidden_size, input_size, num_layers, dropout)
        self.revin = RevIN(self.input_size, affine=True, subtract_last=False, skip=False)

    def forward(self, x):
        raise NotImplementedError

class DSVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            num_layers=self.num_layers,
            temporal_hidden_size=self.temporal_hidden_size,
            spatial_hidden_size=self.spatial_hidden_size,
            dropout_val=self.dropout)

    def forward(self, x):
        #x_norm = x.transpose(2, 1) # (batch_size, time_length, input_size)
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(1, 0)
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x_norm)
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.transpose(1, 0)
        x_hat = self.revin(x_hat, 'denorm')
        return {
            'context_dist': context_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'x_orig': x
        }

    def embed_context(self, x):
        x_norm = self.revin(x, 'norm')
        x_norm = x_norm.transpose(1, 0)
        return self.temporal_encoder.get_context(x_norm)[0]

    def embed_local(self, x):
        x_norm = self.revin(x, 'norm')
        x_norm = x_norm.transpose(1, 0)
        return self.temporal_encoder(x_norm)[1]

    def decode_context(self, x, context_z):
        x_norm = self.revin(x, 'norm')
        x_norm = x_norm.transpose(1, 0)
        z = self.temporal_encoder.generate_local_context(x_norm, context_z)
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.transpose(1, 0)
        x_hat = self.revin(x_hat, 'denorm')
        return x_hat

class DSVAENLN(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            num_layers=self.num_layers,
            temporal_hidden_size=self.temporal_hidden_size,
            spatial_hidden_size=self.spatial_hidden_size,
            dropout_val=self.dropout)

    def forward(self, x):
        #x_norm = x.transpose(2, 1) # (batch_size, time_length, input_size)
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(1, 0)
        context_dist, local_dist, prior_dist, z = self.temporal_encoder(x_norm)
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.transpose(1, 0)
        x_hat = self.revin(x_hat, 'denorm')
        return {
            'context_dist': context_dist,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'x_orig': x
        }

    def embed_context(self, x):
        x_norm = self.revin(x, 'norm')
        x_norm = x_norm.transpose(1, 0)
        return self.temporal_encoder.get_context(x_norm)[0]

class ConvDSVAE(DSVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            num_layers=self.num_layers,
            temporal_hidden_size=self.temporal_hidden_size, independence=False,
            spatial_hidden_size=self.spatial_hidden_size,
            dropout_val=self.dropout, context_model='conv')

class IDSVAE(DSVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            num_layers=self.num_layers,
            temporal_hidden_size=self.temporal_hidden_size, independence=True,
            spatial_hidden_size=self.spatial_hidden_size,
            dropout_val=self.dropout)

class ConvIDSVAE(DSVAE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_encoder = TemporalEncoder(
            self.input_size, self.local_size, self.context_size,
            num_layers=self.num_layers,
            temporal_hidden_size=self.temporal_hidden_size, independence=True,
            spatial_hidden_size=self.spatial_hidden_size,
            dropout_val=self.dropout, context_model='conv')


class LVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_encoder = MLP(
            self.input_size,
            self.spatial_hidden_size, 2 * self.local_size, self.num_layers, self.dropout)
        self.context_size = self.local_size

    def forward(self, x):
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(1, 0)
        mu_lv = self.spatial_encoder(x_norm)
        mu, lv = torch.split(mu_lv, self.local_size, dim=-1)
        local_dist = D.Normal(mu, F.softplus(lv))
        prior_dist = D.Normal(0, 1)
        if self.training:
            z = local_dist.rsample()
        else:
            z = local_dist.mean
        x_hat = self.spatial_decoder(z)
        x_hat = x_hat.transpose(1, 0)
        x_hat = self.revin(x_hat, 'denorm')
        return {
            'context_dist': None,
            'local_dist': local_dist,
            'prior_dist': prior_dist,
            'x_hat': x_hat,
            'x_orig': x
        }

    def embed_context(self, x):
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(1, 0)
        mu_lv = self.spatial_encoder(x_norm)
        mu, _ = torch.split(mu_lv, self.local_size, dim=-1)
        return D.Normal(mu.mean(0), 1.0)

    def embed_local(self, x):
        x_norm = self.revin(x, 'norm')
        x_norm = x_norm.transpose(1, 0)
        mu_lv = self.spatial_encoder(x_norm)
        mu, _ = torch.split(mu_lv, self.local_size, dim=-1)
        return D.Normal(mu, 1.0)

class CVAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = [self.input_size] + [int((2 * self.temporal_hidden_size) / (2 ** i)) for i in range(4, -1, -1)]
        self.temporal_encoder = ResidualCNN(num_channels=self.channels, dropout=self.dropout, conv_type=nn.Conv1d)
        self.mu_lin = nn.Linear(2 * self.temporal_hidden_size, self.context_size)
        self.sd_lin = nn.Linear(2 * self.temporal_hidden_size, self.context_size)
        self.up_lin = nn.Linear(self.context_size, 2 * self.temporal_hidden_size, bias=False)
        self.temporal_decoder = ResidualCNN(num_channels=self.channels[::-1], dropout=self.dropout, conv_type=nn.ConvTranspose1d,
                                            output_sizes=[2, 4, 8, 16, 32])
        self.final_layer = nn.Conv1d(in_channels=self.input_size, out_channels=self.input_size, kernel_size=3, padding=1, stride=1)
        self.revin = RevIN(self.input_size, affine=True, subtract_last=False, skip=False)

    def forward(self, x):
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(2, 1)
        x_norm = self.temporal_encoder(x_norm).squeeze(-1)
        mu = self.mu_lin(x_norm)
        sd = F.softplus(self.sd_lin(x_norm)).clamp(1E-5, 5.0)
        dist = D.Normal(mu, sd)
        if self.training:
            z = dist.rsample()
        else:
            z = dist.mean
        x_hat = self.up_lin(z).unsqueeze(-1)
        x_hat = self.temporal_decoder(x_hat)
        x_hat = x_hat.transpose(2, 1)
        x_hat = self.revin(x_hat, 'denorm')
        return {
            'context_dist': dist,
            'local_dist': None,
            'x_hat': x_hat,
            'x_orig': x
        }

    def embed_context(self, x):
        x_norm = self.revin(x, 'norm') # (batch_size, time_length, input_size)
        x_norm = x_norm.transpose(2, 1)
        x_norm = self.temporal_encoder(x_norm).squeeze(-1)
        mu = self.mu_lin(x_norm)
        return D.Normal(mu, 1.0)
