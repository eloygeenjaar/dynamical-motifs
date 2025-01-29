import torch
import torch.nn.functional as F
from torch import distributions as D


def cross_entropy(output, target):
    logits = output['logits']
    if logits.size(-1) > 1:
        return F.cross_entropy(logits, target.long())
    else:
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), target.float())

def cross_entropy_l1(output, target):
    s = output['s']
    beta = output['beta']
    l1 = (torch.linalg.norm(s, ord=1, dim=-1) / torch.linalg.norm(s, ord=2, dim=-1)).mean(0)
    return cross_entropy(output, target)

def mse(output, target):
    return F.mse_loss(output['x_hat'], output['x_orig'])

def elbo(output, target):
    # Calculate reconstruction loss
    ll = D.Normal(output['x_hat'], 0.1).log_prob(output['x_orig']).sum(-1).mean(1)
    if output['local_dist'] is not None:
        # Calculate the kl-divergence for the local representations
        # KL-local is (num_timesteps, batch_size, latent_dim)
        kl_l = 2 * D.kl.kl_divergence(output['local_dist'], output['prior_dist']).sum(-1).mean(0)
    else:
        kl_l = torch.zeros((1, ), device=output['x_orig'].device)
    if output['context_dist'] is not None:
        # Calculate the context kl-divergence and divide by window_size
        kl_c = 0.05 * D.kl.kl_divergence(
            output['context_dist'], D.Normal(0., 1.)).sum(-1) / output['x_hat'].size(1)
    else:
        kl_c = torch.zeros((1, ), device=output['x_orig'].device)
    elbo = (output['lambda'] * (kl_l + kl_c) - ll).mean()
    return elbo
