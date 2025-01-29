import torch
from torcheval.metrics import functional as TEF
from torch.nn import functional as F
from torch import distributions as D
from sklearn.linear_model import LogisticRegression


def kl_loss(output, target):
    return D.kl.kl_divergence(output['dist'], D.Normal(0., 1.)).mean()

def temporal_kl_loss(output, target):
    return output['lambda'] * D.kl.kl_divergence(
        D.Normal(output['dist'].mean[:-1], output['dist'].stddev[:-1]),
        D.Normal(output['dist'].mean[1:], output['dist'].stddev[1:])).mean()

def mse(output, target):
    return F.mse_loss(output['x_hat'], output['x_orig'])

def nll(output, target):
    return -D.Normal(output['x_hat'], 0.1).log_prob(output['x_orig']).mean()

def kl_context(output, target):
    # Calculate the context kl-divergence
    if output['context_dist'] is not None:
        kl_c = D.kl.kl_divergence(
            output['context_dist'], D.Normal(0., 1.)).sum(-1).mean().detach()
    else:
        kl_c = 0
    return kl_c

def kl_local(output, target):
    if output['local_dist'] is not None:
        # Calculate the kl-divergence for the local representations
        kl_l = D.kl.kl_divergence(output['local_dist'], output['prior_dist']).sum(-1).mean().detach()
    else:
        kl_l = 0
    return kl_l

def linear_sep(output, target):
    if output['context_dist'] is not None and len(torch.unique(target)) > 1:
        z = output['context_dist'].mean.detach().cpu().numpy()
        y = target.cpu().numpy()
        lr = LogisticRegression()
        lr.fit(z, y)
        return lr.score(z, y)
    else:
        return 0
