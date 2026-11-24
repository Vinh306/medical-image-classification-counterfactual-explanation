import torch
import torch.nn.functional as F


# (1) Data consistency
def d_loss(real_logit, fake_logit):
    return F.softplus(-real_logit).mean() + F.softplus(fake_logit).mean()


def g_adv_loss(fake_logit):
    return F.softplus(-fake_logit).mean()


# (2) Classifier consistency: KL(f(x_c) || c)
def bernoulli_kl(p, q, eps = 1e-6):
    p, q = p.float().clamp(eps, 1.0 - eps), q.float().clamp(eps, 1.0 - eps)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def classifier_consistency_loss(f_xc, c_target):
    return bernoulli_kl(f_xc, c_target).mean()


# (3) Self-consistency: reconstruction L1
def reconstruction_loss_l1(x, x_self, x_cyc):
    return F.l1_loss(x_self, x) + F.l1_loss(x_cyc, x)
