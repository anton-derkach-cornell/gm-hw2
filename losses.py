import torch
import torch.nn.functional as F

def gaussian_elbo(x1,x2,z,sigma,mu,logvar):
    
    #
    # Problem 5b: Compute the evidence lower bound for the Gaussian VAE.
    #             Use the closed-form expression for the KL divergence from Problem 1.
    #

    # reconstruction = (1 / (2 * sigma**2)) * torch.sum((x2 - x1)**2)
    # divergence = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar)

    # return reconstruction, divergence

    reconstruction = (1./(2.0*sigma**2))*(x1 - x2).pow(2).view(x1.shape[0],-1).sum(1).mean()
    divergence = .5*(logvar.exp() + mu.pow(2) - 1 - logvar).sum(1).mean()
    return reconstruction, divergence

def mc_gaussian_elbo(x1,x2,z,sigma,mu,logvar):

    #
    # Problem 5c: Compute the evidence lower bound for the Gaussian VAE.
    #             Use a (1-point) monte-carlo estimate of the KL divergence.
    #

    # reconstruction = (1 / (2 * sigma**2)) * torch.sum((x2 - x1)**2)
    
    # log_q = -0.5 * (torch.sum(logvar, dim=1) + torch.sum((z - mu)**2 / torch.exp(logvar), dim=1))
    # log_r = -0.5 * torch.sum(z**2, dim=1)
    
    # divergence = torch.sum(log_q - log_r)

    # return reconstruction, divergence

    reconstruction = (1./sigma*2)*(x1 - x2).pow(2).view(x1.shape[0],-1).mean()
    logpz = 0.5 * z.pow(2).sum(1)
    logqzx = 0.5 * (logvar + torch.pow((z-mu),2)/logvar.exp()).sum(1)
    divergence = (logpz - logqzx).mean()
    return reconstruction, divergence

def cross_entropy(x1,x2):
    return F.binary_cross_entropy_with_logits(x1, x2, reduction='sum')/x1.shape[0]

def discrete_output_elbo(x1,x2,z,logqzx):

    #
    # Problem 6b: Compute the evidence lower bound for a VAE with binary outputs.
    #             Use a (1-point) monte carlo estimate of the KL divergence.
    #


    reconstruction = cross_entropy(x1, x2)

    #    Here, logqzx = - log q(z|x) ignoring the constant,
    #    so we also compute - log p(z) ignoring the same constant.
    #       - log p(z) = 0.5 * ||z||^2
    
    logp_z = 0.5 * (z ** 2).sum(dim=1)  # shape (batch_size,)
    divergence = (logp_z - logqzx).mean()
    
    return reconstruction, divergence
