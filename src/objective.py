import torch
import torch.nn.functional as F
import math

SMALL_NUM=1e-7

def nt_xent_loss(Z, temp=1.0):
    """ Normalized temperature-scaled cross entropy loss
    Compute the nt_xent loss between

    @param Z (torch.FloatTensor of shape (2 * N, zdim)): 2N hidden (`z` in the simclr paper) representations. 

    @return contrastive loss (scalar)
    """
    
    Z = F.normalize(Z)
    batch_size = Z.shape[0]//2

    Z_1_n, Z_n1_2n = Z.split(batch_size, 0)
    cov = torch.mm(Z, Z.T)
    sim = torch.exp(cov / temp)

    neg = sim.sum(dim=-1)
    # since neg counts x^T.x in negative, we remove it there. Correspond to `1_{k=i}` in the simclr paper
    # TODO: should be something like math.exp(1/temp)
    e_t_row = torch.Tensor(neg.shape).fill_(math.exp(1/temp)).to(neg.device)
    neg = torch.clamp(neg - e_t_row, min=SMALL_NUM)

    pos = torch.exp(torch.sum(Z_1_n * Z_n1_2n, dim=-1) / temp)
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(pos / (neg + SMALL_NUM)).mean()
    return loss
