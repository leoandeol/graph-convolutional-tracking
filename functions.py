import torch

def regularized_laplacian(A):
    A_hat = A + torch.eye(A.shape[-1])
    D_hat = torch.diag(A_hat.sum(-1)**(-0.5))
    L = torch.mm(D_hat,torch.mm(A_hat,D_hat))
    return L
