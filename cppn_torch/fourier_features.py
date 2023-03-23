from cppn_torch.activation_functions import gauss

# from https://github.com/tancik/fourier-feature-networks
import torch

def apply_mapping(x, B:torch.tensor):
  if B is None:
    return x
  else:
    x_proj = (2.*torch.pi*x) @ B.T
    x_proj = (2.*torch.pi*x) @ B.T
    f0 = torch.sin(x_proj)
    f1 = torch.cos(x_proj)
    # f1 = gauss(x_proj)
    return torch.cat([f0, f1], dim=-1)

def input_mapping(x, b_scale:float, mapping_size:int, dims:int=2):
    B_gauss = torch.randn((mapping_size//dims, dims), device=x.device)
    B_gauss = B_gauss * b_scale
    return apply_mapping(x, B_gauss)
  

def add_fourier_features(x, n_features, B_scale=10.0, dims=2, include_original=False, mult_percent=.5):
    assert n_features % dims == 0, "mapping_size must be divisible by dims"
    
    
    if mult_percent:
      orig_n_features = n_features
      n_features = orig_n_features - int(orig_n_features * mult_percent)
      
    f_feats = input_mapping(x, B_scale, n_features, dims=dims)

    if mult_percent:
      while f_feats.shape[-1] < orig_n_features:
        two_rand = torch.randint(0, f_feats.shape[-1], (2,))
        m = f_feats[:,:, two_rand[0]] * f_feats[:, :, two_rand[1]]
        f_feats = torch.cat([f_feats, m.unsqueeze(-1)], dim=-1)
        

    if include_original:
      X = torch.cat([x, f_feats], dim=-1)
    else:
      X = f_feats

    return X

