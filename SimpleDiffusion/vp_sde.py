import torch 
import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# based on the variance exploding SDE
# the derivations are given in [https://arxiv.org/pdf/2011.13456.pdf] Appendix C
def marginal_prob_std(t, beta_min=0.1, beta_max=20):
  """Compute the **standard deviation** of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.
  
  Returns:
    The standard deviation.
  """    
  t = torch.tensor(t, device=device)
  log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
  return torch.sqrt(1. - torch.exp(2. * log_mean_coeff))

def marginal_prop_mean(t, beta_min=0.1, beta_max=20):
  """Compute the **mean** of $p_{0t}(x(t) | x(0))$.

  Args:    
    t: A vector of time steps.

  Returns:
    The scaling for the mean.
  """   
  t = torch.tensor(t, device=device)
  log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
  return torch.exp(log_mean_coeff)

def diffusion_coeff(t, beta_min=0.1, beta_max=20):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
  
  Returns:
    The vector of diffusion coefficients.
  """
  t = torch.tensor(t, device=device)
  beta_t = beta_min + t * (beta_max - beta_min)
  return torch.sqrt(beta_t)

def drift_coeff(t, beta_min=0.1, beta_max=20):
  """Compute the drift coefficient of our SDE.

  Args:
    t: A vector of time steps.
  
  Returns:
    The vector of drift coefficients.
  """
  t = torch.tensor(t, device=device)
  beta_t = beta_min + t * (beta_max - beta_min)
  return -0.5 * beta_t