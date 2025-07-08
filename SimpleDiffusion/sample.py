from model import ScoreNet
from vp_sde import marginal_prob_std, marginal_prop_mean, diffusion_coeff, drift_coeff

import numpy as np 
import matplotlib.pyplot as plt 
import torch
from torchvision.utils import make_grid
import functools
from tqdm import tqdm 

device = "cuda" if torch.cuda.is_available() else "cpu"

beta_min = 0.1
beta_max = 20 

marginal_prob_std_fn = functools.partial(marginal_prob_std, 
                                        beta_min=beta_min, beta_max=beta_max)
marginal_prob_mean_fn = functools.partial(marginal_prop_mean, 
                                        beta_min=beta_min, beta_max=beta_max)
diffusion_coeff_fn = functools.partial(diffusion_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)
drift_coeff_fn = functools.partial(drift_coeff, 
                                        beta_min=beta_min, beta_max=beta_max)



score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)


print("Number of Parameters: ", sum([p.numel() for p in score_model.parameters()]))

score_model = score_model.to(device)
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)
score_model.eval() 


T = 1.

def Euler_Maruyama_sampler(score_model, 
                           drift_coeff,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=500, 
                           device=device, 
                           eps=1e-4,
                           T = 1.):
  """Generate samples from score-based models with the Euler-Maruyama solver.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.
  
  Returns:
    Samples.    
  """
  t = torch.ones(batch_size, device=device)
  time_steps = torch.linspace(T, eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x_init = torch.randn(batch_size, 1, 28, 28, device=device)  # Initialize with Gaussian noise.
  x = x_init
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step

      g = diffusion_coeff(batch_time_step)
      f = drift_coeff(batch_time_step)[:, None, None, None]
      mean_x = x - (f*x - (g**2)[:, None, None, None] * score_model(x, batch_time_step)) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)      

  # Do not include any noise in the last sampling step.
  return mean_x




sample_batch_size = 36 
T = 1.
## Generate samples using the specified sampler.
samples = Euler_Maruyama_sampler(score_model, 
                           drift_coeff=drift_coeff_fn,
                           diffusion_coeff=diffusion_coeff_fn, 
                           batch_size=sample_batch_size, 
                           num_steps=500, 
                           device=device, 
                           eps=1e-4,
                           T=T)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()