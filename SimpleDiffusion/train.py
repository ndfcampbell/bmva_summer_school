import numpy as np 
from tqdm import tqdm 

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST



from model import ScoreNet
from vp_sde import marginal_prob_std, marginal_prop_mean, diffusion_coeff, drift_coeff


device = "cuda" if torch.cuda.is_available() else "cpu"

def loss_fn(model, x, marginal_prob_std, marginal_prop_mean, eps=1e-5, T=1.):

    """The loss function for training score-based generative models.

    Args:
        model: A PyTorch model instance that represents a 
        time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of 
        the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (T - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
 
    mean = marginal_prop_mean(random_t)
    perturbed_x = mean[:, None, None, None] * x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
    
    return loss


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
score_model = score_model.to(device)



print("Number of Parameters: ", sum([p.numel() for p in score_model.parameters()]))


# load pre-trained score model
#ckpt = torch.load('ckpt.pth', map_location=device)
#score_model.load_state_dict(ckpt)

n_epochs = 5
batch_size = 32
lr=1e-4

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

T = 1.

optimizer = Adam(score_model.parameters(), lr=lr)
print("Start Training")
for epoch in range(n_epochs):
    avg_loss = 0.
    num_items = 0
    for x, y in tqdm(data_loader):
        x = x.to(device)    
        loss = loss_fn(score_model, x, marginal_prob_std=marginal_prob_std_fn,marginal_prop_mean=marginal_prob_mean_fn, T=T)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    # Print the averaged training loss so far.
    print('Average Loss at epoch {}: {:5f}'.format(epoch, avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'ckpt.pth')