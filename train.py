import torch
from 

## training loop
P_mean = -1.2       # average noise level (logarithmic)
P_std = 1.2     # spread of random noise levels

net = DiscriminatorNet(P_std)

optimizer = torch.optim.Adam


while True:
    
    # Accumulate gradients
    optimizer.zero_grad()