import torch
from torch.utils.tensorboard import SummaryWriter
import actor_critic

writer = SummaryWriter('tensorboard/logs')

#network = actor_critic.MOD_UNET((32, 2), 32, [8], torch.nn.ReLU)
#network = actor_critic.UNET((32, 2), 32, [1, 2], torch.nn.ReLU)
network = actor_critic.ResBlock(4, 3, 1, 1, torch.nn.Conv2d, torch.nn.ReLU)

writer.add_graph(network, torch.rand(10, 4, 32, 2))

writer.close()

