import torch
from torch.utils.tensorboard import SummaryWriter
import actor_critic

writer = SummaryWriter('tensorboard/logs')

#network = actor_critic.MOD_UNET((128, 128, 4), 32, [16, 16, 16, 8, 4], torch.nn.ReLU)
network = actor_critic.UNET((32, 32, 4), 32*32*2, [1, 2], torch.nn.ReLU)
#network = actor_critic.ResBlockRef(4, 3, 1, 1, torch.nn.Conv1d, torch.nn.ReLU)
#network = actor_critic.FCN((32, 2), 32, [70, 60], torch.nn.ReLU)


writer.add_graph(network, torch.rand(1, 32, 32, 4))

writer.close()

