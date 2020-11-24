import torch
from torch.utils.tensorboard import SummaryWriter
from gym_phiflow.envs import networks
import torchviz

writer = SummaryWriter('tensorboard/logs')

#network = networks.CNN_FUNNEL((32, 32, 4), 1, [4, 8, 16, 16, 16], torch.nn.ReLU)
network = networks.RES_FUNNEL((32, 32, 4), 1, [4, 8, 16, 16, 16], torch.nn.ReLU)
#network = networks.ALT_UNET((32, 2), 32, [4, 8, 16, 16, 16], torch.nn.ReLU)
#network = networks.ALT_UNET((128, 128, 2), 32, [4, 8, 16, 16, 16], torch.nn.ReLU)
#network = actor_critic.UNET((32, 32, 4), 32*32*2, [1, 2], torch.nn.ReLU)
#network = actor_critic.ResBlockRef(4, 3, 1, 1, torch.nn.Conv1d, torch.nn.ReLU)
#network = actor_critic.FCN((32, 2), 32, [70, 60], torch.nn.ReLU)
#network = actor_critic.ResBlock2(8, 3, 1, 1, torch.nn.Conv2d, torch.nn.ReLU)

#input_tensor = torch.zeros(1, 128, 128, 2, dtype=torch.float, requires_grad=False)
input_tensor = torch.zeros(1, 32, 32, 4, dtype=torch.float, requires_grad=False)

writer.add_graph(network, input_tensor, verbose=False)
#writer.add_graph(network, torch.rand(3200, 8, 32, 32))
writer.flush()

writer.close()

print('Hello there?')
torchviz.make_dot(network(input_tensor))