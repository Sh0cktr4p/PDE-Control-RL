import numpy as np
import torch
import networks


class Buffer:
    def __init__(self, input_shape, output_shape, batch_size, batches_per_epoch):
        samples_per_epoch = batch_size * batches_per_epoch

        self.input_buffer = np.zeros([samples_per_epoch] + list(input_shape), dtype=np.float32)
        self.target_buffer = np.zeros([samples_per_epoch] + list(output_shape), dtype=np.float32)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

    def store(self, index, network_input, target):
        self.input_buffer[index] = network_input
        self.target_buffer[index] = target

    def get_in_batches(self):
        batches = []

        for batch_index in range(self.batches_per_epoch):
            batch_start_index = batch_index * self.batch_size
            batches.append({
                'x': self.input_buffer[batch_start_index:batch_start_index + self.batch_size],
                'y': self.target_buffer[batch_start_index:batch_start_index + self.batch_size],
            })

        return batches


class Curiosity:
    def __init__(self, network, layer_sizes, activation, obs_shape, act_shape, batch_size, batches_per_epoch, learning_rate):
        # Observation and Action spaces have to be easily stackable
        assert obs_shape[:-1] == act_shape[:-1]

        input_shape = list(obs_shape[:-1]) + [obs_shape[-1] + act_shape[-1]]
        output_shape = obs_shape
        output_dim = np.prod(output_shape)

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.network = network(input_shape, output_dim, layer_sizes, activation)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.buffer = Buffer(input_shape, output_shape, batch_size, batches_per_epoch)
        self.sample_index = 0
        self.batch_index = 0
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def step(self, old_obs, new_obs, act):
        act = act.reshape(self.act_shape)
        network_input = torch.as_tensor(np.concatenate((old_obs, act), -1), dtype=torch.float32)

        prediction = self.network(network_input.unsqueeze(0))
        prediction = prediction.reshape(self.obs_shape)
        loss = self.criterion(prediction, torch.as_tensor(new_obs, dtype=torch.float32).unsqueeze(0))

        self.buffer.store(self.sample_index, network_input.numpy(), new_obs)

        self.sample_index += 1

        if self.sample_index == self.batch_size:
            self.sample_index = 0
            self.batch_index += 1

        if self.batch_index == self.batches_per_epoch:
            self.update()
            self.batch_index = 0

        return loss.detach().numpy()
        
    def update(self):
        print('updating forward dynamics network')

        batches = self.buffer.get_in_batches()

        for batch in batches:
            prediction = self.network(torch.as_tensor(batch['x'], dtype=torch.float32))
            loss = self.criterion(prediction.reshape([self.batch_size] + list(self.obs_shape)), torch.as_tensor(batch['y']))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        



curiosity = Curiosity(networks.FCN, [50, 50], torch.nn.ReLU, (4, 4, 2), (4, 4, 1), 32, 100, 1e-4)

old_obs = np.random.rand(4, 4, 2)
new_obs = np.random.rand(4, 4, 2)
act = np.random.rand(4, 4, 1)

for _ in range(32000000):
    print(curiosity.step(old_obs, new_obs, act))