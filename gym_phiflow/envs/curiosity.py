import numpy as np
import torch


class Buffer:
    def __init__(self, input_shape, output_dim, batch_size, batches_per_epoch):
        self.input_buffer = np.zeros([batch_size * batches_per_epoch] + list(input_shape), dtype=np.float32)
        self.target_buffer = np.zeros([batch_size * batches_per_epoch, output_dim], dtype=np.float32)
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
        output_dim = np.prod(obs_shape)

        self.net = network(input_shape, output_dim, layer_sizes, activation)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.buffer = Buffer(input_shape, output_dim, batch_size * batches_per_epoch)
        self.sample_index = 0
        self.batch_index = 0
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.net.params(), lr=learning_rate)

    def step(self, obs_old, obs_new, act):
        network_input = torch.as_tensor(np.concatenate((obs_old, act), -1), dtype=torch.float32)

        prediction = self.network(obs_old)
        loss = self.criterion(prediction, obs_new)

        self.buffer.store(self.sample_index, network_input.numpy(), target.numpy())

        self.sample_index += 1

        if self.sample_index == self.batch_size:
            self.sample_index = 0
            self.batch_index += 1

        if self.batch_index == self.batches_per_epoch:
            self.update()
            self.batch_index = 0

        return loss
        
    def update(self):
        print('updating forward dynamics network')

        batches = self.buffer.get_in_batches()

        for batch in batches:
            prediction = self.net(torch.as_tensor(batch['x'], dtype=torch.float32))
            loss = self.criterion(prediction, torch.as_tensor(batch['y']))
            self.optimizer.zero_grad()
            loss.backward()
            optimizer.step()



