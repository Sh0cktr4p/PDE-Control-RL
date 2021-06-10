from src.envs.burgers_env import BurgersEnv
from src.envs.burgers_util import GaussianClash
from phi.tf.flow import BurgersVelocity, Domain, box
import matplotlib.pyplot as plt
import numpy as np
import time

def no_forces():
    env = BurgersEnv(10, diffusion_substeps=16, domain=Domain((32,32), box=box[0:1,0:1]))
    print(env._get_init_state())

    obs = env.reset()

    for i in range(3200):
        env.render('live')
        forces = [env.action_space.sample() * 0 + 3 for _ in range(10)]
        #print(forces)
        env.step(forces)
        #time.sleep(0.1)

def gaussian_clash():
    d1 = Domain((128,), box=box[0:1])
    d2 = Domain((32,32), box=box[0:1,0:1])
    
    #print(s1.velocity.data.shape)
    #print(s2.velocity.data.shape)

    for _ in range(100):
        s1 = BurgersVelocity(domain=d1, velocity=GaussianClash(10, 1, True))
        fig_1d = plt.figure()
        plt.ylim(-2, 2)
        x = np.arange(s1.velocity.data[0].size)
        plt.plot(x, s1.velocity.data[0])
        plt.show()
        
        s2 = BurgersVelocity(domain=d2, velocity=GaussianClash(10, 2, True))
        plt.figure()
        x, y = np.mgrid[0:32, 0:32]
        plt.quiver(x, y, s2.velocity.data[0,:,:,0], s2.velocity.data[0,:,:,1])
        plt.show()

if __name__ == "__main__":
    no_forces()
    #gaussian_clash()
