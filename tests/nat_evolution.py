from src.visualization import LiveViz
from src.envs.burgers_env import BurgersEnv
from src.envs.fluid_env import FluidEnv
from src.util.burgers_util import GaussianClash, GaussianForce
from phi.tf.flow import BurgersVelocity, Burgers, Domain, box, CLOSED
import matplotlib.pyplot as plt
import numpy as np
import time

def no_forces_burgers():
    env = BurgersEnv(10, diffusion_substeps=16, domain=Domain((32, 32), box=box[0:1, 0:1]))
    print(env._get_init_state())

    obs = env.reset()

    for i in range(3200):
        env.render('live')
        forces = [env.action_space.sample() for _ in range(10)]
        #print(forces)
        env.step(forces)
        #time.sleep(0.1)

def no_forces_fluid():
    n_envs = 1
    env = FluidEnv(n_envs, step_cnt=128, domain=Domain((127, 127), box=box[0:1,0:1], boundaries=CLOSED))
    viz = LiveViz(2, 1, False)
    obs = env.reset()

    for i in range(3200):
        env.render('live')
        viz.render([env.gt_state.density.data[0]], ['cont'], ['orange'])
        forces = [env.action_space.sample() for _ in range(n_envs)]
        #print(forces)
        env.step(forces)


def gaussian_clash():
    d1 = Domain((128,), box=box[0:1])
    d2 = Domain((32,32), box=box[0:1,0:1])
    
    #print(s1.velocity.data.shape)
    #print(s2.velocity.data.shape)

    plt.ion()
    fig = plt.figure()
    x, y = np.mgrid[0:32, 0:32]
    s2 = BurgersVelocity(domain=d2, velocity=GaussianForce(10, 2), viscosity=0.003)
    plot = plt.quiver(x, y, s2.velocity.data[0,:,:,0], s2.velocity.data[0,:,:,1])
    p = Burgers(diffusion_substeps=16)

    for _ in range(100):
        s2 = BurgersVelocity(domain=d2, velocity=GaussianClash(10, 2), viscosity=0.003)
        for _ in range(32):
        #s1 = BurgersVelocity(domain=d1, velocity=GaussianClash(10, 1, True))
        #fig_1d = plt.figure()
        #plt.ylim(-2, 2)
        #x = np.arange(s1.velocity.data[0].size)
        #plt.plot(x, s1.velocity.data[0])
        #plt.show()
            s2 = p.step(s2, dt=0.03)
            plot.set_UVC(s2.velocity.data[0,:,:,0], s2.velocity.data[0,:,:,1])
            fig.canvas.draw()
            fig.canvas.flush_events()
        

if __name__ == "__main__":
    no_forces_fluid()
    #no_forces_burgers()
    #gaussian_clash()
