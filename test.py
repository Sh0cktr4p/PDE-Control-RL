from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from phi.flow import *

state = Burger(Domain((32, 32)), math.randn(levels=[0, 0, 1]), viscosity=0.2)
goal_state = Burger(Domain((32, 32)), math.randn(levels=[0, 0, 1]), viscosity=0.2)

height, width = state.velocity.shape[-3:-1]
obs = np.linalg.norm(state.velocity, axis=-1).reshape(height, width)
goal_obs = np.linalg.norm(goal_state.velocity, axis=-1).reshape(height, width)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xy = np.meshgrid(np.arange(width), np.arange(height))

ax.plot_surface(xy[0], xy[1], obs)
ax.plot_surface(xy[0], xy[1], goal_obs, color='r')

plt.show()