from gym_phiflow.envs.burger_env import BurgerEnv
from gym_phiflow.envs import visualization
import numpy as np

r = visualization.Renderer()

f = np.zeros((1, 32, 32, 1))
f[0, 0, 0, 0] = 1

while True:
	r.render(f, 15, 1, 500, 500)