from phi.tf.flow import *
import numpy as np

@struct.definition()
class GaussianClash(AnalyticField):
    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.batch_size = batch_size

    def sample_at(self, idx, collapse_dimensions=True):
        leftloc = np.random.uniform(0.2, 0.4, self.batch_size)
        leftamp = np.random.uniform(0, 3, self.batch_size)
        leftsig = np.random.uniform(0.05, 0.15, self.batch_size)
        rightloc = np.random.uniform(0.6, 0.8, self.batch_size)
        rightamp = np.random.uniform(-3, 0, self.batch_size)
        rightsig = np.random.uniform(0.05, 0.15, self.batch_size)
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = leftamp * np.exp(-0.5 * (idx - leftloc) ** 2 / leftsig ** 2)
        right = rightamp * np.exp(-0.5 * (idx - rightloc) ** 2 / rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size):
        AnalyticField.__init__(self, rank=1)
        self.loc = np.random.uniform(0.4, 0.6, batch_size)
        self.amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
        self.sig = np.random.uniform(0.1, 0.4, batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


domain = Domain((32,), box=box[0:1])
dt = 0.03
viscosity=0.003

p = Burgers()

def step_comp_graph(physics, s_ph, f_ph, dt, n_steps):
    for _ in range(n_steps):
        s_ph = physics.step(s_ph, dt=dt, effects=(f_ph,))
    return s_ph

def step_no_fc(physics, s, dt, n_steps):
    for _ in range(n_steps):
        s = physics.step(s, dt)
    return s

print(np.ones(domain.resolution))
s0 = BurgersVelocity(domain, velocity=GaussianClash(10), viscosity=viscosity)
print(s0.velocity.__dict__)
f = FieldEffect(CenteredGrid(np.random.rand(*s0.velocity.data.shape), box=domain.box), ['velocity'])
print(f.field.data.shape)
sY = step_no_fc(p, s0, dt, 1)
sZ = step_comp_graph(p, s0, f, dt, 1)

#print((sZ.velocity.data - sY.velocity.data) / dt - f.field.data)

'''
# Only one simulation step
s0_sim = p.step(s0, dt=dt)
# Simulation step with force application
s0_sim_f = p.step(s0, dt=dt, effects=(f,))        

delta = s0_sim_f.velocity.data - s0_sim.velocity.data

# Reconstructed forces
f_a = delta / dt
# Original forces of field effect
f_b = f.field.at(s0_sim.velocity).data

# Should be close to zero?
print(np.sum(np.abs(f_a - f_b)))

# Using PhiFlow 1.4.0 this produces an error < 1e-4,
# using Phiflow 1.5.1 this error usually is > 1e0

# Reversing the order of advection and diffusion in 1.5.1
# decreases the error to < 1e-4
'''