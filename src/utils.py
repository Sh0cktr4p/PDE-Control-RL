import numpy as np
import phi.flow as phiflow


def infer_forces_from_frames(frames, domain, diffusion_substeps, viscosity, dt):
    frames = np.array(frames)
    step_count = frames.shape[0] - 1

    b = phiflow.Burgers(diffusion_substeps=diffusion_substeps)
    to_state = lambda v: phiflow.BurgersVelocity(domain, velocity=v, viscosity=viscosity)

    # Simulate all timesteps of all trajectories at once
    # => concatenate all frames in batch dimension
    prv = to_state(frames[:-1].reshape((-1,) + frames.shape[2:]))
    prv_sim = b.step(prv, dt=dt)
    
    forces = (frames[1:] - prv_sim.velocity.data.reshape(step_count, -1, *frames.shape[2:])) / dt
    
    # Sanity check, should be able to reconstruct goal state with forces
    s = to_state(frames[0])
    for i in range(step_count):
        f = forces[i].reshape(s.velocity.data.shape)
        effect = phiflow.FieldEffect(phiflow.CenteredGrid(f, box=domain.box), ['velocity'])
        s = b.step(s, dt, (effect,))
    diff = frames[-1] - s.velocity.data
    print('Sanity check - maximum deviation from target state: %f' % np.abs(diff).max())
    return forces

def infer_forces_sum_from_frames(frames, domain, diffusion_substeps, viscosity, dt):
    forces = infer_forces_from_frames(frames, domain, diffusion_substeps, viscosity, dt)
    # Sum along frame and field dimensions, keep batch dimension
    return np.abs(forces).sum(axis=(0, 2)).squeeze()
