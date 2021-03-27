import phi.flow as phiflow

exp_name = 'debug'
domain = phiflow.Domain([32], box=phiflow.box[0:1]) # 1d grid resolution and physical size
viscosity = 0.003 # viscosity constant for Burgers' equation
step_count = 32 # length of each trajectory
dt = 0.03 # time step size
diffusion_substeps = 1 # how many diffusion steps to perform at each solver step

n_envs = 10 # On how many environments to train in parallel, load balancing
final_reward_factor = step_count # How hard to punish the agent for not reaching the goal if that is the case
steps_per_rollout = step_count * 10 # How many steps to collect per environment between agent updates
training_timesteps = steps_per_rollout * 1000 # How long the actual training should be
n_epochs = 10 # How many epochs to perform during agent update
learning_rate = 1e-4 # Learning rate for agent updates
batch_size = 128 # Batch size for agent updates
test_path = '../notebooks/forced-burgers-clash' # Path of the used test set for comparison to cfe method
test_range = range(100) # Test samples inside the dataset