import os
import phi.flow as phiflow
import sys; sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.experiment import BurgersTraining
from src.envs.burgers_util import GaussianClash, GaussianForce
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from config import *


def get_sim_data(domain: phiflow.Domain, viscosity: float, diffusion_substeps: int, step_count: int, dt: float) -> str:
    data_path = os.path.join(os.path.dirname(__file__), '..', 'sim_data', DATA_FOLDER)
    if not os.path.exists(data_path):
        for batch_index in range(SCENE_COUNT // SIM_BATCH_SIZE):
            scene = phiflow.Scene.create(data_path, count=SIM_BATCH_SIZE)
            print(scene)
            world = phiflow.World()
            u0 = phiflow.BurgersVelocity(
                domain, 
                velocity=GaussianClash(SIM_BATCH_SIZE), 
                viscosity=viscosity, 
                batch_size=SIM_BATCH_SIZE, 
                name='burgers'
            )
            u = world.add(u0, physics=phiflow.Burgers(diffusion_substeps=diffusion_substeps))
            force = world.add(phiflow.FieldEffect(GaussianForce(SIM_BATCH_SIZE), ['velocity']))
            scene.write(world.state, frame=0)
            for frame in range(1, step_count + 1):
                world.step(dt=dt)
                scene.write(world.state, frame=frame)

    return data_path


def parse_arguments() -> Namespace:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default=EXP_NAME)
    parser.add_argument('--viscosity', type=float, default=VISCOSITY)
    parser.add_argument('--step_count', type=int, default=STEP_COUNT)
    parser.add_argument('--dt', type=float, default=DT)
    parser.add_argument('--diffusion_substeps', type=int, default=DIFFUSION_SUBSTEPS)
    parser.add_argument('--n_envs', type=int, default=N_ENVS)
    parser.add_argument('--final_reward_factor', type=float, default=FINAL_REWARD_FACTOR)
    parser.add_argument('--steps_per_rollout', type=int, default=STEPS_PER_ROLLOUT)
    parser.add_argument('--n_epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--ppo_batch_size', type=int, default=PPO_BATCH_SIZE)
    parser.add_argument('--n_rollouts', type=int, default=N_ROLLOUTS)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    domain = phiflow.Domain(DOMAIN_SHAPE, box=phiflow.box[0:1])
    exp_path = os.path.join(os.path.dirname(__file__), '..', 'networks', 'rl-models', args.name)
    data_path = get_sim_data(domain, args.viscosity, args.diffusion_substeps, args.step_count, args.dt)

    trainer = BurgersTraining(
        path=exp_path, 
        domain=domain,
        viscosity=args.viscosity,
        step_count=args.step_count,
        dt=args.dt,
        diffusion_substeps=args.diffusion_substeps,
        n_envs=args.n_envs,
        final_reward_factor=args.final_reward_factor,
        steps_per_rollout=args.steps_per_rollout,
        n_epochs=args.n_epochs,
        learning_rate=args.lr,
        batch_size=args.ppo_batch_size,
        data_path=data_path,
        val_range=VAL_RANGE,
        test_range=TEST_RANGE,
    )

    trainer.train(n_rollouts=args.n_rollouts, save_freq=50)
