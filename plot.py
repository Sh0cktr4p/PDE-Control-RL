import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_monitoring_results(**args):
    df = pd.read_csv(args['path'], skiprows=[0])

    avg_reward = df['r']
    avg_length = df['l']
    time = df['t']

    # Handle restarted trainings
    for i in range(len(time) - 1):
        if time[i+1] < time[i]:
            time[i+1] += time[i]

    length = (len(avg_reward) // 100) * 100

    avg_reward = np.sum(np.array(avg_reward[:length]).reshape(-1, 100), axis=-1) / 100

    plt.plot(avg_reward, label="Average return per rollout")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--path', type=str, default='')

    args = parser.parse_args()

    plot_monitoring_results(**args.__dict__)