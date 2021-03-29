import matplotlib.pyplot as plt


def compare_trajectories(
    ground_truth,
    uncontrolled,
    reinforcement_learning,
    control_force_estimator,
    index_in_set,
):
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.6))
    axs[0, 0].set_title('Ground Truth')
    axs[0, 1].set_title('Uncontrolled')
    axs[1, 0].set_title('Reinforcement Learning')
    axs[1, 1].set_title('Supervised Control Force Estimator')

    for subplot_idcs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        axs[subplot_idcs].set_ylim(-2, 2)
        axs[subplot_idcs].set_xlabel('x')
        axs[subplot_idcs].set_ylabel('u(x)')
        axs[subplot_idcs].legend(['Initial state in dark red, final state in dark blue,'])

    def plot_w_label(xy, field):
        color = bplt.gradient_color(0, step_count+1)
        axs[xy].plot(
            field[0][index_in_set].squeeze(), 
            color=color, 
            linewidth=0.8, 
            label='Initial state in dark red, final state in dark blue,'
        )
        axs[xy].legend()

    # Plot the first states and create a legend for each plot
    #plot_w_label((0, 0), gt_frames)
    #plot_w_label((0, 1), unc_frames)
    #plot_w_label((1, 0), rl_frames)
    #plot_w_label((1, 1), cfe_frames)

    # Plot the remaining states
    for frame in range(0, step_count + 1):
        color = bplt.gradient_color(frame, step_count+1)
        plot = lambda xy, field: axs[xy].plot(
            field[frame][index_in_set].squeeze(), 
            color=color, 
            linewidth=0.8
        )
    
        plot((0,0), gt_frames)
        plot((0,1), unc_frames)
        plot((1,0), rl_frames)
        plot((1,1), cfe_frames)