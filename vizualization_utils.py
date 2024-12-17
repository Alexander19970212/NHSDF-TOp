import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from NN_TopOpt.TopOpt import LoadedMesh2D, LoadedMesh2D_ext
import json
from os import listdir
from os.path import isfile, join
from prettytable import PrettyTable

import matplotlib.animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.tri as mtri
from IPython.display import HTML

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time

def extract_scalars_from_event_file(event_file):
    """Extract scalar data from a TensorBoard event file."""
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    
    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        scalars[tag] = event_acc.Scalars(tag)
    
    return scalars

def plot_learning_curves(log_dir, subdirs, metric='loss', filename=None):
    """
    Plot learning curves for a specified metric from TensorBoard logs in a directory.
    
    Parameters:
    - log_dir: str, path to the directory containing TensorBoard log subdirectories.
    - metric: str, the metric to plot (e.g., 'loss', 'accuracy').
    """
    plt.figure(figsize=(10, 6))
    
    for subdir in os.listdir(log_dir):
        if subdir in subdirs:
            subdir_path = os.path.join(log_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.startswith('events.out.tfevents'):
                        event_file = os.path.join(subdir_path, file)
                        scalars = extract_scalars_from_event_file(event_file)
                        
                        if metric in scalars:
                            steps = [s.step for s in scalars[metric]]
                            values = [s.value for s in scalars[metric]]
                            plt.plot(steps, values, label=f'{subdir} - {metric}')
    
    plt.title(f'Learning Curves for {metric}')
    plt.xlabel('Steps')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

# all experiments
experiments_folder = "experiments"

def experiment_table(data=None, comment=None):
    table = PrettyTable()
    table.field_names = ["Experiment", "data", "time", "comment"]

    onlyfiles = [f for f in listdir(experiments_folder) if isfile(join(experiments_folder, f))]

    for experiment_name in onlyfiles:
        experiment_path =experiments_folder + "/" + experiment_name

        with open(experiment_path, 'r') as fp:
            exp_meta = json.load(fp)

        if data != None:
            if exp_meta["data"] == data:
                table_row = [experiment_name[:-5], exp_meta["data"], exp_meta["time"], exp_meta['args']['comment']]
                table.add_row(table_row)

        if comment != None:
            if exp_meta['args']["comment"] == comment:
                table_row = [experiment_name[:-5], exp_meta["data"], exp_meta["time"], exp_meta['args']['comment']]
                table.add_row(table_row)

        if data == None and comment == None:
            table_row = [experiment_name[:-5], exp_meta["data"], exp_meta["time"], exp_meta['args']['comment']]
            table.add_row(table_row)

    print(table)

def last_modified():
    onlyfiles = [f for f in listdir(experiments_folder) if isfile(join(experiments_folder, f))]
    times_m = []

    for exp_name in onlyfiles:
        path = experiments_folder + "/" + exp_name
        ti_m = os.path.getmtime(path)
        times_m.append(ti_m)
    
    times_m = np.array(times_m)
    return onlyfiles[np.argmax(times_m)]

def last_iteration(file_name, save_fig=False):
    experiment_path = experiments_folder + "/" + file_name
    with open(experiment_path, 'r') as fp:
        exp_meta = json.load(fp)

    # load mesh
    Th = LoadedMesh2D(exp_meta['problem']["meshfile"])

    # last_solution
    iterations_str = exp_meta['iter_meta'].keys()
    iterations = np.array([int(iter) for iter in iterations_str])
    last_iter_key = str(iterations[np.argmax(iterations)])

    x = exp_meta['iter_meta'][last_iter_key]['x']
    x = np.array(x)

    if save_fig:
        fig_name = experiment_path = "src/" + file_name[:-5]+'.jpg'
        Th.plot_topology(x, fig_name)

    else:
        Th.plot_topology(x)

def opt_animation(file_name):
    experiment_path = experiments_folder + "/" + file_name

    # plot the final section plot
    fig1, axs = plt.subplots(1, 2, figsize=(16,6), gridspec_kw={'width_ratios': [3, 1]}, layout="constrained")

    div = make_axes_locatable(axs[0])
    cax = div.append_axes('right', '5%', '5%')

    with open(experiment_path, 'r') as fp:
        exp_meta = json.load(fp)

    xs = []
    objs = []
    for iter in exp_meta['iter_meta']:
        xs.append(exp_meta['iter_meta'][iter]['x'])
        try:
            objs.append(exp_meta['iter_meta'][iter]['obj'])
        except:
            # print(iter)
            objs.append(exp_meta['iter_meta']['2']['obj'])

    xs = np.array(xs)
    objs = np.array(objs)

    cb_max = xs.max()
    cb_min = xs.min()

    # load mesh
    Th = LoadedMesh2D_ext(exp_meta['problem']["meshfile"])

    def animate(i):
        axs[0].clear()
        axs[1].clear()
        axs[1].set_ylim([objs.min() - (objs.max() - objs.min())*0.1, objs.max()+(objs.max() - objs.min())*0.1])
        axs[1].set_xlim([0, objs.shape[0]])
        
        axs[0].set_aspect('equal')
        tcf = Th.plot_topology(xs[i], vmax = cb_max, vmin = cb_min, ax = axs[0])
        fig1.colorbar(tcf, cax=cax)
        axs[0].set_title(f'Topology plot, iter = {i}, Obj.={objs[i]}')

        axs[1].plot(np.arange(i), objs[:i])
        axs[1].set_xlabel("iter")
        axs[1].set_ylabel("Obj")
        axs[1].set_title(f'Objective_history_plot')

        asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
        axs[1].set_aspect(asp)

    ani = matplotlib.animation.FuncAnimation(fig1, animate, frames=objs.shape[0])
    # HTML(ani.to_jshtml())
    ani.save(f'src/{file_name[:-5]}.gif', writer = 'pillow')
    plt.show()

def plot_objective_history(file_names):
    """Plot the objective history from multiple experiment files.
    
    Args:
        file_names (list): List of names of the experiment files in the experiments folder
    """
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Find parameters that differ across experiments
    param_values = {}
    
    # First collect all parameters and their values for each file
    for file_name in file_names:
        experiment_path = experiments_folder + "/" + file_name
        with open(experiment_path, 'r') as fp:
            exp_meta = json.load(fp)
            
        for param, value in exp_meta['args'].items():
            if param not in param_values:
                param_values[param] = {}
            param_values[param][file_name] = value
            
    # Find which parameters have different values
    different_params = {}
    for param, values in param_values.items():
        unique_values = set(values.values())
        if len(unique_values) > 1:
            different_params[param] = values
            
    print("Parameters that differ between experiments:")
    for param, values in different_params.items():
        print(f"\n{param}:")
        for fname, value in values.items():
            print(f"  {fname}: {value}")
    
    for file_name in file_names:
        experiment_path = experiments_folder + "/" + file_name

        with open(experiment_path, 'r') as fp:
            exp_meta = json.load(fp)

        # Extract values
        objs = []
        volfrac_losses = []
        gaussian_overlaps = []
        compliances = []
        ff_losses = []
        rs_losses = []
        
        for iter in exp_meta['iter_meta']:
            try:
                objs.append(exp_meta['iter_meta'][iter]['obj'])
                volfrac_losses.append(exp_meta['iter_meta'][iter]['volfrac_loss_pre'])
                gaussian_overlaps.append(exp_meta['iter_meta'][iter]['gaussian_overlap'])
                compliances.append(exp_meta['iter_meta'][iter]['compliance'])
                ff_losses.append(exp_meta['iter_meta'][iter]['ff_loss'])
                rs_losses.append(exp_meta['iter_meta'][iter]['rs_loss'])
            except:
                objs.append(exp_meta['iter_meta']['2']['obj'])
                volfrac_losses.append(exp_meta['iter_meta']['2']['volfrac_loss_pre'])
                gaussian_overlaps.append(exp_meta['iter_meta']['2']['gaussian_overlap']) 
                compliances.append(exp_meta['iter_meta']['2']['compliance'])
                ff_losses.append(exp_meta['iter_meta']['2']['ff_loss'])
                rs_losses.append(exp_meta['iter_meta']['2']['rs_loss'])
        objs = np.array(objs)
        volfrac_losses = np.array(volfrac_losses)
        gaussian_overlaps = np.array(gaussian_overlaps)
        compliances = np.array(compliances)
        ff_losses = np.array(ff_losses)
        rs_losses = np.array(rs_losses)
        # Plot for each file
        label = " ".join([f"{param}={different_params[param][file_name]}" for param in different_params])
        ax1.plot(np.arange(len(objs)), objs, label=label)
        ax2.plot(np.arange(len(volfrac_losses)), volfrac_losses, label=label)
        ax3.plot(np.arange(len(gaussian_overlaps)), gaussian_overlaps, label=label)
        ax4.plot(np.arange(len(compliances)), compliances, label=label)
        ax5.plot(np.arange(len(ff_losses)), ff_losses, label=label)
        ax6.plot(np.arange(len(rs_losses)), rs_losses, label=label)

    # Set labels and titles
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Objective Value") 
    ax1.set_title("Total Objective History")
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Volume Fraction Loss")
    ax2.set_title("Volume Fraction Loss History")
    ax2.grid(True)
    ax2.legend()

    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Gaussian Overlap")
    ax3.set_title("Gaussian Overlap History")
    ax3.grid(True)
    ax3.legend()

    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Compliance")
    ax4.set_title("Compliance History")
    ax4.grid(True)
    ax4.legend()

    ax5.set_xlabel("Iteration")
    ax5.set_ylabel("FF Loss")
    ax5.set_title("FF Loss History")
    ax5.grid(True)
    ax5.legend()

    plt.tight_layout()
    plt.show()
