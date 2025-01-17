import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import json
from os import listdir
from os.path import isfile, join
from prettytable import PrettyTable

import torch

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
                            if len(steps) > 1:
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
    from NN_TopOpt.mesh_utils import LoadedMesh2D, LoadedMesh2D_ext

    loss_names = ['volfrac_loss_pre', 'gaussian_overlap', 'compliance', 'ff_loss', 'rs_loss']

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

    for loss_name in loss_names:
        if loss_name in exp_meta['iter_meta'][last_iter_key]:
            print(f"{loss_name}: {exp_meta['iter_meta'][last_iter_key][loss_name]}")

    if save_fig:
        fig_name = experiment_path = "src/" + file_name[:-5]+'.jpg'
        Th.plot_topology(x, fig_name)

    else:
        Th.plot_topology(x)

def opt_animation(file_name):
    from NN_TopOpt.mesh_utils import LoadedMesh2D, LoadedMesh2D_ext
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
            objs.append(exp_meta['iter_meta'][iter]['obj_real'])
        except:
            # print(iter)
            objs.append(exp_meta['iter_meta']['2']['obj_real'])

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
    

    # Determine the number of metrics to plot
    meta_not_to_plot = ['x', 'ce', 'dc', 'stop_flag']
    sample_file = file_names[0]
    with open(experiments_folder + "/" + sample_file, 'r') as fp:
        sample_meta = json.load(fp)
    metrics = list(sample_meta['iter_meta']['2'].keys())
    metrics = [metric for metric in metrics if metric not in meta_not_to_plot]

    num_metrics = len(metrics)

    # Create subplots dynamically based on the number of metrics
    num_rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    for file_name in file_names:
        experiment_path = experiments_folder + "/" + file_name

        with open(experiment_path, 'r') as fp:
            exp_meta = json.load(fp)

        # Extract values for each metric
        metric_values = {metric: [] for metric in metrics}
        
        for iter in exp_meta['iter_meta']:
            for metric in metrics:
                try:
                    metric_values[metric].append(exp_meta['iter_meta'][iter][metric])
                except KeyError:
                    metric_values[metric].append(exp_meta['iter_meta']['2'][metric])  # Handle missing data gracefully

        # Convert lists to numpy arrays
        for metric in metrics:
            metric_values[metric] = np.array(metric_values[metric])

        # Plot for each file
        label = " ".join([f"{param}={different_params[param][file_name]}" for param in different_params])
        for idx, metric in enumerate(metrics):
            axes[idx].plot(np.arange(len(metric_values[metric])), metric_values[metric], label=label)

    # Set labels and titles for each subplot
    for idx, metric in enumerate(metrics):
        axes[idx].set_xlabel("Iteration")
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(f"{metric.replace('_', ' ').title()} History")
        axes[idx].grid(True)
        axes[idx].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

    plt.tight_layout()
    plt.show()

class_names = ['Ellipse', 'Triangle', 'Quadrangle']

def plot_latent_space(model, dataloader, num_samples=4000, filename = None):
    """Visualize the latent space"""
    model.eval()
    latent_vectors = []
    X = []
    sdf = []
    sdf_target = []
    class_labels = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0])
            # print(output)
            latent_vectors.append(output["z"])
            X.append(batch[0])
            sdf.append(output["sdf_pred"])
            sdf_target.append(batch[1])
            class_labels.append(batch[0][:, 2])
            if len(latent_vectors) * batch[0].shape[0] >= num_samples:
                break
                
    latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples]
    latent_vectors = latent_vectors.cpu().numpy()

    # Use t-SNE for dimensionality reduction
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)

    
    # Concatenate and convert class labels
    class_labels = torch.cat(class_labels, dim=0)[:num_samples].cpu().numpy()
    class_labels = [class_names[int(label*2)] for label in class_labels]
    
    # Plot the reduced dimensions with colors based on class labels
    plt.figure(figsize=(8,8))
    # Convert class labels to numeric values for coloring
    unique_labels = list(set(class_labels))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = [label_to_num[label] for label in class_labels]

    # print(numeric_labels)
    # print(unique_labels)
    for i, label in enumerate(class_names):
        class_bids = [x == label for x in class_labels]
        # Get points for this class
        class_points = latent_2d[class_bids]

        plt.scatter(class_points[:, 0], class_points[:, 1], label=label, alpha=0.5)

    plt.scatter(latent_2d[:,0], latent_2d[:,1], c=numeric_labels, alpha=0.5)

    plt.title('Latent Space Distribution')
    plt.xlabel('First Latent Dimension')
    plt.ylabel('Second Latent Dimension')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)

    plt.show()

def plot_latent_space_radius_sum(model, dataloader, latent_dim=3, num_samples=4000, filename = None):
    """Visualize the latent space"""
    model.eval()
    latent_vectors = []
    X = []
    radius_sum_real = []
    sdf = []
    sdf_target = []
    class_labels = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0])
            latent_vectors.append(output["z"])
            X.append(batch[0])
            radius_sum_real.append(batch[2])
            sdf.append(output["sdf_pred"])
            sdf_target.append(batch[1])
            class_labels.append(batch[0][:, 2])
            if len(latent_vectors) * batch[0].shape[0] >= num_samples:
                break
                
    latent_vectors = torch.cat(latent_vectors, dim=0)[:num_samples]
    latent_vectors = latent_vectors.cpu().numpy()

    latent_vectors_radius_sum = latent_vectors[:, :latent_dim]

    radius_sum_real = torch.cat(radius_sum_real, dim=0)[:num_samples]
    radius_sum_real = radius_sum_real.cpu().numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_vectors_radius_sum[:, 0], latent_vectors_radius_sum[:, 1], c=radius_sum_real, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Radius Sum')
    plt.title('Latent Space with Radius Sum')
    plt.xlabel('First Latent Dimension')
    plt.ylabel('Second Latent Dimension')
    plt.legend()

    if filename is not None:
        plt.savefig(filename)

    plt.show()