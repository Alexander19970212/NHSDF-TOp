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

from matplotlib.patches import Ellipse, Polygon
from dataset_generation.utils_generation import extract_geometry

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import time

import matplotlib.colors as mcolors
    


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
type2ids = {
    "e": 0,
    "t": 1,
    "q": 2
}

def plot_latent_space(model, dataloader, dr_method="tsne", num_samples=500, filename = None):
    """Visualize the latent space"""
    model.eval()
    latent_vectors = []
    # X = []
    # sdf = []
    # sdf_target = []
    class_labels = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch[0])
            # print(output)
            latent_vectors.append(output["z"])
            # X.append(batch[0])
            # sdf.append(output["sdf_pred"])
            # sdf_target.append(batch[1])
            class_labels.append(batch[0][:, 2])
            # if len(latent_vectors) * batch[0].shape[0] >= num_samples:
            #     break
                
    latent_vectors = torch.cat(latent_vectors, dim=0) #[:num_samples]
    latent_vectors = latent_vectors.cpu().numpy()

    # Concatenate and convert class labels
    # class_labels = torch.cat(class_labels, dim=0)[:num_samples].cpu().numpy()
    class_labels = torch.cat(class_labels, dim=0).cpu().numpy()
    class_indices = np.array([int(label*2) for label in class_labels])
    class_labels = [class_names[int(label*2)] for label in class_labels]

    all_indices = np.arange(latent_vectors.shape[0])

    parent_dir = os.path.dirname(os.path.abspath(filename))
    latents_2d_path = parent_dir + f"/latent_2d_{dr_method}.npy"

    scatter_sizes = np.ones(latent_vectors.shape[0])*50

    if os.path.exists(latents_2d_path):
        print(f"Loading latent_2d from {latents_2d_path}")
        latent_2d = np.load(latents_2d_path)
        searching_points_path = parent_dir + "/searching_points.json"
        if os.path.exists(searching_points_path):
            searching_points = json.load(open(searching_points_path))
            tsne_coords = searching_points["tsne_coords"]
            axes_positions = searching_points["axes_positions"]
            gf_types = searching_points["gf_types"]

            specified_ids = searching_points["specified_ids"]
            specified_axes_positions = searching_points["specified_axes_positions"]

            tsne_coords = np.array(tsne_coords) # P x 2

            from scipy.spatial import cKDTree

            # Create a KDTree for efficient nearest neighbor search

            # Find the closest points in latent_2d for each point in tsne_coords
            closests_indices = []
            existing_points = []
            for point, gf_type in zip(tsne_coords, gf_types):
                if gf_type == "n":
                    kdtree = cKDTree(latent_2d)

                    distance, index = kdtree.query(point)
                    closests_indices.append(index)
                    existing_points.append(latent_2d[index])
                else:
                    type_ids = all_indices[class_indices == type2ids[gf_type]]
                    kdtree = cKDTree(latent_2d[type_ids])

                    distance, index = kdtree.query(point)
                    closests_indices.append(type_ids[index])
                    existing_points.append(latent_2d[type_ids[index]])

            # print(closests_indices)
            for specified_id, specified_axes_position in zip(specified_ids, specified_axes_positions):
                closests_indices.append(specified_id)
                existing_points.append(latent_2d[specified_id])
                axes_positions.append(specified_axes_position)

            # print(closests_indices)
            scatter_sizes[closests_indices] = 500
            latents_inner_axes = latent_vectors[closests_indices]

            axes_positions = np.array(axes_positions) # P x 2

            plot_inner_axes = True

        else:
            searching_points = None
            closests_indices = None
            plot_inner_axes = False
    else:
        # Use t-SNE for dimensionality reduction
        if dr_method == "tsne":
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            latent_2d = tsne.fit_transform(latent_vectors)

        elif dr_method == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_vectors)

        elif dr_method == "umap":
            import umap
            umap = umap.UMAP(n_components=2, random_state=42)
            latent_2d = umap.fit_transform(latent_vectors)

        np.save(latents_2d_path, latent_2d)

        plot_inner_axes = False
    
    # Plot the reduced dimensions with colors based on class labels
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Convert class labels to numeric values for coloring
    unique_labels = list(set(class_labels))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    # numeric_labels = [label_to_num[label] for label in class_labels]

    # Use a distinct colormap for better differentiation
    # colors = plt.cm.get_cmap('tab10', len(class_names))
    base_cmap = plt.cm.get_cmap('brg', len(class_names))
    # Darken the color map by reducing the brightness of each RGB channel to 70%
    darkened_colors = [(r * 0.7, g * 0.7, b * 0.7, a) for r, g, b, a in (base_cmap(i) for i in range(len(class_names)))]
    colors = mcolors.ListedColormap(darkened_colors)

    for i, label in enumerate(class_names):
        class_bids_initial = [x == label for x in class_labels]
        # Make only random num_samples of true in class_bids equal to true
        num_samples = min(50, sum(class_bids_initial))  # Adjust the number of samples as needed
        true_indices = [i for i, x in enumerate(class_bids_initial) if x]
        np.random.shuffle(true_indices)
        selected_indices = true_indices[:num_samples]
        class_bids = [i in selected_indices for i in range(len(class_bids_initial))]

        # investigated points are always plotted
        class_bids[closests_indices] = class_bids_initial[closests_indices]
        # Get points for this class
        class_points = latent_2d[class_bids]    
        sc_sizes = scatter_sizes[class_bids]

        plt.scatter(
            class_points[:, 0], 
            class_points[:, 1], 
            label=label, 
            c=[colors(i)], 
            alpha=0.7, 
            edgecolors='w', 
            s=sc_sizes
        )
    # plt.title('t-SNE Visualization of Latent Space Clusters', fontsize=14)
    plt.xlabel('t-SNE Dimension 1', fontsize=20)
    plt.ylabel('t-SNE Dimension 2', fontsize=20)
    plt.legend(title='Shape Class', fontsize=12, title_fontsize='20')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if plot_inner_axes:
        from matplotlib.lines import Line2D

        ax_w = 0.1

        z_batch = torch.from_numpy(latents_inner_axes)
        chis = model.decoder_input(z_batch).detach().cpu().numpy()
        inner_class_indices = class_indices[closests_indices]
    
        # x = np.linspace(-1.0, 1.0, 100)
        # y = np.linspace(-1.0, 1.0, 100)
        # X, Y = np.meshgrid(x, y)
        # grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

        for tsne_idx in range(len(closests_indices)):
            case_color = colors(inner_class_indices[tsne_idx])
            # print(tsne_coord)
            # ax_inset = fig.add_axes([tsne_coord[0], tsne_coord[1], 0.18, 0.18])
            # Transform tsne_coord (data coordinates of the tsne plot) into figure coordinates
            scatter_pos = fig.transFigure.inverted().transform(ax.transData.transform(existing_points[tsne_idx]))
            inset_pos = fig.transFigure.inverted().transform(ax.transData.transform(axes_positions[tsne_idx]))
            # print(inset_pos)

            ax_corners = [[inset_pos[0], inset_pos[1]],
                          [inset_pos[0], inset_pos[1] + ax_w],
                          [inset_pos[0] + ax_w, inset_pos[1] + ax_w],
                          [inset_pos[0] + ax_w, inset_pos[1]]]
            ax_corners = np.array(ax_corners)
            pointer_vectors = ax_corners - scatter_pos
            vector_lengths = np.linalg.norm(pointer_vectors, axis=1)

            # Find the indexes of the two shortest vectors from pointer_vectors
            shortest_idxs = np.argsort(vector_lengths)[:2]

            # Draw an arrow connecting the scatter point (tsne_coord) to the center of the inset axis
            for ax_corner in ax_corners[shortest_idxs]:
                arrow_start = scatter_pos  # scatter point coordinates in figure fraction
                arrow_end = ax_corner
                line = Line2D([arrow_start[0], arrow_end[0]], [arrow_start[1], arrow_end[1]],
                            transform=fig.transFigure, color=case_color, lw=1)
                fig.add_artist(line)

            ax_inset = fig.add_axes([inset_pos[0], inset_pos[1], ax_w, ax_w])
            chi_pred = chis[tsne_idx]
            geometry_type, geometry_params = extract_geometry(chi_pred)
            draw_geometry(geometry_type, geometry_params, ax_inset)

            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.set_xlabel('')
            ax_inset.set_ylabel('')
            
            # Set each spine's edge color to frame_color and increase the line width for visibility
            for spine in ax_inset.spines.values():
                spine.set_edgecolor(case_color)
                spine.set_linewidth(2)


    if filename is not None:
        filename = filename.replace(".png", f"_{dr_method}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')


    plt.show()

# from vizualization_utils import plot_latent_space_radius_sum
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

    # Use t-SNE for dimensionality reduction to get latent_vectors_radius_sum in 2D
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_vectors_radius_sum = tsne.fit_transform(latent_vectors_radius_sum)

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

def plot_predicted_sdf(model, test_loader, num_samples=5):
    """Plot predicted SDF values for sample inputs"""
    model.eval()
    plt.figure(figsize=(15, 5))
    
    with torch.no_grad():
        # Get sample batch
        batch = next(iter(test_loader))
        inputs = batch[0][:8]  # Take first 8 samples

        output = model(inputs, reconstruction=True)
        z = output["z"]
        x_reconstructed = output["x_reconstructed"]
        x_original = inputs[:, 2:]

        x = np.linspace(-1.1, 1.1, 100)
        y = np.linspace(-1.1, 1.1, 100)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Predicted SDF Values for 8 Samples')

        for i in range(8):
            row = i // 4
            col = i % 4

            sdf_pred = model.sdf(z[i], grid_points)
            geometry_type, geometry_params = extract_geometry(x_reconstructed[i].detach().cpu().numpy())
            # extract_geometry(x_original[i].detach().cpu().numpy(), axs[row, col])

            if geometry_type == "ellipse":
                a = geometry_params[1]
                b = geometry_params[2]
                ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='black')
                axs[row, col].add_patch(ellipse)

            elif geometry_type == "polygon":
                vertices = geometry_params[0]
                radiuses = geometry_params[1]
                line_segments = geometry_params[2]
                arc_segments = geometry_params[3]
                # axs[row, col].add_patch(Polygon(vertices, fill=False, color='red', linewidth=4))
                for start, end in line_segments:
                    axs[row, col].plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2)
                    # ax2.plot([start[0], end[0]], [start[1], end[1]], [z_offset, z_offset], 'g-', linewidth=line_width)

                # Plot arc segments
                for center, start_angle, end_angle, radius in arc_segments:
                    # Calculate angles for arc
                    
                    # Ensure we draw the shorter arc
                    if abs(end_angle - start_angle) > np.pi:
                        if end_angle > start_angle:
                            start_angle += 2*np.pi
                        else:
                            end_angle += 2*np.pi
                            
                    # Create points along arc
                    theta = np.linspace(start_angle, end_angle, 100)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    axs[row, col].plot(x, y, 'b-', linewidth=2)

            # geometry_type, geometry_params = extract_geometry(x_reconstructed[i].detach().cpu().numpy(), axs[row, col])
            geometry_type, geometry_params = extract_geometry(x_original[i].detach().cpu().numpy())

            if geometry_type == "ellipse":
                a = geometry_params[1]
                b = geometry_params[2]
                ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='black')
                axs[row, col].add_patch(ellipse)

            elif geometry_type == "polygon":
                vertices = geometry_params[0]
                radiuses = geometry_params[1]
                line_segments = geometry_params[2]
                arc_segments = geometry_params[3]
                # axs[row, col].add_patch(Polygon(vertices, fill=False, color='green', linewidth=4))
                # Plot line segments
                for start, end in line_segments:
                    axs[row, col].plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=4)
                    # ax2.plot([start[0], end[0]], [start[1], end[1]], [z_offset, z_offset], 'g-', linewidth=line_width)

                # Plot arc segments
                for center, start_angle, end_angle, radius in arc_segments:
                    # Calculate angles for arc
                    
                    # Ensure we draw the shorter arc
                    if abs(end_angle - start_angle) > np.pi:
                        if end_angle > start_angle:
                            start_angle += 2*np.pi
                        else:
                            end_angle += 2*np.pi
                            
                    # Create points along arc
                    theta = np.linspace(start_angle, end_angle, 100)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    axs[row, col].plot(x, y, 'r-', linewidth=4)

            # Reshape predictions
            sdf_grid = sdf_pred.reshape(X.shape)

            # Create scatter plot
            im = axs[row, col].imshow(sdf_grid.numpy(),
                                     extent=[-1.1, 1.1, -1.1, 1.1],
                                     cmap='plasma',
                                     origin='lower')
            axs[row, col].set_aspect('equal')
            axs[row, col].set_title(f'Sample {i+1}')
            axs[row, col].set_xlabel('X')
            axs[row, col].set_ylabel('Y')

        # Add colorbar
        # fig.colorbar(scatter, ax=axs.ravel().tolist(), label='Predicted SDF')

    plt.tight_layout()
    plt.show()

# def plot_curve(grid_points, sdf_pred, ax):

#     curve_mask = torch.logical_and(sdf_pred[:, 0] > 0.4, sdf_pred[:, 0] < 0.6)
#     curve_mask_reshaped = curve_mask.reshape(100, 100)[15:85, 15:85]
#     ax.imshow(curve_mask_reshaped, cmap='gray_r', origin='lower')
#     return ax

def plot_curve(grid_points, sdf_pred, ax):

    curve_mask = torch.logical_and(sdf_pred[:, 0] > 0.4, sdf_pred[:, 0] < 0.6)
    curve_mask_reshaped = curve_mask.reshape(100, 100)[13:87, 13:87]
    ax.imshow(curve_mask_reshaped, cmap='gray_r', origin='lower')
    return ax

# def draw_geometry(geometry_type, geometry_params, ax):

#     ax.set_xlim(-0.8, 0.8)
#     ax.set_ylim(-0.8, 0.8)
    
#     if geometry_type == "ellipse":
#         a = geometry_params[1]
#         b = geometry_params[2]
#         ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='red', linewidth=2)
#         ax.add_patch(ellipse)

#     elif geometry_type == "polygon":
#         vertices = geometry_params[0]
#         radiuses = geometry_params[1]
#         line_segments = geometry_params[2]
#         arc_segments = geometry_params[3]
#         # axs[row, col].add_patch(Polygon(vertices, fill=False, color='green', linewidth=4))
#         # Plot line segments
#         for start, end in line_segments:
#             ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=2)
#             # ax2.plot([start[0], end[0]], [start[1], end[1]], [z_offset, z_offset], 'g-', linewidth=line_width)

#         # Plot arc segments
#         for center, start_angle, end_angle, radius in arc_segments:
#             # Calculate angles for arc
            
#             # Ensure we draw the shorter arc
#             if abs(end_angle - start_angle) > np.pi:
#                 if end_angle > start_angle:
#                     start_angle += 2*np.pi
#                 else:
#                     end_angle += 2*np.pi
                    
#             # Create points along arc
#             theta = np.linspace(start_angle, end_angle, 100)
#             x = center[0] + radius * np.cos(theta)
#             y = center[1] + radius * np.sin(theta)
#             ax.plot(x, y, 'r-', linewidth=2)

def draw_geometry(geometry_type, geometry_params, ax):

    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)

    lw = 2
    
    if geometry_type == "ellipse":
        a = geometry_params[1]
        b = geometry_params[2]
        ellipse = Ellipse(np.array([0, 0]), 2*a, 2*b, fill=False, color='red', linewidth=lw)
        ax.add_patch(ellipse)

    elif geometry_type == "polygon":
        vertices = geometry_params[0]
        radiuses = geometry_params[1]
        line_segments = geometry_params[2]
        arc_segments = geometry_params[3]
        # axs[row, col].add_patch(Polygon(vertices, fill=False, color='green', linewidth=4))
        # Plot line segments
        for start, end in line_segments:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'g-', linewidth=lw)
            # ax2.plot([start[0], end[0]], [start[1], end[1]], [z_offset, z_offset], 'g-', linewidth=line_width)

        # Plot arc segments
        for center, start_angle, end_angle, radius in arc_segments:
            # Calculate angles for arc
            
            # Ensure we draw the shorter arc
            if abs(end_angle - start_angle) > np.pi:
                if end_angle > start_angle:
                    start_angle += 2*np.pi
                else:
                    end_angle += 2*np.pi
                    
            # Create points along arc
            theta = np.linspace(start_angle, end_angle, 100)
            x = center[0] + radius * np.cos(theta)
            y = center[1] + radius * np.sin(theta)
            ax.plot(x, y, 'r-', linewidth=lw)

def plot_sdf_transition(model, z_start, z_end, num_steps=10):
    """
    Plots the transition of SDF maps between two latent vectors.
    
    Parameters:
    - model: The VAE model
    - z_start: The starting latent vector
    - z_end: The ending latent vector
    - num_steps: Number of steps in the transition
    """
    z_start = torch.tensor(z_start, dtype=torch.float32)
    z_end = torch.tensor(z_end, dtype=torch.float32)
    
    # Generate intermediate latent vectors
    z_steps = [z_start + (z_end - z_start) * i / (num_steps - 1) for i in range(num_steps)]
    
    x = np.linspace(-1.1, 1.1, 100)
    y = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    fig, axs = plt.subplots(1, num_steps, figsize=(20, 5))
    fig.suptitle('SDF Transition Between Shapes')
    
    with torch.no_grad():
        for i, z in enumerate(z_steps):
            sdf_pred = model.sdf(z, grid_points)
            sdf_grid = sdf_pred.reshape(X.shape)
            
            plot_curve(grid_points, sdf_pred, axs[i])
            axs[i].set_aspect('equal')
            axs[i].set_frame_on(False)
            axs[i].set_xticks([])
            axs[i].set_yticks([])
        
    plt.tight_layout()
    plt.show()

# def plot_sdf_transition_triangle(model, z1, z2, z3, num_steps=10, filename=None, plot_geometry=False):
#     """
#     Plots the transition of SDF maps between three latent vectors in a triangular format.
    
#     Parameters:
#     - model: The VAE model
#     - z1: The first latent vector
#     - z2: The second latent vector
#     - z3: The third latent vector
#     - num_steps: Number of steps in the transition
#     """
#     z1 = torch.tensor(z1, dtype=torch.float32)
#     z2 = torch.tensor(z2, dtype=torch.float32)
#     z3 = torch.tensor(z3, dtype=torch.float32)
    
#     # Generate intermediate latent vectors
#     z_steps = []
#     for i in range(num_steps):
#         for j in range(num_steps - i):
#             z = z1 * (i / (num_steps - 1)) + z2 * (j / (num_steps - 1)) + z3 * ((num_steps - 1 - i - j) / (num_steps - 1))
#             z_steps.append(z)

#     # z_batch = torch.tensor(z_steps).to(model.device)
#     z_batch = torch.stack(z_steps)
#     print(z_batch.shape)
#     chis = model.decoder_input(z_batch).detach().cpu().numpy()
    
#     x = np.linspace(-1.1, 1.1, 100)
#     y = np.linspace(-1.1, 1.1, 100)
#     X, Y = np.meshgrid(x, y)
#     grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
#     fig, ax = plt.subplots(figsize=(5, 4.5))
#     fig.suptitle('SDF Transition Between Shapes in Triangle Format')

#     scale = 0.85 / (num_steps - 1)
    
#     with torch.no_grad():
#         idx = 0
#         for i in range(num_steps):
#             for j in range(num_steps - i):
#                 z = z_steps[idx]
#                 # Calculate the position for each icon
#                 x_pos = (i + 0.5 * j) * scale #- 0.55
#                 y_pos = (np.sqrt(3) / 2 * j) * scale #- 0.55
                
#                 # Plot each SDF as an icon at the calculated position
#                 ax_inset = fig.add_axes([x_pos, y_pos, 0.16, 0.16])

#                 if plot_geometry:
#                     chi_pred = chis[idx]
#                     geometry_type, geometry_params = extract_geometry(chi_pred)
#                     draw_geometry(geometry_type, geometry_params, ax_inset)
                    
#                 else:   
#                     sdf_pred = model.sdf(z, grid_points)
#                     sdf_grid = sdf_pred.reshape(X.shape)
#                     plot_curve(grid_points, sdf_pred, ax_inset)

#                 ax_inset.set_aspect('equal')
#                 ax_inset.set_frame_on(False)
#                 ax_inset.set_xticks([])
#                 ax_inset.set_yticks([])
#                 idx += 1
#     ax.set_frame_on(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     # plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()

def plot_sdf_transition_triangle(model, z1, z2, z3, num_steps=10, filename=None, plot_geometry=False):
    """
    Plots the transition of SDF maps between three latent vectors in a triangular format.
    
    Parameters:
    - model: The VAE model
    - z1: The first latent vector
    - z2: The second latent vector
    - z3: The third latent vector
    - num_steps: Number of steps in the transition
    """
    z1 = torch.tensor(z1, dtype=torch.float32)
    z2 = torch.tensor(z2, dtype=torch.float32)
    z3 = torch.tensor(z3, dtype=torch.float32)
    
    # Generate intermediate latent vectors
    z_steps = []
    for i in range(num_steps):
        for j in range(num_steps - i):
            z = z1 * (i / (num_steps - 1)) + z2 * (j / (num_steps - 1)) + z3 * ((num_steps - 1 - i - j) / (num_steps - 1))
            z_steps.append(z)

    # z_batch = torch.tensor(z_steps).to(model.device)
    z_batch = torch.stack(z_steps)
    # print(z_batch.shape)
    chis = model.decoder_input(z_batch).detach().cpu().numpy()
    
    x = np.linspace(-1.0, 1.0, 100)
    y = np.linspace(-1.0, 1.0, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    fig, ax = plt.subplots(figsize=(10, 9))
    # fig.suptitle('SDF Transition Between Shapes in Triangle Format', fontsize=16, fontweight='bold')

    scale = 0.85 / (num_steps - 1)
    
    with torch.no_grad():
        idx = 0
        for i in range(num_steps):
            for j in range(num_steps - i):
                z = z_steps[idx]
                # Calculate the position for each icon
                x_pos = (i + 0.5 * j) * scale
                y_pos = (np.sqrt(3) / 2 * j) * scale
                
                # Plot each SDF as an icon at the calculated position
                if plot_geometry:
                    ax_inset = fig.add_axes([x_pos, y_pos, 0.18, 0.18])
                    chi_pred = chis[idx]
                    geometry_type, geometry_params = extract_geometry(chi_pred)
                    draw_geometry(geometry_type, geometry_params, ax_inset)
                else:
                    ax_inset = fig.add_axes([x_pos, y_pos, 0.16, 0.16])   
                    sdf_pred = model.sdf(z, grid_points)
                    sdf_grid = sdf_pred.reshape(X.shape)
                    plot_curve(grid_points, sdf_pred, ax_inset)

                ax_inset.set_aspect('equal')
                ax_inset.set_frame_on(False)
                ax_inset.set_xticks([])
                ax_inset.set_yticks([])
                # ax_inset.set_title(f'Step {idx+1}', fontsize=10)
                idx += 1

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05)
    # plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_sdf_surface(model, z, countur=False, filename=None):
    z = torch.tensor(z, dtype=torch.float32)
    x = np.linspace(-1.1, 1.1, 100)
    y = np.linspace(-1.1, 1.1, 100)
    X, Y = np.meshgrid(x, y)
    grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # fig.suptitle('SDF Surface')
    
    with torch.no_grad():
        sdf_pred = model.sdf(z, grid_points)
        sdf_grid = sdf_pred.reshape(X.shape)
        if countur:
            plot_curve(grid_points, sdf_pred, ax)
        else:
            im = ax.imshow(sdf_grid.numpy(),
                       extent=[-1.1, 1.1, -1.1, 1.1],
                       cmap='plasma',
                       origin='lower')
        
        ax.set_aspect('equal')
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def get_latent_subspaces(model, dataloader, num_samples=1000):
    """Visualize the latent space"""

    ellipsoid_label = torch.tensor(0)
    triangle_label = torch.tensor(0.5)
    quadrangle_label = torch.tensor(1)

    model.eval()
    latent_vectors = []
    X = []
    sdf = []
    sdf_target = []
    class_labels = []
    with torch.no_grad():
        for batch in dataloader:
            updated_input = batch[0].clone()
            for col in [6, 7, 8, 13, 14, 15, 16]:
                mask = batch[0][:, col] > 0
                updated_input[mask, col] = 0.03

            output = model(updated_input)
            latent_vectors.append(output["z"])
            X.append(updated_input)
            sdf.append(output["sdf_pred"])
            sdf_target.append(batch[1])
            class_labels.append(batch[0][:, 2])

    class_labels = torch.cat(class_labels)
    latent_vectors = torch.cat(latent_vectors)

    triangle_latent_vectors = latent_vectors[class_labels == triangle_label]
    quadrangle_latent_vectors = latent_vectors[class_labels == quadrangle_label]
    ellipse_latent_vectors = latent_vectors[class_labels == ellipsoid_label]

    return triangle_latent_vectors, quadrangle_latent_vectors, ellipse_latent_vectors