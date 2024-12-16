import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
