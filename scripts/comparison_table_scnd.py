import json
import os
from prettytable import PrettyTable

def create_comparison_table():
    # Define headers and metrics

    headers = ['val_sdf_loss/dataloader_idx_0',
            # 'val_tau_loss/dataloader_idx_0',
            # 'val_reg_loss/dataloader_idx_0',
            'val_smoothness_diff/dataloader_idx_1',
            # 'val_mi_original/dataloader_idx_2', 
            # 'val_mi_tau/dataloader_idx_2', 
            # 'val_mi_ratio/dataloader_idx_2', 
            # 'val_z_std_ratio/dataloader_idx_2',
            # 'val_tau_loss/dataloader_idx_2',
            'val_reconstruction_loss'
            ]

    import pandas as pd

    table_headers = [
        'MSE_sdf',
        # 'MSE_tau', 'L_reg',
        'smoothness_diff',
        # 'MI_original', 'MI_tau', 'ratio_mi', 'ratio_std',
        'MSE_chi'
        #   'MSE_tau_2'
    ]

    latex_table_headers = [
        r"$MSE_{sdf}$",
        # r"$MSE_{\tau}$*", r"$L_{reg}$",
        r"Smooth",
        # r"$MI_{original}$", r"$MI_{\tau}$", r"ratio_{mi}", r"ratio_{std}",
        r"$MSE_{\chi}$"
        #   r"$MSE_{\tau}$"
    ]

    metrics_smaller_is_better = [
        'MSE_sdf',
        # 'MSE_tau', 'L_reg',
        'smoothness_diff',
        # 'MI_original', 'ratio_mi', 'ratio_std',
        'MSE_chi'
    ]

    metrics_larger_is_better = ['MI_tau']


    reconstruction_results_pathes = ["_scnd_strtg_AEs_recon.json",
                                      "_scnd_strtg_MMD_VAEs_recon.json",
                                      "_scnd_strtg_VAEs_recon.json"
                                      ]
    
    result_pathes_prefix = "src/final_metrics_round"
    result_pathes_suffixes = ["_scnd_strtg_AEs.json",
                              "_scnd_strtg_MMD_VAEs.json",
                              "_scnd_strtg_VAEs.json"]

    reconstruction_result_pathes_prefix = "src/final_metrics_round"
    reconstruction_results_pathes_suffixes = ["_scnd_strtg_AEs_recon.json",
                                      "_scnd_strtg_MMD_VAEs_recon.json",
                                      "_scnd_strtg_VAEs_recon.json"
                                      ]

    recon_suffix = "_recon_dec"
    experiment_indeces = [1, 2, 3]

    from collections import defaultdict
    import numpy as np

    # Initialize dictionaries to store all metrics for each model
    all_reconstruction_metrics = defaultdict(lambda: defaultdict(list))
    all_results_metrics = defaultdict(lambda: defaultdict(list))

    for experiment_index in experiment_indeces:
        reconstruction_results_pathes = [reconstruction_result_pathes_prefix + str(experiment_index) + suffix for suffix in reconstruction_results_pathes_suffixes]
        for path in reconstruction_results_pathes:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    recon_results = json.load(file)
                    # Remove suffix in keys of recon_results
                    recon_results = {key.replace(recon_suffix, ''): value for key, value in recon_results.items()}
                    for model_name, metrics in recon_results.items():
                        for metric_name, value in metrics.items():
                            all_reconstruction_metrics[model_name][metric_name].append(value)

        result_pathes = [result_pathes_prefix + str(experiment_index) + suffix for suffix in result_pathes_suffixes]
        for path in result_pathes:
            if os.path.exists(path):
                with open(path, 'r') as file:
                    res = json.load(file)
                    for model_name, metrics in res.items():
                        for metric_name, value in metrics.items():
                            all_results_metrics[model_name][metric_name].append(value)

    # Compute mean and std for each metric
    reconstruction_results = {}
    for model_name, metrics in all_reconstruction_metrics.items():
        reconstruction_results[model_name] = {metric_name: {'mean': np.mean(values), 'std': np.std(values)} for metric_name, values in metrics.items()}

    results = {}
    for model_name, metrics in all_results_metrics.items():
        results[model_name] = {metric_name: {'mean': np.mean(values), 'std': np.std(values)} for metric_name, values in metrics.items()}

    for model_name, metrics in reconstruction_results.items():
        if model_name in results:
            combined_metrics = {**metrics, **results[model_name]}
        else:
            combined_metrics = metrics

        results[model_name] = combined_metrics

    table = PrettyTable()
    table.field_names = ["Model"] + table_headers

    # Find best values for each metric
    best_values = {}
    for header in headers:
        values = [metrics[header]['mean'] if header in metrics else float('inf') for metrics in results.values()]
        if header in [headers[i] for i, h in enumerate(table_headers) if h in metrics_smaller_is_better]:
            best_values[header] = min(values)
        else:
            best_values[header] = max(values)

    # Create lists to store data for pandas DataFrame
    df_data = []
    
    for model_name, metrics in results.items():
        row = [model_name]
        row_data = {'Model': model_name}
        
        for i, header in enumerate(headers):
            mean_value = metrics.get(header, {}).get('mean', "N/A")
            std_value = metrics.get(header, {}).get('std', "N/A")
            if mean_value != "N/A" and std_value != "N/A":
                # Add * if this is the best value
                is_best = abs(mean_value - best_values[header]) < 1e-10
                formatted_value = f"\033[91m{mean_value:.3g} ± {std_value:.3g}\033[0m" if is_best else f"{mean_value:.3g} ± {std_value:.3g}"
                row.append(formatted_value)
                # Store raw value in row_data for DataFrame
                row_data[table_headers[i]] = (mean_value, std_value)
            else:
                row.append("N/A")
                row_data[table_headers[i]] = None
                
        table.add_row(row)
        df_data.append(row_data)

    print(table)
    
    # Create pandas DataFrame
    # df = pd.DataFrame(df_data)
    
    # # Replace column headers with LaTeX versions
    # df.columns = ['Model'] + latex_table_headers
    
    # # Format values and highlight best metrics
    # def format_value(x, header_idx):
    #     if pd.isnull(x):
    #         return 'N/A'
    #     header = table_headers[header_idx-1] if header_idx > 0 else None
    #     if header:
    #         is_best = abs(x - best_values[headers[header_idx-1]]) < 1e-10
    #         if is_best:
    #             return f'\\textbf{{{x:.3g}}}'
    #     return f'{x:.3g}'
    
    # formatted_df = df.copy()
    # for i in range(len(df.columns)):
    #     if i > 0:  # Skip Model column
    #         formatted_df.iloc[:,i] = df.iloc[:,i].apply(lambda x: format_value(x, i))
    
    # # Convert DataFrame to LaTeX table
    # latex_table = formatted_df.to_latex(
    #     index=False,
    #     caption='Comparison of Model Metrics',
    #     label='tab:model_metrics',
    #     escape=False
    # )
    
    # # Add table styling
    # latex_table = latex_table.replace('\\begin{table}', '\\begin{table}[htbp]')
    # latex_table = latex_table.replace('\\toprule', '\\hline')
    # latex_table = latex_table.replace('\\midrule', '\\hline')
    # latex_table = latex_table.replace('\\bottomrule', '\\hline')
    
    # print("\nLaTeX Table:")
    # print(latex_table)

if __name__ == "__main__":
    create_comparison_table()