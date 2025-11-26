# # #!/usr/bin/env python3
# # import sys
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import os

# # def main():
# #     if len(sys.argv) < 2:
# #         print("Usage: python3 compare_results_script.py file1.csv file2.csv ...")
# #         sys.exit(1)

# #     files = sys.argv[1:]
# #     dfs = {}

# #     # Load all CSVs
# #     for file in files:
# #         try:
# #             df = pd.read_csv(file)

# #             # Clean headers
# #             df.columns = [c.strip() for c in df.columns]

# #             # Force numeric (ignore errors -> NaN)
# #             for col in df.columns:    
# #                 df[col] = pd.to_numeric(df[col], errors="ignore")

# #             # Use parent folder as label (e.g. train5)
# #             name = os.path.basename(os.path.dirname(file)) or os.path.basename(file)
# #             dfs[name] = df

# #             print(f"Loaded {file} with {len(df)} rows and {len(df.columns)} columns.")

# #         except Exception as e:
# #             print(f"Error reading {file}: {e}")

# #     if not dfs:
# #         print("No valid CSVs loaded.")
# #         sys.exit(1)

# #     # Plot each column
# #     first_df = next(iter(dfs.values()))
# #     for column in first_df.columns:
# #         if column.lower() == "epoch":
# #             continue

# #         plt.figure(figsize=(10, 6))
# #         for name, df in dfs.items():
# #             if column in df.columns:
# #                 try:
# #                     plt.plot(df["epoch"], df[column], label=name)
# #                 except Exception as e:
# #                     print(f"Could not plot {column} for {name}: {e}")

# #         plt.title(f"Comparison of {column}")
# #         plt.xlabel("Epoch")
# #         plt.ylabel(column)
# #         plt.legend()
# #         plt.grid(True)
# #         plt.tight_layout()
# #         plt.show()

# # if __name__ == "__main__":
# #     main()


# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import argparse
# # import sys
# # from pathlib import Path

# # def load_results_files(file_paths):
# #     """Load all results CSV files into a dictionary of DataFrames"""
# #     results = {}
# #     for file_path in file_paths:
# #         try:
# #             df = pd.read_csv(file_path)
# #             # Extract training run name from file path
# #             run_name = Path(file_path).stem
# #             results[run_name] = df
# #             print(f"Loaded {file_path} with {len(df)} epochs")
# #         except Exception as e:
# #             print(f"Error loading {file_path}: {e}")
# #     return results

# # def compare_metrics(results_dict):
# #     """Compare key metrics across all training runs"""
# #     comparison_data = {}
    
# #     for run_name, df in results_dict.items():
# #         # Get the best epoch for key metrics
# #         best_epoch_map50 = df.loc[df['metrics/mAP50(B)'].idxmax()]
# #         best_epoch_map5095 = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
# #         best_epoch_precision = df.loc[df['metrics/precision(B)'].idxmax()]
# #         best_epoch_recall = df.loc[df['metrics/recall(B)'].idxmax()]
        
# #         # Final epoch metrics
# #         final_epoch = df.iloc[-1]
        
# #         comparison_data[run_name] = {
# #             'best_mAP50': {
# #                 'epoch': best_epoch_map50.name + 1,  # +1 because epochs are 0-indexed in CSV
# #                 'value': best_epoch_map50['metrics/mAP50(B)'],
# #                 'precision': best_epoch_map50['metrics/precision(B)'],
# #                 'recall': best_epoch_map50['metrics/recall(B)'],
# #                 'mAP50_95': best_epoch_map50['metrics/mAP50-95(B)']
# #             },
# #             'best_mAP50_95': {
# #                 'epoch': best_epoch_map5095.name + 1,
# #                 'value': best_epoch_map5095['metrics/mAP50-95(B)']
# #             },
# #             'final_metrics': {
# #                 'mAP50': final_epoch['metrics/mAP50(B)'],
# #                 'mAP50_95': final_epoch['metrics/mAP50-95(B)'],
# #                 'precision': final_epoch['metrics/precision(B)'],
# #                 'recall': final_epoch['metrics/recall(B)']
# #             },
# #             'total_epochs': len(df),
# #             'training_time': df['time'].sum()
# #         }
    
# #     return comparison_data

# # def print_comparison_table(comparison_data):
# #     """Print a formatted comparison table"""
# #     print("\n" + "="*100)
# #     print("COMPARISON OF TRAINING RESULTS")
# #     print("="*100)
    
# #     # Best mAP50 comparison
# #     print("\nBEST mAP50 COMPARISON:")
# #     print("-" * 80)
# #     print(f"{'Run':<15} {'Epoch':<8} {'mAP50':<10} {'Precision':<12} {'Recall':<10} {'mAP50-95':<12}")
# #     print("-" * 80)
    
# #     for run_name, data in comparison_data.items():
# #         best = data['best_mAP50']
# #         print(f"{run_name:<15} {best['epoch']:<8} {best['value']:<10.4f} {best['precision']:<12.4f} {best['recall']:<10.4f} {best['mAP50_95']:<12.4f}")
    
# #     # Final metrics comparison
# #     print("\nFINAL EPOCH METRICS:")
# #     print("-" * 80)
# #     print(f"{'Run':<15} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10}")
# #     print("-" * 80)
    
# #     for run_name, data in comparison_data.items():
# #         final = data['final_metrics']
# #         print(f"{run_name:<15} {final['mAP50']:<10.4f} {final['mAP50_95']:<12.4f} {final['precision']:<12.4f} {final['recall']:<10.4f}")
    
# #     # Training info
# #     print("\nTRAINING INFORMATION:")
# #     print("-" * 50)
# #     print(f"{'Run':<15} {'Epochs':<8} {'Total Time (s)':<15}")
# #     print("-" * 50)
    
# #     for run_name, data in comparison_data.items():
# #         print(f"{run_name:<15} {data['total_epochs']:<8} {data['training_time']:<15.2f}")

# # def plot_comparison(results_dict, output_file=None):
# #     """Create comparison plots"""
# #     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# #     fig.suptitle('YOLO Training Results Comparison', fontsize=16, fontweight='bold')
    
# #     metrics_to_plot = [
# #         ('metrics/mAP50(B)', 'mAP50'),
# #         ('metrics/mAP50-95(B)', 'mAP50-95'),
# #         ('metrics/precision(B)', 'Precision'),
# #         ('metrics/recall(B)', 'Recall'),
# #         ('train/box_loss', 'Training Box Loss'),
# #         ('val/box_loss', 'Validation Box Loss')
# #     ]
    
# #     for idx, (metric_col, title) in enumerate(metrics_to_plot):
# #         ax = axes[idx // 3, idx % 3]
        
# #         for run_name, df in results_dict.items():
# #             epochs = range(1, len(df) + 1)
# #             if metric_col in df.columns:
# #                 ax.plot(epochs, df[metric_col], label=run_name, linewidth=2)
        
# #         ax.set_title(title, fontweight='bold')
# #         ax.set_xlabel('Epoch')
# #         ax.set_ylabel(title)
# #         ax.legend()
# #         ax.grid(True, alpha=0.3)
    
# #     plt.tight_layout()
    
# #     if output_file:
# #         plt.savefig(output_file, dpi=300, bbox_inches='tight')
# #         print(f"\nPlot saved as: {output_file}")
    
# #     plt.show()

# # def main():
# #     parser = argparse.ArgumentParser(description='Compare YOLO training results from CSV files')
# #     parser.add_argument('csv_files', nargs='+', help='Paths to results CSV files')
# #     parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
# #     parser.add_argument('--output', type=str, help='Output file name for plots')
    
# #     args = parser.parse_args()
    
# #     if len(args.csv_files) < 1:
# #         print("Error: Please provide at least one CSV file")
# #         sys.exit(1)
    
# #     print(f"Comparing {len(args.csv_files)} results files: {args.csv_files}")
    
# #     # Load results
# #     results_dict = load_results_files(args.csv_files)
    
# #     if not results_dict:
# #         print("Error: No valid CSV files loaded")
# #         sys.exit(1)
    
# #     # Compare metrics
# #     comparison_data = compare_metrics(results_dict)
    
# #     # Print comparison table
# #     print_comparison_table(comparison_data)
    
# #     # Generate plots if requested
# #     if args.plot:
# #         plot_comparison(results_dict, args.output)

# # if __name__ == "__main__":
# #     main()

# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import argparse
# # import sys
# # from pathlib import Path

# # def load_results_files(file_paths):
# #     """Load all results CSV files into a dictionary of DataFrames"""
# #     results = {}
# #     for file_path in file_paths:
# #         try:
# #             df = pd.read_csv(file_path)
# #             # Extract training run name from file path
# #             run_name = Path(file_path).stem
# #             results[run_name] = df
# #             print(f"✓ Loaded {file_path} with {len(df)} epochs")
# #         except Exception as e:
# #             print(f"✗ Error loading {file_path}: {e}")
# #     return results

# # def calculate_efficiency_metrics(results_dict):
# #     """Calculate time efficiency and performance metrics"""
# #     efficiency_data = {}
    
# #     for run_name, df in results_dict.items():
# #         # Cumulative time
# #         df['cumulative_time'] = df['time'].cumsum()
        
# #         # Find key milestones
# #         milestones = {}
        
# #         # Find epochs where metrics cross certain thresholds
# #         for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
# #             mask = df['metrics/mAP50(B)'] >= threshold
# #             if mask.any():
# #                 first_epoch = mask.idxmax() + 1
# #                 time_to_threshold = df.loc[mask.idxmax(), 'cumulative_time']
# #                 milestones[f'mAP50_{threshold}'] = {
# #                     'epoch': first_epoch,
# #                     'time': time_to_threshold
# #                 }
        
# #         efficiency_data[run_name] = {
# #             'milestones': milestones,
# #             'total_time': df['time'].sum(),
# #             'avg_time_per_epoch': df['time'].mean(),
# #             'final_mAP50': df['metrics/mAP50(B)'].iloc[-1],
# #             'final_mAP50_95': df['metrics/mAP50-95(B)'].iloc[-1]
# #         }
    
# #     return efficiency_data

# # def compare_all_metrics(results_dict):
# #     """Compare all metrics across training runs"""
# #     comparison_data = {}
    
# #     for run_name, df in results_dict.items():
# #         # Best performance metrics
# #         best_epoch_map50 = df.loc[df['metrics/mAP50(B)'].idxmax()]
# #         best_epoch_map5095 = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
# #         best_epoch_precision = df.loc[df['metrics/precision(B)'].idxmax()]
# #         best_epoch_recall = df.loc[df['metrics/recall(B)'].idxmax()]
        
# #         # Best loss metrics (minimum values)
# #         best_epoch_val_loss = df.loc[df['val/box_loss'].idxmin()]
# #         best_epoch_train_loss = df.loc[df['train/box_loss'].idxmin()]
        
# #         # Final epoch metrics
# #         final_epoch = df.iloc[-1]
        
# #         # Time metrics
# #         total_time = df['time'].sum()
# #         avg_time_per_epoch = df['time'].mean()
        
# #         comparison_data[run_name] = {
# #             'best_performance': {
# #                 'mAP50': {
# #                     'epoch': best_epoch_map50.name + 1,
# #                     'value': best_epoch_map50['metrics/mAP50(B)'],
# #                     'time_to_best': df.loc[:best_epoch_map50.name, 'time'].sum()
# #                 },
# #                 'mAP50_95': {
# #                     'epoch': best_epoch_map5095.name + 1,
# #                     'value': best_epoch_map5095['metrics/mAP50-95(B)'],
# #                     'time_to_best': df.loc[:best_epoch_map5095.name, 'time'].sum()
# #                 },
# #                 'precision': best_epoch_precision['metrics/precision(B)'],
# #                 'recall': best_epoch_recall['metrics/recall(B)']
# #             },
# #             'best_losses': {
# #                 'val_box_loss': best_epoch_val_loss['val/box_loss'],
# #                 'train_box_loss': best_epoch_train_loss['train/box_loss'],
# #                 'val_cls_loss': best_epoch_val_loss['val/cls_loss'],
# #                 'train_cls_loss': best_epoch_train_loss['train/cls_loss']
# #             },
# #             'final_metrics': {
# #                 'mAP50': final_epoch['metrics/mAP50(B)'],
# #                 'mAP50_95': final_epoch['metrics/mAP50-95(B)'],
# #                 'precision': final_epoch['metrics/precision(B)'],
# #                 'recall': final_epoch['metrics/recall(B)'],
# #                 'val_box_loss': final_epoch['val/box_loss'],
# #                 'val_cls_loss': final_epoch['val/cls_loss']
# #             },
# #             'time_metrics': {
# #                 'total_time': total_time,
# #                 'avg_time_per_epoch': avg_time_per_epoch,
# #                 'total_epochs': len(df),
# #                 'time_performance_ratio': best_epoch_map50['metrics/mAP50(B)'] / total_time * 1000  # Score per 1000 seconds
# #             },
# #             'convergence_speed': {
# #                 'epoch_to_best_mAP50': best_epoch_map50.name + 1,
# #                 'time_to_best_mAP50': df.loc[:best_epoch_map50.name, 'time'].sum(),
# #                 'final_epoch': len(df)
# #             }
# #         }
    
# #     return comparison_data

# # def print_detailed_comparison(comparison_data):
# #     """Print comprehensive comparison tables"""
    
# #     # Performance Comparison
# #     print("\n" + "="*120)
# #     print("PERFORMANCE COMPARISON")
# #     print("="*120)
    
# #     print("\n📊 BEST PERFORMANCE METRICS:")
# #     print("-" * 120)
# #     print(f"{'Run':<15} {'Best mAP50':<12} {'Epoch':<8} {'Time to Best':<15} {'Best mAP50-95':<15} {'Precision':<12} {'Recall':<10}")
# #     print("-" * 120)
    
# #     for run_name, data in comparison_data.items():
# #         perf = data['best_performance']
# #         print(f"{run_name:<15} {perf['mAP50']['value']:<12.4f} {perf['mAP50']['epoch']:<8} "
# #               f"{perf['mAP50']['time_to_best']:<15.1f} {perf['mAP50_95']['value']:<15.4f} "
# #               f"{perf['precision']:<12.4f} {perf['recall']:<10.4f}")
    
# #     # Final Metrics Comparison
# #     print("\n🎯 FINAL EPOCH METRICS:")
# #     print("-" * 100)
# #     print(f"{'Run':<15} {'mAP50':<10} {'mAP50-95':<12} {'Precision':<12} {'Recall':<10} {'Val Box Loss':<12} {'Val Cls Loss':<12}")
# #     print("-" * 100)
    
# #     for run_name, data in comparison_data.items():
# #         final = data['final_metrics']
# #         print(f"{run_name:<15} {final['mAP50']:<10.4f} {final['mAP50_95']:<12.4f} "
# #               f"{final['precision']:<12.4f} {final['recall']:<10.4f} "
# #               f"{final['val_box_loss']:<12.4f} {final['val_cls_loss']:<12.4f}")
    
# #     # Time Efficiency Comparison
# #     print("\n⏱️  TIME EFFICIENCY ANALYSIS:")
# #     print("-" * 90)
# #     print(f"{'Run':<15} {'Total Time (s)':<15} {'Avg Time/Epoch':<15} {'Total Epochs':<12} {'Time/Perf Ratio':<15} {'Epoch to Best':<12}")
# #     print("-" * 90)
    
# #     for run_name, data in comparison_data.items():
# #         time_metrics = data['time_metrics']
# #         convergence = data['convergence_speed']
# #         print(f"{run_name:<15} {time_metrics['total_time']:<15.1f} {time_metrics['avg_time_per_epoch']:<15.2f} "
# #               f"{time_metrics['total_epochs']:<12} {time_metrics['time_performance_ratio']:<15.4f} "
# #               f"{convergence['epoch_to_best_mAP50']:<12}")
    
# #     # Loss Comparison
# #     print("\n📉 BEST LOSS VALUES:")
# #     print("-" * 80)
# #     print(f"{'Run':<15} {'Val Box Loss':<15} {'Train Box Loss':<15} {'Val Cls Loss':<15} {'Train Cls Loss':<15}")
# #     print("-" * 80)
    
# #     for run_name, data in comparison_data.items():
# #         losses = data['best_losses']
# #         print(f"{run_name:<15} {losses['val_box_loss']:<15.4f} {losses['train_box_loss']:<15.4f} "
# #               f"{losses['val_cls_loss']:<15.4f} {losses['train_cls_loss']:<15.4f}")

# # def print_ranking_tables(comparison_data):
# #     """Print ranked comparisons for quick assessment"""
    
# #     # Rank by mAP50
# #     print("\n🏆 RANKING BY BEST mAP50:")
# #     print("-" * 60)
# #     ranked_by_map50 = sorted(comparison_data.items(), 
# #                            key=lambda x: x[1]['best_performance']['mAP50']['value'], 
# #                            reverse=True)
    
# #     for rank, (run_name, data) in enumerate(ranked_by_map50, 1):
# #         perf = data['best_performance']['mAP50']
# #         print(f"{rank}. {run_name:<12} - mAP50: {perf['value']:.4f} (epoch {perf['epoch']}, time: {perf['time_to_best']:.1f}s)")
    
# #     # Rank by time efficiency (performance per second)
# #     print("\n⚡ RANKING BY TIME EFFICIENCY (mAP50 per 1000s):")
# #     print("-" * 60)
# #     ranked_by_efficiency = sorted(comparison_data.items(), 
# #                                 key=lambda x: x[1]['time_metrics']['time_performance_ratio'], 
# #                                 reverse=True)
    
# #     for rank, (run_name, data) in enumerate(ranked_by_efficiency, 1):
# #         ratio = data['time_metrics']['time_performance_ratio']
# #         total_time = data['time_metrics']['total_time']
# #         best_map50 = data['best_performance']['mAP50']['value']
# #         print(f"{rank}. {run_name:<12} - Score: {ratio:.4f} (mAP50: {best_map50:.4f}, total: {total_time:.1f}s)")
    
# #     # Rank by convergence speed
# #     print("\n🚀 RANKING BY CONVERGENCE SPEED (Time to Best mAP50):")
# #     print("-" * 60)
# #     ranked_by_speed = sorted(comparison_data.items(), 
# #                            key=lambda x: x[1]['best_performance']['mAP50']['time_to_best'])
    
# #     for rank, (run_name, data) in enumerate(ranked_by_speed, 1):
# #         time_to_best = data['best_performance']['mAP50']['time_to_best']
# #         best_map50 = data['best_performance']['mAP50']['value']
# #         epoch = data['best_performance']['mAP50']['epoch']
# #         print(f"{rank}. {run_name:<12} - Time: {time_to_best:.1f}s (mAP50: {best_map50:.4f} at epoch {epoch})")

# # def plot_comprehensive_comparison(results_dict, output_file=None):
# #     """Create comprehensive comparison plots"""
# #     fig, axes = plt.subplots(3, 3, figsize=(20, 15))
# #     fig.suptitle('Comprehensive YOLO Training Results Comparison', fontsize=16, fontweight='bold')
    
# #     # Performance metrics
# #     performance_metrics = [
# #         ('metrics/mAP50(B)', 'mAP50', 'upper left'),
# #         ('metrics/mAP50-95(B)', 'mAP50-95', 'upper left'),
# #         ('metrics/precision(B)', 'Precision', 'upper left'),
# #         ('metrics/recall(B)', 'Recall', 'upper left')
# #     ]
    
# #     # Loss metrics
# #     loss_metrics = [
# #         ('train/box_loss', 'Training Box Loss', 'upper right'),
# #         ('val/box_loss', 'Validation Box Loss', 'upper right'),
# #         ('train/cls_loss', 'Training Class Loss', 'upper right'),
# #         ('val/cls_loss', 'Validation Class Loss', 'upper right')
# #     ]
    
# #     # Plot performance metrics
# #     for idx, (metric_col, title, legend_pos) in enumerate(performance_metrics):
# #         ax = axes[0, idx] if idx < 2 else axes[1, idx-2]
        
# #         for run_name, df in results_dict.items():
# #             epochs = range(1, len(df) + 1)
# #             if metric_col in df.columns:
# #                 ax.plot(epochs, df[metric_col], label=run_name, linewidth=2)
        
# #         ax.set_title(f'{title} Progression', fontweight='bold')
# #         ax.set_xlabel('Epoch')
# #         ax.set_ylabel(title)
# #         ax.legend(loc=legend_pos)
# #         ax.grid(True, alpha=0.3)
    
# #     # Plot time vs performance (scatter plot)
# #     ax = axes[2, 0]
# #     for run_name, df in results_dict.items():
# #         cumulative_time = df['time'].cumsum().values
# #         map50 = df['metrics/mAP50(B)'].values
# #         ax.plot(cumulative_time, map50, label=run_name, linewidth=2, marker='o', markersize=3)
    
# #     ax.set_title('mAP50 vs Training Time', fontweight='bold')
# #     ax.set_xlabel('Cumulative Time (s)')
# #     ax.set_ylabel('mAP50')
# #     ax.legend()
# #     ax.grid(True, alpha=0.3)
    
# #     # Plot learning rates (if available)
# #     ax = axes[2, 1]
# #     for run_name, df in results_dict.items():
# #         if 'lr/pg0' in df.columns:
# #             epochs = range(1, len(df) + 1)
# #             ax.plot(epochs, df['lr/pg0'], label=run_name, linewidth=2)
    
# #     ax.set_title('Learning Rate Progression', fontweight='bold')
# #     ax.set_xlabel('Epoch')
# #     ax.set_ylabel('Learning Rate')
# #     ax.legend()
# #     ax.grid(True, alpha=0.3)
    
# #     # Empty subplot for future use or remove
# #     axes[2, 2].axis('off')
    
# #     plt.tight_layout()
    
# #     if output_file:
# #         plt.savefig(output_file, dpi=300, bbox_inches='tight')
# #         print(f"\n📈 Plot saved as: {output_file}")
    
# #     plt.show()

# # def main():
# #     parser = argparse.ArgumentParser(description='Comprehensive YOLO training results comparison')
# #     parser.add_argument('csv_files', nargs='+', help='Paths to results CSV files')
# #     parser.add_argument('--plot', action='store_true', help='Generate comparison plots')
# #     parser.add_argument('--output', type=str, help='Output file name for plots')
    
# #     args = parser.parse_args()
    
# #     if len(args.csv_files) < 1:
# #         print("Error: Please provide at least one CSV file")
# #         sys.exit(1)
    
# #     print(f"🔍 Comparing {len(args.csv_files)} results files: {args.csv_files}")
    
# #     # Load results
# #     results_dict = load_results_files(args.csv_files)
    
# #     if not results_dict:
# #         print("Error: No valid CSV files loaded")
# #         sys.exit(1)
    
# #     # Compare all metrics
# #     comparison_data = compare_all_metrics(results_dict)
    
# #     # Print detailed comparison
# #     print_detailed_comparison(comparison_data)
    
# #     # Print ranking tables
# #     print_ranking_tables(comparison_data)
    
# #     # Generate plots if requested
# #     if args.plot:
# #         plot_comprehensive_comparison(results_dict, args.output)

# # if __name__ == "__main__":
# #     main()


# import pandas as pd
# import argparse
# from pathlib import Path

# def load_results_files(file_paths):
#     """Load all results CSV files"""
#     results = {}
#     for file_path in file_paths:
#         try:
#             df = pd.read_csv(file_path)
#             run_name = Path(file_path).stem
#             results[run_name] = df
#             print(f"Loaded {file_path} ({len(df)} epochs)")
#         except Exception as e:
#             print(f"Error loading {file_path}: {e}")
#     return results

# def print_time_table(results_dict):
#     """Print a simple time comparison table"""
    
#     # Table headers
#     headers = ["Run", "Total Time", "Avg/Ep", "Epochs", "Time→Best", "Ep→Best"]
    
#     # Calculate column widths
#     col_widths = [len(header) for header in headers]
    
#     # Prepare data rows
#     data_rows = []
#     for run_name, df in results_dict.items():
#         # Find best epoch
#         best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
#         best_mAP50 = df.loc[best_epoch_idx, 'metrics/mAP50(B)']
        
#         # Time calculations
#         total_time = df['time'].sum()
#         avg_time = df['time'].mean()
#         time_to_best = df.loc[:best_epoch_idx, 'time'].sum()
        
#         row = [
#             run_name,
#             f"{total_time:.1f}s",
#             f"{avg_time:.1f}s",
#             str(len(df)),
#             f"{time_to_best:.1f}s",
#             f"{best_epoch_idx + 1}"
#         ]
#         data_rows.append(row)
        
#         # Update column widths
#         for i, cell in enumerate(row):
#             col_widths[i] = max(col_widths[i], len(cell))
    
#     # Print table
#     print("\nTIME COMPARISON")
    
#     # Top border
#     top_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
#     print(top_line)
    
#     # Header
#     header_line = "|"
#     for i, header in enumerate(headers):
#         header_line += f" {header:<{col_widths[i]}} |"
#     print(header_line)
    
#     # Separator
#     print(top_line)
    
#     # Data rows
#     for row in data_rows:
#         row_line = "|"
#         for i, cell in enumerate(row):
#             row_line += f" {cell:<{col_widths[i]}} |"
#         print(row_line)
    
#     # Bottom border
#     print(top_line)

# def main():
#     parser = argparse.ArgumentParser(description='Simple YOLO time comparison')
#     parser.add_argument('csv_files', nargs='+', help='Paths to results CSV files')
    
#     args = parser.parse_args()
    
#     # Load results
#     results_dict = load_results_files(args.csv_files)
    
#     if not results_dict:
#         print("No valid CSV files loaded")
#         return
    
#     # Print time comparison table
#     print_time_table(results_dict)

# if __name__ == "__main__":
#     main()


import pandas as pd
import argparse
from pathlib import Path

def load_results_files(file_paths):
    """Load all results CSV files"""
    results = {}
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            run_name = Path(file_path).stem
            results[run_name] = df
            print(f"Loaded {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return results

def print_time_table(results_dict):
    """Print a simple total time comparison table"""
    
    # Calculate total times
    times = []
    for run_name, df in results_dict.items():
        total_time = df['time'].sum()
        times.append((run_name, total_time))
    
    # Sort by total time (fastest first)
    times.sort(key=lambda x: x[1])
    
    # Print table
    print("\nTOTAL TRAINING TIME COMPARISON")
    print("+----------+------------+")
    print("| Run      | Total Time |")
    print("+----------+------------+")
    
    for run_name, total_time in times:
        print(f"| {run_name:<8} | {total_time:>8.1f}s |")
    
    print("+----------+------------+")

def main():
    parser = argparse.ArgumentParser(description='Compare total training time')
    parser.add_argument('csv_files', nargs='+', help='Paths to results CSV files')
    
    args = parser.parse_args()
    
    # Load results
    results_dict = load_results_files(args.csv_files)
    
    if not results_dict:
        print("No valid CSV files loaded")
        return
    
    # Print time comparison table
    print_time_table(results_dict)

if __name__ == "__main__":
    main()