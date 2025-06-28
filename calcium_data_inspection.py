#!/usr/bin/env python3
"""
Complete analysis of WT_NoStim.mat - C. elegans neural activity data
Single script that performs full analysis from exploration to visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

def check_dependencies():
    """Check and install required dependencies"""
    try:
        import h5py
        print("✓ h5py available")
    except ImportError:
        print("✗ h5py not available. Installing...")
        os.system("pip install h5py")
        try:
            import h5py
            print("✓ h5py installed successfully")
        except ImportError:
            print("✗ Failed to install h5py. Please install manually: pip install h5py")
            return False
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        print("✗ matplotlib not available. Installing...")
        os.system("pip install matplotlib")
        try:
            import matplotlib
            print("✓ matplotlib installed successfully")
        except ImportError:
            print("✗ Failed to install matplotlib. Please install manually: pip install matplotlib")
            return False
    
    return True

def explore_file_structure(filename):
    """Explore the structure of the MATLAB file"""
    
    print(f"\nExploring structure of {filename}...")
    print("=" * 60)
    
    with h5py.File(filename, 'r') as f:
        # Check if it's a MATLAB v7.3 file
        if 'WT_NoStim' in f:
            print("✓ Found WT_NoStim group - this contains the neural activity data")
            wt_group = f['WT_NoStim']
            print(f"Available datasets in WT_NoStim:")
            for key in wt_group.keys():
                print(f"  - {key}")
            return True
        else:
            print("✗ WT_NoStim group not found")
            return False

def load_neural_data(filename):
    """Load all neural data from the file"""
    
    print(f"\nLoading neural activity data...")
    print("-" * 40)
    
    with h5py.File(filename, 'r') as f:
        wt_group = f['WT_NoStim']
        
        # Load deltaFOverF data (calcium imaging)
        deltaf_data = []
        for i in range(wt_group['deltaFOverF'].shape[0]):
            ref = wt_group['deltaFOverF'][i, 0]
            data = f[ref][:]
            deltaf_data.append(data)
            print(f"Neuron dataset {i+1}: {data.shape} (neurons x time points)")
        
        # Load time vectors
        time_data = []
        for i in range(wt_group['tv'].shape[0]):
            ref = wt_group['tv'][i, 0]
            data = f[ref][:]
            time_data.append(data.flatten())  # Flatten to 1D
            print(f"Time vector {i+1}: {data.shape} ({data[0,0]:.2f} to {data[0,-1]:.2f} seconds)")
        
        # Load behavioral states
        states_data = []
        for i in range(wt_group['States'].shape[0]):
            ref = wt_group['States'][i, 0]
            try:
                # States might be stored as groups with multiple datasets
                if isinstance(f[ref], h5py.Group):
                    # Look for common state variable names
                    state_vars = ['fwd', 'rev1', 'rev2', 'revsus', 'slow', 'nostate']
                    for var in state_vars:
                        if var in f[ref]:
                            data = f[ref][var][:]
                            states_data.append(data.flatten())
                            print(f"State {i+1} ({var}): {data.shape}")
                            break
                else:
                    data = f[ref][:]
                    states_data.append(data.flatten())
                    print(f"State {i+1}: {data.shape}")
            except Exception as e:
                print(f"Could not load state {i+1}: {e}")
        
        # Load neuron names (try to decode)
        neuron_names = []
        for i in range(wt_group['NeuronNames'].shape[0]):
            ref = wt_group['NeuronNames'][i, 0]
            try:
                data = f[ref][:]
                # Try to convert from uint16 to string
                name = ''.join([chr(x) for x in data.flatten()])
                neuron_names.append(name)
                print(f"Neuron name {i+1}: {name}")
            except Exception as e:
                neuron_names.append(f"Dataset_{i+1}")
                print(f"Could not decode neuron name {i+1}: {e}")
        
        return deltaf_data, time_data, states_data, neuron_names

def plot_calcium_traces(deltaf_data, time_data, neuron_names):
    """Plot calcium imaging traces for all neurons"""
    
    print(f"\nCreating calcium imaging visualizations...")
    
    # Create a comprehensive plot
    fig, axes = plt.subplots(len(deltaf_data), 1, figsize=(15, 3*len(deltaf_data)))
    if len(deltaf_data) == 1:
        axes = [axes]
    
    for i, (neural_data, time_vec, name) in enumerate(zip(deltaf_data, time_data, neuron_names)):
        ax = axes[i]
        
        # Plot first 10 neurons in this dataset
        for j in range(min(10, neural_data.shape[0])):
            ax.plot(time_vec, neural_data[j, :], alpha=0.7, linewidth=0.8, label=f'Neuron {j+1}')
        
        ax.set_title(f'{name} - Calcium Imaging Traces (ΔF/F)', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('ΔF/F')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add mean activity line
        mean_activity = np.mean(neural_data, axis=0)
        ax.plot(time_vec, mean_activity, 'k-', linewidth=2, label='Mean Activity')
    
    plt.tight_layout()
    plt.savefig('calcium_imaging_traces.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: calcium_imaging_traces.png")

def plot_behavioral_states(states_data, time_data):
    """Plot behavioral state transitions"""
    
    if not states_data:
        print("No behavioral state data available")
        return
    
    print(f"\nCreating behavioral state visualizations...")
    
    fig, axes = plt.subplots(len(states_data), 1, figsize=(15, 3*len(states_data)))
    if len(states_data) == 1:
        axes = [axes]
    
    for i, (state_data, time_vec) in enumerate(zip(states_data, time_data)):
        ax = axes[i]
        
        # Plot state transitions
        ax.plot(time_vec, state_data, 'o-', markersize=2, linewidth=1)
        ax.set_title(f'Behavioral State Transitions - Dataset {i+1}', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('State')
        ax.grid(True, alpha=0.3)
        
        # Add state statistics
        unique_states = np.unique(state_data)
        ax.set_yticks(unique_states)
        ax.set_yticklabels([f'State {s}' for s in unique_states])
        
        # Add state distribution info
        state_counts = np.bincount(state_data.astype(int))
        ax.text(0.02, 0.98, f'States: {unique_states}\nCounts: {state_counts}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('behavioral_states.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: behavioral_states.png")

def plot_activity_heatmap(deltaf_data, time_data, neuron_names):
    """Create heatmap of neural activity"""
    
    print(f"\nCreating activity heatmaps...")
    
    fig, axes = plt.subplots(len(deltaf_data), 1, figsize=(15, 4*len(deltaf_data)))
    if len(deltaf_data) == 1:
        axes = [axes]
    
    for i, (neural_data, time_vec, name) in enumerate(zip(deltaf_data, time_data, neuron_names)):
        ax = axes[i]
        
        # Create heatmap
        im = ax.imshow(neural_data, aspect='auto', cmap='viridis', 
                      extent=[time_vec[0], time_vec[-1], 0, neural_data.shape[0]])
        
        ax.set_title(f'{name} - Neural Activity Heatmap', fontsize=12)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Neuron Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ΔF/F')
        
        # Add time ticks
        time_ticks = np.linspace(0, len(time_vec)-1, 5, dtype=int)
        ax.set_xticks(time_vec[time_ticks])
        ax.set_xticklabels([f'{time_vec[t]:.1f}' for t in time_ticks])
    
    plt.tight_layout()
    plt.savefig('neural_activity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: neural_activity_heatmap.png")

def analyze_statistics(deltaf_data, time_data, neuron_names):
    """Analyze statistical properties of the data"""
    
    print(f"\nAnalyzing statistical properties...")
    
    # Create summary statistics
    stats_summary = []
    
    for i, (neural_data, time_vec, name) in enumerate(zip(deltaf_data, time_data, neuron_names)):
        print(f"\n{name}:")
        print(f"  Recording duration: {time_vec[-1] - time_vec[0]:.2f} seconds")
        print(f"  Sampling rate: {1/(time_vec[1] - time_vec[0]):.2f} Hz")
        print(f"  Number of neurons: {neural_data.shape[0]}")
        print(f"  Number of time points: {neural_data.shape[1]}")
        
        # Calculate statistics for each neuron
        neuron_stats = []
        for j in range(neural_data.shape[0]):
            trace = neural_data[j, :]
            stats = {
                'neuron': j+1,
                'mean': np.mean(trace),
                'std': np.std(trace),
                'min': np.min(trace),
                'max': np.max(trace),
                'range': np.max(trace) - np.min(trace)
            }
            neuron_stats.append(stats)
        
        # Find most active neurons
        mean_activities = [s['mean'] for s in neuron_stats]
        most_active_idx = np.argmax(mean_activities)
        print(f"  Most active neuron: {most_active_idx + 1} (mean ΔF/F: {mean_activities[most_active_idx]:.4f})")
        
        # Calculate correlation matrix
        if neural_data.shape[0] > 1:
            corr_matrix = np.corrcoef(neural_data)
            mean_corr = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
            print(f"  Mean correlation between neurons: {mean_corr:.4f}")
        
        stats_summary.append({
            'name': name,
            'duration': time_vec[-1] - time_vec[0],
            'sampling_rate': 1/(time_vec[1] - time_vec[0]),
            'n_neurons': neural_data.shape[0],
            'n_timepoints': neural_data.shape[1],
            'neuron_stats': neuron_stats
        })
    
    return stats_summary

def create_summary_report(stats_summary):
    """Create a summary report"""
    
    print(f"\nCreating summary report...")
    
    with open('neural_activity_summary.txt', 'w') as f:
        f.write("C. elegans Neural Activity Data Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for stats in stats_summary:
            f.write(f"Dataset: {stats['name']}\n")
            f.write(f"  Recording duration: {stats['duration']:.2f} seconds\n")
            f.write(f"  Sampling rate: {stats['sampling_rate']:.2f} Hz\n")
            f.write(f"  Number of neurons: {stats['n_neurons']}\n")
            f.write(f"  Number of time points: {stats['n_timepoints']}\n")
            
            # Top 5 most active neurons
            sorted_neurons = sorted(stats['neuron_stats'], key=lambda x: x['mean'], reverse=True)
            f.write("  Top 5 most active neurons:\n")
            for i, neuron in enumerate(sorted_neurons[:5]):
                f.write(f"    {i+1}. Neuron {neuron['neuron']}: mean={neuron['mean']:.4f}, std={neuron['std']:.4f}\n")
            
            f.write("\n")
    
    print("✓ Saved: neural_activity_summary.txt")

def create_quick_overview_plot(deltaf_data, time_data, neuron_names):
    """Create a quick overview plot showing sample data"""
    
    print(f"\nCreating quick overview plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot 1: Sample neural traces
    ax1 = axes[0]
    for i, (neural_data, time_vec, name) in enumerate(zip(deltaf_data[:2], time_data[:2], neuron_names[:2])):
        for j in range(min(5, neural_data.shape[0])):
            ax1.plot(time_vec, neural_data[j, :], alpha=0.7, linewidth=0.8, 
                    label=f'{name} Neuron {j+1}')
    ax1.set_title('Sample Neural Traces')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ΔF/F')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Activity distribution
    ax2 = axes[1]
    all_activities = []
    for neural_data in deltaf_data:
        all_activities.extend(neural_data.flatten())
    ax2.hist(all_activities, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_title('Distribution of Neural Activity')
    ax2.set_xlabel('ΔF/F')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Dataset comparison
    ax3 = axes[2]
    dataset_sizes = [data.shape[0] for data in deltaf_data]
    ax3.bar(range(len(dataset_sizes)), dataset_sizes)
    ax3.set_title('Number of Neurons per Dataset')
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Number of Neurons')
    ax3.set_xticks(range(len(dataset_sizes)))
    ax3.set_xticklabels([f'DS{i+1}' for i in range(len(dataset_sizes))])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recording durations
    ax4 = axes[3]
    durations = [time_vec[-1] - time_vec[0] for time_vec in time_data]
    ax4.bar(range(len(durations)), durations)
    ax4.set_title('Recording Duration per Dataset')
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Duration (seconds)')
    ax4.set_xticks(range(len(durations)))
    ax4.set_xticklabels([f'DS{i+1}' for i in range(len(durations))])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: quick_overview.png")

def main():
    """Main analysis function"""
    
    filename = "WT_NoStim.mat"
    
    print("C. elegans Neural Activity Data Analysis")
    print("=" * 50)
    
    # Check if file exists
    if not Path(filename).exists():
        print(f"✗ Error: {filename} not found!")
        print("Please ensure the file is in the current directory.")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("✗ Failed to install required dependencies. Please install manually.")
        return
    
    # Explore file structure
    if not explore_file_structure(filename):
        print("✗ File structure not recognized. This may not be the expected data format.")
        return
    
    # Load neural data
    try:
        deltaf_data, time_data, states_data, neuron_names = load_neural_data(filename)
    except Exception as e:
        print(f"✗ Error loading neural data: {e}")
        return
    
    # Create visualizations
    try:
        plot_calcium_traces(deltaf_data, time_data, neuron_names)
        plot_behavioral_states(states_data, time_data)
        plot_activity_heatmap(deltaf_data, time_data, neuron_names)
        create_quick_overview_plot(deltaf_data, time_data, neuron_names)
    except Exception as e:
        print(f"✗ Error creating visualizations: {e}")
        return
    
    # Analyze statistics
    try:
        stats_summary = analyze_statistics(deltaf_data, time_data, neuron_names)
        create_summary_report(stats_summary)
    except Exception as e:
        print(f"✗ Error analyzing statistics: {e}")
        return
    
    # Final summary
    print(f"\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print("Generated files:")
    print("✓ calcium_imaging_traces.png - Detailed neural activity traces")
    print("✓ behavioral_states.png - Behavioral state transitions")
    print("✓ neural_activity_heatmap.png - Activity heatmaps")
    print("✓ quick_overview.png - Quick overview of all datasets")
    print("✓ neural_activity_summary.txt - Statistical summary")
    print(f"\nData summary:")
    print(f"- {len(deltaf_data)} datasets analyzed")
    print(f"- {sum(data.shape[0] for data in deltaf_data)} total neurons")
    print(f"- ~{time_data[0][-1]:.0f} seconds recording duration")
    print(f"- ~{1/(time_data[0][1] - time_data[0][0]):.1f} Hz sampling rate")

if __name__ == "__main__":
    main() 