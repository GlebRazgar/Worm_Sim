print("Starting C. elegans LIF Neural Network Simulation...")

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from tqdm import tqdm

print("Libraries imported successfully")

# Create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)
print("Outputs directory created/verified")

class LIFNeuron:
    """Leaky Integrate-and-Fire neuron model"""
    def __init__(self, neuron_id, tau_m=20.0, v_rest=-70.0, v_thresh=-50.0, v_reset=-80.0, 
                 r_m=10.0, dt=0.1, refractory_period=2.0):
        self.neuron_id = neuron_id
        self.tau_m = tau_m  # membrane time constant (ms)
        self.v_rest = v_rest  # resting potential (mV)
        self.v_thresh = v_thresh  # spike threshold (mV)
        self.v_reset = v_reset  # reset potential (mV)
        self.r_m = r_m  # membrane resistance (MOhm)
        self.dt = dt  # time step (ms)
        self.refractory_period = refractory_period  # refractory period (ms)
        
        # State variables
        self.v_membrane = v_rest  # current membrane potential
        self.last_spike_time = -np.inf  # time of last spike
        self.spike_times = []  # record of all spike times
        self.membrane_history = []  # history of membrane potential
        self.input_current = 0.0  # external input current
        
    def update(self, current_time, external_current=0.0, synaptic_current=0.0):
        """Update neuron state for one time step"""
        total_current = external_current + synaptic_current
        
        # Check if in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            self.v_membrane = self.v_reset
            self.membrane_history.append(self.v_membrane)
            return False
        
        # Update membrane potential using LIF equation
        dv_dt = (-(self.v_membrane - self.v_rest) + self.r_m * total_current) / self.tau_m
        self.v_membrane += dv_dt * self.dt
        
        # Check for spike
        spike_occurred = False
        if self.v_membrane >= self.v_thresh:
            self.v_membrane = self.v_reset
            self.last_spike_time = current_time
            self.spike_times.append(current_time)
            spike_occurred = True
        
        self.membrane_history.append(self.v_membrane)
        return spike_occurred

class ConnectomeLIFNetwork:
    """C. elegans connectome-based LIF neural network"""
    def __init__(self, positions_path, connections_path, dt=0.1, synaptic_weight_scale=0.1):
        self.dt = dt
        self.synaptic_weight_scale = synaptic_weight_scale
        self.current_time = 0.0
        
        # Load data
        print("Loading neuron positions...")
        self.load_neuron_positions(positions_path)
        
        print("Loading connectivity data...")
        self.load_connectivity_data(connections_path)
        
        print("Initializing neurons...")
        self.initialize_neurons()
        
        print("Setting up connectivity matrix...")
        self.setup_connectivity_matrix()
        
    def load_neuron_positions(self, positions_path):
        """Load neuron position data"""
        self.neurons_df = pd.read_csv(positions_path, header=None, 
                                    names=['neuron_id', 'x', 'y', 'z'])
        self.neuron_ids = list(self.neurons_df['neuron_id'])
        print(f"Loaded {len(self.neuron_ids)} neurons")
        
    def load_connectivity_data(self, connections_path):
        """Load connectivity data from Excel file"""
        try:
            self.connections_df = pd.read_excel(connections_path, 
                                              sheet_name='hermaphrodite chemical', 
                                              header=2, index_col=2)
            print("Successfully loaded connectivity data")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            # Alternative approach if needed
            self.connections_df = pd.read_excel(connections_path, sheet_name=0, header=2, index_col=2)
            print("Loaded connectivity data using alternative approach")
        
        # Clean up the connections dataframe
        if self.connections_df.index.name is None or self.connections_df.index.name != 'neuron_id':
            self.connections_df = self.connections_df.iloc[1:, 1:]
            
    def initialize_neurons(self):
        """Initialize LIF neurons with random input currents"""
        self.neurons = {}
        np.random.seed(42)  # for reproducible results
        
        for neuron_id in self.neuron_ids:
            # Create LIF neuron with slightly randomized parameters
            tau_m = np.random.normal(20.0, 3.0)  # membrane time constant variation
            neuron = LIFNeuron(
                neuron_id=neuron_id,
                tau_m=max(tau_m, 5.0),  # ensure positive time constant
                dt=self.dt
            )
            
            # Set random external input current (represents background activity)
            neuron.input_current = np.random.uniform(0.5, 3.0)  # nA
            self.neurons[neuron_id] = neuron
            
        print(f"Initialized {len(self.neurons)} LIF neurons")
        
    def setup_connectivity_matrix(self):
        """Setup synaptic connectivity matrix"""
        self.connectivity_matrix = {}
        self.synaptic_delays = {}
        
        for source_id in self.neuron_ids:
            self.connectivity_matrix[source_id] = {}
            self.synaptic_delays[source_id] = {}
            
            for target_id in self.neuron_ids:
                # Check if connection exists in data
                weight = 0.0
                delay = 1.0  # default synaptic delay (ms)
                
                try:
                    if source_id in self.connections_df.index and target_id in self.connections_df.columns:
                        num_connections = self.connections_df.loc[source_id, target_id]
                        if pd.notna(num_connections) and num_connections >= 1:
                            # Scale synaptic weight based on number of connections
                            weight = num_connections * self.synaptic_weight_scale
                            # Add some variability to synaptic delay
                            delay = np.random.uniform(1.0, 3.0)
                except (KeyError, TypeError):
                    pass
                
                self.connectivity_matrix[source_id][target_id] = weight
                self.synaptic_delays[source_id][target_id] = delay
                
        print("Connectivity matrix setup complete")
        
    def calculate_synaptic_input(self, target_neuron_id, current_time):
        """Calculate total synaptic input to a target neuron"""
        total_synaptic_current = 0.0
        
        for source_id in self.neuron_ids:
            weight = self.connectivity_matrix[source_id][target_neuron_id]
            if weight > 0:
                # Check for recent spikes from source neuron
                source_neuron = self.neurons[source_id]
                delay = self.synaptic_delays[source_id][target_neuron_id]
                
                # Look for spikes that should affect target now (accounting for delay)
                for spike_time in source_neuron.spike_times:
                    if abs(current_time - spike_time - delay) < self.dt/2:
                        # Exponential decay for synaptic current
                        total_synaptic_current += weight * np.exp(-(current_time - spike_time - delay)/5.0)
        
        return total_synaptic_current
        
    def simulate(self, duration_ms=1000, progress_bar=True):
        """Run network simulation"""
        num_steps = int(duration_ms / self.dt)
        time_points = np.arange(0, duration_ms, self.dt)
        
        print(f"Running simulation for {duration_ms} ms ({num_steps} steps)...")
        
        iterator = tqdm(range(num_steps)) if progress_bar else range(num_steps)
        
        for step in iterator:
            self.current_time = time_points[step]
            
            # Calculate synaptic inputs for all neurons
            synaptic_inputs = {}
            for neuron_id in self.neuron_ids:
                synaptic_inputs[neuron_id] = self.calculate_synaptic_input(neuron_id, self.current_time)
            
            # Update all neurons
            for neuron_id in self.neuron_ids:
                neuron = self.neurons[neuron_id]
                external_current = neuron.input_current + np.random.normal(0, 0.1)  # add noise
                neuron.update(self.current_time, external_current, synaptic_inputs[neuron_id])
        
        print("Simulation complete!")
        
    def plot_raster(self, max_neurons=50, time_window=None):
        """Create raster plot of spike activity"""
        plt.figure(figsize=(12, 8))
        
        # Limit number of neurons for visibility
        neurons_to_plot = self.neuron_ids[:max_neurons]
        
        for i, neuron_id in enumerate(neurons_to_plot):
            spike_times = self.neurons[neuron_id].spike_times
            if time_window:
                spike_times = [t for t in spike_times if time_window[0] <= t <= time_window[1]]
            
            if spike_times:
                plt.plot(spike_times, [i] * len(spike_times), 'k|', markersize=8)
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron Index')
        plt.title(f'C. elegans Connectome: Raster Plot of Spiking Activity (first {max_neurons} neurons)')
        plt.grid(True, alpha=0.3)
        
        if time_window:
            plt.xlim(time_window)
        
        plt.tight_layout()
        plt.savefig('outputs/connectome_raster_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_membrane_potentials(self, neuron_subset=None, time_window=None):
        """Plot membrane potential traces for selected neurons"""
        if neuron_subset is None:
            neuron_subset = self.neuron_ids[:5]  # Plot first 5 neurons by default
            
        plt.figure(figsize=(12, 8))
        time_array = np.arange(0, len(self.neurons[neuron_subset[0]].membrane_history)) * self.dt
        
        if time_window:
            start_idx = int(time_window[0] / self.dt)
            end_idx = int(time_window[1] / self.dt)
            time_array = time_array[start_idx:end_idx]
        else:
            start_idx, end_idx = 0, len(time_array)
        
        for i, neuron_id in enumerate(neuron_subset):
            membrane_trace = self.neurons[neuron_id].membrane_history[start_idx:end_idx]
            plt.plot(time_array, membrane_trace, label=f'{neuron_id}', alpha=0.8)
        
        plt.axhline(y=-50, color='red', linestyle='--', alpha=0.5, label='Spike Threshold')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Potential (mV)')
        plt.title('Membrane Potential Traces')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/connectome_membrane_potentials.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_network_activity(self):
        """Analyze and display network statistics"""
        print("\n" + "="*50)
        print("NETWORK ACTIVITY ANALYSIS")
        print("="*50)
        
        # Calculate firing rates
        total_simulation_time = len(self.neurons[self.neuron_ids[0]].membrane_history) * self.dt / 1000  # convert to seconds
        firing_rates = []
        
        for neuron_id in self.neuron_ids:
            neuron = self.neurons[neuron_id]
            firing_rate = len(neuron.spike_times) / total_simulation_time  # Hz
            firing_rates.append(firing_rate)
        
        firing_rates = np.array(firing_rates)
        
        print(f"Total simulation time: {total_simulation_time:.2f} seconds")
        print(f"Number of neurons: {len(self.neuron_ids)}")
        print(f"Mean firing rate: {np.mean(firing_rates):.2f} ± {np.std(firing_rates):.2f} Hz")
        print(f"Median firing rate: {np.median(firing_rates):.2f} Hz")
        print(f"Active neurons (>0.1 Hz): {np.sum(firing_rates > 0.1)} ({100*np.sum(firing_rates > 0.1)/len(firing_rates):.1f}%)")
        
        # Plot firing rate distribution
        plt.figure(figsize=(10, 6))
        plt.hist(firing_rates, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Firing Rate (Hz)')
        plt.ylabel('Number of Neurons')
        plt.title('Distribution of Firing Rates Across C. elegans Connectome')
        plt.axvline(np.mean(firing_rates), color='red', linestyle='--', label=f'Mean: {np.mean(firing_rates):.2f} Hz')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/connectome_firing_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_activation_data(self, filename='connectome_activations', format='hdf5'):
        """Save all neural activation data to file
        
        Args:
            filename (str): Base filename (without extension)
            format (str): 'hdf5', 'npz', or 'csv'
        """
        print(f"Saving activation data in {format} format...")
        
        if format == 'hdf5':
            import h5py
            
            with h5py.File(f'outputs/{filename}.h5', 'w') as f:
                # Metadata
                f.attrs['dt'] = self.dt
                f.attrs['total_time'] = len(self.neurons[self.neuron_ids[0]].membrane_history) * self.dt
                f.attrs['num_neurons'] = len(self.neuron_ids)
                f.attrs['simulation_parameters'] = str({
                    'dt': self.dt,
                    'synaptic_weight_scale': self.synaptic_weight_scale
                })
                
                # Neuron IDs
                neuron_id_bytes = [n.encode('utf-8') for n in self.neuron_ids]
                f.create_dataset('neuron_ids', data=neuron_id_bytes)
                
                # Time array
                time_array = np.arange(0, len(self.neurons[self.neuron_ids[0]].membrane_history)) * self.dt
                f.create_dataset('time', data=time_array, compression='gzip')
                
                # Membrane potentials (neurons x time)
                membrane_data = np.array([self.neurons[nid].membrane_history for nid in self.neuron_ids])
                f.create_dataset('membrane_potentials', data=membrane_data, compression='gzip')
                
                # Spike times for each neuron
                spikes_group = f.create_group('spike_times')
                for neuron_id in self.neuron_ids:
                    spike_times = np.array(self.neurons[neuron_id].spike_times)
                    spikes_group.create_dataset(neuron_id, data=spike_times, compression='gzip')
                
                # Connectivity matrix
                conn_group = f.create_group('connectivity')
                for source_id in self.neuron_ids:
                    weights = [self.connectivity_matrix[source_id][target_id] for target_id in self.neuron_ids]
                    conn_group.create_dataset(f'{source_id}_weights', data=np.array(weights), compression='gzip')
                
            print(f"Saved to outputs/{filename}.h5")
            
        elif format == 'npz':
            # Create data dictionary
            data_dict = {
                'dt': self.dt,
                'neuron_ids': np.array(self.neuron_ids),
                'time': np.arange(0, len(self.neurons[self.neuron_ids[0]].membrane_history)) * self.dt,
                'membrane_potentials': np.array([self.neurons[nid].membrane_history for nid in self.neuron_ids])
            }
            
            # Add spike times
            for i, neuron_id in enumerate(self.neuron_ids):
                data_dict[f'spikes_{i:03d}_{neuron_id}'] = np.array(self.neurons[neuron_id].spike_times)
            
            np.savez_compressed(f'outputs/{filename}.npz', **data_dict)
            print(f"Saved to outputs/{filename}.npz")
            
        elif format == 'csv':
            # Save membrane potentials
            membrane_df = pd.DataFrame(
                data=np.array([self.neurons[nid].membrane_history for nid in self.neuron_ids]).T,
                columns=self.neuron_ids
            )
            membrane_df.insert(0, 'time_ms', np.arange(len(membrane_df)) * self.dt)
            membrane_df.to_csv(f'outputs/{filename}_membrane_potentials.csv', index=False)
            
            # Save spike times
            spike_data = []
            for neuron_id in self.neuron_ids:
                for spike_time in self.neurons[neuron_id].spike_times:
                    spike_data.append({'neuron_id': neuron_id, 'spike_time_ms': spike_time})
            
            spike_df = pd.DataFrame(spike_data)
            spike_df.to_csv(f'outputs/{filename}_spikes.csv', index=False)
            
            print(f"Saved to outputs/{filename}_membrane_potentials.csv and outputs/{filename}_spikes.csv")
        
        else:
            raise ValueError("Format must be 'hdf5', 'npz', or 'csv'")
    
    def load_activation_data(self, filename, format='hdf5'):
        """Load previously saved activation data
        
        Args:
            filename (str): Filename to load (with extension)
            format (str): 'hdf5', 'npz', or 'csv'
        
        Returns:
            dict: Dictionary containing loaded data
        """
        print(f"Loading activation data from {filename}...")
        
        if format == 'hdf5':
            import h5py
            
            data = {}
            with h5py.File(filename, 'r') as f:
                data['dt'] = f.attrs['dt']
                data['total_time'] = f.attrs['total_time'] 
                data['num_neurons'] = f.attrs['num_neurons']
                data['neuron_ids'] = [nid.decode('utf-8') for nid in f['neuron_ids'][:]]
                data['time'] = f['time'][:]
                data['membrane_potentials'] = f['membrane_potentials'][:]
                
                # Load spike times
                data['spike_times'] = {}
                for neuron_id in data['neuron_ids']:
                    data['spike_times'][neuron_id] = f['spike_times'][neuron_id][:]
                    
            return data
            
        elif format == 'npz':
            loaded = np.load(filename, allow_pickle=True)
            return dict(loaded)
            
        else:
            raise ValueError("Loading only supported for 'hdf5' and 'npz' formats")

def main():
    """Main execution function"""
    # File paths
    positions_path = 'data/NeuroPal/LowResAtlasWithHighResHeadsAndTails.csv'
    connections_path = 'data/Cook/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'
    
    # Check if data files exist
    if not os.path.exists(positions_path):
        print(f"Error: Could not find positions file at {positions_path}")
        return
    if not os.path.exists(connections_path):
        print(f"Error: Could not find connections file at {connections_path}")
        return
    
    # Create and run simulation
    print("Initializing C. elegans LIF Network...")
    network = ConnectomeLIFNetwork(positions_path, connections_path, dt=0.5, synaptic_weight_scale=0.2)
    
    # Run simulation
    network.simulate(duration_ms=500, progress_bar=True)
    
    # Generate visualizations and analysis
    print("\nGenerating visualizations...")
    network.plot_raster(max_neurons=50, time_window=(0, 500))
    network.plot_membrane_potentials(time_window=(0, 200))
    network.analyze_network_activity()
    
    # Save activation data
    print("\nSaving activation data...")
    network.save_activation_data('connectome_activations', format='csv')
    network.save_activation_data('connectome_activations', format='npz')  # Keep NPZ as backup
    
    print("\nSimulation complete! Generated files:")
    print("- outputs/connectome_raster_plot.png")
    print("- outputs/connectome_membrane_potentials.png") 
    print("- outputs/connectome_firing_rate_distribution.png")
    print("- outputs/connectome_activations_membrane_potentials.csv")
    print("- outputs/connectome_activations_spikes.csv")
    print("- outputs/connectome_activations.npz")
    
    return network

if __name__ == "__main__":
    # Run the simulation
    network = main()