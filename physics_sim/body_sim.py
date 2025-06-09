import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import time
import os

# Create a basic model with a finger-like mechanism
MODEL_XML = """
<mujoco>
    <option gravity="0 0 -9.81"/>
    
    <default>
        <joint damping="5.0" stiffness="2.0" springref="0"/>
    </default>
    
    <worldbody>
        <!-- Floor -->
        <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1"/>
        
        <!-- Base segment (fixed to world) -->
        <!-- Note: Elements directly in worldbody are fixed by default -->
        <geom name="base_bone" type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.8 0.3 0.3 1"/>
        <site name="muscle_base" pos="0.05 0 0.02" size="0.01" rgba="0.8 0.8 0.8 1"/>
            
        <!-- Second segment (can rotate) -->
        <body name="finger" pos="0.3 0 0">
            <!-- Hinge joint to allow rotation -->
            <joint name="hinge" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
            <!-- The bone segment -->
            <geom name="finger_bone" type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" rgba="0.3 0.3 0.8 1"/>
            <!-- Muscle attachment point -->
            <site name="muscle_end" pos="0.05 0 0.02" size="0.01" rgba="0.8 0.8 0.8 1"/>
        </body>
    </worldbody>

    <!-- Define the tendon -->
    <tendon>
        <spatial name="muscle_tendon" width="0.003" rgba="0.95 0.3 0.3 1">
            <site site="muscle_base"/>
            <site site="muscle_end"/>
        </spatial>
    </tendon>

    <!-- Define the muscle actuator -->
    <actuator>
        <motor name="motor" joint="hinge" gear="50"/>
    </actuator>
</mujoco>
"""

class FingerSimulation:
    def __init__(self, spike_csv_path='connectome_activations_spikes.csv'):
        # Load the model from the XML string
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        
        # Create the viewer
        self.viewer = None
        
        # Simulation parameters
        self.dt = 0.01  # 10ms timestep
        self.total_time = 10.0  # 10 seconds simulation
        self.n_steps = int(self.total_time / self.dt)
        
        # Load neural spike data and create stimulus
        self.stimulus = self.load_neural_spikes(spike_csv_path)
        
        self.step_counter = 0
        self.last_state = 0  # Track state changes for debug output
        
    def load_neural_spikes(self, csv_path):
        """Load neural spike data and convert to stimulus array"""
        print(f"Loading neural spike data from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"Warning: Neural spike file not found at {csv_path}")
            print("Using dummy stimulus pattern instead...")
            return self.create_dummy_stimulus()
        
        try:
            # Load spike data
            spikes_df = pd.read_csv(csv_path)
            print(f"Loaded {len(spikes_df)} total spikes from neural simulation")
            
            # Get the first neuron's ID
            first_neuron = spikes_df['neuron_id'].iloc[0] if len(spikes_df) > 0 else None
            if first_neuron is None:
                print("No spikes found in CSV file, using dummy stimulus")
                return self.create_dummy_stimulus()
            
            # Filter spikes for the first neuron only
            first_neuron_spikes = spikes_df[spikes_df['neuron_id'] == first_neuron]['spike_time_ms'].values
            print(f"Found {len(first_neuron_spikes)} spikes for neuron '{first_neuron}'")
            
            if len(first_neuron_spikes) == 0:
                print("No spikes found for first neuron, using dummy stimulus")
                return self.create_dummy_stimulus()
            
            # Create stimulus array
            stimulus = np.zeros(self.n_steps)
            
            # Neural simulation was 500ms, physics sim is 10s
            # We'll loop the neural pattern 20 times to fill 10 seconds
            neural_duration_ms = 500  # Neural sim was 500ms
            physics_duration_ms = self.total_time * 1000  # Convert to ms
            
            print(f"Neural simulation duration: {neural_duration_ms}ms")
            print(f"Physics simulation duration: {physics_duration_ms}ms")
            print(f"Looping neural pattern {int(physics_duration_ms/neural_duration_ms)} times")
            
            spike_count = 0
            for loop in range(int(physics_duration_ms / neural_duration_ms)):
                for spike_time_ms in first_neuron_spikes:
                    # Calculate time in physics simulation
                    physics_time_ms = loop * neural_duration_ms + spike_time_ms
                    physics_time_s = physics_time_ms / 1000.0
                    
                    if physics_time_s < self.total_time:
                        # Convert to step index
                        step_idx = int(physics_time_s / self.dt)
                        
                        # Create spike: brief high stimulus followed by decay
                        spike_duration_steps = int(0.05 / self.dt)  # 50ms duration
                        for i in range(spike_duration_steps):
                            if step_idx + i < len(stimulus):
                                # Exponential decay from spike
                                decay_factor = np.exp(-i * self.dt / 0.02)  # 20ms time constant
                                stimulus[step_idx + i] = max(stimulus[step_idx + i], 1.0 * decay_factor)
                        spike_count += 1
            
            print(f"Created stimulus pattern with {spike_count} total spikes")
            print(f"First few spike times: {first_neuron_spikes[:5]} ms")
            
            return stimulus
            
        except Exception as e:
            print(f"Error loading neural data: {e}")
            print("Falling back to dummy stimulus pattern")
            return self.create_dummy_stimulus()
    
    def create_dummy_stimulus(self):
        """Create the original dummy stimulus pattern as fallback"""
        stimulus = np.zeros(self.n_steps)
        
        # Create pulses every second
        for i in range(10):  # 10 pulses over 10 seconds
            start_idx = int((i + 0.5) * 100)  # Start at 0.5s, then every second
            pulse_duration = 30  # 300ms
            stimulus[start_idx:start_idx + pulse_duration] = 1.0
        
        print("Created dummy stimulus array with pulses at:")
        for i in range(10):
            print(f"  Pulse {i+1}: {(i + 0.5):.1f}s to {(i + 0.5 + 0.3):.1f}s")
        
        return stimulus

    def run(self):
        # Create the viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            print("\nStarting neural-controlled finger simulation...")
            print("Time     Neural   Motor   Angle")
            print("         Input    Torque")
            print("--------------------------------")
            
            # Simulation loop
            while viewer.is_running() and self.step_counter < self.n_steps:
                # Get current neural stimulus
                neural_input = self.stimulus[self.step_counter]
                
                # Apply control signal based on neural input
                # Strong neural input causes muscle contraction (flexion)
                if neural_input > 0.1:  # Threshold for activation
                    motor_torque = neural_input * 2.0  # Scale neural input to motor torque
                    self.data.ctrl[0] = motor_torque
                else:
                    self.data.ctrl[0] = -0.3  # Small return force when no neural input
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update the viewer
                viewer.sync()
                
                # Debug print periodically and on significant neural activity changes
                current_time = self.step_counter * self.dt
                neural_active = neural_input > 0.1
                
                if (neural_active != self.last_state) or (self.step_counter % 100 == 0):
                    angle = np.degrees(self.data.qpos[0])  # Convert radians to degrees
                    motor_torque = self.data.ctrl[0]
                    print(f"{current_time:5.2f}s   {neural_input:6.3f}   {motor_torque:6.2f}   {angle:6.1f}°")
                    self.last_state = neural_active
                
                # Increment counter
                self.step_counter += 1
                
                # Add a small delay to not max out CPU
                time.sleep(self.dt)

def main():
    # Look for CSV file in parent directory 
    csv_path = '../connectome_activations_spikes.csv'
    if not os.path.exists(csv_path):
        # Try current directory
        csv_path = 'connectome_activations_spikes.csv'
    
    sim = FingerSimulation(csv_path)
    sim.run()

if __name__ == "__main__":
    main()


