import mujoco
import mujoco.viewer
import numpy as np
import time

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
    def __init__(self):
        # Load the model from the XML string
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        self.data = mujoco.MjData(self.model)
        
        # Create the viewer
        self.viewer = None
        
        # Simulation parameters
        self.dt = 0.01  # 10ms timestep
        self.total_time = 10.0  # 10 seconds simulation
        self.n_steps = int(self.total_time / self.dt)
        
        # Create stimulus array with multiple pulses
        self.stimulus = np.zeros(self.n_steps)
        
        # Create pulses every second
        for i in range(10):  # 10 pulses over 10 seconds
            start_idx = int((i + 0.5) * 100)  # Start at 0.5s, then every second
            pulse_duration = 30  # 300ms
            self.stimulus[start_idx:start_idx + pulse_duration] = 1.0
        
        print("Stimulus array created with pulses at:")
        for i in range(10):
            print(f"  Pulse {i+1}: {(i + 0.5):.1f}s to {(i + 0.5 + 0.3):.1f}s")
        
        self.step_counter = 0
        self.last_state = 0  # Track state changes for debug output

    def run(self):
        # Create the viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            print("\nStarting simulation...")
            print("Time     State    Angle")
            print("-----------------------")
            
            # Simulation loop
            while viewer.is_running() and self.step_counter < self.n_steps:
                # Get current stimulus
                current_stimulus = self.stimulus[self.step_counter]
                
                # Apply control signal (positive to flex up, negative to relax down)
                if current_stimulus > 0:
                    self.data.ctrl[0] = 1.0  # Flex up
                else:
                    self.data.ctrl[0] = -0.5  # Help push down
                
                # Step the simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update the viewer
                viewer.sync()
                
                # Debug print on state changes or every 0.5 seconds
                current_time = self.step_counter * self.dt
                if (current_stimulus != self.last_state) or (self.step_counter % 50 == 0):
                    angle = np.degrees(self.data.qpos[0])  # Convert radians to degrees
                    state = "FLEX" if current_stimulus > 0 else "RELAX"
                    print(f"{current_time:5.2f}s   {state:6}   {angle:6.1f}°")
                    self.last_state = current_stimulus
                
                # Increment counter
                self.step_counter += 1
                
                # Add a small delay to not max out CPU
                time.sleep(self.dt)

def main():
    sim = FingerSimulation()
    sim.run()

if __name__ == "__main__":
    main()


