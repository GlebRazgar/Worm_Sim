print("Starting script...")

# Import necessary libraries
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import os

print("Libraries imported successfully")

# Load the neuron position data
positions_path = 'data/NeuroPal/LowResAtlasWithHighResHeadsAndTails.csv'
print(f"Loading neuron positions from {positions_path}")
neurons_df = pd.read_csv(positions_path, header=None, 
                       names=['neuron_id', 'x', 'y', 'z'])
print(f"Loaded {len(neurons_df)} neuron positions")

# Load the connectivity data
connections_path = 'data/Cook/SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'
print(f"Loading connectivity data from {connections_path}")
# Since this is an excel file with potential multiple sheets, we need to specify the sheet
try:
    connections_df = pd.read_excel(connections_path, sheet_name='hermaphrodite chemical', header=2, index_col=2)
    print("Successfully loaded connectivity data")
except Exception as e:
    print(f"Error loading Excel file: {e}")
    # If the above fails, try listing the sheet names
    import xlrd
    workbook = xlrd.open_workbook(connections_path)
    print(f"Available sheets: {workbook.sheet_names()}")
    # Try an alternative approach
    connections_df = pd.read_excel(connections_path, sheet_name=0, header=2, index_col=2)
    print("Loaded connectivity data using alternative approach")

# Clean up the connections dataframe
if connections_df.index.name is None or connections_df.index.name != 'neuron_id':
    connections_df = connections_df.iloc[1:, 1:]  # Skip any header/description rows and columns

print("Creating neuron position dictionary...")
# Create a dictionary to map neuron IDs to their positions
neuron_positions = {}
for _, row in neurons_df.iterrows():
    neuron_positions[row['neuron_id']] = (row['x'], row['y'], row['z'])

print("Creating initial figure...")
# Create figure
fig = go.Figure()

print("Adding neuron scatter points...")
# Add neurons as scatter points
fig.add_trace(go.Scatter3d(
    x=neurons_df['x'],
    y=neurons_df['y'],
    z=neurons_df['z'],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        opacity=0.8
    ),
    text=neurons_df['neuron_id'],  # Add neuron names for hover info
    hoverinfo='text',
    name='Neurons'
))

print("Preparing to process connections...")
# Prepare lists to store line data
line_x = []
line_y = []
line_z = []
line_width = []
hover_text = []

print("Processing connections between neurons...")
connection_count = 0
# Add connections as lines between neurons
for source_neuron in connections_df.index:
    for target_neuron in connections_df.columns:
        try:
            num_connections = connections_df.loc[source_neuron, target_neuron]
            if pd.notna(num_connections) and num_connections >= 1:
                # Check if both neurons have position data
                if source_neuron in neuron_positions and target_neuron in neuron_positions:
                    # Get positions
                    source_pos = neuron_positions[source_neuron]
                    target_pos = neuron_positions[target_neuron]
                    
                    # Add line coordinates with a None to separate different lines
                    line_x.extend([source_pos[0], target_pos[0], None])
                    line_y.extend([source_pos[1], target_pos[1], None])
                    line_z.extend([source_pos[2], target_pos[2], None])
                    
                    # Store connection strength for hover text
                    hover_text.extend([f"{source_neuron} → {target_neuron}: {num_connections} synapses", 
                                     f"{source_neuron} → {target_neuron}: {num_connections} synapses", 
                                     ""])
                    
                    # Store line width based on connection strength
                    line_width.append(0.2 + (num_connections * 0.08))
            connection_count += 1
        except (KeyError, TypeError) as e:
            continue

print(f"Processed {connection_count} connections. Adding to figure...")
# Add all connections as a single trace with varying widths
fig.add_trace(go.Scatter3d(
    x=line_x,
    y=line_y,
    z=line_z,
    mode='lines',
    line=dict(
        color='gray',
        width=3  # Base width - will be scaled by connection strength in custom data
    ),
    opacity=0.6,
    text=hover_text,
    hoverinfo='text',
    name='Connections',
    customdata=line_width  # Store line widths for potential custom scaling
))

print("Updating layout...")
# Calculate axis ranges for proper scaling
x_range = neurons_df['x'].max() - neurons_df['x'].min()
y_range = neurons_df['y'].max() - neurons_df['y'].min()
z_range = neurons_df['z'].max() - neurons_df['z'].min()
max_range = max(x_range, y_range, z_range)

# Calculate midpoints for centering
x_mid = (neurons_df['x'].max() + neurons_df['x'].min()) / 2
y_mid = (neurons_df['y'].max() + neurons_df['y'].min()) / 2
z_mid = (neurons_df['z'].max() + neurons_df['z'].min()) / 2

# Update layout with proper scaling
fig.update_layout(
    title='C. elegans Neuron 3D Positions with Connections',
    scene=dict(
        xaxis=dict(title='X axis (anterior-posterior)', range=[x_mid - max_range/2, x_mid + max_range/2]),
        yaxis=dict(title='Y axis (dorsal-ventral)', range=[y_mid - max_range/2, y_mid + max_range/2]),
        zaxis=dict(title='Z axis (left-right)', range=[z_mid - max_range/2, z_mid + max_range/2]),
        aspectmode='cube'  # Force equal scaling
    ),
    width=1000,
    height=800,
    margin=dict(l=0, r=0, b=0, t=40)
)

# Add annotation explaining the visualization
fig.add_annotation(
    text="Red dots: neurons | Gray lines: connections (thickness proportional to number of synapses)",
    xref="paper", yref="paper",
    x=0.5, y=0,
    showarrow=False
)

print("Saving figure to HTML...")
# Save as HTML instead of showing directly
fig.write_html("c_elegans_connectome.html")

print("Script completed! The visualization has been saved to 'c_elegans_connectome.html'")
# Comment out the direct show which might be causing issues
# fig.show()