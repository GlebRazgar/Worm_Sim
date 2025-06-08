import neuroglancer
import numpy as np
import zarr
import os

# Configure neuroglancer server
neuroglancer.set_server_bind_address('0.0.0.0')
viewer = neuroglancer.Viewer()

# Use the correct path relative to the parent directory
zarr_path = os.path.join('..', 'data', 'sem.zarr')
print(f"Looking for Zarr store at path: {zarr_path}")

# Check if the path exists
if not os.path.exists(zarr_path):
    raise FileNotFoundError(f"Zarr store not found at {zarr_path}. Current directory: {os.getcwd()}")

# Open your Zarr data
zarr_store = zarr.open(zarr_path, mode='r')

# Get the raw EM data and segmentation data
raw_data = zarr_store['raw']
seg_data = zarr_store['gt_labels']

# Add the raw EM data to the viewer
with viewer.txn() as s:
    s.dimensions = neuroglancer.CoordinateSpace(
        names=['z', 'y', 'x'],
        units='nm',
        scales=raw_data.attrs['voxel_size']
    )
    
    # Add raw data
    s.layers['raw'] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(
            data=raw_data,
            dimensions=s.dimensions,
        )
    )
    
    # Add segmentation data
    s.layers['segmentation'] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(
            data=seg_data,
            dimensions=s.dimensions,
        )
    )

# Print the URL to access the viewer
print(f"Neuroglancer URL: {viewer.get_viewer_url()}")
print("Open the URL above in your web browser to view the data.")
input("Press Enter to exit...")