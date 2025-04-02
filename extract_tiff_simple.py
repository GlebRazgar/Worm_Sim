import zarr
import numpy as np
from skimage import io
import os

def get_zarr_info(zarr_path):
    """Get information about the zarr volume."""
    # Open the zarr data
    zarr_store = zarr.open(zarr_path, mode='r')
    
    # Print available arrays
    print(f"Available arrays: {list(zarr_store.keys())}")
    
    # Get info for each array
    datasets = {}
    for key in zarr_store.keys():
        datasets[key] = {
            "shape": zarr_store[key].shape,
            "dtype": zarr_store[key].dtype,
            "voxel_size": zarr_store[key].attrs.get('voxel_size', None)
        }
        print(f"{key} shape: {datasets[key]['shape']}, dtype: {datasets[key]['dtype']}")
        print(f"{key} voxel size: {datasets[key]['voxel_size']}")
    
    return zarr_store, datasets

def extract_window(zarr_store, dataset_name, z, y, x, d, h, w):
    """
    Extract a window from the zarr dataset.
    
    Args:
        zarr_store: The zarr store
        dataset_name: Name of the dataset to extract from (e.g., 'raw', 'gt_labels')
        z, y, x: Top-left corner coordinates
        d, h, w: Depth, height, width of the window
    
    Returns:
        Extracted window as numpy array
    """
    # Get the dataset
    dataset = zarr_store[dataset_name]
    
    # Ensure coordinates are within bounds
    z_max, y_max, x_max = dataset.shape
    print(f"Dataset shape: {dataset.shape}")
    
    # Clamp start coordinates to valid range
    z_start = max(0, min(z, z_max-1))
    y_start = max(0, min(y, y_max-1))
    x_start = max(0, min(x, x_max-1))
    
    # Clamp end coordinates to valid range
    z_end = max(z_start+1, min(z_start+d, z_max))
    y_end = max(y_start+1, min(y_start+h, y_max))
    x_end = max(x_start+1, min(x_start+w, x_max))
    
    # Extract the window
    window = dataset[z_start:z_end, y_start:y_end, x_start:x_end]
    print(f"Extracted window shape: {window.shape}")
    
    return window

def save_to_tiff(data, output_path):
    """Save numpy array to TIFF file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving data to {output_path}")
    io.imsave(output_path, data, check_contrast=False)

def colorize_labels(labels):
    """Create a colored version of the segmentation for visualization."""
    unique_labels = np.unique(labels)
    colored_labels = np.zeros((*labels.shape, 3), dtype=np.uint8)
    
    for z in range(labels.shape[0]):
        for label in unique_labels[1:]:  # Skip background (usually 0)
            mask = labels[z] == label
            if np.any(mask):
                # Generate a pseudorandom color based on the label
                r = (label * 37) % 256
                g = (label * 73) % 256
                b = (label * 151) % 256
                
                colored_labels[z, mask, 0] = r
                colored_labels[z, mask, 1] = g
                colored_labels[z, mask, 2] = b
    
    return colored_labels

# Example usage
if __name__ == "__main__":
    zarr_path = "data/sem.zarr"
    
    # Get information about the zarr volume
    zarr_store, datasets = get_zarr_info(zarr_path)
    
    # Example: Extract a window from the raw data
    # Extract a 50×200×200 window starting at position (10, 100, 100)
    
    # Region of brain:
    dims = [126, 100, 200, 10000, 364, 464]
    
    # Pick the slices where the green/grey neuron is
    dims[0] += 50
    # dims[3] = 290 - 50
    
    # Pick the bottom right corner of the window
    dims[1] += dims[4] / 2
    dims[2] += dims[5] / 2
    dims[4] /= 2
    dims[5] /= 2
    
    # Convert all dimensions to integers
    dims = tuple(int(d) for d in dims)
    dim_str = "_".join(map(str, dims))
    raw_window = extract_window(zarr_store, 'raw', *dims)
    
    # Example: Extract the same window from the labels
    label_window = extract_window(zarr_store, 'gt_labels', *dims)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Save raw data as TIFF
    save_to_tiff(raw_window, f"output/{dim_str}_raw_window.tif")
    
    # Create colored version of labels for better visualization
    colored_labels = colorize_labels(label_window)
    
    # Save colored labels
    save_to_tiff(colored_labels, f"output/{dim_str}_colored_labels_window.tif")