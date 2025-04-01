import zarr
import numpy as np
from skimage import io
import os
from scipy import ndimage

# Path to the high-resolution zarr data
zarr_path = "data/sem_highres.zarr"

# Open the zarr data
print(f"Opening Zarr store at path: {zarr_path}")
zarr_store = zarr.open(zarr_path, mode='r')

# Print available arrays
print(f"Available arrays: {list(zarr_store.keys())}")

# Access both the raw EM data and segmentation labels
raw_data = zarr_store['raw']
gt_labels = zarr_store['gt_labels']

print(f"Raw data shape: {raw_data.shape}, dtype: {raw_data.dtype}")
print(f"Label data shape: {gt_labels.shape}, dtype: {gt_labels.dtype}")

# Check and print the voxel resolution
voxel_size = raw_data.attrs.get('voxel_size', None)
print(f"Original voxel size (nm): {voxel_size}")

# SegNeuron expects voxel sizes in range:
# x,y: ~6-9 nm
# z: ~8-50 nm
print(f"SegNeuron expects x,y resolution ~6-9 nm and z resolution ~8-50 nm")

# Extract a small subvolume
# Choose a center slice on z-axis for 2D, or a small range for 3D
z_center = raw_data.shape[0] // 2
z_range = 20  # Get more slices for higher resolution

# Extract a region of 512x512 pixels from the center for higher resolution
y_center = raw_data.shape[1] // 2
x_center = raw_data.shape[2] // 2
size = 512  # Larger region for higher resolution

# Define region to extract
y_start = max(0, y_center - size//2)
y_end = min(raw_data.shape[1], y_center + size//2)
x_start = max(0, x_center - size//2)
x_end = min(raw_data.shape[2], x_center + size//2)

# Extract the data
raw_subvol = raw_data[z_center:z_center+z_range, y_start:y_end, x_start:x_end]
labels_subvol = gt_labels[z_center:z_center+z_range, y_start:y_end, x_start:x_end]

print(f"Extracted raw subvolume shape: {raw_subvol.shape}")
print(f"Extracted labels subvolume shape: {labels_subvol.shape}")

# Determine if resampling is needed
need_resampling = False
target_voxel_size = None

if voxel_size:
    # For high-resolution EM data, we want to:
    # 1. Keep x,y resolution if it's already in the right range (6-9 nm)
    # 2. Downsample x,y if it's too high resolution (< 6 nm) to avoid excessive data size
    # 3. Upsample x,y if it's too low resolution (> 9 nm)
    # 4. Ensure z resolution is in the right range (8-50 nm)
    
    # Check if x,y resolution needs adjustment
    x_out_of_range = voxel_size[2] < 6 or voxel_size[2] > 9  # x is index 2 in [z,y,x]
    y_out_of_range = voxel_size[1] < 6 or voxel_size[1] > 9  # y is index 1 in [z,y,x]
    z_out_of_range = voxel_size[0] < 8 or voxel_size[0] > 50  # z is index 0 in [z,y,x]
    
    if x_out_of_range or y_out_of_range or z_out_of_range:
        need_resampling = True
        
        # Calculate target voxel size within SegNeuron range
        target_voxel_size = [
            voxel_size[0],  # z resolution
            voxel_size[1],  # y resolution
            voxel_size[2],  # x resolution
        ]
        
        # Adjust z dimension if needed
        if z_out_of_range:
            target_voxel_size[0] = 30.0 if voxel_size[0] < 8 else 50.0
        
        # For the high-res case (2nm), we want to downsample to 6nm (the lower bound of SegNeuron's range)
        # The 2nm resolution is too detailed for SegNeuron's expected input
        if voxel_size[1] < 6:  # y dimension too high resolution
            target_voxel_size[1] = 6.0
        elif voxel_size[1] > 9:  # y dimension too low resolution
            target_voxel_size[1] = 9.0
            
        if voxel_size[2] < 6:  # x dimension too high resolution
            target_voxel_size[2] = 6.0
        elif voxel_size[2] > 9:  # x dimension too low resolution
            target_voxel_size[2] = 9.0
        
        print(f"Current voxel size {voxel_size} is outside SegNeuron's expected range.")
        print(f"Resampling to target voxel size: {target_voxel_size} nm")
        
        # Calculate the scaling factors for resampling (original / target)
        # For downsampling, this will be < 1; for upsampling, this will be > 1
        zoom_factors = [voxel_size[i] / target_voxel_size[i] for i in range(3)]
        print(f"Resampling zoom factors: {zoom_factors}")
        
        # Resample the raw data (use order=1 for linear interpolation)
        raw_subvol = ndimage.zoom(raw_subvol, zoom_factors, order=1)
        
        # Resample the labels (use nearest neighbor interpolation to preserve label values)
        labels_subvol = ndimage.zoom(labels_subvol, zoom_factors, order=0)
        
        print(f"Resampled raw subvolume shape: {raw_subvol.shape}")
        print(f"Resampled labels subvolume shape: {labels_subvol.shape}")
    else:
        print(f"Current voxel size {voxel_size} is within SegNeuron's expected range. No resampling needed.")

# Get unique labels in the region (these should correspond to neurons)
unique_labels = np.unique(labels_subvol)
print(f"Number of unique labels (including background): {len(unique_labels)}")
print(f"First few labels: {unique_labels[:10]}")

# Create output directory if it doesn't exist
os.makedirs("output", exist_ok=True)

# Save the raw EM data as a TIFF
raw_tiff_path = "output/em_data_highres.tif"
print(f"Saving raw EM data to {raw_tiff_path}")
io.imsave(raw_tiff_path, raw_subvol, check_contrast=False)

# For visualization, create a colored segmentation
# We'll use a simpler approach - mod the label values to create an RGB image
# This isn't ideal but should help with visualization
colored_labels = np.zeros((*labels_subvol.shape, 3), dtype=np.uint8)
for z in range(labels_subvol.shape[0]):
    for label in unique_labels[1:]:  # Skip background (usually 0)
        mask = labels_subvol[z] == label
        if np.any(mask):
            # Generate a pseudorandom color based on the label
            r = (label * 37) % 256
            g = (label * 73) % 256
            b = (label * 151) % 256
            
            colored_labels[z, mask, 0] = r
            colored_labels[z, mask, 1] = g
            colored_labels[z, mask, 2] = b

# Save the colored segmentation as a TIFF
labels_tiff_path = "output/segmentation_highres.tif"
print(f"Saving colored segmentation to {labels_tiff_path}")
io.imsave(labels_tiff_path, colored_labels, check_contrast=False)

# If we resampled, also save metadata about the resampling
if need_resampling and target_voxel_size:
    metadata_path = "output/voxel_info_highres.txt"
    with open(metadata_path, "w") as f:
        f.write(f"Original voxel size (nm): {voxel_size}\n")
        f.write(f"Resampled voxel size (nm): {target_voxel_size}\n")
        f.write(f"Zoom factors used: {zoom_factors}\n")
    print(f"Saved resampling metadata to {metadata_path}")

print("Done!") 