import os
import numpy as np
import zarr
from cloudvolume import CloudVolume, Bbox
import zarr.storage

print("Importing CloudVolume and dependencies")

def download_and_save_as_zarr(em_volume, seg_volume, save_path, mip_level=0, squeeze_c_dim=True):
    """
    Downloads two cloud volumes (EM and segmentation) at their bounding boxes,
    finds the intersection if the bounding boxes are mismatched, and saves
    the resulting subvolume data in a Zarr directory store (with .zarray, .zgroup, etc.).
    
    Using mip_level=0 for highest resolution.
    """
    print(f"Starting download with mip_level={mip_level}")
    
    # 1) Get full bounding boxes for EM and segmentation
    em_bbox = Bbox.from_delta(em_volume.bounds.minpt, em_volume.bounds.size3())
    seg_bbox = Bbox.from_delta(seg_volume.bounds.minpt, seg_volume.bounds.size3())

    print(f"EM bounding box: {em_bbox}")
    print(f"Segmentation bounding box: {seg_bbox}")

    # 2) Calculate the intersection of the bounding boxes
    intersection_bbox = em_bbox.intersection(em_bbox, seg_bbox)
    print(f"Intersection bounding box: {intersection_bbox}")
    
    # Since the high-res version is very large, we'll extract a smaller region
    # Get 10% of the volume centered on the middle
    min_pt = intersection_bbox.minpt
    max_pt = intersection_bbox.maxpt
    size = intersection_bbox.size3()
    
    # Calculate center
    center = [(min_pt[i] + max_pt[i]) // 2 for i in range(3)]
    
    # Calculate a region that's 10% of the original size around the center
    size_fraction = 0.1
    small_size = [int(s * size_fraction) for s in size]
    
    # Create a new bbox that's centered on the same point but smaller
    small_min = [center[i] - small_size[i] // 2 for i in range(3)]
    small_max = [center[i] + small_size[i] // 2 for i in range(3)]
    
    # Create a new bounding box
    small_bbox = Bbox(small_min, small_max)
    print(f"Reduced bounding box (10% of original size): {small_bbox}")
    
    # Use the smaller bounding box for download
    download_bbox = small_bbox

    # 3) Download data from the intersection region
    print("Downloading EM data...")
    em_data = em_volume.download(download_bbox)
    print("Downloading segmentation data...")
    seg_data = seg_volume.download(download_bbox)

    # 4) Convert to numpy arrays if necessary and print dtype and check if not all zero
    if not isinstance(em_data, np.ndarray):
        em_data = np.array(em_data)
    print(f"EM data shape: {em_data.shape}, dtype: {em_data.dtype}")
    if np.any(em_data != 0):
        print("EM data is not all zero.")
    
    if not isinstance(seg_data, np.ndarray):
        seg_data = np.array(seg_data)
    print(f"Segmentation data shape: {seg_data.shape}, dtype: {seg_data.dtype}")
    if np.any(seg_data != 0):
        print("Segmentation data is not all zero. ✅")

    # 5) Create a standard Zarr directory store
    print(f"Creating Zarr store at {save_path}")
    store = zarr.DirectoryStore(save_path)
    root = zarr.group(store=store, overwrite=True)
    
    # 6) Create datasets with chunked layout
    # Remove channel dimension if present
    if squeeze_c_dim and em_data.ndim == 4:
        assert em_data.shape[-1] == 1, "EM data channel axis is not 1"
        em_data = em_data.squeeze(axis=-1)
    
    if squeeze_c_dim and seg_data.ndim == 4:
        assert seg_data.shape[-1] == 1, "Segmentation data channel axis is not 1"
        seg_data = seg_data.squeeze(axis=-1)

    # Choose reasonable chunk sizes
    chunks = (64, 64, 64)
    print(f"Creating raw array with chunks {chunks}")
    raw_array = root.create_dataset('raw', data=em_data, chunks=chunks, dtype=em_data.dtype)
    print(f"Creating gt_labels array with chunks {chunks}")
    seg_array = root.create_dataset('gt_labels', data=seg_data, chunks=chunks, dtype=seg_data.dtype)
    
    # 7) Extract metadata for the specified mip level
    em_info = em_volume.info['scales'][mip_level]
    seg_info = seg_volume.info['scales'][mip_level]
    
    # 8) Create .zattrs for metadata
    print("Updating array attributes")
    raw_array.attrs.update({
        'description': 'Raw electron microscopy data',
        'encoding': em_info['encoding'],
        'offset': em_info['voxel_offset'],
        'axis_names': ['x', 'y', 'z'],
        'units': ['nm', 'nm', 'nm'],
        'voxel_size': em_info['resolution']
    })
    seg_array.attrs.update({
        'description': 'Ground truth labels for segmentation',
        'encoding': seg_info['encoding'],
        'offset': seg_info['voxel_offset'],
        'axis_names': ['x', 'y', 'z'],
        'units': ['nm', 'nm', 'nm'],
        'voxel_size': seg_info['resolution']
    })

    print(f"Zarr datasets saved in '{save_path}'")
    return root

def reorder_xyz_to_zyx(zarr_path):
    """Reorders arrays from XYZ to ZYX order"""
    print(f"Opening Zarr store at path: {zarr_path}")
    store = zarr.open(zarr_path, mode='r+')
    
    # Recursively handle groups
    def _reorder_group(group):
        print(f"Processing group: {group.name}")
        for k, v in group.items():
            if isinstance(v, zarr.Group):
                print(f"Found subgroup: {k}")
                _reorder_group(v)
            elif isinstance(v, zarr.Array):
                print(f"Found array: {k}")
                axes = v.attrs.get('axis_names', None)
                print(f"Current axes: {axes}")
                if axes == ['x', 'y', 'z']:
                    print(f"Reordering array: {k}")
                    data = v[...]
                    # Reorder
                    data = np.transpose(data, (2,1,0))
                    # Update data & axes
                    v.resize(data.shape)
                    v[...] = data
                    v.attrs['axis_names'] = ['z', 'y', 'x']
                    v.attrs['offset'] = [v.attrs['offset'][2], v.attrs['offset'][1], v.attrs['offset'][0]]
                    v.attrs['voxel_size'] = [v.attrs['voxel_size'][2], v.attrs['voxel_size'][1], v.attrs['voxel_size'][0]]
                    v.attrs['units'] = [v.attrs['units'][2], v.attrs['units'][1], v.attrs['units'][0]]
                    print(f"Reordered axes: {v.attrs['axis_names']}")
    
    _reorder_group(store)
    print(f"Completed reordering for Zarr store at path: {zarr_path}")
    return store

if __name__ == "__main__":
    # Create destination directory
    base_download_path = "data"
    os.makedirs(base_download_path, exist_ok=True)
    
    # Set target paths
    zarr_path = os.path.join(base_download_path, "sem_highres.zarr")
    
    try:
        print("Initializing CloudVolume connections")
        # Initialize with mip=0 for highest resolution (2nm x 2nm x 30nm)
        sem = CloudVolume('s3://bossdb-open-data/witvliet2020/Dataset_5/em/', 
                         mip=0, 
                         secrets=None, 
                         use_https=True,
                         fill_missing=True)
        
        sem_labels = CloudVolume('s3://bossdb-open-data/witvliet2020/Dataset_5/segmentation/', 
                                mip=0, 
                                secrets=None, 
                                use_https=True,
                                fill_missing=True)
        
        print("Refreshing CloudVolume info")
        sem.refresh_info()
        sem_labels.refresh_info()
        
        # Print resolution information
        print("\n---- SEM (nm) ----")
        resolution = sem.scales[0]['resolution']
        print(f"Resolution at mip=0: {resolution}")
        
        print("\n---- SEM Labels (nm) ----")
        resolution = sem_labels.scales[0]['resolution']
        print(f"Resolution at mip=0: {resolution}")
        
        print(f"\nDownloading and saving to {zarr_path}")
        download_and_save_as_zarr(
            sem, 
            sem_labels, 
            save_path=zarr_path,
            mip_level=0,
            squeeze_c_dim=True
        )
        
        print(f"\nReordering axes in {zarr_path}")
        reorder_xyz_to_zyx(zarr_path)
        
        print("\nSuccess downloading and reordering high-resolution dataset!")
        
    except Exception as e:
        print(f"Error: {e}") 