import numpy as np


def affinities_to_segmentation(aff, threshold=0.5):
    """
    Convert a 3-channel affinity map into a labeled volume by threshold + union-find.

    Parameters
    ----------
    aff : numpy.ndarray
        A float array of shape (3, D, H, W), the predicted affinities along x, y, z axes.
    threshold : float
        If aff[channel, d, h, w] exceeds this threshold, we union the voxel with its neighbor.

    Returns
    -------
    seg : numpy.ndarray
        An int array of shape (D, H, W), where each distinct integer corresponds to one segment.
    """

    # aff has shape (3, D, H, W)
    c, D, H, W = aff.shape
    assert c == 3, "Expected 3 channels for x-, y-, z-affinities."

    # Flatten each (d, h, w) voxel index into a single integer for union-find
    def index(d, h, w):
        return d * (H * W) + h * W + w

    # Union-Find (Disjoint Set) data structures
    parent = np.arange(D * H * W)
    rank = np.zeros(D * H * W, dtype=np.int32)

    def find_set(x):
        if parent[x] != x:
            parent[x] = find_set(parent[x])
        return parent[x]

    def union_set(a, b):
        rootA = find_set(a)
        rootB = find_set(b)
        if rootA != rootB:
            if rank[rootA] < rank[rootB]:
                parent[rootA] = rootB
            elif rank[rootA] > rank[rootB]:
                parent[rootB] = rootA
            else:
                parent[rootB] = rootA
                rank[rootA] += 1

    # Go through each voxel and union it with neighbors if affinity > threshold
    for d in range(D):
        for h in range(H):
            for w in range(W):
                # Current voxel's flattened index
                curr_idx = index(d, h, w)

                # 1) Along X-axis: check neighbor at (d, h, w+1)
                if w < W - 1 and aff[0, d, h, w] > threshold:
                    union_set(curr_idx, index(d, h, w + 1))

                # 2) Along Y-axis: check neighbor at (d, h+1, w)
                if h < H - 1 and aff[1, d, h, w] > threshold:
                    union_set(curr_idx, index(d, h + 1, w))

                # 3) Along Z-axis: check neighbor at (d+1, h, w)
                if d < D - 1 and aff[2, d, h, w] > threshold:
                    union_set(curr_idx, index(d + 1, h, w))

    # Second pass: assign unique segment IDs by root representative
    seg = np.zeros((D, H, W), dtype=np.int32)
    label_map = {}
    next_label = 1

    for d in range(D):
        for h in range(H):
            for w in range(W):
                root = find_set(index(d, h, w))
                # Assign each root a unique integer ID
                if root not in label_map:
                    label_map[root] = next_label
                    next_label += 1
                seg[d, h, w] = label_map[root]

    return seg


# -----------------------
# Example usage:
if __name__ == "__main__":
    # Suppose 'pred_aff' is your (3, D, H, W) numpy array of predicted affinities in [0, 1].
    D, H, W = 32, 96, 96
    pred_aff = np.random.rand(3, D, H, W)

    # Convert to integer-labeled segmentation volume
    seg_vol = affinities_to_segmentation(pred_aff, threshold=0.5)
    print("Segmentation volume shape:", seg_vol.shape)
    print("Unique segment IDs:", np.unique(seg_vol))
