import numpy as np
import skimage.segmentation
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.opt.multicut as nmc


def segment_neuron_affinities(
    affinity_map: np.ndarray, foreground_mask: np.ndarray = None, threshold: float = 0.3
) -> np.ndarray:
    """
    Segment neuron instance IDs from a 3D affinity map using a typical
    boundary-based pipeline (watershed + multicut).

    Parameters
    ----------
    affinity_map : np.ndarray
        A float32 array of shape (3, D, H, W), each channel is affinity
        (likelihood that adjacent voxels along x, y, or z axis belong
        to the same object).
    foreground_mask : np.ndarray, optional
        A binary array of shape (D, H, W). If given, any voxel with
        foreground_mask==0 is forced to have zero affinity.
    threshold : float, optional
        Threshold on the boundary probability used in watershed
        initialization. Typically 0.5 or thereabouts.

    Returns
    -------
    seg : np.ndarray
        A 3D integer-labeled segmentation of shape (D, H, W), where each
        connected neuron segment has a unique integer ID.
    """
    print(
        f"Affinity map shape: {affinity_map.shape}, min: {np.min(affinity_map)}, max: {np.max(affinity_map)}"
    )

    # 1. Foreground restriction
    #    (replace predicted affinity with zero if outside the neuron).
    #    This step helps clamp spurious high-affinity values in the background.
    if foreground_mask is not None:
        print(f"Using foreground mask with shape {foreground_mask.shape}")
        # broadcast (3, D, H, W) & (D, H, W)
        affinity_map = np.minimum(affinity_map, foreground_mask[None])
        print(f"After mask: min: {np.min(affinity_map)}, max: {np.max(affinity_map)}")

    # 2. Convert affinity -> boundary probability
    #    The boundary probability is typically something like 1 - affinity.
    #    We combine the x/y/z boundary signals, for example by taking
    #    the maximum across channels:
    boundary_prob = 1.0 - np.max(affinity_map, axis=0)  # shape (D, H, W)
    print(f"Boundary prob: min: {np.min(boundary_prob)}, max: {np.max(boundary_prob)}")

    # 3. Create an over-segmentation via watershed
    #    We can interpret boundary_prob as a height-map for watershed.
    #    A small sigma is sometimes used to smooth boundary before watershed.
    #    We'll do a simple threshold to define "seeds".
    seed_map = (boundary_prob < threshold).astype(np.uint8)
    print(f"Seed map has {np.sum(seed_map)} seed pixels")

    # If no seeds are found, reduce threshold
    if np.sum(seed_map) < 100:
        print(f"Few seeds found, reducing threshold to 0.7")
        seed_map = (boundary_prob < 0.7).astype(np.uint8)
        print(f"Seed map now has {np.sum(seed_map)} seed pixels")

    # Label the seeds
    seeds, num_seeds = skimage.measure.label(seed_map, return_num=True)
    print(f"Found {num_seeds} distinct seed regions")

    # Watershed from those seeds
    overseg = skimage.segmentation.watershed(boundary_prob, markers=seeds, mask=None)
    print(
        f"Overseg: min: {np.min(overseg)}, max: {np.max(overseg)}, unique labels: {len(np.unique(overseg))}"
    )

    # Check if oversegmentation worked
    if len(np.unique(overseg)) <= 1:
        print("WARNING: Oversegmentation failed, trying again with different threshold")
        # Try with more aggressive threshold
        seed_map = (boundary_prob < 0.8).astype(np.uint8)
        seeds, num_seeds = skimage.measure.label(seed_map, return_num=True)
        print(f"New attempt found {num_seeds} distinct seed regions")
        overseg = skimage.segmentation.watershed(
            boundary_prob, markers=seeds, mask=None
        )
        print(f"New overseg: unique labels: {len(np.unique(overseg))}")

    # 4. Build Region Adjacency Graph (RAG)
    rag = nifty.graph.rag.gridRag(overseg.astype("uint64"))
    print(f"RAG has {rag.numberOfNodes} nodes and {rag.numberOfEdges} edges")

    # 5. Compute edge "affinity" or "boundary" values for each pair of regions in the RAG.
    #    Typically, we take the average or minimum boundary prob along the boundary
    #    between two segments as an edge cost. Here, we'll do a simple mean boundary:
    #    The function "nifty.graph.rag.accumulateEdgeMeanAndLength" accumulates
    #    the boundary prob for each edge and returns edge features (mean, pixel count).
    features = nifty.graph.rag.accumulateEdgeMeanAndLength(rag, boundary_prob)

    # Our edge cost is the mean boundary probability on that edge.
    edge_cost = features[:, 0]  # first column is the mean boundary
    print(
        f"Edge costs: min: {np.min(edge_cost)}, max: {np.max(edge_cost)}, mean: {np.mean(edge_cost)}"
    )

    # For multicut, we want to cut edges with high boundary probability (close to 1)
    # and keep edges with low boundary probability (close to 0)
    # We need to convert edge_cost to edge_weights where:
    # - positive weights encourage keeping edges connected (same segment)
    # - negative weights encourage cutting edges (different segments)
    edge_weights = 0.5 - edge_cost  # Transform [0,1] to [-0.5,0.5]
    print(
        f"Edge weights: min: {np.min(edge_weights)}, max: {np.max(edge_weights)}, mean: {np.mean(edge_weights)}"
    )

    # 6. Solve multicut with Kernighan–Lin to get the final partition of the RAG.
    objective = nmc.multicutObjective(
        rag, edge_weights
    )  # Using edge_weights instead of edge_cost
    solver_factory = nmc.KernighanLinFactoryMulticutObjectiveUndirectedGraph()
    solver = solver_factory.create(objective)
    # run solver
    solved_node_labels = solver.optimize()
    print(f"Solved node labels: {len(np.unique(solved_node_labels))} unique labels")

    # 7. Project the partitioning on the RAG back to a full 3D segmentation.
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, solved_node_labels)

    print("seg.min(): ", np.min(seg))
    print("seg.max(): ", np.max(seg))
    print("seg unique values: ", np.unique(seg))

    return seg.astype(np.int32)


def segment_to_image(segment):
    max_label = np.max(segment) + 1  # Add 1 to accommodate 0
    print(f"Creating colors for {max_label} segments")
    colors = np.random.randint(0, 255, (max_label, 3))
    # Make sure 0 is black
    if max_label > 0:
        colors[0] = [0, 0, 0]

    image = np.zeros((segment.shape[0], segment.shape[1], segment.shape[2], 3))
    for i in range(max_label):
        mask = segment == i
        if np.any(mask):
            image[mask] = colors[i]

    return image
