import numpy as np
import skimage.segmentation
import nifty
import nifty.graph
import nifty.graph.rag
import nifty.graph.opt.multicut as nmc


def segment_neuron_affinities(
    affinity_map: np.ndarray, foreground_mask: np.ndarray = None, threshold: float = 0.5
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

    # 1. Foreground restriction
    #    (replace predicted affinity with zero if outside the neuron).
    #    This step helps clamp spurious high-affinity values in the background.
    if foreground_mask is not None:
        # broadcast (3, D, H, W) & (D, H, W)
        affinity_map = np.minimum(affinity_map, foreground_mask[None])

    # 2. Convert affinity -> boundary probability
    #    The boundary probability is typically something like 1 - affinity.
    #    We combine the x/y/z boundary signals, for example by taking
    #    the maximum across channels:
    boundary_prob = 1.0 - np.max(affinity_map, axis=0)  # shape (D, H, W)

    # 3. Create an over-segmentation via watershed
    #    We can interpret boundary_prob as a height-map for watershed.
    #    A small sigma is sometimes used to smooth boundary before watershed.
    #    We'll do a simple threshold to define "seeds".
    seed_map = (boundary_prob < threshold).astype(np.uint8)
    # Label the seeds
    seeds, _ = skimage.measure.label(seed_map, return_num=True)
    # Watershed from those seeds
    overseg = skimage.segmentation.watershed(boundary_prob, markers=seeds, mask=None)

    # 4. Build Region Adjacency Graph (RAG)
    rag = nifty.graph.rag.gridRag(overseg.astype("uint64"))

    # 5. Compute edge "affinity" or "boundary" values for each pair of regions in the RAG.
    #    Typically, we take the average or minimum boundary prob along the boundary
    #    between two segments as an edge cost. Here, we'll do a simple mean boundary:
    #    The function "nifty.graph.rag.accumulateEdgeMeanAndLength" accumulates
    #    the boundary prob for each edge and returns edge features (mean, pixel count).
    features = nifty.graph.rag.accumulateEdgeMeanAndLength(rag, boundary_prob)

    # Our edge cost is the mean boundary probability on that edge.
    edge_cost = features[:, 0]  # first column is the mean boundary

    # 6. Solve multicut with Kernighan–Lin to get the final partition of the RAG.
    objective = nmc.multicutObjective(rag, edge_cost)
    solver_factory = nmc.KernighanLinFactoryMulticutObjectiveUndirectedGraph()
    solver = solver_factory.create(objective)
    # run solver
    solved_node_labels = solver.optimize()

    # 7. Project the partitioning on the RAG back to a full 3D segmentation.
    seg = nifty.graph.rag.projectScalarNodeDataToPixels(rag, solved_node_labels)

    print("seg.min(): ", np.min(seg))
    print("seg.max(): ", np.max(seg))

    return seg.astype(np.int32)


def segment_to_image(segment):
    colors = np.random.randint(0, 255, (np.max(segment), 3))
    image = np.zeros((segment.shape[0], segment.shape[1], segment.shape[2], 3))
    print("range:", np.max(segment))
    for i in range(np.max(segment)):
        image[segment == i] = colors[i]
    return image
