import numpy as np 
from scipy.optimize import lsq_linear
from sklearn.cluster import KMeans
import ipdb 
from scipy.optimize import minimize

def _cluster_distances(
    centers,
    anchor_cluster,
):
    """
    Return cluster IDs ordered by distance from anchor cluster.
    """

    distances = np.linalg.norm(
        centers
        - centers[anchor_cluster],
        axis=1,
    )

    return np.argsort(
        distances
    )
    
def __cluster_users(
    user_latents,
    n_clusters=5,
):
    """
    Cluster users in baseline latent space.

    Returns
    -------
    labels : np.ndarray
        Cluster assignment for every user.

    cluster_dict : dict[int, np.ndarray]
        User indices belonging to each cluster.

    centers : np.ndarray
        KMeans cluster centers.
    """

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init="auto",
    ).fit(user_latents)

    labels = kmeans.labels_

    cluster_dict = {
        cluster_id: np.where(
            labels == cluster_id
        )[0]
        for cluster_id in range(n_clusters)
    }

    return (
        labels,
        cluster_dict,
        kmeans.cluster_centers_,
    )
    
def sample_low_dispersion(
    rng,
    baseline_U,
    alpha,
    n_users,
    n_clusters=5,
):
    """
    Construct a low-dispersion coalition.

    Prefer users from one cluster. If that cluster is too
    small, fill the remaining coalition positions from the
    nearest clusters in latent space.
    """

    coalition_size = max(
        1,
        int(n_users * alpha),
    )

    (
        _,
        user_clusters,
        centers,
    ) = __cluster_users(
        baseline_U,
        n_clusters=n_clusters,
    )

    # --------------------------------------------------------
    # Prefer a cluster large enough to contain the coalition.
    # --------------------------------------------------------

    big_enough = [
        cluster_id
        for cluster_id, members
        in user_clusters.items()
        if len(members) >= coalition_size
    ]

    if len(big_enough) > 0:

        anchor_cluster = int(
            rng.choice(big_enough)
        )

    else:

        # If none is large enough, start with a random cluster.
        anchor_cluster = int(
            rng.choice(
                list(user_clusters.keys())
            )
        )

    # --------------------------------------------------------
    # Visit clusters from nearest -> furthest.
    # --------------------------------------------------------

    cluster_order = _cluster_distances(
        centers,
        anchor_cluster,
    )

    coalition_idx = []

    for cluster_id in cluster_order:

        remaining = (
            coalition_size
            - len(coalition_idx)
        )

        if remaining <= 0:
            break

        members = user_clusters[
            int(cluster_id)
        ]

        n_take = min(
            remaining,
            len(members),
        )

        selected = rng.choice(
            members,
            size=n_take,
            replace=False,
        )

        coalition_idx.extend(
            selected.tolist()
        )

    return np.asarray(
        coalition_idx,
        dtype=int,
    )

def sample_med_dispersion(
    rng,
    baseline_U,
    alpha,
    n_users,
    n_clusters=5,
    n_selected_clusters=3,
):
    """
    Construct a medium-dispersion coalition by deliberately
    spreading coalition members across several user clusters.

    Users are allocated approximately evenly across the
    selected clusters.
    """

    coalition_size = max(
        1,
        int(n_users * alpha),
    )

    (
        _,
        user_clusters,
        _,
    ) = __cluster_users(
        baseline_U,
        n_clusters=n_clusters,
    )

    # Never ask for more clusters than users in coalition.
    n_selected_clusters = min(
        n_selected_clusters,
        n_clusters,
        coalition_size,
    )

    # --------------------------------------------------------
    # Select several distinct clusters.
    # --------------------------------------------------------

    available_clusters = [
        cluster_id
        for cluster_id, members
        in user_clusters.items()
        if len(members) > 0
    ]

    selected_clusters = rng.choice(
        available_clusters,
        size=n_selected_clusters,
        replace=False,
    )

    coalition_idx = []

    # --------------------------------------------------------
    # Try to distribute users approximately evenly.
    # --------------------------------------------------------

    base_per_cluster = (
        coalition_size
        // n_selected_clusters
    )

    remainder = (
        coalition_size
        % n_selected_clusters
    )

    for position, cluster_id in enumerate(
        selected_clusters
    ):

        desired = (
            base_per_cluster
            + int(position < remainder)
        )

        members = user_clusters[
            int(cluster_id)
        ]

        n_take = min(
            desired,
            len(members),
        )

        selected = rng.choice(
            members,
            size=n_take,
            replace=False,
        )

        coalition_idx.extend(
            selected.tolist()
        )

    # --------------------------------------------------------
    # Some selected clusters might not have had enough users.
    # Fill any remaining slots from users not yet selected.
    # --------------------------------------------------------

    remaining = (
        coalition_size
        - len(coalition_idx)
    )

    if remaining > 0:

        selected_set = set(
            coalition_idx
        )

        eligible = np.array([
            user_idx
            for user_idx in range(n_users)
            if user_idx not in selected_set
        ])

        extra = rng.choice(
            eligible,
            size=remaining,
            replace=False,
        )

        coalition_idx.extend(
            extra.tolist()
        )

    return np.asarray(
        coalition_idx,
        dtype=int,
    )

def sample_high_dispersion(rng, baseline_U, alpha, n_users, n_clusters = 5): 
    coalition_size = int(n_users * alpha) 
    coalition_idx = rng.choice(
            n_users,
            size=coalition_size,
            replace=False,
        )
    return coalition_idx 

def _interleave_clusters(
    rng,
    cluster_ids,
    user_clusters,
):
    """
    Randomly shuffle users within each cluster, then draw from
    clusters in round-robin order.

    Example:
        C1 user, C2 user, C3 user,
        C1 user, C2 user, C3 user, ...
    """

    shuffled = {
        int(cluster_id): list(
            rng.permutation(
                user_clusters[int(cluster_id)]
            )
        )
        for cluster_id in cluster_ids
    }

    order = []

    active_clusters = [
        int(c)
        for c in cluster_ids
    ]

    while active_clusters:

        next_active = []

        for cluster_id in active_clusters:

            if len(
                shuffled[cluster_id]
            ) > 0:

                order.append(
                    shuffled[cluster_id].pop()
                )

            if len(
                shuffled[cluster_id]
            ) > 0:

                next_active.append(
                    cluster_id
                )

        active_clusters = (
            next_active
        )

    return order

def sample_med_dispersion_order(
    rng,
    user_clusters,
    cluster_centers,
    n_selected_clusters=3,
):
    """
    Create a nested medium-dispersion ordering.

    Choose an anchor cluster and several nearby clusters, then
    interleave users across those clusters.

    Once those clusters are exhausted, append the remaining
    clusters from nearest to furthest.
    """

    n_clusters = len(
        user_clusters
    )

    n_selected_clusters = min(
        n_selected_clusters,
        n_clusters,
    )

    cluster_ids = np.arange(
        n_clusters
    )

    # --------------------------------------------------------
    # Pick an anchor.
    # --------------------------------------------------------

    anchor_cluster = int(
        rng.choice(cluster_ids)
    )

    distances = np.linalg.norm(
        cluster_centers
        - cluster_centers[anchor_cluster],
        axis=1,
    )

    nearest_clusters = np.argsort(
        distances
    )

    # Anchor + nearest neighboring clusters.
    selected_clusters = (
        nearest_clusters[
            :n_selected_clusters
        ]
    )

    # --------------------------------------------------------
    # Interleave the selected clusters.
    # --------------------------------------------------------

    user_order = _interleave_clusters(
        rng=rng,
        cluster_ids=selected_clusters,
        user_clusters=user_clusters,
    )

    # --------------------------------------------------------
    # Add remaining clusters after the medium-dispersion
    # region has been exhausted.
    # --------------------------------------------------------

    remaining_clusters = [
        int(cluster_id)
        for cluster_id in nearest_clusters
        if cluster_id
        not in set(selected_clusters)
    ]

    for cluster_id in remaining_clusters:

        members = rng.permutation(
            user_clusters[
                cluster_id
            ]
        )

        user_order.extend(
            members.tolist()
        )

    return np.asarray(
        user_order,
        dtype=int,
    )
    
def sample_high_dispersion_order(
    rng,
    n_users,
):
    """
    Random nested coalition ordering over the full population.
    """

    return rng.permutation(
        n_users
    )

def sample_low_dispersion_order(
    rng,
    user_clusters,
    cluster_centers,
):
    """
    Create a nested low-dispersion user ordering.

    Start from one anchor cluster. Once that cluster is exhausted,
    expand outward through clusters in increasing centroid distance.

    Thus small coalition prefixes are highly concentrated, while
    larger coalitions gradually become more heterogeneous.
    """

    cluster_ids = np.array(
        list(user_clusters.keys()),
        dtype=int,
    )

    # Random anchor so seeds generate different low-dispersion coalitions.
    anchor_cluster = int(
        rng.choice(cluster_ids)
    )

    distances = np.linalg.norm(
        cluster_centers
        - cluster_centers[anchor_cluster],
        axis=1,
    )

    cluster_order = np.argsort(
        distances
    )

    user_order = []

    for cluster_id in cluster_order:

        members = np.asarray(
            user_clusters[int(cluster_id)]
        )

        # Randomize users within each cluster.
        members = rng.permutation(
            members
        )

        user_order.extend(
            members.tolist()
        )

    return np.asarray(
        user_order,
        dtype=int,
    )