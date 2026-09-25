import numpy as np 

import numpy as np 

from sklearn.cluster import KMeans
import ipdb 


def _cluster_items(item_latents, n_clusters): 
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(item_latents)
    return kmeans.labels_

def __cluster_users(user_latents, n_clusters):

    # Fit KMeans clustering on the user-item data
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(user_latents)

    # Get cluster labels for each user
    cluster_labels = kmeans.labels_
    
    # Create a dictionary to hold user indices for each cluster
    cluster_dict = {i: [] for i in range(n_clusters)}
    for idx, label in enumerate(cluster_labels):
        cluster_dict[label].append(idx) #cluster id: user indexes belonging to that cluster 

    return cluster_labels, cluster_dict

def __find_target_item_clusters(
    user_item_data,
    target_items,
    cluster_dict,
    positive_threshold=3.0,
):
    cluster_avg_interactions = {}

    for cluster, users in cluster_dict.items():

        if len(users) == 0:
            cluster_avg_interactions[cluster] = 0.0
            continue

        group_ratings = user_item_data[users][:, target_items]

        positive_ratings = group_ratings[
            group_ratings > positive_threshold
        ]

        if positive_ratings.size > 0:
            avg_interaction = np.mean(positive_ratings)
        else:
            avg_interaction = 0.0

        cluster_avg_interactions[cluster] = avg_interaction

    return cluster_avg_interactions

def generate_strategy_items_and_ratigns(params, rng, strategy, item_popularity, item_latents, user_latents, baseline_ratings, budget=10): 
    if strategy == 'max_rating': 
        strategy_items = np.array([
            params.target_item
        ])
        strategy_ratings = np.array([
            params.rating_max
        ])
    if strategy == 'random_filler':
        candidate_items = np.delete(
            np.arange(params.n_items),
            params.target_item,
        )

        filler_items = rng.choice(
            candidate_items,
            size=budget - 1,
            replace=False,
        )

        strategy_items = np.concatenate([
            np.array([params.target_item]),
            filler_items,
        ])

        strategy_ratings = np.full(
            budget,
            params.rating_max,
            dtype=float,
        )
    if strategy == 'mainstream_mimic': 
        candidate_items = np.delete(
            np.arange(params.n_items),
            params.target_item,
        )

        # Sort remaining items from most to least popular.
        ranked_candidates = candidate_items[
            np.argsort(
                item_popularity[candidate_items]
            )[::-1]
        ]

        most_popular_items = ranked_candidates[
            :budget - 1
        ]

        strategy_items = np.concatenate([
            np.array(
                [params.target_item],
                dtype=int,
            ),
            most_popular_items,
        ])

        strategy_ratings = np.full(
            len(strategy_items),
            params.rating_max,
            dtype=float,
        )
    if strategy == 'target_neighborhood': 
        clusters = _cluster_items(
        item_latents,
        n_clusters=5,
        )

        target_cluster = clusters[params.target_item]

        # ---------------------------------------------------------
        # Compute cluster centroids
        # ---------------------------------------------------------
        unique_clusters = np.unique(clusters)

        cluster_centroids = {
            c: item_latents[clusters == c].mean(axis=0)
            for c in unique_clusters
        }

        target_centroid = cluster_centroids[target_cluster]

        # ---------------------------------------------------------
        # Order clusters by distance from target cluster
        # Target cluster itself comes first
        # ---------------------------------------------------------
        cluster_distances = {
            c: np.linalg.norm(
                cluster_centroids[c] - target_centroid
            )
            for c in unique_clusters
        }

        ordered_clusters = sorted(
            unique_clusters,
            key=lambda c: cluster_distances[c],
        )

        # ---------------------------------------------------------
        # Add candidates cluster-by-cluster until we have enough
        # ---------------------------------------------------------
        candidate_items = []

        for cluster_id in ordered_clusters:

            cluster_items = np.flatnonzero(
                clusters == cluster_id
            )

            # Never include the target item as a filler
            cluster_items = cluster_items[
                cluster_items != params.target_item
            ]

            candidate_items.extend(
                cluster_items.tolist()
            )

            if len(candidate_items) >= budget - 1:
                break

        candidate_items = np.asarray(
            candidate_items,
            dtype=int,
        )

        # ---------------------------------------------------------
        # Select filler items
        # ---------------------------------------------------------
        filler_items = rng.choice(
            candidate_items,
            size=budget - 1,
            replace=False,
        )

        strategy_items = np.concatenate([
            [params.target_item],
            filler_items,
        ])

        strategy_ratings = np.full(
            len(strategy_items),
            params.rating_max,
            dtype=float,
        )
    if strategy == 'target_supporter_mimic': 
        # ---------------------------------------------------------
        # 1. Cluster users in latent space
        # ---------------------------------------------------------
        cluster_labels, clusters = __cluster_users(
            user_latents=user_latents,
            n_clusters=5,
        )

        # ---------------------------------------------------------
        # 2. Find which user cluster most strongly supports
        #    the target item
        # ---------------------------------------------------------
        avg_interactions = __find_target_item_clusters(
            user_item_data=baseline_ratings,
            target_items=[params.target_item],
            cluster_dict=clusters,
            positive_threshold=3.0,
        )

        target_cluster = max(
            avg_interactions,
            key=avg_interactions.get,
        )

        mimic_users = clusters[target_cluster]
        # ipdb.set_trace()
        # ---------------------------------------------------------
        # 3. Construct average rating profile of those users
        # ---------------------------------------------------------
        mimic_profile = np.mean(
            baseline_ratings[mimic_users, :],
            axis=0,
        )

        # ---------------------------------------------------------
        # 4. Candidate items = everything except target
        # ---------------------------------------------------------
        candidate_items = np.arange(
            baseline_ratings.shape[1]
        )

        candidate_items = candidate_items[
            candidate_items != params.target_item
        ]

        # ---------------------------------------------------------
        # 5. Select items most highly rated by target supporters
        # ---------------------------------------------------------
        # ipdb.set_trace()
        candidate_scores = baseline_ratings[
            np.ix_(
                np.array(mimic_users),
                candidate_items,
            )
        ]
        
        # One score per candidate item
        candidate_scores = np.asarray(np.sum(
            candidate_scores,
            axis=0,
        )).ravel()

        order = np.argsort(candidate_scores)[::-1]
        # ipdb.set_trace() 
        
        filler_items = candidate_items[
            order[:budget - 1]
        ]

        # ---------------------------------------------------------
        # 6. Build strategy
        # ---------------------------------------------------------
        strategy_items = np.concatenate([
            [params.target_item],
            filler_items.flatten(),
        ])

        strategy_ratings = np.full(
            len(strategy_items),
            params.rating_max,
            dtype=float,
        )
    
    return strategy_items, strategy_ratings

def compute_strategy_vector(
        baseline_item_latents,
        strategy_items,
        strategy_ratings,
        reg,
    ):
    V = baseline_item_latents[strategy_items]
    r = np.asarray(strategy_ratings, dtype=float)

    A = (
        V.T @ V
        + reg * np.eye(V.shape[1])
    )

    b = V.T @ r

    u_c_star = np.linalg.solve(A, b)

    return u_c_star