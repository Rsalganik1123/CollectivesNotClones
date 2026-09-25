from scipy import sparse
import numpy as np 

from general_utils.math_helpers import _compute_interaction_probabilities 

def generate_sparse_ratings(
    params, 
    rng,
    true_users,
    true_items,
    item_popularity,
) :
    """
    Generate a scipy CSR ratings matrix.

    Assumptions embedded in our code:
    - Missing ratings are absent sparse entries.
    - There are no NaNs.
    - All observed ratings are in [1, 5], so zero safely means missing.
    """
    preference_scores, probabilities = (
        _compute_interaction_probabilities(
            true_users=true_users,
            true_items=true_items,
            item_popularity=item_popularity,
            params = params
        )
    )

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    observation_mask = np.zeros(
        (params.n_users, params.n_items),
        dtype=bool,
    )

    noise = rng.normal(
        0.0,
        params.rating_noise_std,
        size=(params.n_users, params.n_items),
    )

    dense_rating_values = np.clip(
        params.rating_mean
        + params.rating_scale * preference_scores
        + noise,
        params.rating_min,
        params.rating_max,
    )

    for user_idx in range(params.n_users):
        observed_items = rng.choice(
            params.n_items,
            size=params.interactions_per_user,
            replace=False,
            p=probabilities[user_idx],
        )

        observation_mask[
            user_idx,
            observed_items,
        ] = True

        rows.extend(
            [user_idx] * len(observed_items)
        )
        cols.extend(observed_items.tolist())
        data.extend(
            dense_rating_values[
                user_idx,
                observed_items,
            ].tolist()
        )

    ratings = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(params.n_users, params.n_items),
        dtype=float,
    )

    return ratings, observation_mask

def enforce_target_item_max_rating(
    ratings,
    coalition_indices,
    target_item,
    target_rating,
):
    """
    Coalition members all assign r_C to the target item.

    If an entry already exists, this replaces it.
    If it does not exist, this adds it.
    """

    ratings_post = ratings.copy()

    ratings_post[
        coalition_indices,
        target_item
    ] = target_rating

    return ratings_post

def apply_strategic_action_to_ratings(
    params, 
    baseline_ratings,
    coalition_indices,
    coordinated_latents,
    baseline_item_factors,
    strategic_items,
):
    """
    Preserve historical behavior and add the pre-selected
    collective strategy items.

    Strategic_items are chosen once for the condition and are
    shared by all coalition members.

    Ratings for the strategic items generated from each user's desired coordinated
    latent representation u'_i.
    """
    modified = baseline_ratings.tolil(copy=True)

    strategic_items = np.asarray(
        strategic_items,
        dtype=int,
    )

    for local_idx, user_idx in enumerate(
        coalition_indices
    ):
        # What this desired latent representation predicts
        # on the already-selected strategy items.
        strategic_ratings = (
            coordinated_latents[local_idx]
            @ baseline_item_factors[strategic_items].T
        )

        strategic_ratings = np.clip(
            strategic_ratings,
            params.rating_min,
            params.rating_max,
        )

        # Explicit collective objective:
        # maximally support the target.
        target_position = np.flatnonzero(
            strategic_items == params.target_item
        )

        strategic_ratings[
            target_position[0]
        ] = params.rating_max

        #Add strategic behavior.
        for item_idx, rating in zip(
            strategic_items,
            strategic_ratings,
        ):
            modified[
                user_idx,
                item_idx,
            ] = rating
        

    result = modified.tocsr()
    result.eliminate_zeros()

    return result