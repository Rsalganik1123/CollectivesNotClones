import numpy as np 

def _stable_softmax(x, axis = -1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def _compute_interaction_probabilities(
    params, 
    true_users: np.ndarray,
    true_items: np.ndarray,
    item_popularity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    preference_scores = true_users @ true_items.T

    log_weights = (
        params.popularity_bias_tau
        * np.log(item_popularity)[None, :]
        + params.preference_selectivity_eta
        * preference_scores
    )

    probabilities = _stable_softmax(
        log_weights,
        axis=1,
    )
    return preference_scores, probabilities

#For making sure that our ALS solutions didn't weirdly rotate
def orthogonal_rotation(
    source: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    cross_covariance = source.T @ target
    left, _, right_t = np.linalg.svd(
        cross_covariance,
        full_matrices=False,
    )
    return left @ right_t

def align_factorization_to_baseline(
    post_users: np.ndarray,
    post_items: np.ndarray,
    baseline_users: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = orthogonal_rotation(
        source=post_users,
        target=baseline_users,
    )
    return (
        post_users @ rotation,
        post_items @ rotation,
        rotation
    )
    
    