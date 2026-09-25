
import numpy as np 
from dataclasses import dataclass
from typing import Iterable

from dataclasses import replace
import numpy as np
import pandas as pd


from general_utils.reproducibility import _condition_seed
from real.cluster_construction import __cluster_users, sample_high_dispersion_order, sample_low_dispersion_order, sample_med_dispersion_order
@dataclass(frozen=True)
class MLParams:
    # Environment
    n_users: int = 100
    n_items: int = 100
    n_genres: int = 5
    n_items_per_genre: int = 20
    latent_dim: int = 5

    # True item/user generation
    sigma_v: float = 0.2
    user_mean: float = 0.0
    user_std: float = 1.0

    # Popularity
    sigma_item_pop: float = 0.5
    genre_popularity: tuple[float, ...] = (
        0.35,
        0.25,
        0.18,
        0.13,
        0.09,
    )

    # Interaction sampling
    interactions_per_user: int = 20
    popularity_bias_tau: float = 1.0
    preference_selectivity_eta: float = 1.0

    # Initial synthetic ratings
    rating_mean: float = 3.0
    rating_min: float = 1.0
    rating_max: float = 5.0
    rating_scale: float = 0.75
    rating_noise_std: float = 0.35

    # Intervention ratings generated from fitted ALS factors
    intervention_noise_std: float = 0.05
    force_max_target_rating: bool = True
    replace_existing_coalition_ratings: bool = False

    # ALS
    als_reg: float = 0.1
    als_iters: int = 15
    als_patience: int = 5
    als_seed: int = 42

    # Target
    target_item: int = 111
    target_rating: int = 5


def params_from_dataframe(
    ratings_df,
    base_params=None,
    user_col="user_id",
    item_col="movie_id",
    rating_col="rating",
    genre_col=None,
):
    """
    Populate dataset-dependent SyntheticParams fields from a dataframe.

    Parameters
    ----------
    ratings_df : pd.DataFrame
        Interaction dataframe.

    base_params : SyntheticParams, optional
        Existing parameter object. Dataset-dependent values are replaced
        while all other parameters are preserved.

    user_col : str
        User ID column.

    item_col : str
        Item ID column.

    rating_col : str
        Rating/value column.

    genre_col : str or None
        Optional genre/category column. If provided, genre-related
        parameters are estimated from the dataframe.

    Returns
    -------
    SyntheticParams
    """

    if base_params is None:
        base_params = MLParams()

    # --------------------------------------------------
    # Basic dataset statistics
    # --------------------------------------------------

    n_users = ratings_df[user_col].nunique()
    n_items = ratings_df[item_col].nunique()

    interactions_per_user = (
        ratings_df.groupby(user_col)
        .size()
        .mean()
    )

    rating_mean = ratings_df[rating_col].mean()
    rating_min = ratings_df[rating_col].min()
    rating_max = ratings_df[rating_col].max()

    updates = {
        "n_users": int(n_users),
        "n_items": int(n_items),

        # SyntheticParams expects an integer here.
        # Mean interactions/user is usually the most useful analogue.
        "interactions_per_user": int(
            round(interactions_per_user)
        ),

        "rating_mean": float(rating_mean),
        "rating_min": float(rating_min),
        "rating_max": float(rating_max),
    }

    # --------------------------------------------------
    # Optional genre statistics
    # --------------------------------------------------

    if genre_col is not None:

        item_genres = (
            ratings_df[
                [item_col, genre_col]
            ]
            .drop_duplicates(subset=item_col)
        )

        genre_counts = (
            item_genres[genre_col]
            .value_counts()
        )

        n_genres = len(genre_counts)

        updates["n_genres"] = int(n_genres)

        # This assumes approximately equal-sized genres.
        # Useful if the synthetic generator requires one scalar.
        updates["n_items_per_genre"] = int(
            round(
                n_items / n_genres
            )
        )

        # Empirical interaction share by genre
        interaction_genre_counts = (
            ratings_df[genre_col]
            .value_counts()
        )

        genre_popularity = (
            interaction_genre_counts
            / interaction_genre_counts.sum()
        )

        updates["genre_popularity"] = tuple(
            genre_popularity.values.astype(float)
        )

    return replace(
        base_params,
        **updates,
    )
    
BASE_TT_CONFIG = {
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 3,
    "seed": 42,
    "verbose": 1,
}

def sample_target_items(ratings_df, setting='Medium'):
    movies = ratings_df.movie_id.unique() 
    num_interactions = ratings_df.groupby('movie_id')['user_id'].apply(lambda x: len(list(x))).reset_index(name='num_users')
    num_interactions['bin'] = pd.cut(num_interactions['num_users'], bins=3, labels=['Low', 'Medium', 'High'])
    candidates = num_interactions[num_interactions['bin'] == 'Medium']
    
def organize_coalition_orderings(baseline_U, seed, num_clusters=5): 
    user_cluster_labels, user_clusters,user_cluster_centers, = __cluster_users(
        baseline_U,
        n_clusters=num_clusters,
    )
    low_rng = np.random.default_rng(
    _condition_seed(
        seed=seed,
        alpha_index=0,
        p0_index=0,
        stream=101,
    )
    )

    med_rng = np.random.default_rng(
        _condition_seed(
            seed=seed,
            alpha_index=0,
            p0_index=1,
            stream=101,
        )
    )

    high_rng = np.random.default_rng(
        _condition_seed(
            seed=seed,
            alpha_index=0,
            p0_index=2,
            stream=101,
        )
    )

    low_order = sample_low_dispersion_order(
        rng=low_rng,
        user_clusters=user_clusters,
        cluster_centers=user_cluster_centers,
    )

    med_order = sample_med_dispersion_order(
        rng=med_rng,
        user_clusters=user_clusters,
        cluster_centers=user_cluster_centers,
        n_selected_clusters=3,
    )

    high_order = sample_high_dispersion_order(
        rng=high_rng,
        n_users=baseline_U.shape[0],
    )
    
    coalition_orders = {
        "low": low_order,
        "med": med_order,
        "high": high_order,
    }
    
    return coalition_orders

