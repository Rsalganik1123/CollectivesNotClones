
import numpy as np 

from general_utils.math_helpers import _stable_softmax


from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class SyntheticParams:
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
    target_item: int = 35
    target_rating: int = 5
    


class SyntheticWorld: 
    def __init__(self, params, seed):
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def get_seed(self):
        return self.seed
    
    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def define_item_popularity(self, item_genres): 

        item_noise = self.rng.normal(
            0.0,
            self.params.sigma_item_pop,
            size=self.params.n_items,
        )

        genre_popularity = np.asarray(
            self.params.genre_popularity,
            dtype=float,
        )
        item_popularity = np.zeros(
            self.params.n_items,
            dtype=float,
        )

        for genre in range(self.params.n_genres):
            indices = np.flatnonzero(
                item_genres == genre
            )
            within_genre = _stable_softmax(
                item_noise[indices]
            )
            item_popularity[indices] = (
                genre_popularity[genre] * within_genre
            )

        if not np.isclose(item_popularity.sum(), 1.0):
            raise RuntimeError(
                "Item popularity does not sum to one."
            )

        return item_popularity

    def get_coalition_indices(self, coalition_order, alpha):
        coalition_size = round(alpha * self.params.n_users)
        return np.sort(coalition_order[:coalition_size])

    # Creating the stable synthetic environment --> one per seed 
    def generate_items(self): 
        prototypes = np.eye(
            self.params.n_genres,
            self.params.latent_dim,
            dtype=float,
        )

        true_items = np.empty(
            (self.params.n_items, self.params.latent_dim),
            dtype=float,
        )
        
        item_genres = np.empty(
            self.params.n_items,
            dtype=int,
        )

        start = 0
        for genre in range(self.params.n_genres):
            end = start + self.params.n_items_per_genre
            true_items[start:end] = self.rng.normal(
                loc=prototypes[genre],
                scale=self.params.sigma_v,
                size=(
                    self.params.n_items_per_genre,
                    self.params.latent_dim,
                ),
            )
            item_genres[start:end] = genre
            start = end

        return true_items, item_genres

    def generate_users(self,): 
        true_users = self.rng.normal(
            loc=self.params.user_mean,
            scale=self.params.user_std,
            size=(
                self.params.n_users,
                self.params.latent_dim,
            ),
        )
        return true_users 
        
    def generate_synthetic_environment(self):
        true_items, item_genres = self.generate_items()
        item_popularity = self.define_item_popularity(
            item_genres,
        )
        base_true_users = self.generate_users()
        coalition_order = self.rng.permutation(self.params.n_users)
        
        return true_items, item_genres, item_popularity, base_true_users, coalition_order
