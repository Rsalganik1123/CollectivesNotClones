"""
Shared two-tower collective-action pipeline pieces, used by both
real_TT_CA.py and synthetic_TT_CA.py.

These are lifted directly from real_TT_CA.py with one fix: functions that
previously closed over a module-level `ratings_df` / `user_id_to_idx`
(fine for a single-entrypoint script, unsafe once a second script imports
them with different data) now take those as explicit arguments.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from general_utils.data_loading import build_sparse_rating_matrix, build_tf_datasets
from models.two_tower import (
    build_finetune_model,
    build_two_tower_from_pieces,
    get_user_model,
)


def build_histories_from_ratings(ratings_df, max_history_length):
    """
    Convert a ratings_df into per-user left-padded history arrays,
    in the format the TT user models consume.
    """
    ratings_df = (
        ratings_df
        .astype({"user_id": int, "movie_id": int})
        .sort_values(["user_id", "timestamp"])
    )

    user_ids, histories, history_ratings = [], [], []

    for user_id, group in ratings_df.groupby("user_id"):
        group = group.sort_values("timestamp")

        movie_ids = group["movie_id"].astype(str).tolist()
        ratings = group["rating"].tolist()

        movie_ids = movie_ids[-max_history_length:]
        ratings = ratings[-max_history_length:]

        n_pad = max_history_length - len(movie_ids)

        user_ids.append(str(user_id))
        histories.append([""] * n_pad + movie_ids)
        history_ratings.append([0.0] * n_pad + ratings)

    user_idx_to_id = {idx: uid for idx, uid in enumerate(user_ids)}
    user_id_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}

    ratings_df = ratings_df.astype({"user_id": str, "movie_id": str})

    return (
        ratings_df,
        np.asarray(user_ids),
        np.asarray(histories),
        np.asarray(history_ratings, dtype=np.float32),
        user_idx_to_id,
        user_id_to_idx,
    )


def gather_ratings(ratings_df, user_id_to_idx, movie_id_to_idx):
    baseline_ratings = build_sparse_rating_matrix(
        ratings_df=ratings_df,
        user_id_to_idx=user_id_to_idx,
        movie_id_to_idx=movie_id_to_idx,
    )

    observation_mask = np.zeros(baseline_ratings.shape, dtype=bool)
    observation_mask[baseline_ratings.nonzero()] = True

    item_popularity = np.asarray(
        baseline_ratings.getnnz(axis=0)
    ).ravel()

    return baseline_ratings, observation_mask, item_popularity


def compute_strategy_vector(
    *,
    model,
    user_model_name,
    strategic_items,
    strategic_ratings,
    history_length,
):
    """
    The latent representation induced by adopting the strategy items
    alone (no prior history) -- the TT analogue of the ALS bot-farm
    representation u_C^*.
    """
    strategic_movie_ids = [str(item) for item in strategic_items]
    ratings = [float(r) for r in strategic_ratings]

    strategic_movie_ids = strategic_movie_ids[-history_length:]
    ratings = ratings[-history_length:]

    n_pad = history_length - len(strategic_movie_ids)

    history = np.asarray([[""] * n_pad + strategic_movie_ids])
    rating_history = np.asarray(
        [[0.0] * n_pad + ratings], dtype=np.float32
    )

    if user_model_name == "weighted":
        return model.user_model(
            {
                "history": tf.constant(history),
                "ratings": tf.constant(rating_history),
            },
            training=False,
        ).numpy()[0]

    elif user_model_name == "basic":
        # No meaningful history-only strategic representation
        # for the ID baseline.
        return None

    else:
        return model.user_model(
            tf.constant(history),
            training=False,
        ).numpy()[0]


def apply_strategy_to_histories(
    ratings_df,
    histories,
    history_ratings,
    strategic_items,
    strategic_ratings,
    collective_indices,
    max_history_length,
):
    new_histories = histories.copy()
    new_ratings = history_ratings.copy()

    strategic_movie_ids = [str(item_idx) for item_idx in strategic_items]
    strategic_ratings = [float(x) for x in strategic_ratings]

    last_timestamp = ratings_df.timestamp.max()
    new_rows = []

    for user_idx in collective_indices:

        old_movies = [
            x for x in new_histories[user_idx] if x != ""
        ]
        old_ratings = [
            float(r)
            for movie, r in zip(
                new_histories[user_idx], new_ratings[user_idx]
            )
            if movie != ""
        ]

        updated_movies = old_movies + strategic_movie_ids
        updated_ratings = old_ratings + strategic_ratings

        updated_movies = updated_movies[-max_history_length:]
        updated_ratings = updated_ratings[-max_history_length:]

        n_pad = max_history_length - len(updated_movies)

        new_histories[user_idx] = [""] * n_pad + updated_movies
        new_ratings[user_idx] = [0.0] * n_pad + updated_ratings

        for action_idx, (movie_id, rating) in enumerate(
            zip(strategic_movie_ids, strategic_ratings)
        ):
            new_rows.append({
                "user_id": str(user_idx),
                "movie_id": str(movie_id),
                "rating": float(rating),
                "timestamp": last_timestamp + action_idx + 1,
            })

    strategic_df = pd.DataFrame(new_rows)

    return strategic_df, new_histories, new_ratings


def refit_two_tower(
    *,
    baseline_model,
    model_config,
    movies,
    ratings_df_post,
    user_ids_vocab,
    movie_ids_vocab,
    epochs=3,
):
    """
    Thin wrapper around build_finetune_model + .fit(), warm-started
    from the frozen baseline model.
    """
    post_model, post_train_ds = build_finetune_model(
        baseline_model=baseline_model,
        config=model_config,
        movies=movies,
        ratings_df_post=ratings_df_post,
        user_ids_vocab=user_ids_vocab,
        movie_ids_vocab=movie_ids_vocab,
    )

    post_model.fit(post_train_ds, epochs=epochs, verbose=0)

    return post_model


def train_two_tower_from_scratch(
    *,
    config,
    movies,
    ratings_df,
    user_ids_vocab,
    movie_ids_vocab,
    epochs=15,
):
    """
    Train a fresh two-tower model on `ratings_df`.

    Unlike real_TT_CA.py -- which loads a checkpoint tuned once on the
    fixed MovieLens population -- the synthetic pipeline needs to retrain
    per (alpha, p0) condition, since construct_users_with_dispersion
    changes the *baseline* population itself. With n_users/n_items in the
    hundreds this is cheap (seconds), so we skip the full validation /
    early-stopping machinery in TT_tuning.py and just run a fixed number
    of epochs. Swap in build_tf_datasets' val split + EarlyStopping
    (see TT_tuning.py) if you want that safety net back.
    """
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config["seed"])

    user_model = get_user_model(
        u_model=config["user_model_name"],
        user_ids_vocab=user_ids_vocab,
        movie_ids_vocab=movie_ids_vocab,
        embedding_dim=config["embedding_dim"],
        max_history_length=config["history_length"],
    )

    movie_model = tf.keras.Sequential([
        movie_ids_vocab,
        tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            config["embedding_dim"],
        ),
    ])

    model = build_two_tower_from_pieces(
        user_model=user_model,
        movie_model=movie_model,
        movies=movies,
        user_model_name=config["user_model_name"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adagrad(
            learning_rate=config["learning_rate"]
        )
    )

    train_ds, _, _ = build_tf_datasets(
        ratings_df=ratings_df,
        history_length=config["history_length"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    model.fit(train_ds, epochs=epochs, verbose=0)

    return model
