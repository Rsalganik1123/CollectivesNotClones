import numpy as np
import pandas as pd
import tensorflow as tf


def movies_df_from_params(params):
    movie_ids = np.arange(params.n_items).astype(str)
    return pd.DataFrame({"movie_id": movie_ids})


def synthetic_ratings_to_df(ratings_csr, params, rng):
    ratings_csr = ratings_csr.tocoo()

    df = pd.DataFrame({
        "user_id": ratings_csr.row.astype(str),
        "movie_id": ratings_csr.col.astype(str),
        "rating": ratings_csr.data.astype(np.float32),
    })

    def _assign_order(group):
        order = rng.permutation(len(group))
        return group.assign(timestamp=order)

    df = (
        df.groupby("user_id", group_keys=False)
        .apply(_assign_order)
    )

    return df


def build_tt_artifacts(ratings_csr, params, rng):
    movies_df = movies_df_from_params(params)
    ratings_df = synthetic_ratings_to_df(ratings_csr, params, rng)

    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(str)

    ratings = tf.data.Dataset.from_tensor_slices({
        "user_id": ratings_df["user_id"].values,
        "movie_id": ratings_df["movie_id"].values,
    })

    movies = tf.data.Dataset.from_tensor_slices(
        movies_df["movie_id"].values
    )

    return movies, ratings, movies_df, ratings_df

