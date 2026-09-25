import pandas as pd 
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from scipy.sparse import csr_matrix
import numpy as np 
import ipdb 

from paths import * 

#TT dataloading for the tuning protocol 
def load_movielens_for_tuning():
    
    # ---------------------------------------------------------
    # Ratings
    # u.data columns:
    # user_id | item_id | rating | timestamp
    # ---------------------------------------------------------

    ratings_df = pd.read_csv(
        f"{DATA_DIR}/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    
    # ipdb.set_trace() 

    # TFRS quickstart expects string IDs.
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(str)

    # ---------------------------------------------------------
    # Movies
    # u.item is pipe-separated and encoded as latin-1
    # ---------------------------------------------------------

    movies_df = pd.read_csv(
        f"{DATA_DIR}/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["movie_id", "movie_title"],
    )

    movies_df["movie_id"] = movies_df["movie_id"].astype(str)


    ratings_df = ratings_df.sort_values(['user_id', 'timestamp'])
    # print(ratings_df.head())
    # print(movies_df.head())
    
    ratings = tf.data.Dataset.from_tensor_slices(
    {
        "user_id": ratings_df["user_id"].values,
        "movie_id": ratings_df["movie_id"].values,
    }
    )

    movies = tf.data.Dataset.from_tensor_slices(
        movies_df["movie_id"].values
    )

    return movies, ratings, movies_df, ratings_df  

def tf_data_split(
    ratings_df,
    max_history_length=None,
    train_frac=0.8,
    val_frac=0.1,
):

    ratings_df = ratings_df.sort_values(
        ["user_id", "timestamp"]
    )

    splits = {
        "train": [],
        "val": [],
        "test": [],
    }

    for user_id, group in ratings_df.groupby("user_id"):

        movie_ids = (
            group["movie_id"]
            .astype(str)
            .tolist()
        )

        ratings = (
            group["rating"]
            .astype(np.float32)
            .tolist()
        )

        n = len(movie_ids)

        if n < 3:
            continue

        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_end = max(train_end, 2)
        val_end = max(val_end, train_end + 1)
        val_end = min(val_end, n - 1)

        for t in range(1, n):

            # Full prefix
            history = movie_ids[:t]
            history_ratings = ratings[:t]

            # Only sequential models truncate
            if max_history_length is not None:
                history = history[-max_history_length:]
                history_ratings = history_ratings[
                    -max_history_length:
                ]

            example = {
                "user_id": user_id,
                "history": history,
                "ratings": history_ratings,
                "movie_id": movie_ids[t],
            }

            if t < train_end:
                splits["train"].append(example)

            elif t < val_end:
                splits["val"].append(example)

            else:
                splits["test"].append(example)

    return splits

def tf_dataset_build(
    examples,
    padded_history_length=None,
):

    if padded_history_length is None:
        padded_history_length = max(
            len(example["history"])
            for example in examples
        )

    user_ids = []
    histories = []
    history_ratings = []
    targets = []

    for example in examples:

        history = example["history"]
        ratings = example["ratings"]

        n_pad = (
            padded_history_length
            - len(history)
        )

        padded_history = (
            [""] * n_pad
            + history
        )

        padded_ratings = (
            [0.0] * n_pad
            + ratings
        )

        user_ids.append(
            example["user_id"]
        )

        histories.append(
            padded_history
        )

        history_ratings.append(
            padded_ratings
        )

        targets.append(
            example["movie_id"]
        )

    return tf.data.Dataset.from_tensor_slices({
        "user_id": user_ids,
        "history": histories,
        "ratings": np.asarray(
            history_ratings,
            dtype=np.float32,
        ),
        "movie_id": targets,
    })    

def get_tf_datasets(
    ratings_df,
    history_length,
    batch_size,
    seed,
):

    splits = tf_data_split(
        ratings_df,
        max_history_length=history_length,
    )

    if history_length is None:

        all_examples = (
            splits["train"]
            + splits["val"]
            + splits["test"]
        )

        padded_history_length = max(
            len(x["history"])
            for x in all_examples
        )

    else:
        padded_history_length = history_length

    train_ds = tf_dataset_build(
        splits["train"],
        padded_history_length=
            padded_history_length,
    )

    val_ds = tf_dataset_build(
        splits["val"],
        padded_history_length=
            padded_history_length,
    )

    test_ds = tf_dataset_build(
        splits["test"],
        padded_history_length=
            padded_history_length,
    )

    train_ds = (
        train_ds
        .shuffle(
            100_000,
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_ds = (
        test_ds
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds

#TT dataloading for the finetuning protocol 
def ratings_to_history(
    ratings_df,
    max_history_length=None,
):
   
    ratings_df = ratings_df.sort_values(
        ["user_id", "timestamp"]
    )

    examples = []

    for user_id, group in ratings_df.groupby("user_id"):

        movie_ids = (
            group["movie_id"]
            .astype(str)
            .tolist()
        )

        ratings = (
            group["rating"]
            .astype(np.float32)
            .tolist()
        )

        n = len(movie_ids)

        if n < 2:
            continue

        for t in range(1, n):

            history = movie_ids[:t]
            history_ratings = ratings[:t]

            if max_history_length is not None:
                history = history[-max_history_length:]
                history_ratings = history_ratings[
                    -max_history_length:
                ]

            examples.append({
                "user_id": user_id,
                "history": history,
                "ratings": history_ratings,
                "movie_id": movie_ids[t],
            })

    return examples

def build_finetune_dataset(
    ratings_df,
    history_length,
    batch_size,
    seed,
):
    """
    Fine-tuning counterpart to build_tf_datasets.

    This builds every example from ratings_df as train, no split.
    """
    examples = ratings_to_history(
        ratings_df,
        max_history_length=history_length,
    )

    dataset = tf_dataset_build(
        examples,
        padded_history_length=history_length,
    )

    dataset = (
        dataset
        .shuffle(
            100_000,
            seed=seed,
            reshuffle_each_iteration=True,
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset

#ALS dataloading for real experiments 
def load_real_ratings(data_dir):
    """
    Load MovieLens100K as a sparse user x item matrix.

    Rows/columns follow the same canonical order as the two-tower
    experiments (users sorted by id, items in u.item order), so the
    target item index refers to the same movie in both pipelines.
    """
    ratings_df = pd.read_csv(
        f"{data_dir}/u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    
    ratings_df["user_id"] = ratings_df["user_id"].astype(str)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(str)
    
    genre_columns = [
        "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies_df = pd.read_csv(
        f"{data_dir}/u.item",
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1] + list(range(5, 24)),  # Movie ID, Title, and Genres
        names=["movie_id", "movie_title"] + genre_columns,
    )
    
    user_ids = np.sort(
    ratings_df["user_id"].unique()
    )

    movie_ids = np.sort(
        ratings_df["movie_id"].unique()
    )

    user_id_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    movie_id_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }
    baseline_ratings = csr_matrix(
        (
            ratings_df["rating"].astype(np.float64).to_numpy(),
            (
                ratings_df["user_id"].map(user_id_to_idx).to_numpy(),
                ratings_df["movie_id"].map(movie_id_to_idx).to_numpy(),
            ),
        ),
        shape=(len(user_id_to_idx), len(movie_id_to_idx)),
    )
    baseline_ratings.eliminate_zeros()

    observation_mask = np.zeros(
        baseline_ratings.shape,
        dtype=bool,
    )
    observation_mask[baseline_ratings.nonzero()] = True

    item_popularity = np.asarray(
        baseline_ratings.getnnz(axis=0)
    ).ravel()

    return ratings_df, movies_df, baseline_ratings, observation_mask, item_popularity, user_id_to_idx, movie_id_to_idx

def split_movielens_per_user(
    ratings_df,
    train_frac=0.8,
    val_frac=0.1,
):
    train_rows = []
    val_rows = []
    test_rows = []

    for _, group in ratings_df.groupby("user_id"):
        group = group.sort_values("timestamp")

        n = len(group)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))

        train_rows.append(group.iloc[:train_end])
        val_rows.append(group.iloc[train_end:val_end])
        test_rows.append(group.iloc[val_end:])

    return (
        pd.concat(train_rows, ignore_index=True),
        pd.concat(val_rows, ignore_index=True),
        pd.concat(test_rows, ignore_index=True),
    )

def build_sparse_rating_matrix(
    *,
    ratings_df,
    user_id_to_idx,
    movie_id_to_idx,
):
    """
    Construct a sparse user-item rating matrix whose rows
    correspond EXACTLY to user_ids and whose columns correspond
    EXACTLY to movie_ids.

    This guarantees alignment with baseline_U and baseline_V.
    """
    # ipdb.set_trace() 
    # -----------------------------------------
    # Canonical ID -> latent row mappings
    # -----------------------------------------

    df = ratings_df.copy()

    df["user_id"] = (
        df["user_id"]
        .astype(str)
    )

    df["movie_id"] = (
        df["movie_id"]
        .astype(str)
    )

    # -----------------------------------------
    # Convert external IDs -> matrix positions
    # -----------------------------------------

    row_indices = (
        df["user_id"]
        .map(user_id_to_idx)
    )

    col_indices = (
        df["movie_id"]
        .map(movie_id_to_idx)
    )

    # -----------------------------------------
    # Fail loudly if IDs don't line up
    # -----------------------------------------

    if row_indices.isna().any():

        missing_users = (
            df.loc[
                row_indices.isna(),
                "user_id",
            ]
            .unique()
        )

        raise ValueError(
            "Some rating users are not present "
            f"in user_ids: {missing_users[:10]}"
        )

    if col_indices.isna().any():

        missing_movies = (
            df.loc[
                col_indices.isna(),
                "movie_id",
            ]
            .unique()
        )

        raise ValueError(
            "Some rating movies are not present "
            f"in movie_ids: {missing_movies[:10]}"
        )

    # -----------------------------------------
    # Construct CSR
    # -----------------------------------------

    baseline_ratings = csr_matrix(
        (
            df["rating"]
            .astype(np.float64)
            .to_numpy(),

            (
                row_indices
                .astype(np.int32)
                .to_numpy(),

                col_indices
                .astype(np.int32)
                .to_numpy(),
            ),
        ),
        shape=(
            len(user_id_to_idx),
            len(movie_id_to_idx),
        ),
    )

    baseline_ratings.eliminate_zeros()

    return baseline_ratings