import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import json 
import itertools
import pandas as pd 
import os 
import shutil
import numpy as np 


def build_vocabularies(movies, ratings):

    user_ids_vocab = tf.keras.layers.StringLookup(
        mask_token=None
    )

    user_ids_vocab.adapt(
        ratings.map(lambda x: x["user_id"])
    )

    # "" -> index 0, which works with Embedding(mask_zero=True)
    movie_ids_vocab = tf.keras.layers.StringLookup(
        mask_token=""
    )

    movie_ids_vocab.adapt(movies)

    return user_ids_vocab, movie_ids_vocab


def load_checkpoint_and_config_for_umodel(u_model): 
    checkpoint_path = f'./checkpoints/best/{u_model}/best.weights.h5'
    model_config = json.load(open(f'./checkpoints/best/{u_model}/config.json'))
    return checkpoint_path, model_config

def _json_safe(value):
    """
    Convert NumPy / Pandas types into normal
    Python types that json.dump understands.
    """

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        if np.isnan(value):
            return None

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value

def save_best_checkpoints_by_model(
    results_df,
    output_root="checkpoints/best",
):

    best_configs = get_best_configs_by_model(
        results_df
    )

    os.makedirs(
        output_root,
        exist_ok=True,
    )

    saved = []

    for _, row in best_configs.iterrows():

        model_name = row[
            "user_model_name"
        ]

        source_checkpoint = row[
            "checkpoint_path"
        ]

        if (
            source_checkpoint is None
            or not os.path.exists(
                source_checkpoint
            )
        ):
            raise FileNotFoundError(
                "Could not find checkpoint for "
                f"{model_name}: "
                f"{source_checkpoint}"
            )

        # =====================================
        # Model-specific output directory
        # =====================================

        model_dir = os.path.join(
            output_root,
            model_name,
        )

        os.makedirs(
            model_dir,
            exist_ok=True,
        )

        # =====================================
        # Copy winning weights
        # =====================================

        destination_checkpoint = (
            os.path.join(
                model_dir,
                "best.weights.h5",
            )
        )

        shutil.copy2(
            source_checkpoint,
            destination_checkpoint,
        )

        # =====================================
        # Save training parameters + selection
        # information
        # =====================================

        config = {
            "user_model_name":
                row["user_model_name"],

            "learning_rate":
                row["learning_rate"],

            "embedding_dim":
                row["embedding_dim"],

            "history_length":
                row["history_length"],

            "batch_size":
                row["batch_size"],

            "max_epochs":
                row["max_epochs"],

            "patience":
                row["patience"],

            "seed":
                row["seed"],
        }

        selection = {
            "best_epoch":
                row["best_epoch"],

            "best_val_top10":
                row["best_val_top10"],

            "best_val_loss":
                row["best_val_loss"],

            "epochs_trained":
                row["epochs_trained"],
        }

        metadata = {
            "config": {
                k: _json_safe(v)
                for k, v in config.items()
            },

            "selection": {
                k: _json_safe(v)
                for k, v in selection.items()
            },

            "checkpoint":
                "best.weights.h5",

            "original_checkpoint":
                source_checkpoint,
        }

        config_path = os.path.join(
            model_dir,
            "config.json",
        )

        with open(
            config_path,
            "w",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        saved.append({
            "user_model_name":
                model_name,

            "checkpoint_path":
                destination_checkpoint,

            "config_path":
                config_path,
        })

        print(
            f"\nSaved best {model_name}:"
        )

        print(
            f"  weights: {destination_checkpoint}"
        )

        print(
            f"  config:  {config_path}"
        )

    return pd.DataFrame(
        saved
    )
    
def get_best_configs_by_model(
    results_df,
):

    successful = results_df[
        results_df["status"] == "ok"
    ].copy()

    if len(successful) == 0:
        raise RuntimeError(
            "No successful runs found."
        )

    best_indices = (
        successful
        .groupby(
            "user_model_name"
        )["best_val_top10"]
        .idxmax()
    )

    best_configs = (
        successful
        .loc[best_indices]
        .sort_values(
            "user_model_name"
        )
        .reset_index(drop=True)
    )

    return best_configs

def generate_configs(
    user_model_name,
    search_space,
    base_config,
):

    keys = list(
        search_space.keys()
    )

    value_combinations = (
        itertools.product(
            *[
                search_space[key]
                for key in keys
            ]
        )
    )

    configs = []

    for values in value_combinations:

        config = base_config.copy()

        config.update(
            dict(
                zip(
                    keys,
                    values,
                )
            )
        )

        config[
            "user_model_name"
        ] = user_model_name

        configs.append(config)

    return configs
