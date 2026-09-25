from real.world import BASE_TT_CONFIG
from general_utils.data_loading import load_movielens_for_tuning, get_tf_datasets
from real.TT_utils import build_vocabularies,generate_configs, get_best_configs_by_model, save_best_checkpoints_by_model
from models.two_tower import get_user_model, build_two_tower_from_pieces


import numpy as np 
import pandas as pd 
import os 
import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger


def train_model(
    config, 
    movies, 
    train_ds, 
    val_ds, 
    user_ids_vocab, 
    movie_ids_vocab, 
    checkpoint_path = None 
): 
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
    
    model.compile( #Note I'm using legacy cause I run on M1. switch to tf.keras.optimizers.Adagrad if you aren't :) 
        optimizer=tf.keras.optimizers.legacy.Adagrad(
            learning_rate=config["learning_rate"]
        )
    )
    
    # ----------------------------------------
    # Early stopping -- We want this to make sure the model isn't overfitting 
    # ----------------------------------------

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=VAL_METRIC,
        mode="max",
        patience=config["patience"],
        restore_best_weights=True,
    )

    callbacks = [
        early_stopping,
        # WandbMetricsLogger( #log checkpoints 
        #     log_freq="epoch"
        # ),
    ]
    
    if checkpoint_path is not None:

        checkpoint_dir = os.path.dirname(
            checkpoint_path
        )

        if checkpoint_dir:
            os.makedirs(
                checkpoint_dir,
                exist_ok=True,
            )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=VAL_METRIC,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )

        callbacks.append(
            checkpoint
        )
    
    
    #TRAIN GHE MODEL 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["max_epochs"],
        callbacks=callbacks,
        verbose=config.get("verbose", 1),
    )
    

    #GET BEST RESULTS 
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)
        
    metric_values = history.history[
        VAL_METRIC
    ]

    best_idx = int(
        np.argmax(metric_values)
    )

    best_result = {
        **config,

        "best_epoch":
            best_idx + 1,

        "best_val_top10":
            float(metric_values[best_idx]),

        "best_val_loss":
            float(
                history.history[
                    "val_loss"
                ][best_idx]
            ),

        "epochs_trained":
            len(metric_values),

        "early_stopped":
            len(metric_values)
            < config["max_epochs"],

        "checkpoint_path":
            checkpoint_path,
    }
    return model, history, best_result
    
def run_config(
    config, 
    movies, 
    ratings_df, 
    user_ids_vocab, 
    movie_ids_vocab
): 
    run_name = (
        f"{config['user_model_name']}"
        f"_lr{config['learning_rate']}"
        f"_d{config['embedding_dim']}"
        f"_h{config['history_length']}"
        f"_seed{config['seed']}"
    )
    checkpoint_path = os.path.join(
        "checkpoints",
        "sweep_runs",
        f"{run_name}.weights.h5",
    )
    
    print("\n")
    print("=" * 70)
    print(
        f"MODEL: {config['user_model_name']} | "
        f"LR: {config['learning_rate']} | "
        f"DIM: {config['embedding_dim']} | "
        f"HISTORY: {config['history_length']} | "
        f"SEED: {config['seed']}"
    )
    print("=" * 70)
    
    train_ds, val_ds, test_ds = get_tf_datasets(
            ratings_df=ratings_df,
            history_length=config["history_length"],
            batch_size=config["batch_size"],
            seed=config["seed"],
        )
    
    model, history, result = train_model(
            config=config,
            movies=movies,
            train_ds=train_ds,
            val_ds=val_ds,
            user_ids_vocab=user_ids_vocab,
            movie_ids_vocab=movie_ids_vocab,
            checkpoint_path=checkpoint_path,
        )

    return (
            model,
            history,
            result,
            test_ds,
        )
    
    return 0 

def run_sweep(
    movies, 
    ratings_df, 
    user_ids_vocab, 
    movie_ids_vocab, 
    search_spaces, 
    base_config, 
    results_path): 
    
    
    all_results = []
    for user_model_name, search_sapce in search_spaces.items(): 
        configs = generate_configs(user_model_name, search_sapce, base_config)

        print("\n\n")
        print("#" * 70)
        print(
            f"STARTING {user_model_name.upper()}"
        )
        print(
            f"{len(configs)} configurations"
        )
        print("#" * 70)
        
        for run_idx, config in enumerate(configs, start=1): 
            try:

                _, _, result, _ = (
                    run_config(
                        config=config,
                        movies=movies,
                        ratings_df=ratings_df,
                        user_ids_vocab=
                            user_ids_vocab,
                        movie_ids_vocab=
                            movie_ids_vocab,
                    )
                )

                result["status"] = "ok"
                result["error"] = None
            except Exception as exc:

                print(
                    "\nFAILED CONFIG:"
                )

                print(config)
                print(repr(exc))

                result = {
                    **config,
                    "status": "failed",
                    "error": repr(exc),
                    "best_epoch": np.nan,
                    "best_val_top10": np.nan,
                    "best_val_loss": np.nan,
                    "epochs_trained": np.nan,
                }
            all_results.append(result)
    return pd.DataFrame(all_results)


#NOTE: I AM USING THESE FOR EARLY STOPPING
VAL_METRIC = (
    "val_factorized_top_k/"
    "top_10_categorical_accuracy"
)

#NOTE: THIS IS WHERE I STORE THE HYPERPARAMS TO SWEEP 
SEARCH_SPACES = {
    # "mean": {
    #     "learning_rate": [
    #         0.001,
    #         0.01,
    #         0.05,
    #         0.1,
    #     ],
    #     "embedding_dim": [
    #         28,
    #         64,
    #         128,
    #     ],
    #     "history_length": [
    #         None,
    #     ],
    # },
    # "attention": {
    #     "learning_rate": [
    #         0.0001,
    #         0.01,
    #         0.05,
    #         0.1,
            
    #     ],
    #     "embedding_dim": [
    #         64,
    #         128,
    #     ],
    #     "history_length": [
            
    #         None,
            
    #     ],
    # },
    "sequential": {
        "learning_rate": [
            0.001,
            0.01,
            0.05,
            0.1,
        ],
        "embedding_dim": [
            128,
        ],
        "history_length": [
            20,
            50,
            100
        ],
    },
}
      
if __name__ == "__main__": 
   
    # ========================================
    # Load data ONCE
    # ========================================

    (
        movies,
        ratings,
        movies_df,
        ratings_df,
    ) = load_movielens_for_tuning()

    # ========================================
    # Build vocabularies ONCE
    # ========================================

    (
        user_ids_vocab,
        movie_ids_vocab,
    ) = build_vocabularies(
        movies,
        ratings,
    )
    
    #Just making sure that our padding is a character that isn't being used 
    assert movie_ids_vocab(tf.constant("")).numpy() == 0

    # ========================================
    # Sweep
    # ========================================

    results_df = run_sweep(
        movies=movies,
        ratings_df=ratings_df,
        user_ids_vocab=user_ids_vocab,
        movie_ids_vocab=movie_ids_vocab,
        search_spaces=SEARCH_SPACES,
        base_config=BASE_TT_CONFIG,
        results_path=(
            "two_tower_hyperparameter_sweep.csv"
        ),
    )


    # ========================================
    # Print rankings
    # ========================================

    successful = results_df[
        results_df["status"] == "ok"
    ].copy()

    print("\n\n")
    print("=" * 70)
    print("BEST CONFIGURATIONS")
    print("=" * 70)

    print(
        successful[
            [
                "user_model_name",
                "learning_rate",
                "embedding_dim",
                "history_length",
                "best_epoch",
                "best_val_top10",
                "best_val_loss",
            ]
        ]
        .sort_values(
            [
                "user_model_name",
                "best_val_top10",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )
    
    # ========================================
    # Best configuration PER architecture
    # ========================================

    best_configs = get_best_configs_by_model(
        results_df
    )

    print("\n\n")
    print("=" * 70)
    print("BEST CONFIGURATION PER MODEL")
    print("=" * 70)

    print(
        best_configs[
            [
                "user_model_name",
                "learning_rate",
                "embedding_dim",
                "history_length",
                "best_epoch",
                "best_val_top10",
                "best_val_loss",
                "checkpoint_path",
            ]
        ]
        .sort_values(
            "user_model_name"
        )
        .to_string(
            index=False
        )
    )

    # ========================================
    # Save clean copies of each winner
    # together with configuration metadata
    # ========================================

    saved_models = save_best_checkpoints_by_model(
        results_df,
        output_root="checkpoints/best",
    )

    saved_models.to_csv(
        "checkpoints/best_models.csv",
        index=False,
    )
