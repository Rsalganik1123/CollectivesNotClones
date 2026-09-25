import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
import numpy as np 

from general_utils.data_loading import build_finetune_dataset

class MeanHistoryUserModel(tf.keras.Model): #Option 1 -- average over all embeddings in history 
    "Conceptually: u_i = \frac{1}{|H_i|} = \sum_{j \in H_i} e_j"
    def __init__(self, movie_ids_vocab, embedding_dim=64):
        super().__init__()

        self.movie_lookup = movie_ids_vocab

        self.movie_embedding = tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            embedding_dim,
            mask_zero=True,
        )

    def call(self, history):

        # history shape:
        # (batch_size, history_length)
        movie_indices = self.movie_lookup(history) #loads the movies in the user rating history

        # shape:
        # (batch_size, history_length, embedding_dim)
        movie_embeddings = self.movie_embedding(movie_indices) #collects their embeddings 

        # mask shape:
        # (batch_size, history_length)
        mask = self.movie_embedding.compute_mask(movie_indices) #computes a mask over all embeddings for efficent summing
        
        mask = tf.cast(mask, tf.float32) 
        
        # make it broadcast over embedding dimension
        mask = tf.expand_dims(mask, axis=-1)

        # shape:
        # (batch_size, embedding_dim)
        summed_embeddings = tf.reduce_sum(
            movie_embeddings * mask,
            axis=1,
        )

        # shape:
        # (batch_size, 1)
        counts = tf.reduce_sum(
            mask,
            axis=1,
        )

        # shape:
        # (batch_size, embedding_dim)
        user_embedding = (
            summed_embeddings /
            tf.maximum(counts, 1.0)
        )
        # ipdb.set_trace() 
        # tf.print(
        # "history:", tf.shape(history),
        # "movie_embeddings:", tf.shape(movie_embeddings),
        # "user_embedding:", tf.shape(user_embedding),
        # )

        return user_embedding 
    
class RatingWeightedUserModel(tf.keras.Model): #Option 2 -- *weighted* ratings-based average over all embeddings in history 
    '''u_i = \frac{\sum_{j \in H_i} r_{ij} e_j}{\sum_{j \in H_i}r_{ij}}'''
    def __init__(self, movie_ids_vocab, embedding_dim=64):
        super().__init__()

        self.movie_lookup = movie_ids_vocab

        self.movie_embedding = tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            embedding_dim,
            mask_zero=True,
        )

    def call(self, inputs):

        history = inputs["history"]
        ratings = inputs["ratings"]

        movie_indices = self.movie_lookup(history) #get all movies in the history 

        movie_embeddings = self.movie_embedding(movie_indices) #get the embeddings 

        mask = self.movie_embedding.compute_mask(movie_indices) #generate mask 

        mask = tf.cast(mask, tf.float32) #case to float just in case 

        weights = ratings * mask

        weights = tf.expand_dims(weights, axis=-1) #reshape the weights 

        weighted_sum = tf.reduce_sum(
            movie_embeddings * weights,
            axis=1,
        )

        weight_sum = tf.reduce_sum(
            weights,
            axis=1,
        )

        return weighted_sum / tf.maximum(weight_sum, 1e-8)

class AttentionHistoryUserModel(tf.keras.Model): #Option 3 -- attention-based history encoder 
    '''u_i = \sum_{j \in H_i} a_{ij}e_j where a_{ij} = softmax(g(e_j))'''
    def __init__(self, movie_ids_vocab, embedding_dim=64):
        super().__init__()

        self.movie_lookup = movie_ids_vocab

        self.movie_embedding = tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            embedding_dim,
            mask_zero=True,
        )

        self.attention_score = tf.keras.Sequential([
            tf.keras.layers.Dense(embedding_dim, activation="tanh"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, history):

        movie_indices = self.movie_lookup(history) #get movies

        movie_embeddings = self.movie_embedding(movie_indices) #get emb

        mask = self.movie_embedding.compute_mask(movie_indices) #gen mask 

        logits = self.attention_score(movie_embeddings) #initialize attention 

        logits = tf.squeeze(logits, axis=-1) #reshape 

        if mask is not None:
            large_negative = tf.constant(-1e9, dtype=logits.dtype)

            logits = tf.where(
                mask,
                logits,
                large_negative,
            )

        attention_weights = tf.nn.softmax(
            logits,
            axis=1,
        )

        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        return tf.reduce_sum(
            movie_embeddings * attention_weights,
            axis=1,
        )

class SASRecUserModel(tf.keras.Model): #Option 4 -- basically a sequential model... mimicing sasrec but not fully. 

    def __init__(
        self,
        movie_ids_vocab,
        embedding_dim=64,
        max_history_length=50,
        num_heads=2,
        ff_dim=128,
    ):
        super().__init__()

        self.movie_lookup = movie_ids_vocab

        self.movie_embedding = tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            embedding_dim,
            mask_zero=True,
        )

        self.position_embedding = tf.keras.layers.Embedding(
            max_history_length,
            embedding_dim,
        )

        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_dim // num_heads,
        )

        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embedding_dim),
        ])

        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(
        self,
        history,
        training=False,
    ):

        movie_indices = (
            self.movie_lookup(history)
        )

        x = self.movie_embedding(
            movie_indices
        )

        seq_len = tf.shape(
            history
        )[1]

        positions = tf.range(
            seq_len
        )

        position_embeddings = (
            self.position_embedding(
                positions
            )
        )

        x = (
            x
            + position_embeddings
        )

        # ========================================
        # Padding mask
        # ========================================

        padding_mask = tf.not_equal(
            movie_indices,
            0,
        )

        # shape:
        # (batch, 1, history_length)
        padding_mask = (
            padding_mask[:, tf.newaxis, :]
        )

        # ========================================
        # Causal mask
        # ========================================

        causal_mask = (
            tf.linalg.band_part(
                tf.ones(
                    (
                        seq_len,
                        seq_len,
                    ),
                    dtype=tf.bool,
                ),
                -1,
                0,
            )
        )

        # shape:
        # (1, history_length, history_length)
        causal_mask = (
            causal_mask[tf.newaxis, :, :]
        )

        # Broadcasts to:
        # (batch, history_length, history_length)
        attention_mask = (
            padding_mask
            & causal_mask
        )

        # ========================================
        # Transformer block
        # ========================================

        attention_output = (
            self.attention(
                query=x,
                value=x,
                key=x,
                attention_mask=
                    attention_mask,
                training=training,
            )
        )

        x = self.norm1(
            x + attention_output
        )

        ffn_output = (
            self.ffn(
                x,
                training=training,
            )
        )

        x = self.norm2(
            x + ffn_output
        )

        # Because histories are LEFT padded,
        # the final position is always the
        # newest real interaction.
        return x[:, -1, :]
    
class MovieLensModel(tfrs.Model):

    def __init__(
        self,
        user_model,
        movie_model,
        task,
        user_model_name,
    ):
        super().__init__()

        self.user_model = user_model
        self.movie_model = movie_model
        self.task = task

        self.user_model_name = (
            user_model_name
        )

    def compute_loss(
        self,
        features,
        training=False,
    ):

        # =====================================
        # Construct user embedding
        # =====================================

        if self.user_model_name == "basic":

            user_embeddings = (
                self.user_model(
                    features["user_id"],
                    training=training,
                )
            )

        elif self.user_model_name == "weighted":

            user_embeddings = (
                self.user_model(
                    {
                        "history":
                            features["history"],

                        "ratings":
                            features["ratings"],
                    },
                    training=training,
                )
            )

        elif self.user_model_name in [
            "mean",
            "attention",
            "sequential",
        ]:

            user_embeddings = (
                self.user_model(
                    features["history"],
                    training=training,
                )
            )

        else:

            raise ValueError(
                "Unknown user model: "
                f"{self.user_model_name}"
            )

        # =====================================
        # Construct target-item embedding
        # =====================================

        movie_embeddings = (
            self.movie_model(
                features["movie_id"],
                training=training,
            )
        )

        # =====================================
        # Retrieval objective
        # =====================================

        return self.task(
            user_embeddings,
            movie_embeddings,
        )

def get_user_model(
    u_model,
    user_ids_vocab,
    movie_ids_vocab,
    embedding_dim=64,
    max_history_length=20,
):

    if u_model == "basic":

        user_model = tf.keras.Sequential([
            user_ids_vocab,

            tf.keras.layers.Embedding(
                user_ids_vocab.vocabulary_size(),
                embedding_dim,
            ),
        ])

    elif u_model == "mean":

        user_model = MeanHistoryUserModel(
            movie_ids_vocab=movie_ids_vocab,
            embedding_dim=embedding_dim,
        )

    elif u_model == "weighted":

        user_model = RatingWeightedUserModel(
            movie_ids_vocab=movie_ids_vocab,
            embedding_dim=embedding_dim,
        )

    elif u_model == "attention":

        user_model = AttentionHistoryUserModel(
            movie_ids_vocab=movie_ids_vocab,
            embedding_dim=embedding_dim,
        )

    elif u_model == "sequential":

        user_model = SASRecUserModel(
            movie_ids_vocab=movie_ids_vocab,
            embedding_dim=embedding_dim,
            max_history_length=max_history_length,
        )

    else:

        raise ValueError(
            f"Unknown user model: {u_model}"
        )

    return user_model

def build_two_tower_from_pieces(
    user_model,
    movie_model,
    movies,
    user_model_name,
):

    task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=(
                movies
                .batch(128)
                .map(movie_model)
            )
        )
    )

    model = MovieLensModel(
        user_model=user_model,
        movie_model=movie_model,
        task=task,
        user_model_name=user_model_name,
    )

    return model

def build_finetune_model(
    *,
    baseline_model,
    config,
    movies,
    ratings_df_post,
    user_ids_vocab,
    movie_ids_vocab,
):
    """
    Rebuild the same architecture and initialize it
    from the baseline model weights.
    """

    user_model = get_user_model(
        u_model=config["user_model_name"],
        user_ids_vocab=user_ids_vocab,
        movie_ids_vocab=movie_ids_vocab,
        embedding_dim=int(
            config["embedding_dim"]
        ),
        max_history_length=int(
            config["history_length"]
        ),
    )

    movie_model = tf.keras.Sequential([
        movie_ids_vocab,
        tf.keras.layers.Embedding(
            movie_ids_vocab.vocabulary_size(),
            int(config["embedding_dim"]),
        ),
    ])

    post_model = build_two_tower_from_pieces(
        user_model=user_model,
        movie_model=movie_model,
        movies=movies,
        user_model_name=
            config["user_model_name"],
    )

    post_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adagrad(
            learning_rate=float(
                config["learning_rate"]
            )
        )
    )

    # Build model variables
    train_ds = build_finetune_dataset(
        ratings_df=ratings_df_post,
        history_length=int(
            config["history_length"]
        ),
        batch_size=int(
            config["batch_size"]
        ),
        seed=int(
            config["seed"]
        ),
    )

    sample_batch = next(
        iter(train_ds.take(1))
    )

    _ = post_model.compute_loss(
        sample_batch,
        training=False,
    )

    # Warm start from baseline
    post_model.set_weights(
        baseline_model.get_weights()
    )

    return (
        post_model,
        train_ds,
    )

def encode_users_from_state(
    *,
    model,
    user_model_name,
    user_ids,
    histories,
    history_ratings,
    batch_size=256,
):
    all_embeddings = []

    for start in range(
        0,
        len(user_ids),
        batch_size,
    ):

        end = start + batch_size

        batch_users = tf.constant(
            user_ids[start:end]
        )

        batch_histories = tf.constant(
            histories[start:end]
        )

        batch_ratings = tf.constant(
            history_ratings[start:end],
            dtype=tf.float32,
        )

        if user_model_name == "basic":

            emb = model.user_model(
                batch_users,
                training=False,
            )

        elif user_model_name == "weighted":

            emb = model.user_model(
                {
                    "history":
                        batch_histories,

                    "ratings":
                        batch_ratings,
                },
                training=False,
            )

        else:

            emb = model.user_model(
                batch_histories,
                training=False,
            )

        all_embeddings.append(
            emb.numpy()
        )

    return np.concatenate(
        all_embeddings,
        axis=0,
    )

def get_all_item_embeddings(
    model,
    movies_df,
):

    movie_ids = movies_df["movie_id"].values

    V = model.movie_model(
        tf.constant(movie_ids),
        training=False,
    ).numpy()

    movie_id_to_idx = {
        movie_id: idx
        for idx, movie_id in enumerate(movie_ids)
    }
    movie_idx_to_id = {
        idx: movie_id
        for  movie_id, idx in movie_id_to_idx.items()
    }

    return V, movie_ids, movie_id_to_idx, movie_idx_to_id
