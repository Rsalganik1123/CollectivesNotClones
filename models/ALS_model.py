import numpy as np 
from tqdm import tqdm 
from scipy import sparse
import wandb 

class ALS:
    def __init__(self, R_train, n_factors=20, reg=0.1, seed=42, log=True):
        self.n_factors = n_factors
        self.reg = reg
        self.seed = seed
        self.log = log
        self.n_users, self.n_items = R_train.shape

        rng = np.random.default_rng(seed)

        self.user_factors = 0.01 * rng.standard_normal(
            (self.n_users, self.n_factors)
        )
        self.item_factors = 0.01 * rng.standard_normal(
            (self.n_items, self.n_factors)
        )

        self.history = {
            "iteration": [],
            "train_rmse": [],
            "train_loss": [],
        }
    def solve_embedding(self, X_obs, r_obs):
        I = np.eye(self.n_factors)

        A = X_obs.T @ X_obs + self.reg * I
        b = X_obs.T @ r_obs

        return np.linalg.solve(A, b)

    def update_users(self, R_train):
        R_csr = R_train.tocsr()

        for u in range(self.n_users):
            item_ids = R_csr[u].indices
            ratings = R_csr[u].data

            if len(item_ids) == 0:
                continue

            V_obs = self.item_factors[item_ids]
            self.user_factors[u] = self.solve_embedding(V_obs, ratings)

    def update_items(self, R_train):
        R_csc = R_train.tocsc()

        for i in range(self.n_items):
            user_ids = R_csc[:, i].indices
            ratings = R_csc[:, i].data

            if len(user_ids) == 0:
                continue

            U_obs = self.user_factors[user_ids]
            self.item_factors[i] = self.solve_embedding(U_obs, ratings)

    def fit(
        self,
        R_train,
        n_iters=100,
        R_val=None,
        patience=5,
        update_users=True,
        update_items=True,
        version='bsln'
    ):
        best_val = np.inf
        best_user_factors = None
        best_item_factors = None
        bad_epochs = 0
        
        self.history = {
            "iteration": [],
            "train_rmse": [],
            "train_loss": [],
            "val_rmse": [],
        }
        
        
        for it in range(n_iters):

            if update_users:
                self.update_users(R_train)

            if update_items:
                self.update_items(R_train)

            train_rmse = self.rmse(R_train)

            train_loss = self.compute_loss(R_train)

            self.history["iteration"].append(it + 1)
            self.history["train_rmse"].append(train_rmse)
            self.history["train_loss"].append(train_loss)
            
            
            if R_val is not None:
                val_rmse = self.rmse(R_val)
                self.history["val_rmse"].append(val_rmse)
                # Log training metrics to wandb
                if self.log:
                    wandb.log({
                        f"{version}_iteration": it + 1,
                        f"{version}_train_rmse": train_rmse,
                        f"{version}_val_rmse": val_rmse,
                    })

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_user_factors = self.user_factors.copy()
                    best_item_factors = self.item_factors.copy()
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if bad_epochs >= patience:
                    print(f"Early stopping at iteration {it + 1}")
                    break

                # pbar.update(1)
                # pbar.set_postfix({f"{version}_train_rmse": train_rmse, f"{version}_val_rmse": val_rmse if R_val is not None else None})

        if R_val is not None and best_user_factors is not None:
            self.user_factors = best_user_factors
            self.item_factors = best_item_factors
            
    def rmse(self, R):
        se = 0.0
        n = 0

        R = R.tocsr()

        for u in range(R.shape[0]):
            item_ids = R[u].indices
            ratings = R[u].data

            if len(item_ids) == 0:
                continue

            preds = self.user_factors[u] @ self.item_factors[item_ids].T
            se += np.sum((ratings - preds) ** 2)
            n += len(ratings)

        return np.sqrt(se / n)

    def predict(self, user_id, item_id):
        return self.user_factors[user_id] @ self.item_factors[item_id]

    def predict_all(self):
        return self.user_factors @ self.item_factors.T

    def recommend(self, user_id, R_train, k=10):
        scores = self.user_factors[user_id] @ self.item_factors.T

        seen_items = R_train[user_id].indices
        scores[seen_items] = -np.inf

        recs = np.argsort(scores)[::-1][:k]
        return recs, scores[recs]
    
    def recommend_all(self, R_train=None, k=10):
        scores = self.user_factors @ self.item_factors.T

        if R_train is not None:
            R_train = R_train.tocsr()
            for u in range(R_train.shape[0]):
                seen_items = R_train[u].indices
                scores[u, seen_items] = -np.inf

        recs = np.argsort(scores, axis=1)[:, ::-1][:, :k]
        rec_scores = scores[np.arange(scores.shape[0])[:, None], recs]

        return recs, rec_scores

    def compute_loss(self, R):
        loss = 0.0

        for u in range(R.shape[0]):
            item_ids = R[u].indices
            ratings = R[u].data

            if len(item_ids) == 0:
                continue

            preds = self.user_factors[u] @ self.item_factors[item_ids].T
            loss += np.sum((ratings - preds) ** 2)

        loss += self.reg * (
            np.sum(self.user_factors ** 2)
            + np.sum(self.item_factors ** 2)
        )

        return loss


def fit_als(
    params,
    ratings,
    version,
    R_val=None,
    initial_user_factors=None,
    initial_item_factors=None,
    n_iters = None,
):

    ratings = ratings.tocsr()

    model = ALS(
        ratings,
        n_factors=params.latent_dim,
        reg=params.als_reg,
        seed=params.als_seed,
        log=False,
    )

    #Warm start option 
    if initial_user_factors is not None:
        model.user_factors = initial_user_factors.copy()

    if initial_item_factors is not None:
        model.item_factors = initial_item_factors.copy()

    # Default to normal experiment setting
    if n_iters is None:
        n_iters = params.als_iters

    model.fit(
        ratings,
        R_val=R_val,
        n_iters=n_iters,
        patience=params.als_patience,
        update_users=True,
        update_items=True,
        version=version,
    )

    return (
        model,
        model.user_factors.copy(),
        model.item_factors.copy(),
    )
    
