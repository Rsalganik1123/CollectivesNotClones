from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
import ipdb 
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



from general_utils.ratings import generate_sparse_ratings
from general_utils.reproducibility import _condition_seed
from general_utils.heuristics import generate_strategy_items_and_ratigns

from synthetic.world import SyntheticWorld, SyntheticParams

from models.ALS_model import ALS , fit_als

from plot_utils.bot_farm_plots import plot_success_results, plot_diminishing_returns, plot_epsilon_results, plot_monotonocity_results

'''
run_botfarm_exp(seed)
│
├── generate baseline/synthetic world 
│
├── for each strategy:
│   │
│   ├── generate u_C (coalition vector)
│   │
│   ├── compute_botfarm_geometry()
│   │
│   ├── test_success_formula_accuracy() -- Prop1 
│   │
│   └── test_epsilon() 
│
├── concatenate alpha results
│
├── test_monotonicity(alpha_results) 
│
└── test_diminishing_returns(alpha_results) 
'''


def find_optimal_uc_theory(
    A,
    population_mean,
    v_target,
    coalition_rating,
    coalition_size,
    u_init=None,
    norm_bound=None,
):
    """
    Numerically find u_C that maximizes the one-step theoretical
    population success:

        S(u_C)
        =
        s * c(u_C) * Delta(u_C)
        / (1 + s * q(u_C))

    where
        c(u)     = population_mean^T A^{-1} u
        Delta(u) = coalition_rating - u^T v_target
        q(u)     = u^T A^{-1} u
    """
    A = np.asarray(A, dtype=float)
    population_mean = np.asarray(
        population_mean,
        dtype=float,
    ).reshape(1, -1)
    v_target = np.asarray(
        v_target,
        dtype=float,
    ) #.reshape(-1)

    d = A.shape[0]
    s = coalition_size

    if u_init is None:
        u_init = population_mean.copy()
    else:
        u_init = np.asarray(
            u_init,
            dtype=float,
        ).reshape(-1)

    # More stable than explicitly computing inverse.
    A_inv = np.linalg.inv(A)

    def success(u):
        c = (
            population_mean
            @ A_inv
            @ u
        )

        delta = (
            coalition_rating
            - u @ v_target
        )

        q = (
            u
            @ A_inv
            @ u
        )

        return (
            s * c * delta
            / (1.0 + s * q)
        )

    def objective(u):
        return -success(u)

    constraints = []

    # Optional: don't allow optimizer to invent a
    # ridiculous latent vector with enormous norm.
    if norm_bound is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda u: (
                norm_bound**2
                - np.dot(u, u)
            ),
        })

    result = minimize(
        objective,
        x0=u_init,
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    u_star = result.x

    return {
        "u_star": u_star,
        "S_star": success(u_star),
        "success": result.success,
        "message": result.message,
        "result": result,
    }

def test_diminishing_returns(
    results,
    params,
):
    output = []

    for (seed, strategy), g in results.groupby(
        ["seed", "strategy"]
    ):
        g = g.sort_values("alpha").copy()

        g["d2S_dalpha2_exact"] = np.gradient(
            g["dS_dalpha_exact"].to_numpy(),
            g["alpha"].to_numpy(),
            edge_order=2,
        )

        g["d2S_dalpha2_theory"] = (
            -2
            * params.n_users**2
            * g["c"]
            * g["q"]
            * g["delta"]
            / (
                1
                + g["alpha"]
                * params.n_users
                * g["q"]
            )**3
        )

        # MUST be inside loop
        output.append(g)

    return pd.concat(
        output,
        ignore_index=True,
    )
    
def test_monotonicity(results, params):
    output = []

    for (seed, strategy), g in results.groupby(
        ["seed", "strategy"]
    ):
        g = g.sort_values("alpha").copy()

        alpha = g["alpha"].to_numpy()
        S_exact = g["S_exact"].to_numpy()

        g["dS_dalpha_exact"] = np.gradient(
            S_exact,
            alpha,
            edge_order=2,
        )

        g["dS_dalpha_theory"] = (
            params.n_users
            * g["c"]
            * g["delta"]
            / (
                1
                + g["alpha"]
                * params.n_users
                * g["q"]
            )**2
        )

        output.append(g)

    return pd.concat(
        output,
        ignore_index=True,
    )
    
def test_epsilon(uc_results,
    m,
    c,
    q,
    delta,
    seed,
    strategy,
    n_epsilons=20,
):
    """
    Compare theoretical and empirical minimum coalition
    proportions required to achieve different epsilon-success levels.
    """

    S_max = (
        m * c * delta
        / (1 + m * q)
    )

    epsilons = np.linspace(
        0.05 * S_max,
        0.95 * S_max,\
        n_epsilons
    )

    rows = []

    for epsilon in epsilons:

        denominator = (
            m * (c * delta - epsilon * q)
        )

        if denominator <= 0:
            alpha_theory = np.nan
        else:
            alpha_theory = epsilon / denominator

        successful_rows = uc_results[
            uc_results["S_exact"] > epsilon
        ]

        if len(successful_rows) == 0:
            alpha_empirical = np.nan
        else:
            alpha_empirical = (
                successful_rows["alpha"].min()
            )

        rows.append({
            "epsilon": epsilon,
            "alpha_min_theory": alpha_theory,
            "alpha_min_exact": alpha_empirical,
            'c': c, 
            'q': q, 
            'delta': delta, 
            'seed': seed, 
            'strategy': strategy
        })
    return pd.DataFrame(rows)  

def test_success_conditions(): 
    return 0 

def test_success_formula_accuracy(params, strategy, seed, A, b, v_target_closed_form, u_bar, coalition_vector, c, q, delta, m, alphas): 
    u_C = coalition_vector 
    all_alpha_results = [] 
    for alpha in alphas: 
        s = alpha * m

        # -----------------------------------
        # Closed-form theoretical prediction
        # -----------------------------------
        theory_success = (
            s * c * delta
            / (1.0 + s * q)
        )
        # -----------------------------------
        # Exact UC ALS item update
        # -----------------------------------
        A_post = A + s * np.outer(u_C, u_C)
        b_post = b + s * params.target_rating * u_C

        v_target_post = np.linalg.solve(
            A_post,
            b_post
        )
        
        # EXACT success quantity 
        exact_success = (
            u_bar
            @ (
                v_target_post
                - v_target_closed_form
            )
        )

        all_alpha_results.append({
            'strategy': strategy, 
            'seed': seed, 
            "alpha": alpha,

            "c": c,
            "q": q,
            "delta": delta,

            "alignment_condition": c > 0,
            "feasibility_condition": delta > 0,

            "S_theory": theory_success,
            "S_exact": exact_success,
            
        }) 

    return pd.DataFrame(all_alpha_results)  

def calculate_success(params, strategy, seed, A, b, v_target_closed_form, u_bar, coalition_vector, c, q, delta, m, alpha): 
    u_C = coalition_vector 
    
    s = alpha * m

    # -----------------------------------
    # Closed-form theoretical prediction
    # -----------------------------------
    theory_success = (
        s * c * delta
        / (1.0 + s * q)
    )
    # -----------------------------------
    # Exact UC ALS item update
    # -----------------------------------
    A_post = A + s * np.outer(u_C, u_C)
    b_post = b + s * params.target_rating * u_C

    v_target_post = np.linalg.solve(
        A_post,
        b_post
    )
    
    # EXACT success quantity 
    exact_success = (
        u_bar
        @ (
            v_target_post
            - v_target_closed_form
        )
    )

    return {
        'strategy': strategy, 
        'seed': seed, 
        "alpha": alpha,

        "c": c,
        "q": q,
        "delta": delta,

        "alignment_condition": c > 0,
        "feasibility_condition": delta > 0,

        "S_theory": theory_success,
        "S_exact": exact_success,
        
    }

def compute_baseline_target_geometry(
    params,
    target_item,
    baseline_ratings,
    observation_mask,
    baseline_U,
    baseline_V
):
    """
    Compute quantities that depend ONLY on the baseline world.
    """

    # m, d = baseline_U.shape

    # # Users who already rated target
    # target_observers = np.where(
    #     observation_mask[:, target_item]
    # )[0]

    # U_target = baseline_U[
    #     target_observers
    # ]

    # r_target = np.asarray(
    #     baseline_ratings[
    #         target_observers,
    #         target_item
    #     ]
    # ) #.reshape(-1)

    # A = (
    #     U_target.T @ U_target
    #     + params.als_reg * np.eye(d)
    # )

    # b = (
    #     U_target.T @ r_target
    # )

    # # This is the target vector implied exactly
    # # by the baseline item-update normal equations
    # v_target = np.linalg.solve(
    #     A,
    #     b,
    # )

    # u_bar = baseline_U.mean(
    #     axis=0
    # )
    m, d = baseline_U.shape
    
    v_target = baseline_V[target_item]


    # Existing users who rated the target item
    target_observers = np.where(
        observation_mask[:, target_item]
    )[0]

    U_target = baseline_U[target_observers]

    r_target = baseline_ratings[target_observers, target_item]
    #.reshape(-1)

    A = (
        U_target.T @ U_target
        + params.als_reg * np.eye(d)
    )

    b = (
        U_target.T @ r_target
    ).reshape(-1)

    v_target_closed_form = np.linalg.solve(A, b)
        
    # Average existing user.
    u_bar = baseline_U.mean(axis=0)

    return A, b, v_target_closed_form, u_bar, m 

def compute_uc_geometry(
    A,
    v_target,
    u_bar,
    u_C,
    target_rating,
):
    """
    Geometry induced by a particular candidate u_C.
    """

    A_inv_uC = np.linalg.solve(
        A,
        u_C,
    )

    c = (
        u_bar
        @ A_inv_uC
    )

    q = (
        u_C
        @ A_inv_uC
    )

    delta = (
        target_rating
        - u_C @ v_target
    )

    return c, q, delta

def compute_strategy_vector(
        baseline_item_latents,
        strategy_items,
        strategy_ratings,
        reg,
    ):
    V = baseline_item_latents[strategy_items]
    r = np.asarray(strategy_ratings, dtype=float)

    A = (
        V.T @ V
        + reg * np.eye(V.shape[1])
    )

    b = V.T @ r

    u_c_star = np.linalg.solve(A, b)

    return u_c_star

def compute_botfarm_geometry(params,
    target_item, 
    target_rating,
    baseline_ratings,
    observation_mask,
    baseline_U,
    baseline_V,
    coalition_vector): 
    
    m, d = baseline_U.shape
    u_C = coalition_vector
    v_target = baseline_V[target_item]


    # Existing users who rated the target item
    target_observers = np.where(
        observation_mask[:, target_item]
    )[0]

    U_target = baseline_U[target_observers]

    r_target = baseline_ratings[target_observers, target_item]
    #.reshape(-1)

    A = (
        U_target.T @ U_target
        + params.als_reg * np.eye(d)
    )

    b = (
        U_target.T @ r_target
    ).reshape(-1)

    v_target_closed_form = np.linalg.solve(A, b)
        
    # Average existing user.
    u_bar = baseline_U.mean(axis=0)

    # Fixed geometry
    A_inv_uC = np.linalg.solve(A, u_C)

    c = u_bar @ A_inv_uC
    q = u_C @ A_inv_uC
    delta = target_rating - u_C @ v_target_closed_form

    return A, b, v_target_closed_form, u_bar, coalition_vector, c, q, delta, m

def run_botfarm_exp(
    params,
    seed,
    strategies=(
        "mainstream_mimic",
        "target_neighborhood",
        "target_supporter_mimic",
    ),
    alphas = np.linspace(
        0.001,
        1.0,
        1000,
    )
): 
    world = SyntheticWorld(
        params,
        seed=seed,
    )

    (
        true_items,
        item_genres,
        item_popularity,
        true_users,
        coalition_order,
    ) = world.generate_synthetic_environment()

    print("seed:", seed)

    baseline_seed = _condition_seed(
        seed=seed,
        alpha_index=0,
        p0_index=0,
        stream=101,
    )

    baseline_rng = np.random.default_rng(
        baseline_seed
    )

    baseline_ratings, observation_mask = (
        generate_sparse_ratings(
            true_users=true_users,
            true_items=true_items,
            item_popularity=item_popularity,
            rng=baseline_rng,
            params=world.params,
        )
    )

    _, baseline_U, baseline_V = fit_als(
        ratings=baseline_ratings,
        version="baseline",
        params=params,
    )    
    
    # ------------------------------------------------
    # ONE SHARED TARGET
    # ------------------------------------------------

    target_item = int(
        params.target_item
    )  

    # ------------------------------------------------
    # STRATEGY LOOP
    # ------------------------------------------------

    strategy_seed = _condition_seed(
        seed=seed,
        alpha_index=0,
        p0_index=0,
        stream=202,
    )
    
    all_epsilon_results, all_alpha_results = [], [] 
    for strategy in strategies:

        strategy_rng = np.random.default_rng(
            strategy_seed
        )

        (
        strategic_items,
        strategic_ratings,
            ) = generate_strategy_items_and_ratigns(
                strategy=strategy,
                item_popularity=item_popularity,
                item_latents=baseline_V,
                user_latents=baseline_U,
                baseline_ratings=baseline_ratings,
                budget=10,
                params=params,
                rng=strategy_rng,
            )

        strategic_items = np.asarray(
            strategic_items,
            dtype=int,
        )

        
        population_mean = np.mean(baseline_U, axis=0)
        
        strategy_uc = compute_strategy_vector(
                baseline_V,
                strategic_items,
                strategic_ratings,
                reg=params.als_reg,
            )
        
        A, b, v_target, u_bar, m  = (
                compute_baseline_target_geometry(
                    params=params,
                    target_item=target_item,
                    baseline_ratings=baseline_ratings,
                    observation_mask=observation_mask,
                    baseline_U=baseline_U,
                    baseline_V=baseline_V
                )
            )
    
        c, q, delta = compute_uc_geometry(
                            A = A,
                            v_target=v_target,
                            u_bar = u_bar,
                            u_C= strategy_uc,
                            target_rating=params.target_rating) 
        
        
        strategy_alpha_results = [] 
        for alpha in alphas:
            coalition_size = int(baseline_U.shape[0] * alpha) 
            
            # A, b, v_target, u_bar, m  = (
            #     compute_baseline_target_geometry(
            #         params=params,
            #         target_item=target_item,
            #         baseline_ratings=baseline_ratings,
            #         observation_mask=observation_mask,
            #         baseline_U=baseline_U,
            #         baseline_V=baseline_V
            #     )
            # )

            # optimal_res = find_optimal_uc_theory(
            #     A = A, 
            #     population_mean=population_mean, 
            #     v_target = baseline_V[target_item], 
            #     coalition_rating=params.target_rating, 
            #     coalition_size=coalition_size, 
            #     u_init = strategy_uc
            #     ) 
            
            # c, q, delta = compute_uc_geometry(
            #                 A = A,
            #                 v_target=v_target,
            #                 u_bar = u_bar,
            #                 u_C= optimal_res['u_star'],
            #                 target_rating=params.target_rating) 
            strategy_alpha_results.append(calculate_success(params, strategy, seed, A, b, v_target, u_bar, strategy_uc, c, q, delta, m, alpha))
        
        
            # alpha_results = test_success_formula_accuracy(params, strategy, seed, A, b, v_target_closed_form, u_bar, coalition_vector, c, q, delta, m, alphas)
        alpha_results = pd.DataFrame(strategy_alpha_results)
            
        alpha_results['baseline_target_error'] = np.linalg.norm(
            baseline_V[target_item]
            - v_target
        )
        epsilon_results = test_epsilon(
            uc_results=alpha_results,
            m=m,
            c=c,
            q=q,
            delta=delta,
            seed=seed,
            strategy=strategy, 
            n_epsilons=20,
        )

        all_epsilon_results.append(epsilon_results)
        all_alpha_results.append(alpha_results)        

    epsilon_df = pd.concat(all_epsilon_results)
    epsilon_df['epsilon_error'] = np.abs(epsilon_df.alpha_min_theory - epsilon_df.alpha_min_exact)
    alpha_df = pd.concat(all_alpha_results)
    alpha_df['success_calculation_error'] = np.abs(alpha_df.S_exact - alpha_df.S_theory)
    alpha_df= test_monotonicity(alpha_df, params)
    alpha_df =test_diminishing_returns(alpha_df, params)
    
    
    plt.show()
    return alpha_df, epsilon_df, pd.DataFrame()


if __name__ == "__main__":

    params = SyntheticParams()

    strategies = (
        "mainstream_mimic",
        "target_neighborhood",
        "target_supporter_mimic",
    )

    all_alpha_results, all_epsilon_results, all_sign_results = [], [], [] 

    for seed in range(5):

        alpha_results, epsilon_results, sign_results = (
            run_botfarm_exp(
                params=params,
                seed=seed,
            )
        )
        all_alpha_results.append(alpha_results) 
        all_epsilon_results.append(epsilon_results)
        all_sign_results.append(sign_results)

    alpha_results_df = pd.concat(all_alpha_results)
    epsilon_results_df = pd.concat(all_epsilon_results)
    sign_results_df = pd.concat(all_sign_results)
 
    plot_success_results(alpha_results_df) 
    plot_epsilon_results(epsilon_results_df)
    plot_monotonocity_results(alpha_results_df)
    plot_diminishing_returns(alpha_results_df)

    
    print("****error stats***")
    print(f"success_error:\n{alpha_results_df.success_calculation_error.agg(['mean', 'std'])}")
    print(f"epsilon_error:\n{epsilon_results_df.epsilon_error.agg(['mean', 'std'])}")
    