import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix

import paths
from general_utils.reproducibility import _condition_seed
from real.world import MLParams, params_from_dataframe
from real.cluster_construction import (
    __cluster_users as cluster_users,
    sample_low_dispersion_order,
    sample_med_dispersion_order,
    sample_high_dispersion_order,
)
from general_utils.CA_stats import coalition_moments, get_initial_coalition_stats
from general_utils.CA_stats import estimate_realized_coordination_and_cohesion
from models.ALS_model import fit_als
from general_utils.heuristics import generate_strategy_items_and_ratigns, compute_strategy_vector
from general_utils.ratings import apply_strategic_action_to_ratings, enforce_target_item_max_rating
from general_utils.math_helpers import align_factorization_to_baseline
from general_utils.success_metrics import run_all_ALS_success_metrics, recall_at_k
from general_utils.data_loading import load_real_ratings, split_movielens_per_user, build_sparse_rating_matrix
from real.world import organize_coalition_orderings
# The theory / diagnostics / optimal-solver helpers do not depend on where the
# ratings come from, so we reuse them from the synthetic experiment as-is.
from synthetic_ALS_CA import (
    evaluate_optimal_uc_empirically,
    target_normal_equations_without_coalition,
    construct_coordination,
    theory_target_update,
    run_theory_validation,
    run_als_diagnostics,
    run_strategy_diagnostics,
)
from synthetic.optimal_solvers import find_optimal_uc_collective

'''
Experiment flow (identical to synthetic_ALS_CA.py except for step 1-2)

REAL MovieLens ratings R0                         <-- replaces generate_sparse_ratings
      |
fit baseline ALS  ->  U0, V0                      <-- fit once, reused for every seed/condition
      |
choose coalition (alpha) with dispersion p0 in {low, med, high}
      |
strategy (items + ratings) -> strategy vector u_C
      |
for (beta, gamma):  desired coalition latents -> ratings -> refit ALS
      |
theory / exact / empirical success, top-k, realized beta/gamma, optimal u_C^*
'''

def run_optimal_variations(params, strategy, alpha, p0, baseline_ratings, baseline_U, baseline_V, A, b, coalition_vector, population_mean, v_target_pre_theory, c_mean_0, c_cov_0, U_C_0, coalition_indices, beta, gamma, target_item, budget):

    R_h_bound = np.linalg.norm(
        coalition_vector - c_mean_0
    )

    coalition_size = len(coalition_indices)

    optimal_res = {}
    for name, distance_bound in (("unbounded", None), ("bounded", R_h_bound)):
        optimal_res[name] = find_optimal_uc_collective(
            params=params,
            baseline_V=baseline_V,
            baseline_U=baseline_U,
            A=A,
            b=b,
            population_mean=population_mean,
            v_target_pre=v_target_pre_theory,
            c_mean_0=c_mean_0,
            c_cov_0=c_cov_0,
            coalition_size=coalition_size,
            target_rating=params.target_rating,
            beta=beta,
            gamma=gamma,
            distance_bound=distance_bound,
            u_init=coalition_vector,
        )

    optimal_empirical = {}
    for name, res in optimal_res.items():
        optimal_empirical[name] = evaluate_optimal_uc_empirically(
            params=params,
            u_c_star=res["u_star"],
            alpha=alpha,
            p0=p0,
            beta=beta,
            gamma=gamma,
            U_C_0=U_C_0,
            baseline_ratings=baseline_ratings,
            baseline_U=baseline_U,
            baseline_V=baseline_V,
            coalition_indices=coalition_indices,
            population_mean=population_mean,
            v_target_pre_theory=v_target_pre_theory,
            target_item=target_item,
            budget=budget,
            title=f'{name}_{strategy}_{alpha}_{p0}_{beta}_{gamma}'
        )

    return (
        optimal_res["bounded"]["S_star"],
        optimal_res["unbounded"]["S_star"],
        R_h_bound,
        optimal_empirical["bounded"],
        optimal_empirical["unbounded"],
    )

def run_geometry_diagnostics(params, A, coalition_vector, population_mean, score_uc):
    #We're gonna use these to solve for our theoretical exact success
    A_inv_uc = np.linalg.solve(A, coalition_vector)

    strategy_alignment_Ainv = ( # equiv. to c in our proofs: c = \bar{u}^TA^{-1}u_C
        population_mean
        @ A_inv_uc
    )

    strategy_leverage_Ainv = ( #equiv. to q in out proofs: q = u_C^TA^{-1}u_C
        coalition_vector
        @ A_inv_uc
    )

    strategy_target_residual = ( #quiv. to delta in our proofs: r_C=u_C^Tv_{j^*}
        params.target_rating
        - score_uc
    )
    return A_inv_uc, {'c': strategy_alignment_Ainv,
                      'q': strategy_leverage_Ainv,
                      'delta': strategy_target_residual}

# Columns copied verbatim from run_theory_validation's output so it's easier to make plots
IDENTITY_KEYS = (
    "mean_identity_error",
    "cov_identity_error",
    "target_update_identity_error",
    "dispersion_identity_error",
    "realized_dispersion",
    "realized_dispersion_ratio",
    "mean_realization_error",
    "covariance_realization_error",
    "target_vector_error",
    "rating_beta_realized",
    "rating_gamma_realized",
    "rating_coordination_residual",
    "post_beta_realized",
    "post_gamma_realized",
    "post_coordination_residual",
    "rating_mean_error",
    "rating_covariance_error",
    "rating_user_error",
    "rating_target_vector_error",
    "rating_dispersion_ratio",
    "als_mean_drift",
    "als_covariance_drift",
    "als_user_drift",
    "als_target_vector_drift",
    "rating_mean_relative_error",
    "rating_cov_relative_error",
    "rating_target_relative_error",
    "final_mean_relative_error",
    "final_cov_relative_error",
    "final_target_relative_error",
)

SUCCESS_KEYS = (
    "theory_success",
    "empirical_success",
    "exact_success",
    "topk_change",
    "topk_change_noncoalition",
)

# ============================================================
# Per-condition run
# ============================================================

def _run_strategy_on_shared_condition(
    params,
    seed,
    alpha,
    p0,
    strategy,
    strategy_rng,
    item_popularity,
    baseline_ratings,
    baseline_U,
    baseline_V,
    coalition_indices,
    coalition_size,
    U_C_0,
    c_mean_0,
    c_cov_0,
    initial_dispersion,
    population_dispersion,
    normalized_initial_dispersion,
    population_mean,
    target_item,
    A,
    b,
    v_target_pre_theory,
    beta_values,
    gamma_values,
    budget, 
    pbar=None):

    results = []
    #Calculate strategic action w.r.t. items and ratings
    (
        strategic_items,
        strategic_ratings,
    ) = generate_strategy_items_and_ratigns(
        strategy=strategy,
        item_popularity=item_popularity,
        item_latents=baseline_V,
        user_latents=baseline_U,
        baseline_ratings=baseline_ratings,
        budget=budget,
        params=params,
        rng=strategy_rng,
    )

    coalition_vector = compute_strategy_vector(
            baseline_V,
            strategic_items,
            strategic_ratings,
            reg=params.als_reg,
        )

    strategy_diagnostics, noncoalition_mask = run_strategy_diagnostics(c_mean_0, v_target_pre_theory, coalition_vector, baseline_U, coalition_indices)

    A_inv_uc, geometry_diagnostics = run_geometry_diagnostics(params, A, coalition_vector, population_mean, strategy_diagnostics['score_uc'])

    # ========================================================
    # COORDINATION / HETEROGENEITY GRID
    # ========================================================

    for beta in beta_values:
        for gamma in gamma_values:

            # ================================================
            # 1. DESIRED LATENT COORDINATION
            # ================================================
            desired_U_C = construct_coordination(
                initial_coalition_latents=U_C_0,
                coalition_vector=coalition_vector,
                beta=beta,
                gamma=gamma,
            )
            #basically this is \bar{u}_C^{(t)}, \Sigma_C^{(t)}
            c_mean_t, c_cov_t = coalition_moments(desired_U_C)

            # ================================================
            # 2. THEORY PREDICTIONS
            # ================================================

            theory = theory_target_update(
                A=A,
                b=b,
                coalition_c_mean_0=c_mean_0,
                coalition_c_cov_0=c_cov_0,
                coalition_vector=coalition_vector,
                beta=beta,
                gamma=gamma,
                coalition_size=coalition_size,
                target_rating=params.target_rating,
            )

            # ================================================
            # 3. APPLY HEURISTIC STRATEGY
            # ================================================
            intervention_ratings = (
                apply_strategic_action_to_ratings(
                    baseline_ratings=baseline_ratings,
                    coalition_indices=coalition_indices,
                    coordinated_latents=desired_U_C,
                    baseline_item_factors=baseline_V,
                    strategic_items=strategic_items,
                    params=params,
                )
            )

            # Explicit target action required by theorem.
            intervention_ratings = (
                enforce_target_item_max_rating(
                    ratings=intervention_ratings,
                    coalition_indices=coalition_indices,
                    target_item=target_item,
                    target_rating=params.target_rating,
                )
            )

            # ================================================
            # 4. REFIT ALS
            # ================================================
            post_model, post_U_raw, post_V_raw = fit_als(
                ratings=intervention_ratings,
                version="intervention",
                params=params,
                initial_item_factors=baseline_V,
                initial_user_factors=baseline_U
            )
            
            post_recall10 = recall_at_k(
                model=post_model,
                R_train=intervention_ratings,
                R_test=test_ratings,
                k=10,
                relevance_threshold=3.0,
            )

            post_U, post_V, alignment_rotation = (
                align_factorization_to_baseline(
                    post_users=post_U_raw,
                    post_items=post_V_raw,
                    baseline_users=baseline_U,
                )
            )

            als_diagnostics, realized_U_C, realized_mean, realized_cov = run_als_diagnostics(alignment_rotation, post_U, coalition_indices)

            # ================================================
            # 5. CALCULATE AN OPTIMAL U_C
            # ================================================

            S_star_distance_bounded, S_star_unbounded, R_h_bound, optimal_empirical_distance_bounded, optimal_empirical_unbounded = run_optimal_variations(params, strategy, alpha, p0, baseline_ratings, baseline_U, baseline_V, A, b, coalition_vector, population_mean, v_target_pre_theory, c_mean_0, c_cov_0, U_C_0, coalition_indices, beta, gamma, target_item, budget)

            # ================================================
            # 6. THEORY IDENTITY CHECKS
            # ================================================

            identity_diagnostics, v_target_exact = (
                run_theory_validation(
                    params=params,

                    c_mean_t=c_mean_t,
                    c_cov_t=c_cov_t,

                    theory=theory,
                    gamma=gamma,
                    c_cov_0=c_cov_0,
                    c_mean_0=c_mean_0,
                    A=A,
                    b=b,

                    desired_U_C=desired_U_C,

                    intervention_ratings=intervention_ratings,
                    coalition_indices=coalition_indices,
                    baseline_V=baseline_V,

                    realized_U_C=realized_U_C,
                    realized_cov=realized_cov,
                    realized_mean=realized_mean,

                    initial_dispersion=initial_dispersion,

                    v_target_empirical=post_V[target_item],
                    collective_vector=coalition_vector
                )
            )

            # ================================================
            # 7. SUCCESS METRICS
            # ================================================

            success_metrics = run_all_ALS_success_metrics(population_mean, theory['v_target_theory'], v_target_pre_theory, v_target_exact, post_V[target_item], baseline_U, post_U, baseline_V, post_V, noncoalition_mask, target_item)

            # ================================================
            # 8. STORE
            # ================================================
            row = {
                # Shared condition
                "seed": seed,
                "alpha": alpha,
                "p0": p0,
                "beta": beta,
                "gamma": gamma,
                "budget": budget,
                
                #Performance
                'post_recall': post_recall10,

                # Strategy
                "strategy": strategy,
                "target_item": target_item,
                "n_strategy_items": len(strategic_items),
                "strat_alignment (c)": geometry_diagnostics['c'],
                "strat_leverage (q)": geometry_diagnostics['q'],
                "strat_target_residual (delta)": geometry_diagnostics['delta'],

                # Initial coalition
                "initial_dispersion": initial_dispersion,
                "population_dispersion": population_dispersion,
                "normalized_initial_dispersion": normalized_initial_dispersion, #Used for checking that we're aligned with p0

                "alignment_rotation_error": als_diagnostics['alignment_rotation_error'],

                # Optimality checks
                "optimal_theory_success_bounded": S_star_distance_bounded,
                "optimal_empirical_success_bounded": optimal_empirical_distance_bounded["empirical_success"],
                "optimal_topk_change_bounded": optimal_empirical_distance_bounded["topk_change"],
                "optimal_theory_success_unbounded": S_star_unbounded,
                "optimal_empirical_success_unbounded": optimal_empirical_unbounded["empirical_success"],
                "optimal_topk_change_unbounded": optimal_empirical_unbounded["topk_change"],
                "heuristic_target_displacement": R_h_bound,
                "realized_mean_displacement": beta * R_h_bound,
            }
            row.update({k: identity_diagnostics[k] for k in IDENTITY_KEYS})
            row.update({k: success_metrics[k] for k in SUCCESS_KEYS})
            results.append(row)
            if pbar is not None:
                pbar.set_postfix(
                    alpha=alpha,
                    p0=p0,
                    strategy=strategy,
                    beta=beta,
                    gamma=gamma,
                )
                pbar.update(1)
    
    return results

def launch_strategy_comparison(
    params,
    seed,
    baseline_ratings,
    observation_mask,
    item_popularity,
    baseline_U,
    baseline_V,
    budget_values=(10,),
    strategies = (
        "mainstream_mimic",
        "target_neighborhood",
        "target_supporter_mimic",
        "random_filler",
    ),
    alphas=(0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9),
    p0_values=("low", "med", "high"),
    beta_values=(0.0, 0.5, 1.0),
    gamma_values=(0.0, 0.5, 1.0),
):
    # Real ratings are fixed, so U, V are shared across seeds.
    # The seed controls the coalition ordering and the strategy sampling.
    n_users = baseline_U.shape[0]
    coalition_orders = organize_coalition_orderings(baseline_U, seed)

    #We freeze target item for all runs
    target_item = int(
        params.target_item
    )

    results = []
    total_conditions = (
    len(alphas)
    * len(p0_values)
    * len(strategies)
    * len(budget_values)
    * len(beta_values)
    * len(gamma_values)
)

    with tqdm(
        total=total_conditions,
        desc=f"Seed {seed}",
        unit="exp setting",
    ) as pbar:
        for alpha_index, alpha in enumerate(alphas):
            coalition_size = max(1, int(n_users * alpha))

            for p0_index, p0 in enumerate(p0_values):
                coalition_indices = coalition_orders[p0][:coalition_size]

                coalition_stats, population_mean = get_initial_coalition_stats(
                    baseline_U,
                    coalition_indices,
                )

                # ALS solving
                A, b = (
                    target_normal_equations_without_coalition(
                        baseline_U=baseline_U,
                        baseline_ratings=baseline_ratings,
                        observation_mask=observation_mask,
                        coalition_indices=coalition_indices,
                        target_item=target_item,
                        als_reg=params.als_reg,
                    )
                )

                v_target_pre_theory = np.linalg.solve(A, b)

                strategy_seed = _condition_seed(
                    seed=seed,
                    alpha_index=alpha_index,
                    p0_index=p0_index,
                    stream=202,
                )

                for strategy in strategies:
                    strategy_rng = np.random.default_rng(strategy_seed)

                    for budget in budget_values:
                        strategy_rows = _run_strategy_on_shared_condition(
                            params,
                            seed,
                            alpha,
                            p0,
                            strategy,
                            strategy_rng,
                            item_popularity,
                            baseline_ratings,
                            baseline_U,
                            baseline_V,
                            coalition_indices,
                            coalition_size,
                            coalition_stats["U_C_0"],
                            coalition_stats["c_mean_0"],
                            coalition_stats["c_cov_0"],
                            coalition_stats["initial_dispersion"],
                            coalition_stats["population_dispersion"],
                            coalition_stats["normalized_initial_dispersion"],
                            population_mean,
                            target_item,
                            A,
                            b,
                            v_target_pre_theory,
                            beta_values,
                            gamma_values,
                            budget,
                            pbar, 
                        )
                        results.extend(strategy_rows)
                        
    return pd.DataFrame(results)

if __name__ == "__main__":
    ratings_df, movies_df, baseline_ratings, observation_mask, item_popularity, user_id_to_idx, movie_id_to_idx = load_real_ratings(paths.DATA_DIR)
    train_df, val_df, test_df = split_movielens_per_user(
        ratings_df
    )

    train_ratings = build_sparse_rating_matrix(
        ratings_df=train_df,
        user_id_to_idx=user_id_to_idx,
        movie_id_to_idx=movie_id_to_idx,
    )
    observation_mask = np.zeros(
        train_ratings.shape,
        dtype=bool,
    )
    observation_mask[train_ratings.nonzero()] = True
    item_popularity = np.asarray(
        train_ratings.getnnz(axis=0)
    ).ravel()
    
    baseline_ratings = train_ratings
    
    val_ratings = build_sparse_rating_matrix(
        ratings_df=val_df,
        user_id_to_idx=user_id_to_idx,
        movie_id_to_idx=movie_id_to_idx,
    )

    test_ratings = build_sparse_rating_matrix(
        ratings_df=test_df,
        user_id_to_idx=user_id_to_idx,
        movie_id_to_idx=movie_id_to_idx,
    )
    
    
    params = params_from_dataframe(ratings_df, MLParams(), 'user_id', 'movie_id', 'rating')

    #Fit the model to the real ratings
    baseline_model, baseline_U, baseline_V = fit_als(
        ratings=baseline_ratings,
        R_val=val_ratings, 
        version="baseline",
        params=params,
        
    )
    
    baseline_recall10 = recall_at_k(
        model=baseline_model,
        R_train=train_ratings,
        R_test=test_ratings,
        k=10,
    )

    all_results = []
    for seed in range(3):
        seed_results = (
            launch_strategy_comparison(
                params,
                seed,
                baseline_ratings,
                observation_mask,
                item_popularity,
                baseline_U,
                baseline_V,
            )
        )
        
        

        if len(seed_results) > 0:
            all_results.append(
                seed_results
            )

    if len(all_results) == 0:
        raise RuntimeError(
            "No successful experimental runs."
        )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )
    results['base_recall'] = baseline_recall10

    output_path = (
        "real_ALS_CA_V2.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )
