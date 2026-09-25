import pandas as pd 
from tqdm import tqdm 
import numpy as np 
import ipdb 

from general_utils.reproducibility import _condition_seed
from synthetic.world import SyntheticParams, SyntheticWorld
from synthetic.cohesion import construct_users_with_dispersion
from general_utils.ratings import generate_sparse_ratings
from general_utils.CA_stats import coalition_moments, compute_dispersion, estimate_realized_coordination_and_cohesion
from models.ALS_model import fit_als
from general_utils.heuristics import generate_strategy_items_and_ratigns, compute_strategy_vector
from general_utils.CA_helpers import get_noncollective_mask
from general_utils.ratings import apply_strategic_action_to_ratings, enforce_target_item_max_rating
from general_utils.math_helpers import align_factorization_to_baseline 
from synthetic.optimal_solvers import find_optimal_uc_collective, select_items_for_uc_star
from general_utils.success_metrics import top_k_exposure_rate, run_all_ALS_success_metrics

#HELPERS/UTILS FOR OPTIMAL RUNS 
def evaluate_optimal_uc_empirically(
    *,
    params,
    u_c_star,
    alpha, 
    p0,
    beta,
    gamma,
    U_C_0,
    baseline_ratings,
    baseline_U,
    baseline_V,
    coalition_indices,
    population_mean,
    v_target_pre_theory,
    target_item,
    title, 
    budget=10,
):
    """
    Take an optimized latent direction u_C^* and evaluate
    its realized effect after translating it into feasible
    behavior and refitting ALS.

    Returns:
        empirical_success
        topk_change
        post_U
        post_V
        strategic_items
        desired_U_C
    """

    # ========================================================
    # 1. Construct desired coordinated coalition around u_C^*
    # ========================================================

    desired_U_C = construct_coordination(
        initial_coalition_latents=U_C_0,
        coalition_vector=u_c_star,
        beta=beta,
        gamma=gamma,
    )

    # ========================================================
    # 2. Give u_C^* its own behavioral item set
    # ========================================================

    optimal_strategic_items = (
        select_items_for_uc_star(
            u_c_star=u_c_star,
            baseline_V=baseline_V,
            target_item=target_item,
            budget=budget,
        )
    )

    # ========================================================
    # 3. Translate desired latent state into ratings
    # ========================================================

    intervention_ratings = (
        apply_strategic_action_to_ratings(
            baseline_ratings=baseline_ratings,
            coalition_indices=coalition_indices,
            coordinated_latents=desired_U_C,
            baseline_item_factors=baseline_V,
            strategic_items=optimal_strategic_items,
            params=params,
        )
    )

    # Explicit target action, same as heuristics
    intervention_ratings = (
        enforce_target_item_max_rating(
            ratings=intervention_ratings,
            coalition_indices=coalition_indices,
            target_item=target_item,
            target_rating=params.target_rating,
        )
    )

    # ========================================================
    # 4. Refit ALS
    # ========================================================

    _, post_U_raw, post_V_raw = fit_als(
        ratings=intervention_ratings,
        version="intervention",
        params=params,
        initial_item_factors=baseline_V,
        initial_user_factors=baseline_U,
    )

    post_U, post_V, rotation = (
        align_factorization_to_baseline(
            post_users=post_U_raw,
            post_items=post_V_raw,
            baseline_users=baseline_U,
        )
    )

    # ========================================================
    # 5. Empirical target success
    # ========================================================

    v_target_empirical = post_V[
        target_item
    ]

    empirical_success = float(
        population_mean
        @ (
            v_target_empirical
            - v_target_pre_theory
        )
    )

    # ========================================================
    # 6. Top-k exposure
    # ========================================================

    noncoalition_mask = np.ones(
        baseline_U.shape[0],
        dtype=bool,
    )

    noncoalition_mask[
        coalition_indices
    ] = False

    baseline_noncoalition = baseline_U[
        noncoalition_mask
    ]

    post_noncoalition = post_U[
        noncoalition_mask
    ]

    topk_pre, topk_pre_noncoalition = (
        top_k_exposure_rate(
            users=baseline_U,
            items=baseline_V,
            noncoalition_users=baseline_noncoalition,
            target_item=target_item,
        )
    )

    topk_post, topk_post_noncoalition = (
        top_k_exposure_rate(
            users=post_U,
            items=post_V,
            noncoalition_users=post_noncoalition,
            target_item=target_item,
        )
    )

    topk_change = (
        topk_post
        - topk_pre
    )

    topk_change_noncoalition = (
        topk_post_noncoalition
        - topk_pre_noncoalition
    )

    # ========================================================
    # 7. Diagnostics
    # ========================================================

    realized_U_C = post_U[
        coalition_indices
    ]

    realized_mean, realized_cov = (
        coalition_moments(
            realized_U_C
        )
    )

    desired_mean, desired_cov = (
        coalition_moments(
            desired_U_C
        )
    )
    
    return {
        "empirical_success":
            empirical_success,

        "topk_change":
            topk_change,

        "topk_change_noncoalition":
            topk_change_noncoalition,

        "topk_pre":
            topk_pre,

        "topk_post":
            topk_post,

        "strategic_items":
            optimal_strategic_items,

        "desired_U_C":
            desired_U_C,

        "realized_mean_error":
            np.linalg.norm(
                realized_mean
                - desired_mean
            ),

        "realized_cov_error":
            np.linalg.norm(
                realized_cov
                - desired_cov,
                ord="fro",
            ),

        "post_U":
            post_U,

        "post_V":
            post_V,
    }

def run_optimal_variations(strategy, alpha, p0, baseline_ratings, baseline_U, baseline_V, A, b, coalition_vector, population_mean, v_target_pre_theory, c_mean_0, c_cov_0, U_C_0, coalition_indices, beta, gamma, target_item, budget): 
    
    R_h_bound = np.linalg.norm(
        coalition_vector - c_mean_0
        )
    
    coalition_size = len(coalition_indices)
    
    optimal_res_unbounded = find_optimal_uc_collective(
        params = params, 
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
        u_init=coalition_vector,
    )
    
    optimal_res_distance_bounded = find_optimal_uc_collective(
        params = params, 
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
        distance_bound=R_h_bound,
        u_init=coalition_vector,
    )
    
    
    
    u_C_star_distance_bounded = optimal_res_distance_bounded["u_star"]
    S_star_distance_bounded = optimal_res_distance_bounded["S_star"]
    
    
    u_C_star_unbounded = optimal_res_unbounded["u_star"]
    S_star_unbounded = optimal_res_unbounded["S_star"]
    
    optimal_empirical_distance_bounded = (
        evaluate_optimal_uc_empirically(
            params=params,
            u_c_star=u_C_star_distance_bounded,
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
            budget=10,
            title=f'bounded_{strategy}_{alpha}_{p0}_{beta}_{gamma}'
        )
    )
    optimal_empirical_unbounded = (
        evaluate_optimal_uc_empirically(
            params=params,
            u_c_star=u_C_star_unbounded,
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
            budget=10,
            title=f'unbounded_{strategy}_{alpha}_{p0}_{beta}_{gamma}'
        )
    )
    
    return S_star_distance_bounded, S_star_unbounded, R_h_bound, optimal_empirical_distance_bounded, optimal_empirical_unbounded

# Helpers for synthetic strategic action 
def construct_coordination( #generates the theoretical collective u'_i = (1-\beta)\bar{u}_C + \beta u_C + \epsilon_i
    initial_coalition_latents,
    coalition_vector,
    beta,
    gamma,
):
    """
    Disentangled collective-action parameterization.

    Let

        u_i^(0) = c_mean_0 + epsilon_i,

    where the coalition deviations epsilon_i average to zero. Then

        u_i^(t)
            = (1-beta) c_mean_0
            + beta u_c
            + gamma epsilon_i.

    beta  controls movement of the coalition mean toward u_c.
    gamma controls residual within-coalition heterogeneity.

    Therefore:

        mean_t = (1-beta) c_mean_0 + beta u_c
        cov_t  = gamma^2 c_cov_0

    Special cases:
        beta=0, gamma=1 : original heterogeneous coalition
        beta=1, gamma=0 : traditional unified bot farm
        beta=1, gamma>0 : fully aligned mean (high coordination), residual heterogeneity
    """
    
    c_mean_0 = initial_coalition_latents.mean(axis=0)
    deviations_0 = initial_coalition_latents - c_mean_0

    mean_t = (
        (1 - beta) * c_mean_0
        + beta * coalition_vector
    )

    return (
        mean_t
        + gamma * deviations_0
    )

#Helpers for analysis            
def target_normal_equations_without_coalition(
    baseline_U,
    baseline_ratings,
    observation_mask,
    coalition_indices,
    target_item,
    als_reg,
):
    """
    Construct A and b using target-item observations from
    NON-COALITION users only.

    This makes the generalized coalition update exactly:

        A_post = A + sum_{i in C} u_i u_i^T
        b_post = b + r_C sum_{i in C} u_i
    """

    m, d = baseline_U.shape

    target_observers = np.where(
        observation_mask[:, target_item]
    )[0]

    coalition_set = set(
        np.asarray(coalition_indices).tolist()
    )

    noncoalition_target_observers = np.array([
        i
        for i in target_observers
        if i not in coalition_set
    ], dtype=int)

    U_target = baseline_U[
        noncoalition_target_observers
    ]

    r_target = baseline_ratings[
        noncoalition_target_observers,
        target_item
    ]

    A = (
        U_target.T @ U_target
        + als_reg * np.eye(d)
    )

    b = (
        U_target.T @ r_target
    ).reshape(-1)

    return A, b

def theory_target_update( #applies the theoretical generalized target update formula for CA 
    A,
    b,
    coalition_c_mean_0,
    coalition_c_cov_0,
    coalition_vector,
    beta,
    gamma,
    coalition_size,
    target_rating,
):
    """
    Closed-form target update under the disentangled model:

        mean_t =
            (1-beta) c_mean_0 + beta u_c

        cov_t =
            gamma^2 c_cov_0
    """

    mean_theory = (
        (1 - beta) * coalition_c_mean_0
        + beta * coalition_vector
    )

    cov_theory = (
        gamma**2
        * coalition_c_cov_0
    )

    A_post = (
        A
        + coalition_size
        * (
            np.outer(
                mean_theory,
                mean_theory,
            )
            + cov_theory
        )
    )

    b_post = (
        b
        + coalition_size
        * target_rating
        * mean_theory
    )

    v_target_theory = np.linalg.solve(
        A_post,
        b_post,
    )

    return {
        "mean_theory": mean_theory,
        "cov_theory": cov_theory,
        "v_target_theory": v_target_theory,
    }

def exact_latent_target_update(
    A,
    b,
    desired_coalition_latents,
    target_rating,
):
    """
    Directly construct the target-item normal equations using
    every coordinated coalition latent.

    This should agree numerically with theory_target_update().
    """

    A_post = (
        A
        + desired_coalition_latents.T
        @ desired_coalition_latents
    )

    b_post = (
        b
        + target_rating
        * desired_coalition_latents.sum(axis=0)
    )

    return np.linalg.solve(
        A_post,
        b_post,
    )

def infer_coalition_latents_from_ratings(
    *,
    intervention_ratings,
    coalition_indices,
    baseline_V,
    als_reg,
):
    """
    Recover the coalition latents implied by the ACTUAL
    intervention ratings while holding item factors fixed
    at baseline_V.

    This isolates the latent-to-behavior realization step:

        desired_U_C
            ->
        intervention ratings
            ->
        rating_implied_U_C

    before allowing the full ALS model to refit.
    """

    rating_implied_U_C = []

    for user_idx in coalition_indices:

        u_rating = solve_user_latent_from_ratings(
            rating_profile=intervention_ratings[
                user_idx
            ],
            item_factors=baseline_V,
            als_reg=als_reg,
        )

        rating_implied_U_C.append(
            u_rating
        )

    return np.asarray(
        rating_implied_U_C
    )

def solve_user_latent_from_ratings(
    rating_profile,
    item_factors,
    als_reg,
):
    """
    Given one user's observed ratings and fixed item factors,
    solve the ALS user update:

        u = (V_O^T V_O + lambda I)^(-1) V_O^T r_O

    Missing ratings are encoded as 0.
    """

    # Sparse row -> dense
    if hasattr(rating_profile, "toarray"):
        rating_profile = rating_profile.toarray()

    rating_profile = np.asarray(
        rating_profile,
        dtype=float,
    ).reshape(-1)

    observed = np.where(
        ~np.isclose(
            rating_profile,
            0.0,
        )
    )[0]

    if len(observed) == 0:
        raise ValueError(
            "Cannot solve user latent: no observed ratings."
        )

    V_obs = item_factors[
        observed
    ]

    r_obs = rating_profile[
        observed
    ]

    d = item_factors.shape[1]

    A_u = (
        V_obs.T @ V_obs
        + als_reg * np.eye(d)
    )

    b_u = (
        V_obs.T @ r_obs
    )

    return np.linalg.solve(
        A_u,
        b_u,
    )

#Analysis 
def run_theory_validation(
    *,
    params,
    c_mean_t,
    c_cov_t,
    theory,
    gamma,
    c_cov_0,
    c_mean_0,
    A,
    b,
    desired_U_C,
    intervention_ratings,
    coalition_indices,
    baseline_V,
    realized_U_C,
    realized_cov,
    realized_mean,
    initial_dispersion,
    v_target_empirical,
    collective_vector,
):
    """
    Validate three distinct stages:

    1. THEORY / IDENTITY
       Desired coalition geometry vs analytical moments.

    2. BEHAVIORAL REALIZATION
       Desired coalition geometry vs the latents implied by
       the actual intervention ratings, holding V fixed.

    3. FULL ALS REALIZATION
       Rating-implied geometry vs the final coalition geometry
       after jointly refitting ALS.

    This lets us distinguish:

        desired latent
            ->
        feasible ratings
            ->
        full ALS equilibrium
    """

    # ========================================================
    # 1. THEORY IDENTITIES
    # ========================================================

    mean_theory = theory[
        "mean_theory"
    ]

    cov_theory = theory[
        "cov_theory"
    ]

    v_target_theory = theory[
        "v_target_theory"
    ]

    # --------------------------------------------------------
    # Desired moments should exactly match theoretical moments
    # --------------------------------------------------------

    mean_identity_error = np.linalg.norm(
        c_mean_t
        - mean_theory
    )

    cov_identity_error = np.linalg.norm(
        c_cov_t
        - cov_theory,
        ord="fro",
    )

    # --------------------------------------------------------
    # Dispersion identity
    # --------------------------------------------------------

    dispersion_ratio_theory = (
        gamma**2
    )

    dispersion_ratio_latent = (
        np.trace(c_cov_t)
        / np.trace(c_cov_0)
        if np.trace(c_cov_0) > 0
        else np.nan
    )

    dispersion_identity_error = np.abs(
        dispersion_ratio_latent
        - dispersion_ratio_theory
    )

    # --------------------------------------------------------
    # Exact target update from desired latent vectors
    # --------------------------------------------------------

    v_target_exact = exact_latent_target_update(
        A=A,
        b=b,
        desired_coalition_latents=desired_U_C,
        target_rating=params.target_rating,
    )

    target_update_identity_error = np.linalg.norm(
        v_target_exact
        - v_target_theory
    )

    # ========================================================
    # 2. BEHAVIORAL REALIZATION
    #
    # Freeze baseline item factors and ask:
    #
    # "What user latents do the ACTUAL generated ratings imply?"
    # ========================================================

    rating_implied_U_C = (
        infer_coalition_latents_from_ratings(
            intervention_ratings=intervention_ratings,
            coalition_indices=coalition_indices,
            baseline_V=baseline_V,
            als_reg=params.als_reg,
        )
    )

    (
        rating_implied_mean,
        rating_implied_cov,
    ) = coalition_moments(
        rating_implied_U_C
    )

    # --------------------------------------------------------
    # Desired -> ratings gap
    # --------------------------------------------------------

    rating_mean_error = np.linalg.norm(
        rating_implied_mean
        - mean_theory
    )

    rating_covariance_error = np.linalg.norm(
        rating_implied_cov
        - cov_theory,
        ord="fro",
    )

    rating_dispersion = np.trace(
        rating_implied_cov
    )

    rating_dispersion_ratio = (
        rating_dispersion
        / initial_dispersion
        if initial_dispersion > 0
        else np.nan
    )

    # Average per-user latent realization error.
    rating_user_error = np.mean(
        np.linalg.norm(
            rating_implied_U_C
            - desired_U_C,
            axis=1,
        )
    )

    # --------------------------------------------------------
    # Target update implied by ACTUALLY REALIZABLE user latents
    #
    #
    # theory target
    #     ->
    # target using rating-implied coalition geometry
    # --------------------------------------------------------

    v_target_rating_implied = (
        exact_latent_target_update(
            A=A,
            b=b,
            desired_coalition_latents=rating_implied_U_C,
            target_rating=params.target_rating,
        )
    )

    rating_target_vector_error = np.linalg.norm(
        v_target_rating_implied
        - v_target_theory
    )

    # ========================================================
    # 3. FULL ALS REALIZATION
    # ========================================================

    realized_dispersion = np.trace(
        realized_cov
    )

    realized_dispersion_ratio = (
        realized_dispersion
        / initial_dispersion
        if initial_dispersion > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Total desired -> final ALS error
    # --------------------------------------------------------

    mean_realization_error = np.linalg.norm(
        realized_mean
        - mean_theory
    )

    covariance_realization_error = np.linalg.norm(
        realized_cov
        - cov_theory,
        ord="fro",
    )

    target_vector_error = np.linalg.norm(
        v_target_empirical
        - v_target_theory
    )

    # ========================================================
    # 4. ADDITIONAL ERROR INTRODUCED BY FULL ALS
    #
    # ratings-implied state -> final ALS state
    # ========================================================

    als_mean_drift = np.linalg.norm(
        realized_mean
        - rating_implied_mean
    )

    als_covariance_drift = np.linalg.norm(
        realized_cov
        - rating_implied_cov,
        ord="fro",
    )

    als_user_drift = np.mean(
        np.linalg.norm(
            realized_U_C
            - rating_implied_U_C,
            axis=1,
        )
    )

    als_target_vector_drift = np.linalg.norm(
        v_target_empirical
        - v_target_rating_implied
    )

    # ========================================================
    # 5. RELATIVE ERRORS
    # ========================================================

    eps = 1e-12

    mean_theory_norm = np.linalg.norm(
        mean_theory
    )

    cov_theory_norm = np.linalg.norm(
        cov_theory,
        ord="fro",
    )

    target_theory_norm = np.linalg.norm(
        v_target_theory
    )

    rating_mean_relative_error = (
        rating_mean_error
        / mean_theory_norm
        if mean_theory_norm > eps
        else np.nan
    )

    rating_cov_relative_error = (
        rating_covariance_error
        / cov_theory_norm
        if cov_theory_norm > eps
        else np.nan
    )

    rating_target_relative_error = (
        rating_target_vector_error
        / target_theory_norm
        if target_theory_norm > eps
        else np.nan
    )

    final_mean_relative_error = (
        mean_realization_error
        / mean_theory_norm
        if mean_theory_norm > eps
        else np.nan
    )

    final_cov_relative_error = (
        covariance_realization_error
        / cov_theory_norm
        if cov_theory_norm > eps
        else np.nan
    )

    final_target_relative_error = (
        target_vector_error
        / target_theory_norm
        if target_theory_norm > eps
        else np.nan
    )
    # ipdb.set_trace()
    
    rating_coordination =  estimate_realized_coordination_and_cohesion(
        initial_mean=c_mean_0,
        initial_cov=c_cov_0,
        realized_mean=rating_implied_mean,
        realized_cov=rating_implied_cov,
        coalition_vector=collective_vector,
    )
    
    post_coordination =  estimate_realized_coordination_and_cohesion(
        initial_mean=c_mean_0,
        initial_cov=c_cov_0,
        realized_mean=realized_mean,
        realized_cov=realized_cov,
        coalition_vector=collective_vector,
    )

    return {

        # ----------------------------------------------------
        # Theory / identity validation
        # ----------------------------------------------------

        "mean_identity_error":
            mean_identity_error,

        "cov_identity_error":
            cov_identity_error,

        "target_update_identity_error":
            target_update_identity_error,

        "dispersion_identity_error":
            dispersion_identity_error,

        "dispersion_ratio_theory":
            dispersion_ratio_theory,

        "dispersion_ratio_empirical":
            dispersion_ratio_latent,

        # ----------------------------------------------------
        # Desired -> feasible ratings
        # ----------------------------------------------------

        "rating_mean_error":
            rating_mean_error,

        "rating_covariance_error":
            rating_covariance_error,

        "rating_user_error":
            rating_user_error,

        "rating_dispersion":
            rating_dispersion,

        "rating_dispersion_ratio":
            rating_dispersion_ratio,

        "rating_target_vector_error":
            rating_target_vector_error,

        # ----------------------------------------------------
        # Feasible ratings -> full ALS
        # ----------------------------------------------------

        "als_mean_drift":
            als_mean_drift,

        "als_covariance_drift":
            als_covariance_drift,

        "als_user_drift":
            als_user_drift,

        "als_target_vector_drift":
            als_target_vector_drift,

        # ----------------------------------------------------
        # Desired -> final ALS
        # ----------------------------------------------------

        "realized_dispersion":
            realized_dispersion,

        "realized_dispersion_ratio":
            realized_dispersion_ratio,

        "mean_realization_error":
            mean_realization_error,

        "covariance_realization_error":
            covariance_realization_error,

        "target_vector_error":
            target_vector_error,

        # ----------------------------------------------------
        # Relative errors
        # ----------------------------------------------------

        "rating_mean_relative_error":
            rating_mean_relative_error,

        "rating_cov_relative_error":
            rating_cov_relative_error,

        "rating_target_relative_error":
            rating_target_relative_error,

        "final_mean_relative_error":
            final_mean_relative_error,

        "final_cov_relative_error":
            final_cov_relative_error,

        "final_target_relative_error":
            final_target_relative_error,
            
        # ----------------------------------------------------
        # Realized beta
        # ----------------------------------------------------

        "rating_beta_realized":
            rating_coordination["beta_realized"],

        "rating_gamma_realized":
            rating_coordination["gamma_realized"],

        "rating_coordination_residual":
            rating_coordination["mean_identity_error"],

        "post_beta_realized":
            post_coordination["beta_realized"],

        "post_gamma_realized":
            post_coordination["gamma_realized"],

        "post_coordination_residual":
            post_coordination["mean_identity_error"],

    }, v_target_exact

def run_als_diagnostics(alignment_rotation, post_U, coalition_indices): 
    #Basically we just want to make sure that when we're comparing the targets pre and post we're alignging them correctly cause sometimes ALS rotates things that would make there seem like there are errors when there arent'
    alignment_rotation_error = np.linalg.norm(
        alignment_rotation
        - np.eye(alignment_rotation.shape[0]),
        ord="fro",
    )

    realized_U_C = post_U[
        coalition_indices
    ]

    realized_mean, realized_cov = (
        coalition_moments(
            realized_U_C
        )
    )
    return {'alignment_rotation_error': alignment_rotation_error}, realized_U_C, realized_mean, realized_cov

def run_geometry_diagnostics(A, coalition_vector, population_mean, score_uc): 
    #We're gonna use these to solve for our theoretical exact success 
    A_inv_uc = np.linalg.solve(A,coalition_vector)

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

def run_strategy_diagnostics(c_mean_0, v_target_pre_theory, coalition_vector, baseline_U, coalition_indices): 
    
    #Ideally you want score_uc > score_initial_mean --> this means heuristic works well 
    score_initial_mean = (c_mean_0@ v_target_pre_theory) #how well aligned the coalition mean is to target (before movement)
    score_uc = (coalition_vector@ v_target_pre_theory ) #how well aligned the heristic strategy is to target (before movement)

    
    noncoalition_mask = get_noncollective_mask(coalition_indices, np.arange(baseline_U.shape[0]))
    noncoalition_users = baseline_U[noncoalition_mask]
    
    
    score_noncoalition = (noncoalition_users @ v_target_pre_theory) #later we assses how the strategc action will move target
    noncoalition_alignment = (noncoalition_users@ coalition_vector) #later we look at how the strategic alignment will move alignment between coalition and general user base 
    
    return {'score_initial_mean': score_initial_mean, 
            'score_noncoalition': score_noncoalition, 
            'score_uc': score_uc, 
            'noncoalition_alignmnet': noncoalition_alignment}, noncoalition_mask 

def collect_collective_stats(baseline_U, coalition_indices): 
    U_C_0 = baseline_U[ 
        coalition_indices
    ]

    c_mean_0, c_cov_0 = coalition_moments(
        U_C_0
    )

    initial_dispersion = np.trace(
        c_cov_0
    )

    population_dispersion = compute_dispersion(
        baseline_U
    )

    normalized_initial_dispersion = ( #used for checking that we're following p0 input
        initial_dispersion
        / population_dispersion
        if population_dispersion > 0
        else np.nan
    )

    population_mean = baseline_U.mean(
        axis=0
    )
    
    return U_C_0, c_mean_0, c_cov_0, initial_dispersion, population_dispersion, normalized_initial_dispersion, population_mean

def _run_strategy_on_shared_condition(
    params,
    world,
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
    budget): 
    
    
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

    A_inv_uc, geometry_diagnostics = run_geometry_diagnostics(A, coalition_vector, population_mean, strategy_diagnostics['score_uc'])
    
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
            
            #basically this is \bar{u}_C^', \Sigma_C^'
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
                    params=world.params,
                )
            )

            # Explicit target action required by theorem. (r_{cJ^*} = 5)
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
                initial_user_factors=baseline_U,
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
            
            S_star_distance_bounded, S_star_unbounded, R_h_bound, optimal_empirical_distance_bounded, optimal_empirical_unbounded = run_optimal_variations(strategy, alpha, p0, baseline_ratings, baseline_U, baseline_V, A, b, coalition_vector, population_mean, v_target_pre_theory, c_mean_0, c_cov_0, U_C_0, coalition_indices, beta, gamma, target_item, budget)
            
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
                    c_mean_0 = c_mean_0, 
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
            results.append({

                # -------------------------
                # Shared condition
                # -------------------------
                "seed":
                    seed,

                "alpha":
                    alpha,

                "p0":
                    p0,
                    
                "beta":
                    beta,

                "gamma":
                    gamma,
                
                'budget': budget, 
                
                # -------------------------
                # Strategy
                # -------------------------
                "strategy":
                    strategy,

                "target_item":
                    target_item,

                "n_strategy_items":
                    len(strategic_items),

                "strat_alignment (c)": 
                    geometry_diagnostics['c'],

                "strat_leverage (q)":
                    geometry_diagnostics['q'],

                "strat_target_residual (delta)":
                    geometry_diagnostics['delta'],

                # -------------------------
                # Initial coalition
                # -------------------------
                "initial_dispersion":
                    initial_dispersion,

                "population_dispersion":
                    population_dispersion,

                "normalized_initial_dispersion": #Used for checking that we're aligned with p0
                    normalized_initial_dispersion,

                # -------------------------
                # Identity checks
                # -------------------------
                "mean_identity_error":
                    identity_diagnostics['mean_identity_error'],

                "cov_identity_error":
                    identity_diagnostics['cov_identity_error'],

                "target_update_identity_error":
                    identity_diagnostics['target_update_identity_error'],

                "dispersion_identity_error":
                    identity_diagnostics['dispersion_identity_error'],

                # -------------------------
                # Realized coordination and cohesion
                # -------------------------
                "realized_dispersion":
                    identity_diagnostics['realized_dispersion'],

                "realized_dispersion_ratio":
                    identity_diagnostics['realized_dispersion_ratio'],

                "mean_realization_error":
                    identity_diagnostics['mean_realization_error'],

                "covariance_realization_error":
                    identity_diagnostics['covariance_realization_error'],

                "target_vector_error":
                    identity_diagnostics['target_vector_error'],

                "alignment_rotation_error":
                    als_diagnostics['alignment_rotation_error'],
                    
                "rating_beta_realized":
                    identity_diagnostics["rating_beta_realized"],

                "rating_gamma_realized":
                    identity_diagnostics["rating_gamma_realized"],

                "rating_coordination_residual":
                    identity_diagnostics["rating_coordination_residual"],

                "post_beta_realized":
                    identity_diagnostics["post_beta_realized"],

                "post_gamma_realized":
                    identity_diagnostics["post_gamma_realized"],

                "post_coordination_residual":
                    identity_diagnostics["post_coordination_residual"],
                    
                # -------------------------
                # Success
                # -------------------------
                "theory_success":
                    success_metrics['theory_success'],

                "empirical_success":
                    success_metrics['empirical_success'],
                
                "exact_success": 
                    success_metrics['exact_success'], 
                    
                "topk_change": 
                    success_metrics['topk_change'],
                
                
                # -------------------------
                # Optimality checks
                # -------------------------
                
                "optimal_theory_success_bounded":
                    S_star_distance_bounded,
                    
                "optimal_empirical_success_bounded":
                    optimal_empirical_distance_bounded[
                        "empirical_success"
                    ],

                "optimal_topk_change_bounded":
                    optimal_empirical_distance_bounded[
                        "topk_change"
                    ],
                
                "optimal_theory_success_unbounded":
                    S_star_unbounded,
                
                "optimal_empirical_success_unbounded":
                    optimal_empirical_unbounded[
                        "empirical_success"
                    ],

                "optimal_topk_change_unbounded":
                    optimal_empirical_unbounded[
                        "topk_change"
                    ],
                    
                "heuristic_target_displacement": R_h_bound,
                "realized_mean_displacement": beta * R_h_bound,     
                    
                    
                # -------------------------
                # Desired -> ratings
                # -------------------------

                "rating_mean_error":
                    identity_diagnostics[
                        "rating_mean_error"
                    ],

                "rating_covariance_error":
                    identity_diagnostics[
                        "rating_covariance_error"
                    ],

                "rating_user_error":
                    identity_diagnostics[
                        "rating_user_error"
                    ],

                "rating_target_vector_error":
                    identity_diagnostics[
                        "rating_target_vector_error"
                    ],

                "rating_dispersion_ratio":
                    identity_diagnostics[
                        "rating_dispersion_ratio"
                    ],

                # -------------------------
                # Ratings -> ALS
                # -------------------------

                "als_mean_drift":
                    identity_diagnostics[
                        "als_mean_drift"
                    ],

                "als_covariance_drift":
                    identity_diagnostics[
                        "als_covariance_drift"
                    ],

                "als_user_drift":
                    identity_diagnostics[
                        "als_user_drift"
                    ],

                "als_target_vector_drift":
                    identity_diagnostics[
                        "als_target_vector_drift"
                    ],

                # -------------------------
                # Relative errors
                # -------------------------

                "rating_mean_relative_error":
                    identity_diagnostics[
                        "rating_mean_relative_error"
                    ],

                "rating_cov_relative_error":
                    identity_diagnostics[
                        "rating_cov_relative_error"
                    ],

                "rating_target_relative_error":
                    identity_diagnostics[
                        "rating_target_relative_error"
                    ],

                "final_mean_relative_error":
                    identity_diagnostics[
                        "final_mean_relative_error"
                    ],

                "final_cov_relative_error":
                    identity_diagnostics[
                        "final_cov_relative_error"
                    ],

                "final_target_relative_error":
                    identity_diagnostics[
                        "final_target_relative_error"
                    ],
                })

    return results 

def launch_strategy_comparison(
    params, 
    seed,
    budget_values=(5, 10, 15), 
    strategies = (
        "mainstream_mimic",
        "target_neighborhood",
        "target_supporter_mimic",
        "random_filler", 
    ),
    alphas=(0.03, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9),
    p0_values=(0.0,0.3, 0.5, 0.7, 1.0),
    beta_values=(0.0, 0.3, 0.5, 0.7, 1.0),
    gamma_values=(0.0,0.5, 1.0),
): 
    #Generate synthetic world and draw env settings 
    world = SyntheticWorld(params, seed=seed)
    
    (
        true_items,
        item_genres,
        item_popularity,
        true_users,
        coalition_order,
    ) = world.generate_synthetic_environment()


    results = [] 
    
    for alpha_index, alpha in enumerate(tqdm(alphas)): 
        #Calculate collective members 
        coalition_indices = world.get_coalition_indices(coalition_order, alpha)
        coalition_size = len(coalition_indices)
        
        #Initial Dispersion 
        for p0_index, p0 in enumerate(p0_values): 
            
            condition_id = ( #this is just for bookkeeping 
                f"seed={seed}|"
                f"alpha={alpha}|"
                f"p0={p0}"
            )
            
            users_p0 = construct_users_with_dispersion(
                true_users,
                coalition_indices,
                p0,
            )
            
            #We want all the baseline runs to share a seed so that observed ratings set stays consistent irrespective of alpha
            baseline_seed = _condition_seed(
                seed=seed,
                alpha_index=alpha_index,
                p0_index=p0_index,
                stream=101,
            )

            baseline_rng = np.random.default_rng(baseline_seed)
            
            #SIMULATE the baseline ratings that would have created our user/item representations  
            baseline_ratings, observation_mask = (
                generate_sparse_ratings(
                    true_users=users_p0,
                    true_items=true_items,
                    item_popularity=item_popularity,
                    rng=baseline_rng,
                    params=world.params,
                )
            )
            
            #Fit the model to simulated ratings U, V
            baseline_model, baseline_U, baseline_V = fit_als(
                ratings=baseline_ratings,
                version="baseline",
                params=params,
            )
            
            U_C_0, c_mean_0, c_cov_0, initial_dispersion, population_dispersion, normalized_initial_dispersion, population_mean = collect_collective_stats(baseline_U, coalition_indices)
            
            #We freeze target item for all runs 
            target_item = int(
                params.target_item
            )
            
            # ALS solving for PRE strategic action
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
            
            v_target_pre_theory = np.linalg.solve(A,b)
            
            strategy_seed = _condition_seed(
                seed=seed,
                alpha_index=alpha_index,
                p0_index=p0_index,
                stream=202,
            )
            
            for strategy in strategies: 
                strategy_rng = np.random.default_rng(strategy_seed)
                
                for budget_index, budget in enumerate(
                    budget_values
                ):
                    strategy_rows = (_run_strategy_on_shared_condition(
                                            params,
                                            world,
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
                                            budget) 
                        )
                    results.extend(strategy_rows)

                # except Exception as exc:
                    # ipdb.set_trace() 
                    # print(
                    #     f"seed={seed}, "
                    #     f"alpha={alpha}, "
                    #     f"p0={p0}, "
                    #     f"strategy={strategy}: "
                    #     f"FAILED: {exc}"
                    # )
                    
    return pd.DataFrame(results)
            
if __name__ == "__main__": 
    params = SyntheticParams()
    
    all_results = [] 
    for seed in range(5): 
        seed_results = (
            launch_strategy_comparison(
                params, 
                seed, 
            )
        )
    
        if len(seed_results) > 0:
            all_results.append(
                seed_results
            )

    results = pd.concat(
        all_results,
        ignore_index=True,
    )
    
    output_path = (
        "synthetic_ALS_CA_V4.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )