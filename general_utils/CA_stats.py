import numpy as np 
import ipdb 

def compute_covariance(vectors):
    vectors = np.asarray(vectors, dtype=float)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    return centered.T @ centered / vectors.shape[0]

def compute_dispersion(vectors):
    return float(np.trace(compute_covariance(vectors)))

def collective_moments(U_C):
    """
    Return collective mean and population covariance:

        mean = (1/s) sum_i u_i
        cov  = (1/s) sum_i (u_i - mean)(u_i - mean)^T
    """
    mean = U_C.mean(axis=0)

    centered = U_C - mean

    cov = (
        centered.T @ centered
        / len(U_C)
    )

    return mean, cov

def get_initial_collective_stats(baseline_U, collective_indices): 
    U_C_0 = baseline_U[ 
        collective_indices
    ]

    c_mean_0, c_cov_0 = collective_moments(
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
    
    return {
        "U_C_0":
            U_C_0,

        "c_mean_0":
            c_mean_0,

        "c_cov_0":
            c_cov_0,

        "initial_dispersion":
            initial_dispersion,

        "population_dispersion":
            population_dispersion,

        "normalized_initial_dispersion":
            normalized_initial_dispersion,

    }, population_mean

def estimate_realized_coordination_and_cohesion(
    initial_mean,
    initial_cov,
    realized_mean,
    realized_cov,
    collective_vector,
):
    """
    Estimate beta and gamma implied by an observed collective state.

    beta:
        projection of realized mean movement onto
        collective_vector - initial_mean

    gamma:
        inferred from the theoretical dispersion identity
            tr(Sigma_t) = gamma^2 tr(Sigma_0)
    """
    # ipdb.set_trace() 
    # ========================================================
    # REALIZED BETA
    # ========================================================

    direction = (
        collective_vector
        - initial_mean
    )

    movement = (
        realized_mean
        - initial_mean
    )

    denom = (
        direction @ direction
    )

    if denom > 1e-12:
        beta_realized = float(
            movement @ direction
            / denom
        )
    else:
        beta_realized = np.nan

    beta_mean_prediction = (
        initial_mean
        + beta_realized * direction
        if np.isfinite(beta_realized)
        else np.full_like(initial_mean, np.nan)
    )

    mean_identity_error = (
        np.linalg.norm(
            realized_mean
            - beta_mean_prediction
        )
        if np.isfinite(beta_realized)
        else np.nan
    )

    # ========================================================
    # REALIZED GAMMA
    # ========================================================

    initial_dispersion = np.trace(
        initial_cov
    )

    realized_dispersion = np.trace(
        realized_cov
    )

    if initial_dispersion > 1e-12:

        gamma_sq_realized = (
            realized_dispersion
            / initial_dispersion
        )

        gamma_realized = np.sqrt(
            max(
                gamma_sq_realized,
                0.0,
            )
        )

        gamma_cov_prediction = (
            gamma_realized**2
            * initial_cov
        )

        cov_identity_error = (
            np.linalg.norm(
                realized_cov
                - gamma_cov_prediction,
                ord="fro",
            )
        )

    else:

        gamma_sq_realized = np.nan
        gamma_realized = np.nan
        cov_identity_error = np.nan

    return {
        "beta_realized":
            beta_realized,

        "gamma_realized":
            gamma_realized,

        "realized_dispersion":
            realized_dispersion,

        "realized_dispersion_ratio":
            gamma_sq_realized,

        "mean_identity_error":
            mean_identity_error,

        "cov_identity_error":
            cov_identity_error,
    }