import numpy as np 
from scipy.optimize import minimize

#OPTIMAL STRAT. HELPERS
def find_optimal_uc_collective(
    params, 
    A,
    b,
    baseline_U,
    baseline_V, 
    population_mean,
    v_target_pre,
    c_mean_0,
    c_cov_0,
    coalition_size,
    target_rating,
    beta,
    gamma,
    distance_bound=None, 
    u_init=None,
):
    """
    Find u_C maximizing theoretical collective-action success
    for fixed beta and gamma.
    """

    A = np.asarray(
        A,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    ).reshape(-1)

    population_mean = np.asarray(
        population_mean,
        dtype=float,
    ).reshape(-1)

    v_target_pre = np.asarray(
        v_target_pre,
        dtype=float,
    ).reshape(-1)

    c_mean_0 = np.asarray(
        c_mean_0,
        dtype=float,
    ).reshape(-1)

    c_cov_0 = np.asarray(
        c_cov_0,
        dtype=float,
    )

    d = A.shape[0]

    if u_init is None:
        u_init = c_mean_0.copy()
    else:
        u_init = np.asarray(
            u_init,
            dtype=float,
        ).reshape(-1)

    def success(u_C):

        mean_beta = (
            (1 - beta) * c_mean_0
            + beta * u_C
        )

        cov_gamma = (
            gamma**2
            * c_cov_0
        )

        A_post = (
            A
            + coalition_size
            * (
                np.outer(
                    mean_beta,
                    mean_beta,
                )
                + cov_gamma
            )
        )

        b_post = (
            b
            + coalition_size
            * target_rating
            * mean_beta
        )

        v_target_post = np.linalg.solve(
            A_post,
            b_post,
        )

        return float(
            population_mean
            @ (
                v_target_post
                - v_target_pre
            )
        )

    def get_v_post(u_C): #you can use this to enforce "realistic ratings", we took it out cause our parameters UV are unbounded 

        mean_beta = (
            (1 - beta) * c_mean_0
            + beta * u_C
        )

        cov_gamma = (
            gamma**2
            * c_cov_0
        )

        A_post = (
            A
            + coalition_size
            * (
                np.outer(
                    mean_beta,
                    mean_beta,
                )
                + cov_gamma
            )
        )

        b_post = (
            b
            + coalition_size
            * target_rating
            * mean_beta
        )

        return np.linalg.solve(
            A_post,
            b_post,
        )
    
    V = np.asarray(
        baseline_V,
        dtype=float,
        )    
    constraints = [] 
    if distance_bound is not None: 
        constraints.append({
            "type": "eq",
            "fun": lambda u: (
                distance_bound**2
                - np.dot(
                    u - c_mean_0,
                    u - c_mean_0,
                )
            ),
        })
    
    else: 
        constraints = [] 
    
    result = minimize(
        lambda u: -success(u),
        x0=u_init,
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": 2000,
            "ftol": 1e-12,
        },
    )

    return {
        "u_star": result.x,
        "S_star": success(result.x),
        "success": result.success,
        "message": result.message,
        "result": result,
    }
    
def select_items_for_uc_star(
    u_c_star,
    baseline_V,
    target_item,
    budget=10,
):
    """
    Construct a behavioral profile for an optimized latent
    direction u_C^*.

    Select the target plus the B-1 non-target items most
    strongly preferred by u_C^*.
    """

    candidate_items = np.delete(
        np.arange(
            baseline_V.shape[0]
        ),
        target_item,
    )

    scores = (
        baseline_V[candidate_items]
        @ u_c_star
    )

    order = np.argsort(
        scores
    )[::-1]

    filler_items = candidate_items[
        order[:budget - 1]
    ]

    strategic_items = np.concatenate([
        [target_item],
        filler_items,
    ])

    return strategic_items