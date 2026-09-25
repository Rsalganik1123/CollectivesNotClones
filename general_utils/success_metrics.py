import numpy as np 
import ipdb 

def run_all_ALS_success_metrics(population_mean, v_target_theory, v_target_pre_theory, v_target_exact, v_target_empirical, baseline_U, post_U, baseline_V, post_V, noncoalition_mask, target_item, k=10): 
   
#     population_mean,
#     v_target_theory,
#     v_target_pre_theory,
#     v_target_exact,
#     baseline_U,
#     post_U,
#     baseline_V,
#     post_V,
#     noncoalition_mask,
#     target_item,
#     k=10,
# ):
    # ========================================================
    # TARGET ITEM EMBEDDINGS
    # ========================================================

    v_target_pre = baseline_V[target_item]
    v_target_post = post_V[target_item]

    # ========================================================
    # 1. THEORY SUCCESS
    #
    # Analytical target-item movement.
    # ========================================================

    theory_success = float(
        population_mean
        @ (
            v_target_theory
            - v_target_pre_theory
        )
    )

    # ========================================================
    # 2. EXACT SUCCESS
    #
    # Exact latent-space target update used to validate theory.
    # ========================================================

    exact_success = float(
        population_mean
        @ (
            v_target_exact
            - v_target_pre_theory
        )
    )

    exact_success_error = abs(
        exact_success
        - theory_success
    )

    # ========================================================
    # 3. ITEM-SHIFT SUCCESS
    #
    # Empirical movement of the target item after full ALS
    # refitting, evaluated against the baseline population.
    #
    # This is the empirical analogue of theory_success.
    # ========================================================
    # ipdb.set_trace() 
    item_shift_success = float(
        population_mean
        @ (
            v_target_post
            - v_target_pre
        )
    )

    # ========================================================
    # 4. EMPIRICAL / POST-REFIT SUCCESS
    #
    # Full end-to-end change after refitting ALS:
    # both users and items are allowed to move.
    # ========================================================

    score_pre = (
        baseline_U
        @ v_target_pre
    )

    score_post = (
        post_U
        @ v_target_post
    )

    empirical_success = float(
        np.mean(
            score_post
            - score_pre
        )
    )

    # ========================================================
    # 5. TOP-K EXPOSURE
    # ========================================================

    pre_noncoalition_users = (
        baseline_U[
            noncoalition_mask
        ]
    )

    post_noncoalition_users = (
        post_U[
            noncoalition_mask
        ]
    )

    (
        topk_pre,
        topk_pre_noncoalition,
    ) = top_k_exposure_rate(
        users=baseline_U,
        items=baseline_V,
        noncoalition_users=pre_noncoalition_users,
        target_item=target_item,
        k=k,
    )

    (
        topk_post,
        topk_post_noncoalition,
    ) = top_k_exposure_rate(
        users=post_U,
        items=post_V,
        noncoalition_users=post_noncoalition_users,
        target_item=target_item,
        k=k,
    )

    topk_change = (
        topk_post
        - topk_pre
    )

    topk_change_noncoalition = (
        topk_post_noncoalition
        - topk_pre_noncoalition
    )

    return {
        "theory_success":
            theory_success,

        "exact_success":
            exact_success,

        "exact_success_error":
            exact_success_error,

        "item_shift_success":
            item_shift_success,

        "empirical_success":
            empirical_success,

        "topk_change":
            topk_change,

        "topk_change_noncoalition":
            topk_change_noncoalition,
    }

def run_all_TT_success_metrics(
    population_mean,
    baseline_U,
    action_U,
    post_U,
    baseline_V,
    post_V,
    noncoalition_mask,
    target_item,
    k
):
    """
    Compute success metrics for the two-tower collective-action experiment.

    Metrics
    -------
    immediate_success:
        Change in target-item score immediately after the coalition
        changes its histories, with the recommender frozen.

            mean_i [
                u_i^action^T v_j*
                -
                u_i^pre^T v_j*
            ]

    empirical_success:
        End-to-end change in target-item score after retraining.

            mean_i [
                u_i^post^T v_j*^post
                -
                u_i^pre^T v_j*^pre
            ]

    item_shift_success:
        Change attributable to movement of the target item embedding,
        evaluated using the baseline population mean.

            u_bar^T (
                v_j*^post
                -
                v_j*^pre
            )

        This is the closest analogue to the success metric used in
        the ALS experiments.

    topk_change:
        Change in target-item top-k exposure after retraining.

    topk_change_noncoalition:
        Change in target-item top-k exposure among non-coalition users.
    """

    # ========================================================
    # TARGET ITEM EMBEDDINGS
    # ========================================================

    v_target_pre = (
        baseline_V[target_item]
    )

    v_target_post = (
        post_V[target_item]
    )


    # ========================================================
    # 1. IMMEDIATE SUCCESS
    #
    # User histories have changed, but the recommender
    # parameters are still frozen.
    # ========================================================

    score_pre = (
        baseline_U
        @ v_target_pre
    )

    score_action = (
        action_U
        @ v_target_pre
    )

    immediate_success = float(
        np.mean(
            score_action
            - score_pre
        )
    )
    
    # theory_success = float(
    #     population_mean
    #     @ (
    #         v_target_theory
    #         - v_target_pre_theory
    #     )
    # )


    # ========================================================
    # 2. EMPIRICAL SUCCESS
    #
    # End-to-end score change after retraining.
    # Both user and item embeddings may now differ.
    # ========================================================

    score_post = (
        post_U
        @ v_target_post
    )

    empirical_success = float(
        np.mean(
            score_post
            - score_pre
        )
    )


    # ========================================================
    # 3. ITEM-SHIFT SUCCESS
    #
    # Closest analogue to our original ALS success metric:
    #
    #   u_bar^T (v'_j* - v_j*)
    # ========================================================

    item_shift_success = float(
        population_mean
        @ (
            v_target_post
            - v_target_pre
        )
    )


    # ========================================================
    # 4. TOP-K EXPOSURE
    # ========================================================

    pre_noncoalition_users = (
        baseline_U[
            noncoalition_mask
        ]
    )

    post_noncoalition_users = (
        post_U[
            noncoalition_mask
        ]
    )

    (
        topk_pre,
        topk_pre_noncoalition,
    ) = top_k_exposure_rate(
        users=baseline_U,
        items=baseline_V,
        noncoalition_users=
            pre_noncoalition_users,
        target_item=target_item,
        k=k
    )

    (
        topk_post,
        topk_post_noncoalition,
    ) = top_k_exposure_rate(
        users=post_U,
        items=post_V,
        noncoalition_users=
            post_noncoalition_users,
        target_item=target_item,
        k=k
    )
    # ipdb.set_trace()
    topk_change = (
        topk_post
        - topk_pre
    )

    topk_change_noncoalition = (
        topk_post_noncoalition
        - topk_pre_noncoalition
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {
        "immediate_success":
            immediate_success,

        "empirical_success":
            empirical_success,

        "item_shift_success":
            item_shift_success,

        "topk_change":
            topk_change,

        "topk_change_noncoalition":
            topk_change_noncoalition,
    }

# ============================================================
# Longterm RecSys Performance
# ============================================================

def top_k_exposure_rate(
    users: np.ndarray,
    items: np.ndarray,
    noncoalition_users: np.ndarray, 
    target_item: int,
    k: int = 10,
) -> float:
    scores = users @ items.T
    noncoalition_scores = noncoalition_users @ items.T
    top_k_items_general = np.argpartition(
        scores,
        kth=-k,
        axis=1,
    )[:, -k:]
    top_k_items_noncoalition = np.argpartition(
        noncoalition_scores,
        kth=-k,
        axis=1,
    )[:, -k:]
    
    general_pop = float(
        np.mean(
            np.any(
                top_k_items_general == target_item,
                axis=1,
            )
        )
    )
    noncoalition_pop = float(
        np.mean(
            np.any(
                top_k_items_noncoalition == target_item,
                axis=1,
            )
        )
    )
    return general_pop, noncoalition_pop

def recall_at_k(
    model,
    R_train,
    R_test,
    k=10,
    relevance_threshold=3.0,
    user_mask=None,
):
    recs, _ = model.recommend_all(
        R_train=R_train,
        k=k,
    )

    users = (
        np.arange(R_test.shape[0])
        if user_mask is None
        else np.flatnonzero(user_mask)
    )

    recalls = []

    for u in users:
        test_row = R_test[u]

        relevant = test_row.indices[
            test_row.data >= relevance_threshold
        ]

        if len(relevant) == 0:
            continue

        hits = np.isin(
            recs[u],
            relevant,
        ).sum()

        recalls.append(
            hits / len(relevant)
        )

    return float(np.mean(recalls))