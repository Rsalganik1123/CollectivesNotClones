import numpy as np 

from general_utils.CA_stats import compute_dispersion

def construct_users_with_dispersion(true_users, coalition_indices, p0): 
    true_users = true_users.copy()
    
    coalition_size = len(coalition_indices)
    
    reference_population_dispersion = (
        compute_dispersion(true_users)
    )
    
    coalition_users = true_users[coalition_indices]
    coalition_mean = coalition_users.mean(axis=0)
    deviations = coalition_users - coalition_mean
    current_dispersion = compute_dispersion(coalition_users)
    target_dispersion = p0 * reference_population_dispersion
    if np.isclose(current_dispersion, 0.0):
        scale = 0.0
        # if coalition_size <= 1:
        #     current_dispersion = 0.0
    else:
        scale = np.sqrt(
            target_dispersion / current_dispersion
        )
    true_users[coalition_indices] = (
        coalition_mean + scale * deviations
    )
            
    #OLD VERSION THAT RANDOMLY SAMPLED NOISE -- too much randomness introduced 
    # target_coalition_dispersion = (
    #     p0 * reference_population_dispersion
    # )
    
    # deviations = generate_centered_deviations(
    #     n_vectors=coalition_size,
    #     latent_dim=params.latent_dim,
    #     target_dispersion=(
    #         target_coalition_dispersion
    #     ),
    #     rng=rng,
    # )
    # coalition_mean = np.mean(true_users[coalition_indices], axis=0)
    
    # true_users[coalition_indices] = (
    #     coalition_mean[None, :] + deviations
    # )
    return true_users

