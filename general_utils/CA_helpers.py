import numpy as np 

def get_noncollective_mask(collective_indices, user_ids): 
    noncollective_mask = np.ones(
            len(user_ids),
            dtype=bool,
        )

    noncollective_mask[
            collective_indices
        ] = False
    return noncollective_mask