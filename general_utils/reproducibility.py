import numpy as np 

def _condition_seed(
    seed,
    alpha_index,
    p0_index,
    stream,
):
    ss = np.random.SeedSequence([
        int(seed),
        int(alpha_index),
        int(p0_index),
        int(stream),
    ])

    return int(
        ss.generate_state(
            1,
            dtype=np.uint32,
        )[0]
    )
