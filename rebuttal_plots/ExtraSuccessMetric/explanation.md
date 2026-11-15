

Please note that the plots in this folder differ in their naming structure from the one used in the original paper. Note: We omit the $j^*$ subscript for clarity and use pre/post to consider before and after refitting the ALS model.

`theory_success` = $\bar{u}^T(v^{theory}-v^{pre})$

`exact_success` = $\bar{u}^T(v^{exact} - v^{pre})$ —> called Item Shift Success in our plotting. 

`item_shift_success` = $\bar{u}^T(v^{post} - v^{pre})$ —> what the reviewer seems to be referring to but is not listed in our Section 5.1

`empirical_success` = $\frac{1}{|U|}\sum_{i \in U} (u_i^{post}v^{post} - u_i^{pre}-v^{pre})$ —> called ALS Refit in our plotting.