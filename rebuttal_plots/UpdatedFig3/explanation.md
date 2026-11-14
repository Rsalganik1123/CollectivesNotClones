
In an attempt to make sure that all the results were fully updated when considering figure 3(a) and 3(b), we reran the entire pipeline and removed the optimal lines from the graphs for better visualization. 

Below, we concretely compare a fully cohesive and coordinated collective with a heterogeneous and uncoordinated one. 

First, we showcase the setting where $\alpha=0.1, \beta = 1.0, \text{and~}  \Sigma_C = \gamma=0$.
![Local Screenshot](vary_alpha_beta_1.0_p0_0.0_gamma_0.0_multi_metric.png)

Then, we relax the cohesion assumption and showcase  $\alpha=0.1, \beta = 1.0, \text{and~} \Sigma_C = \gamma=1.0$.
![Local Screenshot](vary_alpha_beta_1.0_p0_1.0_gamma_1.0_multi_metric.png)

Finally, we relax both the cohesion and coordination assumptions, showcasing $\alpha=0.1, \beta = 0.0, \text{and~}  \Sigma_C = \gamma=1.0$.
![Local Screenshot](vary_alpha_beta_0.0_p0_1.0_gamma_1.0_multi_metric.png)

Crucially, the goal of this comparison is to strengthen the key findings of our work, which is that full cohesiveness and coordination may not necessarily achieve the highest success. Our goal is to motivate the explicit formulation of heterogeneity (both initial and action-induced) in the approximation of a collective's success.

As we can see from these plots, relaxing the cohesion and coordination assumptions can actually achieve higher success in the Item-Shift Success. 