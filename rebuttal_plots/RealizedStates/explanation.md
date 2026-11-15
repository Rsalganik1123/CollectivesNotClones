
# Plotting realized parameters $\hat\beta, \hat\gamma$ as they relate to downstream metrics. 

Below, we report performance on the "realized" parameters $\hat\beta, \hat\gamma$ in the setting where $\alpha=0.1, \Sigma_C = 0.0$. Just for clarity, we present two of the most "realistic" heuristics. However, we also provide the plots for the remaining heuristics in this folder. 

We can see that, even with imperfectly realized coordination, we can see that the coordination trend (perfect coordination seems to yield lower performance for the strategies we're studying) persists. We wish to note that since $\gamma$ is a scaling parameter on the initial heterogeneity $\Sigma_C$, setting this parameter to 0 means that the $\hat\gamma$ values should remain relatively unaffected and unable to rise. 
![Local Screenshot](success_vs_realized_rating_alpha_0.1_p0_0.0.png)

We also present the results for a heterogeneous collective by setting $\alpha=0.1, \Sigma_C = 0.5$. We believe that these plots, too, confirm our findings from the perfect parameter plots presented in Figure 3 of our paper. We can see that increased coordination seems to lower success and higher heterogeneity (seen in realized $\hat \gamma$ values) increases success. 
![Local Screenshot](success_vs_realized_rating_alpha_0.1_p0_0.5.png)


For a more direct comparison with the plots in Figure 3(d) and 3(e) we present the following plots as well. 

First, we showcase the setting where $\alpha=0.1, \Sigma_C = 0, \gamma=0.0$ and plotted the realized beta $\hat\beta$. This corresponds with figure 3(d). 
![Local Screenshot](success_vs_realized_rating_realized_beta_alpha_0.1_p0_0.0_gamma_1.0.png)
![Local Screenshot](success_vs_realized_rating_realized_beta_alpha_0.1_p0_0.0_gamma_1.0_all_strat.png)

Second, we showcase the setting where $\alpha=0.1, \Sigma_C = 1.0, \beta=1.0$ and plotted the realized beta $\hat\gamma$. This corresponds with figure 3(e).
![Local Screenshot](success_vs_realized_rating_realized_gamma_alpha_0.1_beta_1.0_p0_1.0.png)
![Local Screenshot](success_vs_realized_rating_realized_gamma_alpha_0.1_beta_1.0_p0_1.0_all_strat.png)