
This code contains three concrete components which make up the empricial section of our paper: (1) Bot Farm experiments (2) Synthetic Collective Action experiments, and (3) Real Collective Action experiments (which consist of the ALS and more general Two Tower (TT) frameworks). 

# Codebase overview 
* data: basically just meant to contain MovieLens100K
* general_utils: 
    - CA_helpers: contains functions to return collective/noncollective mask
    - CA_stats: all the functions which are used to monitor the cohesion/coordination of a collective (compatible with both real and synthetic experiemnts)
    - data_loading: all the functions for data loading. Mostly made for the tensorflow dataset build used in the TT experiments. 
    - heuristics: code for generating strategic items/ratings for our heuristic strategies 
    - math_helpers: numerically stable softmax 
    - ratings: code for generating sparse ratings (ALS compatible) from user / item representations. used very heavily in synthetic expereimnts 
    - reproducibility: seeds 
    - success_metrics: eval -- both the single-update, full-update, and topk metrics 
* models (pretty self explanatory)
    - ALS 
* plot_utils: included to reproduce our visuals 
* real 
    - cluster_construction: since we cannot perfectly control the heterogeneity/cohesiveness of the collectives within the real data, we have some functions that are able to sample between low/med/high dispersion 
    - logging: just used for generating/storing/loading config models and checkpoints 
    - TT_utils: just helpers that are specific to the tf implementation of our user models
    - world: in order to make our strategic action pipeline compatible with both synthetic and real experiments, we store a bunch of information in params which remain frozen throughout all the runs. If you want to change things like the target item, etc. then you can modify the code here directly. 
* synthetic 
    - TBD after clean. 
* paths: change dataset directory here. 


# Preliminaries 

## Data 
If you would like to use the same dataset that we present in our experiments you can easily download it from: https://grouplens.org/datasets/movielens/100k/ . Basiclly just unzip this file and then change its location in ```paths.py``` (currently this is set as: ```DATA_DIR='/data/ml-100k' ```) but you want the full file path

## Environment 
We encourage you to use the exact libraries specified in our ```requirements.txt```. Particularly two tower experiments rely heavily on tensorflow which can be finicky if you try to use the wrong version of Python. More formally (hahah!) you want to use something like ```conda create -n tfrs python=3.10``` --> this is the version of python that plays nicely with all the key libraries in our emprical setup. Then you can pip install all the libraries in our requirements. 

# (1) Bot Farm Experiments 
Our BotFarm results can be reproduced by running ```synthetic_ALS_BF.py```. If you want to generate your own synthetic setup you will need to modify ```CollectiveAction/synthetic/world.py``` which is where we construct the settings which encompass our experimentation environment. For example, you might want to change the number of users/distribution of genres, etc. 

# (2) Synthetic Collective Action Experiments 
Our synthetic CA results can be reproduced by running ```synthetic_ALS_CA.py```. Here, again we use the environment variables specified in ```CollectiveAction/synthetic/world.py```. 

# (3) Real Collective Action Experiments 
Crucially, the difference between (2)/(3) or the Synthetic/Real setups is *where the ratings are coming from*. In the Synthetic setup we generate a distribution over user and item representations and then *back-engineer* the baseline ratings that would have yielded this kind of initial representation set. Meanwhile, in the Real setup we load the MovieLens100K dataset ratings and use these as our baseline ratings. 

## ALS 
TBD after full clean  

