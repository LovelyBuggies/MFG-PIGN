# MFG-PIGN

To use MPNN and physic loss to predict the MFG.

rho_{i,t}=rho_{i,t-1}+rho_{i-1,t-1}u_{i-1,t-1}-rho_{i,t-1}u_{i,t-1}.


To run:

```
python3 main.y --network braess --config_path config/braess/braess-simple.ymal
```

Citation:

```
@Article{g15020012,
AUTHOR = {Chen, Xu and Liu, Shuo and Di, Xuan},
TITLE = {Physics-Informed Graph Neural Operator for Mean Field Games on Graph: A Scalable Learning Approach},
JOURNAL = {Games},
VOLUME = {15},
YEAR = {2024},
NUMBER = {2},
ARTICLE-NUMBER = {12},
URL = {https://www.mdpi.com/2073-4336/15/2/12},
ISSN = {2073-4336},
DOI = {10.3390/g15020012}
}
```
