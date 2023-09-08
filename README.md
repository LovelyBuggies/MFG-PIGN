# MFG-PIGN

To use MPNN and physic loss to predict the MFG.

rho_{i,t}=rho_{i,t-1}+rho_{i-1,t-1}u_{i-1,t-1}-rho_{i,t-1}u_{i,t-1}.


To run:

```
python3 main.y --network braess --config_path config/braess/braess-simple.ymal
```
