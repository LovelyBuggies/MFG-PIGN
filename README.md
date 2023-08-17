# MFG-PIGN

To use MPNN and physic loss to predict the MFG.

rho_{i,t}=rho_{i,t-1}+rho_{i-1,t-1}u_{i-1,t-1}-rho_{i,t-1}u_{i,t-1}.


To run:

```
python3 mainl.y --config_path config/config_classic.ymal --run rho_v
```
