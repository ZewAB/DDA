# DDA
Debiased and data-adaptive (DDA) estimation of the average treatment effect on the treated (ATT) in high
dimensions


## Overview

The project provides simulation and empirical analysis code for estimating ATT using the DDA framework.
It includes multiple simulation settings and an application to the IHDP benchmark dataset.

---

## Repository Structure

The repository includes the following main scripts:

- **DDA_main_sim1.jl**  
  First simulation setting.

- **DDA_main_sim2.jl**  
  Second simulation setting.

- **DDA_main_many_cluster.jl**  
  Simulation under many cluster configurations.

- **DDA_ihdp.jl**  
  Empirical analysis using the IHDP dataset.

Wrapper scripts automate experiment execution:

- `DDA_wrapper_sim1.jl`
- `DDA_wrapper_sim2.jl`
- `DDA_wrapper_many_cluster.jl`

---

**IHDP dataset source:** [CEVAE GitHub Repository](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/IHDP)

