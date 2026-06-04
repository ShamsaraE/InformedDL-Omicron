#  InformedDL-Omicron  
An informed deep learning model for analyzing the Omicron wave and assessing the impact of vaccination.

##  Overview  
The **InformedDL-Omicron** project utilizes **Physics-Informed Neural Networks (PINNs) [1](#1)** and **Disease-Informed Neural Networks (DINNs)[2](#2)** to model the **Omicron variant spread** and its epidemiological dynamics in Germany, France, and Italy.

By integrating **real-world COVID-19 data** with **epidemiological constraints**, this model dynamically estimates **transmission rates (β), reinfection rates (η), vaccine efficacy (δ),** and other key parameters, while ensuring physically meaningful solutions.

---
[1]: Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational physics 378 (2019): 686-707.

[2] Shaier, Sagi, Maziar Raissi, and Padmanabhan Seshaiyer. "Data-driven approaches for predicting spread of infectious diseases through DINNs: Disease Informed Neural Networks." arXiv preprint arXiv:2110.05445 (2021).

###  Key Contributions  
- Developed a **DINN model with sliding-window approach** to understand the dynamic behavior of Omicron wave and **real-time estimation of epidemic parameters**.
- Assessed the **impact of vaccination and mutations on Omicron transmission**.  

---

##  Citation  
If you use this repository in your work, please cite the following:  

E. Shamsara, F. König, N. Pfeifer, "An Informed Deep Learning Model of the Omicron Wave and the Impact of Vaccination". Computers in Biology and Medicine (2025)


  



---

## Repository structure

```
InformedDL-Omicron.ipynb   Main notebook: model, sliding-window training, parameter estimation
data/                      Daily compartment data per country (DEU, FRA, ITA)
results/                   Outputs: checkpoints, reconstructed compartments, parameter tables
                           + example_observed_compartments.png (no-training preview)
requirements.txt           Python dependencies
```

## Data

`data/{DEU,FRA,ITA}_Omicron_check.csv` — daily values over the Omicron period (Jan 2022 – Apr 2023,
477 days each). Columns: `Country, Variant, date, I, D, R, V, H, ICU, Population, S, Stringency`,
where S/I/H/R/D/V are the SIHRDV compartments.

## How to run

```bash
pip install -r requirements.txt
jupyter lab            # open InformedDL-Omicron.ipynb
```

- **Quick look (seconds, CPU):** run Sections 1–3 then the **Example** cell — it plots the observed
  Omicron waves directly from the data, no training.
- **Full training (heavy, GPU recommended):** Sections 1–6 train a DINN on every 90-day sliding
  window (387 windows per country × tens of thousands of epochs). For a fast functional test, set
  `WINDOW_START, WINDOW_END = 0, 2` and `EPOCHS = 200` in Section 5.

## Model notes

- A **Disease-Informed Neural Network** maps time `t` to the six SIHRDV compartments and to seven
  epidemiological parameters; the SIHRDV ODE residuals are enforced as a physics loss alongside the
  data-fit (MSE) loss, balanced by an adaptive weight.
- The network uses **tanh** activations and **no batch normalisation** — both required so that the
  autograd time-derivatives used in the physics loss are well-defined per sample.
