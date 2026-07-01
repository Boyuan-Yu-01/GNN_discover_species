# Literature Search 3: GNN and Machine Learning for Reaction Model and Mechanism Reduction

Search date: 2026-06-17

## Scope

This list focuses on machine learning methods used to reduce chemical kinetic models, classify local chemistry states, choose local reduced mechanisms, cluster cells, or replace parts of detailed chemistry with lower-cost learned models. The papers are mostly from combustion and reacting-flow modeling.

## Direct ML-Assisted Mechanism Reduction

| Paper | ML tool | What is reduced | Brief description |
|---|---|---|---|
| [Wang et al., "Deep mechanism reduction method for fuel chemical kinetics," Combustion and Flame 2024](https://doi.org/10.1016/j.combustflame.2024.113373) | Deep learning, DeePMR | Species and reactions in fuel mechanisms | Learns from local sensitivity information and generates reduced mechanisms. A direct example of deep learning used for kinetic mechanism reduction. |
| [Xia and Gou, "A chemical kinetic mechanism reduction method with autoencoder for hydrocarbon fuels," Energy and AI 2025](https://doi.org/10.1016/j.egyai.2024.100464) | Autoencoder | Kinetic mechanism size | Uses learned low-dimensional representations to support mechanism reduction for hydrocarbon combustion. |
| [Zhang et al., "Graph neural network-based chemical mechanism reduction for combustion modeling," arXiv 2026](https://arxiv.org/abs/2603.22318) | Graph neural network | Combustion mechanism species/reactions | Represents the mechanism as a graph and learns which species/reactions are important under local conditions. This is one of the most directly relevant GNN papers for mechanism reduction. |
| [Lu et al., "A kinetic mechanism reduction method for fuel combustion based on random forests model," Fuel 2026](https://doi.org/10.1016/j.fuel.2025.137399) | Random forest | Fuel combustion kinetic mechanism | Uses tree-based ML to identify important species/reactions and reduce fuel mechanisms. |

## ML Classification Plus Local Reduced Mechanisms

| Paper | ML tool | What is reduced | Brief description |
|---|---|---|---|
| [D'Alessio et al., "Adaptive chemistry via pre-partitioning of composition space and mechanism reduction," Combustion and Flame 2020](https://doi.org/10.1016/j.combustflame.2019.09.010) | LPCA clustering plus DRGEP | Local mechanism size | Introduces SPARC: cluster composition space offline with LPCA, build one DRGEP reduced mechanism per cluster, then classify CFD cells online. |
| [D'Alessio et al., "Impact of the partitioning method on multidimensional adaptive-chemistry simulations," Energies 2020](https://doi.org/10.3390/en13102567) | Partitioning/classification methods | Local mechanism assignment | Studies how the clustering/partitioning method affects adaptive chemistry accuracy and cost. Important because the classifier controls which reduced mechanism a cell uses. |
| [D'Alessio et al., "Feature extraction and artificial neural networks for the on-the-fly classification of high-dimensional thermochemical spaces in adaptive-chemistry simulations," Data-Centric Engineering 2021](https://doi.org/10.1017/dce.2021.2) | Feature extraction plus artificial neural networks | Online cluster classification | Replaces or accelerates the classification step used to select local reduced mechanisms. Useful for SPARC-type workflows. |
| [Amaduzzi et al., "Automated adaptive chemistry for large eddy simulations of turbulent reacting flows," Combustion and Flame 2024](https://doi.org/10.1016/j.combustflame.2023.113136) | Automated LPCA/VQPCA-style clustering, Bayesian optimization, CSP-based target selection | Local mechanism size | Automates the selection of clustering and reduction settings for LES. The CFD cell is classified during the chemistry step and uses the mechanism of its assigned cluster. |
| [Pagani et al., "An enhanced Sample-Partitioning Adaptive Reduced Chemistry method with a-priori error estimation," Combustion and Flame 2024](https://doi.org/10.1016/j.combustflame.2023.113221) | Enhanced SPARC, error estimation | Local reduced mechanisms | Extends SPARC by adding a-priori error estimation to improve reliability of local mechanism selection. |

## Cell Clustering and Spatial-Domain Treatment

| Paper | ML or clustering tool | What is reduced | Brief description |
|---|---|---|---|
| [Perini, "High-dimensional, unsupervised cell clustering for computationally efficient engine simulations with detailed combustion chemistry," Fuel 2013](https://doi.org/10.1016/j.fuel.2012.11.015) | Bounding-box-constrained k-means | Number of chemistry ODE integrations | Does not reduce the mechanism itself. Instead, CFD cells are clustered every timestep, chemistry is integrated once per cluster, and the result is remapped to cells. |
| [Liang et al., "A pre-partitioned adaptive chemistry methodology for the efficient implementation of combustion chemistry," Combustion and Flame 2009](https://doi.org/10.1016/j.combustflame.2009.05.003) | Pre-partitioning, adaptive chemistry | Number and cost of local chemistry calculations | Early adaptive chemistry framework using pre-partitioned composition space. Important ancestor of SPARC-type methods. |
| [Feng and Zhang, "A dynamic adaptive chemistry and dynamic cell clustering model for reactive flow simulations," Combustion Theory and Modelling 2019](https://doi.org/10.1080/13647830.2019.1575871) | Dynamic adaptive chemistry plus clustering | Local mechanism size and number of ODE solves | Combines local mechanism adaptation with dynamic cell clustering, so both chemistry size and number of integrations are reduced. |
| [Zhou et al., "Application of cell agglomeration algorithm coupled with dynamic adaptive chemistry in spray combustion simulation," Fuel 2020](https://doi.org/10.1016/j.fuel.2019.116754) | Cell agglomeration plus DAC | Local mechanism size and number of integrations | Applies spatial/composition grouping together with dynamic adaptive chemistry in spray combustion. |
| [Cuoci et al., "Tabulation-based sample-partitioning adaptive reduced chemistry and cell agglomeration," Proceedings of the Combustion Institute 2024](https://doi.org/10.1016/j.proci.2024.105386) | SPARC, tabulation, cell agglomeration | Local chemistry cost and repeated ODE integrations | Combines pre-partitioned adaptive reduced chemistry with tabulation and cell agglomeration, connecting local mechanism selection with spatial grouping. |

## Learned Surrogates and Reduced Reaction Models

| Paper | ML tool | What is reduced | Brief description |
|---|---|---|---|
| [Lapeyre et al., "Training convolutional neural networks to estimate turbulent sub-grid scale reaction rates," Combustion and Flame 2019](https://doi.org/10.1016/j.combustflame.2019.02.019) | Convolutional neural network | Reaction-rate closure cost | Does not reduce a kinetic mechanism directly, but replaces expensive closure evaluations with learned reaction-rate estimates. |
| [Pulga et al., "A machine learning methodology for improving the accuracy of laminar flame simulations with reduced chemical kinetics mechanisms," Combustion and Flame 2020](https://doi.org/10.1016/j.combustflame.2020.02.021) | Neural networks and ML workflow | Reduced-mechanism flame-speed modeling | Uses ML to improve the accuracy and cost tradeoff when reduced chemical mechanisms are used for laminar flame-speed tables. |
| [Wan et al., "Machine learning for detailed chemistry reduction in DNS of a syngas turbulent oxy-flame with side-wall effects," Proceedings of the Combustion Institute 2021](https://doi.org/10.1016/j.proci.2020.06.047) | Convolutional neural network | Detailed chemistry source-term prediction | Uses CNNs to predict chemical source terms from local thermochemical images, reducing detailed chemistry cost in DNS. |

## Best Reading Order

1. Start with Perini 2013 if you want to understand online cell clustering.
2. Read Liang 2009 and D'Alessio 2020 to understand pre-partitioned adaptive chemistry.
3. Read De Paola 2021 and Amaduzzi 2024 for ML-assisted classification and automated adaptive chemistry.
4. Read DeePMR 2024, autoencoder 2025, random forest 2026, and GNN 2026 for direct ML-based mechanism reduction.
5. Read the surrogate-model papers only after the mechanism-reduction papers, because they reduce the reaction model in a broader sense rather than pruning species/reactions directly.

## Relationship Between These Approaches

There are three different meanings of "reduction" in this literature:

1. Mechanism-size reduction:
   Fewer species and reactions are integrated.

2. Local adaptive chemistry:
   Different cells use different locally reduced mechanisms.

3. Chemistry-solve reduction:
   The detailed mechanism may remain unchanged, but fewer chemistry ODE systems are solved because similar cells are clustered.

The GNN and ML papers are beginning to connect these ideas: graph models can learn species/reaction importance, classifiers can select local mechanisms, and clustering models can reduce repeated chemistry solves.
