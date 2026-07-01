# Literature Search 2: GNN and Machine Learning for Protein, Catalysis, Battery Chemistry, and Materials

Search date: 2026-06-17

## Scope

This list collects representative machine learning and graph neural network papers in protein structure/design, catalysis, battery chemistry, and materials discovery. The common thread is learning from molecular, protein, crystal, surface, or reaction graph structure to predict properties or design new candidates.

## Protein Structure, Interaction, and Design

| Paper | ML tool | What it does | Brief description |
|---|---|---|---|
| [Jumper et al., "Highly accurate protein structure prediction with AlphaFold," Nature 2021](https://doi.org/10.1038/s41586-021-03819-2) | Deep learning with attention and structure modules | Predicts protein 3D structures from sequence. | Landmark result showing that neural networks can learn protein structure with near-experimental accuracy for many cases. |
| [Baek et al., "Accurate prediction of protein structures and interactions using a three-track neural network," Science 2021](https://doi.org/10.1126/science.abj8754) | RoseTTAFold, three-track neural network | Predicts protein structures and protein-protein interactions. | Important alternative architecture to AlphaFold; useful for understanding sequence-structure-distance coupling. |
| [Evans et al., "Protein complex prediction with AlphaFold-Multimer," bioRxiv 2021](https://doi.org/10.1101/2021.10.04.463034) | Deep learning for protein complexes | Predicts multimeric protein complex structures. | Extends protein ML from single-chain folding to interaction prediction. |
| [Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3," Nature 2024](https://doi.org/10.1038/s41586-024-07487-w) | Diffusion-style biomolecular structure model | Predicts structures of proteins, nucleic acids, small molecules, ions, and modified residues. | Broadens protein ML toward general biomolecular complex modeling. |
| [Dauparas et al., "Robust deep learning-based protein sequence design using ProteinMPNN," Science 2022](https://doi.org/10.1126/science.add2187) | Message-passing neural network | Designs protein sequences for target backbone structures. | A clean GNN-style example: learn sequence design on protein residue graphs. |
| [Watson et al., "De novo design of protein structure and function with RFdiffusion," Nature 2023](https://doi.org/10.1038/s41586-023-06415-8) | Diffusion model | Generates new protein backbones and functional motifs. | Representative of generative ML for protein design. |
| [Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model," Science 2023](https://doi.org/10.1126/science.ade2574) | Protein language model | Predicts protein structures from learned sequence representations. | Shows that protein sequence language models can encode structural information. |

## Catalysis and Surface Chemistry

| Paper | ML tool | What it does | Brief description |
|---|---|---|---|
| [Chanussot et al., "The Open Catalyst 2020 (OC20) Dataset and Community Challenges," ACS Catalysis 2021](https://doi.org/10.1021/acscatal.0c04525) | Large dataset for GNNs and ML potentials | Provides DFT relaxations and adsorption energies for catalyst surfaces. | Major benchmark for GNNs in heterogeneous catalysis. |
| [Tran et al., "The Open Catalyst 2022 (OC22) Dataset and Challenges for Oxide Electrocatalysts," ACS Catalysis 2023](https://doi.org/10.1021/acscatal.2c05426) | Dataset for oxide electrocatalysis | Extends catalyst ML benchmarks to oxide surfaces and electrocatalysis. | Useful for battery/electrocatalysis-adjacent surface chemistry. |
| [Gasteiger et al., "Directional Message Passing for Molecular Graphs," ICLR 2020](https://openreview.net/forum?id=B1eWbxStPH) | DimeNet, directional message passing | Learns molecular energies and forces using angular information. | Not only catalysis, but central to many later atomistic GNNs used for surfaces. |
| [Gasteiger et al., "GemNet: Universal Directional Graph Neural Networks for Molecules," NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/35cf8659cfcb13224cbd47863a34fc58-Abstract.html) | Directional GNN | Predicts molecular and atomistic properties with geometric information. | Influential architecture family for atomistic chemistry and catalysis tasks. |
| [Jinnouchi et al., "On-the-fly active learning of interatomic potentials for large-scale atomistic simulations," Journal of Physical Chemistry Letters 2020](https://doi.org/10.1021/acs.jpclett.0c01061) | Active learning ML potential | Builds potentials during simulation. | Important for catalysis and materials simulations where DFT is too expensive. |
| [Gasteiger et al., "GemNet-OC: Developing Graph Neural Networks for Large and Diverse Molecular Simulation Datasets," arXiv 2022](https://arxiv.org/abs/2204.02782) | Directional/equivariant graph neural network | Targets large catalyst-surface datasets such as OC20. | A practical GNN architecture paper for adsorption-energy and catalyst-surface learning. |

## Battery Chemistry, Solid Electrolytes, and Materials Discovery

| Paper | ML tool | What it does | Brief description |
|---|---|---|---|
| [Xie and Grossman, "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties," Physical Review Letters 2018](https://doi.org/10.1103/PhysRevLett.120.145301) | Crystal graph convolutional neural network | Predicts crystal material properties from graph representations. | Foundational GNN paper for inorganic materials, including battery-relevant crystals. |
| [Chen et al., "Graph Networks as a Universal Machine Learning Framework for Molecules and Crystals," Chemistry of Materials 2019](https://doi.org/10.1021/acs.chemmater.9b01294) | MEGNet graph network | Predicts molecular and crystal properties using graph networks. | Important general framework for materials chemistry and crystal property prediction. |
| [Chen and Ong, "A universal graph deep learning interatomic potential for the periodic table," Nature Computational Science 2022](https://doi.org/10.1038/s43588-022-00349-3) | M3GNet graph neural network potential | Predicts energies, forces, stresses, and relaxations across the periodic table. | Useful for screening battery materials and solid-state chemistry at scale. |
| [Deng et al., "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling," Nature Machine Intelligence 2023](https://doi.org/10.1038/s42256-023-00716-3) | Charge-informed graph neural network potential | Learns atomistic energies and magnetic/charge-informed behavior. | Relevant for transition-metal oxides and battery electrode materials. |
| [Merchant et al., "Scaling deep learning for materials discovery," Nature 2023](https://doi.org/10.1038/s41586-023-06735-9) | GNoME, graph networks for materials exploration | Discovers many stable inorganic crystals using deep learning and DFT validation. | Large-scale example of ML-driven inorganic materials discovery, including battery-relevant chemistries. |
| [Sendek et al., "Holistic computational structure screening of more than 12,000 candidates for solid lithium-ion conductor materials," Energy and Environmental Science 2017](https://doi.org/10.1039/C6EE02697D) | Machine learning screening | Screens solid Li-ion conductor candidates. | Early and influential ML workflow for battery solid electrolytes. |
| [Sendek et al., "Machine learning-assisted discovery of solid Li-ion conducting materials," Chemistry of Materials 2019](https://doi.org/10.1021/acs.chemmater.8b03272) | ML-guided materials discovery | Identifies candidate lithium-ion conductors. | Directly relevant to battery materials discovery. |
| [Cubuk et al., "Unsupervised discovery of solid-state lithium ion conductors," Nature Communications 2019](https://doi.org/10.1038/s41467-019-09492-8) | Unsupervised learning | Finds solid-state Li-ion conductor candidates from materials data. | Shows unsupervised ML can reveal battery electrolyte candidates without explicit labels. |

## Useful Reading Path

1. For proteins: read AlphaFold2, RoseTTAFold, ProteinMPNN, and RFdiffusion.
2. For catalysis: read OC20 first, then DimeNet/GemNet-style geometric GNN papers.
3. For battery/materials chemistry: read CGCNN, MEGNet, M3GNet, GNoME, and the solid electrolyte ML screening papers.

## Connection To Chemical Reaction Discovery

The same ideas recur across domains:

- represent chemistry as a graph;
- learn local environments;
- predict energy, force, stability, activity, or sequence compatibility;
- use the model for screening, design, or accelerated simulation.

For combustion or reaction mechanism work, the most transferable ideas are graph representations, active learning, uncertainty-aware prediction, and combining ML predictions with physics-based validation.
