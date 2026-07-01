## Jun. 10th, 2026

Three types of GNN layers [Link](https://www.youtube.com/watch?v=uF53xsT7mjc)
[summary](08062026/summary.md)

### Potential Scope of the project
<span style="color:blue">For chemistry discovery purpose, can we describe the scope of the project to be: finding the intermediate chemistry species, incorporating which may have significant improvement in combustion modeling.</span>
OR
<span style="color:blue">Can we produce a surrogate model that 1. satisfy chemistry laws and 2. the existence of this 1 species may replace several other species that presents in a given kinetic model</span>
IRREVALENT
<span style="color:blue">Use GNN for producing skeleton model </span>
### Read through ``obsolete/tutorial/3_train_small.py``
* <span style="color:red">Question: </span> about ``3_train_small.py``: Is there a repeated definition about the edge index between ``line 77-78``?
* <span style="color:red">Question: </span> higher bond order -> shorter bond length -> stronger bond. <span style="color:green">why do we want to train B.O. and bond existence separately? </span> <span style="color:red">Existence is for nodal existence</span> 
<span style="color:blue">Understanding </span> about the ```3_train_small.py```: create 100 datasets, each dataset contains 4 nodes (H, H, O, O) the bond are randomly generated s.t. either all atoms are connected or non of them are connect. The dist and other parameters are defined uniformly too.

The training is to train the bond_predictor & existence_predictor to recognise (predict) the bond order and bond existence.

The training loss consists of MSELoss on bond order and BCELoss on existence

<span style="color:red">Caveats: </span>
* As mentioned in question before
* Uniformly definition of distance, B.O. etc.
* y_bond & y_exist are very simple

### Read through ``obsolete/tutorial/6_bo_4node.py``
* <span style="color:green">Major difference:</span> ``CombustionConv`` has aggregator of mean instead of add
* <span style="color:green">Major difference:</span> Instead of having 16 input features each for two atoms, it now has 193 inputs. 
* <span style="color:green">Major difference:</span> introduce sum, abs_diff, multi, 4 types of commutative operator manufactured features.

<span style="color:blue">Understanding </span>about ``/tutorial/4_bo_4node.py``: still have a small 4 atoms training data.

A sample_type template has been given as a list that contains ``2 H + 2 O``. The algorithm then generate 1000 sample examples from the sample_type template. Sample example differ from each other by:
* having different indices for each atom (for example: template(0) has ('H', 'H', 'O', 'O'), but the indices can be (1,2,3,4) or (2,4,1,3)) <span style="color:red">CHALLENGE but haven't proof it : permutation invariance and equivariance </span>
* for each case, when there is a bond formation, the distance is then between 0.7 to 1.6, and when there is no bond formation, the distance is between 2.2 to 3.5
* assign the bond value of each connected pair using bonded Pauling's formula (taken into account of the pairing between different species)

## Thoughts
* for any atom that has $\textit{n}$ features, let's have $m$ features that is identical across same elements and $n-m$ species that has features related to the bond formation.
* How to distinguish OH and OH*?

## Jun. 17th, 2026
### Concept of graph reinforcement learning: potentially applicable to GNN based skeletal kinetic model production [Link](../../tutorial/graphReinforcement/graph_reinforcement_learning_case.md)

### Literature Search on Adaptive Chemistry
* Further literature for the Adaptive Chemistry [Link](../../../AdaptiveChemistry/model_reduction_papers.md)
* Exciting discovery on Spatial Reduction papers [Link](../../../SpatialReduction/literature/adaptive_chemistry_spatial_treatment_literature.md) (in total 27 papers)

## July. 1st, 2026
### Read GNN-RL [CodeLink](../gnn_combustion/Xu_training/RL/grow_train_animation)     [README_gpt](../gnn_combustion/Xu_training/RL/README_gpt.md)
#### class AdvanceMoleculeEnv
* *def get_pyg_data*: <span style="color:blue">Encode info</span>
	* nodal encoding: one hot feature (H:[1,0]; O:[0,1])
	* edge encoding: a <span style="color:red">2 by 2n matrix, undirected edge encoding</span>
* def *get_valid_action_masks*: <span style="color:blue">Mask out using valence rule</span>
	* valance rule hard encoded. Only bonds between atoms with spare capacity and that aren't already connected are allowed. <span style="color:red">CAVEAT?</span>
	* <span style="color:red">The "node_grow_mask" is either 1 (able to grow on) or 0 (max valency),could be problematic since the information is not enough</span> 
	* <span style="color:blue">This may also prevent species such as "H*" appear as valid species</span> (though not appeared in the valid species list)
	* refactor local nodal indices to global nodal indices for different epoch, also create possible bonding
* def *step*:<span style="color:blue"> Grow, connect, or terminate</span>
	* Define three update actions to the environment: <span style="color:green">growth, connection, and terminate</span> 
	* <span style="color:blue">growth</span>: add a new atom and create its bond-count entry. 
	* <span style="color:blue">connection</span>: add the global edge, update valency counts, and merge the two connected components so the environment knows those atoms now belong to the same molecule. After the growth, apply function <span style="color:red">get_valid_action_masks</span> to mask out atoms that is no longer available. 
* def *evaluate_inventory*: <span style="color:blue"> Convert each component into a formula string</span> 

#### class GrowthGNN
* def _init_: <span style="color:blue"> define MLP architecture</span> 
	* 2 MLP layer with _"hidden_dim"_ number of protons, then this architecture is connected to <span style="color:blue"> grow_head, connect_head, and termination_layer</span> 
* def _forward_: <span style="color:blue"> forward passing the neural network to generate grow_logits, connect_logits, & term_logit</span> 
	* In *grow_logit*, a very large negative number has been added to curb an invalid chemical growth <span style="color:red">CAVEAT?</span>
	* Similar for connect_logit <span style="color:red">CAVEAT?</span>
	* Termination layer uses the mean of all atoms with the termination_layer output

![](plot/MLP_structure_30062026.pdf)

#### Training Loop
*This is closer to a compact policy-gradient-style toy loop than to supervised training: there are no labeled target graphs. The model is rewarded when the final disconnected fragments are all valid H/O species.*

