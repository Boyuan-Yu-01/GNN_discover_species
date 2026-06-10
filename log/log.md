## Jun. 7th, 2026

Three types of GNN layers [Link](https://www.youtube.com/watch?v=uF53xsT7mjc)
[summary](08062026/summary.md)

### Potential Scope of the project
<span style="color:blue">For chemistry discovery purpose, can we describe the scope of the project to be: finding the intermediate chemistry species, incorporating which may have significant improvement in combustion modeling.</span>
OR
<span style="color:blue">Can we produce a surrogate model that 1. satisfy chemistry laws and 2. the existence of this 1 species may replace several other species that presents in a given kinetic model</span>
IRREVALENT
<span style="color:blue">Use GNN for producing skeleton model </span>
### Read through ``/tutorial/3_train_small.py``
* <span style="color:red">Question: </span> about ``3_train_small.py``: Is there a repeated definition about the edge index between ``line 77-78``?
* <span style="color:red">Question: </span> higher bond order -> shorter bond length -> stronger bond. <span style="color:green">why do we want to train B.O. and bond existence separately? </span> 
<span style="color:blue">Understanding </span> about the ```3_train_small.py```: create 100 datasets, each dataset contains 4 nodes (H, H, O, O) the bond are randomly generated s.t. either all atoms are connected or non of them are connect. The dist and other parameters are defined uniformly too.

The training is to train the bond_predictor & existence_predictor to recognise (predict) the bond order and bond existence.

The training loss consists of MSELoss on bond order and BCELoss on existence

<span style="color:red">Caveats: </span>
* As mentioned in question before
* Uniformly definition of distance, B.O. etc.
* y_bond & y_exist are very simple

### Read through ``/tutorial/4_bo_4node.py``
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