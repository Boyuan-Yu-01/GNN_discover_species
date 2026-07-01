### A graph-convolutional neural network model for the prediction of chemical reactivity, Connor, Bill, etc. 2019 [Main](ConnorBill2019/ConnorBill2019.pdf) [SP**](ConnorBill2019/ESP.pdf)
*  use a graph-based representation of reactant species to propose changes in bond order for organic reactions
* Predicting target: bond formation
* Use up to 5 simultaneous bond changes in each individual reaction <span style="color:blue">Can we use "collision time" to justify? exp: 1 collision needs Xs, 2 collision needs Ys, the system captures below or above certain threshold, so we only counts up to m bond changes...</span>
* <span style="color:red">Important info in SP</span>
#### Thoughts
* <span style="color:blue">Nodal Encoding:</span> Should it change with bond formation? If yes->1-layer Weisfeiler-Lehman encoding. If not->2+-layer Weisfeiler-Lehman encoding
* <span style="color:blue">Potential:</span>The method introduced in this paper might be directly implemented into <span style="color:red">Finding reaction job</span>: Imagine a dewar that contains A, B, C. There are sets of nodes & edges that undergo change in connectivity

![](plot/1.jpg)