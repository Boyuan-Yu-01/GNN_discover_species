### A graph-convolutional neural network model for the prediction of chemical reactivity, Connor, Bill, etc. 2019 [Main](ConnorBill2019/ConnorBill2019.pdf) [SP**](ConnorBill2019/ESP.pdf)
*  use a graph-based representation of reactant species to propose changes in bond order for organic reactions
* Predicting target: bond formation
* Use up to 5 simultaneous bond changes in each individual reaction <span style="color:blue">Can we use "collision time" to justify? exp: 1 collision needs Xs, 2 collision needs Ys, the system captures below or above certain threshold, so we only counts up to m bond changes...</span>
* <span style="color:red">Important info in SP</span>
#### Thoughts
* <span style="color:blue">Nodal Encoding:</span> Should it change with bond formation? If yes->1-layer Weisfeiler-Lehman encoding. If not->2+-layer Weisfeiler-Lehman encoding
* <span style="color:blue">Potential:</span>The method introduced in this paper might be directly implemented into <span style="color:red">Finding reaction job</span>: Imagine a dewar that contains A, B, C. There are sets of nodes & edges that undergo change in connectivity

![](plot/1.jpg)

### Neural Message Passing for Quantum Chemistry [main](NeuralMessagePassing4QM.pdf)

#### Convolutional network is unable to identify correlations between edge states and node states
The message from neighbour $w$ to centre atom $v$ is 
$$
M(h_v, h_w, e_{vw}=(h_w, e_{vw})
$$
After summing over neighbours:
$$
m_v^{t+1}
=
\left(
\sum_{w \in N(v)} h_w^t,\;
\sum_{w \in N(v)} e_{vw}
\right)
$$

So the result remembers:
- Which neighbor states exist.
- Which bond types exist.
- But it forgets which bond belongs to which neighbor.

#### GG-NN (Gated Graph Neural Network)
 __Numerical Example of a Gated Graph Neural Network:__
This example demonstrates one message-passing step of a Gated Graph Neural Network (GGNN) using a simplified formaldehyde molecule:

```text
    H₁
     \
      C = O
     /
    H₂
```
The molecule contains four atoms:

| Node | Atom | Connected atoms |
|---:|:---:|:---|
| 1 | C | O, H₁, H₂ |
| 2 | O | C |
| 3 | H₁ | C |
| 4 | H₂ | C |

The C–O bond is a double bond, while both C–H bonds are single bonds.

---
## 1. Initial node states

Suppose every atom is represented by a two-dimensional hidden-state vector.
$$
h_{\mathrm C}^{(0)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
h_{\mathrm O}^{(0)}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
$$
$$
h_{\mathrm{H_1}}^{(0)}
=
\begin{bmatrix}
1\\
1
\end{bmatrix},
\qquad
h_{\mathrm{H_2}}^{(0)}
=
\begin{bmatrix}
1\\
1
\end{bmatrix}
$$

These vectors are only illustrative. In a real molecular GNN, the initial node features could contain information such as:
- Element identity
- Formal charge
- Remaining valence
- Radical state
- Aromaticity
---
## 2. Bond-dependent message matrices
A GGNN can use a different trainable matrix for each bond type.
For a single bond, let
$$
A_{\mathrm{single}}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}.
$$
For a double bond, let
$$
A_{\mathrm{double}}
=
\begin{bmatrix}
2 & 1\\
0 & 2
\end{bmatrix}.
$$
These matrices control how the hidden state of a neighboring atom is transformed before it is sent as a message.
The general message equation is

$$
m_v^{(t+1)}
=
\sum_{w\in\mathcal{N}(v)}
A_{e_{vw}}h_w^{(t)},
$$
where:
- $v$ is the receiving node.
- $w$ is a neighboring node.
- $\mathcal{N}(v)$ is the set of neighbors of $v$.
- $e_{vw}$ is the bond type between $v$ and $w$.
- $A_{e_{vw}}$ is the matrix associated with that bond type.
---

## 3. Message received by carbon
Carbon is connected to oxygen by a double bond and to two hydrogen atoms by single bonds.
Therefore,
$$
m_{\mathrm C}^{(1)}
=
A_{\mathrm{double}}h_{\mathrm O}^{(0)}
+
A_{\mathrm{single}}h_{\mathrm{H_1}}^{(0)}
+
A_{\mathrm{single}}h_{\mathrm{H_2}}^{(0)}.
$$
The message from oxygen is
$$
A_{\mathrm{double}}h_{\mathrm O}^{(0)}
=
\begin{bmatrix}
2 & 1\\
0 & 2
\end{bmatrix}
\begin{bmatrix}
0\\
1
\end{bmatrix}
=
\begin{bmatrix}
1\\
2
\end{bmatrix}.
$$
The messages from the two hydrogen atoms are

$$
A_{\mathrm{single}}h_{\mathrm{H_1}}^{(0)}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
1\\
1
\end{bmatrix},
$$

and

$$
A_{\mathrm{single}}h_{\mathrm{H_2}}^{(0)}
=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$

The total message received by carbon is therefore

$$
m_{\mathrm C}^{(1)}
=
\begin{bmatrix}
1\\
2
\end{bmatrix}
+
\begin{bmatrix}
1\\
1
\end{bmatrix}
+
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}.
$$

---

## 4. Message received by oxygen
Oxygen has only one neighbor: carbon. The bond between them is a double bond.
Therefore,
$$
m_{\mathrm O}^{(1)}
=
A_{\mathrm{double}}h_{\mathrm C}^{(0)}.
$$
Numerically,
$$
m_{\mathrm O}^{(1)}
=
\begin{bmatrix}
2 & 1\\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
2\\
0
\end{bmatrix}.
$$

---

## 5. Messages received by the hydrogen atoms
Each hydrogen atom is connected to carbon by a single bond.
For the first hydrogen atom,
$$
m_{\mathrm{H_1}}^{(1)}
=
A_{\mathrm{single}}h_{\mathrm C}^{(0)}
=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
1\\
0
\end{bmatrix}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$
Similarly,
$$
m_{\mathrm{H_2}}^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$
The four aggregated message vectors are therefore
$$
m_{\mathrm C}^{(1)}
=
\begin{bmatrix}
3\\
4
\end{bmatrix},
\qquad
m_{\mathrm O}^{(1)}
=
\begin{bmatrix}
2\\
0
\end{bmatrix},
$$
$$
m_{\mathrm{H_1}}^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix},
\qquad
m_{\mathrm{H_2}}^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

---

## 6. Whole-graph matrix representation
Stack the node states into one vector:
$$
H^{(0)}
=
\begin{bmatrix}
h_{\mathrm C}^{(0)}\\
h_{\mathrm O}^{(0)}\\
h_{\mathrm{H_1}}^{(0)}\\
h_{\mathrm{H_2}}^{(0)}
\end{bmatrix}
=
\begin{bmatrix}
1\\
0\\
0\\
1\\
1\\
1\\
1\\
1
\end{bmatrix}.
$$
The block message matrix is
$$
\mathcal A
=
\begin{bmatrix}
0 & A_{\mathrm{double}} & A_{\mathrm{single}} & A_{\mathrm{single}}\\
A_{\mathrm{double}} & 0 & 0 & 0\\
A_{\mathrm{single}} & 0 & 0 & 0\\
A_{\mathrm{single}} & 0 & 0 & 0
\end{bmatrix}.
$$

Each entry in this matrix is a $2\times2$ block.
Expanding the block matrix gives
$$
\mathcal A
=
\begin{bmatrix}
0&0&2&1&1&0&1&0\\
0&0&0&2&0&1&0&1\\
2&1&0&0&0&0&0&0\\
0&2&0&0&0&0&0&0\\
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0
\end{bmatrix}.
$$

The complete message calculation is
$$
M^{(1)}
=
\mathcal A H^{(0)}.
$$

Substituting the numerical values gives
$$
M^{(1)}
=
\begin{bmatrix}
0&0&2&1&1&0&1&0\\
0&0&0&2&0&1&0&1\\
2&1&0&0&0&0&0&0\\
0&2&0&0&0&0&0&0\\
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0\\
1&0&0&0&0&0&0&0\\
0&1&0&0&0&0&0&0
\end{bmatrix}
\begin{bmatrix}
1\\
0\\
0\\
1\\
1\\
1\\
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
3\\
4\\
2\\
0\\
1\\
0\\
1\\
0
\end{bmatrix}.
$$

Thus,
$$
M^{(1)}
=
\begin{bmatrix}
m_{\mathrm C}^{(1)}\\
m_{\mathrm O}^{(1)}\\
m_{\mathrm{H_1}}^{(1)}\\
m_{\mathrm{H_2}}^{(1)}
\end{bmatrix}.
$$

---

## 7. Gated node update
In a complete GGNN, a GRU (Gated Recurrent Unit) updates each node state:
$$
h_v^{(1)}
=
\operatorname{GRU}
\left(
h_v^{(0)},m_v^{(1)}
\right).
$$
For an easily calculated numerical example, use a simplified gated update.
Let the update gate be

$$
z_v=0.5.
$$

Define the candidate hidden state as
$$
\widetilde h_v^{(1)}
=
\tanh\left(0.1m_v^{(1)}\right).
$$
The updated node state is
$$
h_v^{(1)}
=
(1-z_v)h_v^{(0)}
+
z_v\widetilde h_v^{(1)}.
$$

Because $z_v=0.5$, this becomes
$$
h_v^{(1)}
=
0.5h_v^{(0)}
+
0.5\tanh\left(0.1m_v^{(1)}\right).
$$
This simplified equation retains the main idea of a gated update: the new state combines the previous node state with information received from neighboring nodes.

---
## 8. Updated carbon state
For carbon,
$$
m_{\mathrm C}^{(1)}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}.
$$
The candidate state is
$$
\widetilde h_{\mathrm C}^{(1)}
=
\tanh
\left(
0.1
\begin{bmatrix}
3\\
4
\end{bmatrix}
\right)
=
\tanh
\left(
\begin{bmatrix}
0.3\\
0.4
\end{bmatrix}
\right).
$$
Numerically,
$$
\widetilde h_{\mathrm C}^{(1)}
\approx
\begin{bmatrix}
0.2913\\
0.3799
\end{bmatrix}.
$$
The updated carbon state is
$$
h_{\mathrm C}^{(1)}
=
0.5
\begin{bmatrix}
1\\
0
\end{bmatrix}
+
0.5
\begin{bmatrix}
0.2913\\
0.3799
\end{bmatrix}.
$$
Therefore,
$$
h_{\mathrm C}^{(1)}
\approx
\begin{bmatrix}
0.6457\\
0.1900
\end{bmatrix}.
$$
---
## 9. Updated oxygen state
For oxygen,

$$
m_{\mathrm O}^{(1)}
=
\begin{bmatrix}
2\\
0
\end{bmatrix}.
$$

The candidate state is

$$
\widetilde h_{\mathrm O}^{(1)}
=
\tanh
\left(
\begin{bmatrix}
0.2\\
0
\end{bmatrix}
\right)
\approx
\begin{bmatrix}
0.1974\\
0
\end{bmatrix}.
$$
The updated oxygen state is
$$
h_{\mathrm O}^{(1)}
=
0.5
\begin{bmatrix}
0\\
1
\end{bmatrix}
+
0.5
\begin{bmatrix}
0.1974\\
0
\end{bmatrix}.
$$
Therefore,
$$
h_{\mathrm O}^{(1)}
\approx
\begin{bmatrix}
0.0987\\
0.5000
\end{bmatrix}.
$$

---

## 10. Updated hydrogen states

For each hydrogen atom,

$$
m_{\mathrm H}^{(1)}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}.
$$

The candidate state is

$$
\widetilde h_{\mathrm H}^{(1)}
=
\tanh
\left(
\begin{bmatrix}
0.1\\
0
\end{bmatrix}
\right)
\approx
\begin{bmatrix}
0.0997\\
0
\end{bmatrix}.
$$

The updated hydrogen state is

$$
h_{\mathrm H}^{(1)}
=
0.5
\begin{bmatrix}
1\\
1
\end{bmatrix}
+
0.5
\begin{bmatrix}
0.0997\\
0
\end{bmatrix}.
$$

Therefore,

$$
h_{\mathrm{H_1}}^{(1)}
=
h_{\mathrm{H_2}}^{(1)}
\approx
\begin{bmatrix}
0.5498\\
0.5000
\end{bmatrix}.
$$
---

## 11. Final node-state matrix

After one message-passing step, the node-state matrix is approximately

$$
H_{\mathrm{node}}^{(1)}
=
\begin{bmatrix}
0.6457 & 0.1900\\
0.0987 & 0.5000\\
0.5498 & 0.5000\\
0.5498 & 0.5000
\end{bmatrix}.
$$

The rows correspond to

$$
\left[
\mathrm C,\,
\mathrm O,\,
\mathrm{H_1},\,
\mathrm{H_2}
\right].
$$

---

## 12. Why this GGNN preserves node–edge correlations
Consider the original bonding assignment around carbon:
- C=O is a double bond.
- C–H₁ is a single bond.
- C–H₂ is a single bond.
The carbon message is
$$
m_{\mathrm C}^{(1)}
=
A_{\mathrm{double}}h_{\mathrm O}^{(0)}
+
A_{\mathrm{single}}h_{\mathrm{H_1}}^{(0)}
+
A_{\mathrm{single}}h_{\mathrm{H_2}}^{(0)}
=
\begin{bmatrix}
3\\
4
\end{bmatrix}.
$$

Now consider a hypothetical reassignment:
- C–O is a single bond.
- C=H₁ is treated mathematically as a double bond.
- C–H₂ remains a single bond.
This is not intended to be a chemically realistic formaldehyde structure. It is only used to test whether the network can recognize which bond belongs to which neighboring atom.
The new carbon message would be
$$
m_{\mathrm C,\mathrm{swapped}}^{(1)}
=
A_{\mathrm{single}}h_{\mathrm O}^{(0)}
+
A_{\mathrm{double}}h_{\mathrm{H_1}}^{(0)}
+
A_{\mathrm{single}}h_{\mathrm{H_2}}^{(0)}.
$$
Calculate each contribution:
$$
A_{\mathrm{single}}h_{\mathrm O}^{(0)}
=
\begin{bmatrix}
0\\
1
\end{bmatrix},
$$

$$
A_{\mathrm{double}}h_{\mathrm{H_1}}^{(0)}
=
\begin{bmatrix}
2 & 1\\
0 & 2
\end{bmatrix}
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
3\\
2
\end{bmatrix},
$$
and
$$
A_{\mathrm{single}}h_{\mathrm{H_2}}^{(0)}
=
\begin{bmatrix}
1\\
1
\end{bmatrix}.
$$
Therefore,
$$
m_{\mathrm C,\mathrm{swapped}}^{(1)}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
+
\begin{bmatrix}
3\\
2
\end{bmatrix}
+
\begin{bmatrix}
1\\
1
\end{bmatrix}
=
\begin{bmatrix}
4\\
4
\end{bmatrix}.
$$
The two structures produce different carbon messages:
$$
m_{\mathrm C,\mathrm{original}}^{(1)}
=
\begin{bmatrix}
3\\
4
\end{bmatrix},
\qquad
m_{\mathrm C,\mathrm{swapped}}^{(1)}
=
\begin{bmatrix}
4\\
4
\end{bmatrix}.
$$
Therefore, the GGNN can distinguish which edge type is associated with which neighboring node.
This happens because each neighboring node state is transformed by its corresponding bond matrix before aggregation:
$$
m_v^{(t+1)}
=
\sum_{w\in\mathcal N(v)}
A_{e_{vw}}h_w^{(t)}.
$$

The node information and bond information interact inside each individual message.
By comparison, separately aggregating nodes and edges would give something like
$$
m_v^{(t+1)}
=
\left(
\sum_{w\in\mathcal N(v)}h_w^{(t)},
\,
\sum_{w\in\mathcal N(v)}e_{vw}
\right).
$$

Both hypothetical structures contain the same neighboring atoms and the same total collection of bond types. Consequently, this separate-sum representation cannot determine which bond type belongs to which neighbor.

---
## 13. Connection to a chemical-reaction GNN

In a chemical-reaction environment, a message could be written as
$$
m_{i\leftarrow j}^{(t)}
=
A_{b_{ij}}h_j^{(t)},
$$
where:
- $h_j^{(t)}$ represents atom $j$.
- $b_{ij}$ represents the bond order between atoms $i$ and $j$.
- $A_{b_{ij}}$ is a trainable transformation for that bond type.
The total message received by atom $i$ is
$$
m_i^{(t+1)}
=
\sum_{j\in\mathcal N(i)}
m_{i\leftarrow j}^{(t)}.
$$
The atom state is then updated by
$$
h_i^{(t+1)}
=
\operatorname{GRU}
\left(
h_i^{(t)},m_i^{(t+1)}
\right).
$$
After several message-passing steps, the final node embeddings contain information about the local chemical environments of the atoms. These embeddings can then be used to predict actions such as:
- Increase the bond order between two atoms.
- Decrease the bond order between two atoms.
- Form a bond between two molecules.
- Break a bond and split a molecule.
- Estimate the value or reward of the current reaction state.

#### Graph-level output (Interaction Network)
The final node states are summed:
$$
h_G = \sum_{v\in G}h_v^T$$
A neural network $f$ then converts this graph representation into a prediction:
$$
R = f(\sum_{v\in G}h_v^T)
$$
<span style="color:red">For our case, we could consider the situation where the graph representation is the summation of all subgraph information.</span>

<span style="color:green">Different from GGNN, the interaction network uses a more general form:</span> 
$$
m_{v<-w} = \mathcal{MLP}([h_v, h_w, e_{vw}])
$$
And therefore the interaction network becomes  <span style="color:red">more expressive</span> 