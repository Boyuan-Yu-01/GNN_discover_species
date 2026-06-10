Three types of GNN layers [Link](https://www.youtube.com/watch?v=uF53xsT7mjc)
* Convolutional GNNs: Chapter 7.1
    * [Link](https://www.youtube.com/watch?v=uF53xsT7mjc): convolutional respect *translational invariance* i.e. patterns are interesting irrespective of where they are in the network
    * [Link](https://www.youtube.com/watch?v=uF53xsT7mjc): *Locality*: neighbouring pixels relate much more strongly than distant ones
* Message-passing GNNs: Chapter 7, Weisfeiler-Lehman
* Attentional GNNs: Chapter 5.2.3 Neighborhood Attention
![](plot/3Flavours.png)


### Blue Print
challenge for non-image graph, so define <span style="color:red">Permutation Invariance and Equivariance</span> (from video 10:37) For function that is <span style="color:red">Permutation Invariance</span>, $\mathcal{f}(\underline{\underline{P}}\cdot\underline{X})= \mathcal{f}(\underline{X})$. For function that is <span style="color:red">Permutation equivariance</span>, $\mathcal{f}(\underline{\underline{P}}\cdot\underline{X})= \underline{\underline{P}}\mathcal{f}(\underline{X})$

![](plot/blueprint.png)
![](plot/2.png)
![](plot/3.png)
$\underline{\underline{X}}$ is a matrix where each row $i$ contains information for $i$th node.
$\underline{\underline{A}}$ is the adjacency matrix  
![](plot/4.png)
![](plot/5.png)
![](plot/6.png)

![](plot/convolutional.png)
<span style="color:red">A is merely just the adjaceny matrix (Kronecker delta.) Neighbour weights are determined by the graph structure, not learned individually for each edge.</span>

![](plot/attentional.png)
<span style="color:red">Attentional GNN: the model learns an attention weight for each neighbour.</span> $\alpha_{ij}=a(x_i,x_j)$ and $\alpha_{ij}$ is learnt and depends on node features. This weights can change during training

![](plot/messagePassing.png)

<span style="color:red">Instead of passing scalar weights like in attentional GNN and convolutional GNN, Message-passing GNN passes arbitrary vectors across edges</span>
The $\underline{\psi}(\underline{x_i}, \underline{x_j})$  is a vector, or $\psi$ can also be $\underline{\psi}(\underline{x_i}, \underline{x_j},\underline{e_{i,j}})$ 
<span style="color:red">More expressive than the previous two</span>

### Perspectives on GNNs
#### Node Embedding Techniques
![](plot/7.png)

![](plot/8.png)

![](plot/9.png)
Random walk refines condition from <span style="color:blue">'nodes are close together if they are connected by an edge'</span> to <span style="color:red">'they should be close together if they co-occur in a short random walk'</span>
![](plot/10.png)

![](plot/11.png)

### Spectral GNNs (Graph Transform)
![](plot/12.png)

![](plot/13.png)
Circulant matrices commute and are jointly diagonalisable 
![](plot/14.png)
![](plot/15.png)

![](plot/16.png)

![](plot/17.png)

![](plot/18.png)

![](plot/19.png)

![](plot/20.png)

![](plot/21.png) 
![](plot/22.png)
![](plot/23.png)

![](plot/24.png)

![](plot/25.png)

### Graph Isomorphism Testing
...
