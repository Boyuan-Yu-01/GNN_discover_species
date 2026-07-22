## Parameter of interest:
$$
A \qquad E_{\mathrm{ref}} \qquad E_{\mathrm{b.d.e}} \qquad k
$$
* Tunable parameters: `A`, `E_ref`, & `k` are tunable parameters. The aim of conducting sensitivity analysis of those parameter is to find the most robust parameters.
* The aim of conducting sensitivity analysis on `E_b.d.e.` is that <span style="color:red">not all bonds are equal</span>
Sensitivity analysis provides an additional way to compare these triples.

<span style="color:red">??? ALSO, THE PREVIOUS DETERMINATION OF $P_{(O-O)}$ & $P_{(C-C)}$ ARE QUITE RANDOM</span>

Sensitivity measures how much a predicted probability changes when a parameter changes slightly. A low-sensitivity parameter set is locally robust. A high-sensitivity set is fragile even if it matches the target probabilities exactly.

We can use the sensitivity analysis to therefore:
* distinguish robust parameter triples from fragile ones
* quantify how uncertainty in fitted parameters propagates into $P_{break}$ 

## Derive analytical sensitivity
#### Equation definition
$$
P = \frac{1}{1+k\ exp{-A\ exp(-E_{b.d.e}/E_{ref})}}
$$
#### Define Auxiliary Equation
$$
x_b = A \cdot exp(-\frac{E_b}{E_{ref}})
$$
$$
z=x_b-ln(k)
$$
So that,
$$
P_b(z)=\sigma(z)=\frac{1}{1+exp(-z)}
$$
And,
$$
\dot{P_b}(z) = P_b \cdot (1-P_b)
$$ A easier illustration of the functions' relationship is:
$$P_b=\mathcal{f}(z)$$
$$z=\mathcal{g}(x_b,k)$$
$$x_b=\mathcal{h}(A,E_b,E_{ref})$$
With these, the sensitivity terms are easy to derive:
#### Analytical Sensitivity Terms
The Formula for sensitivity analysis:
$$
\frac{\partial P}{\partial\log\theta}=\theta\frac{\partial P}{\partial\theta}
$$
And the derived sensitivity analysis for each term is:

Sensitivity to $k$:
$$
\boxed{S_{k}=\frac{\partial P_b}{\partial\log k}=-P_b(1-P_b)}
$$
Sensitivity to $A$:
$$
\boxed{S_{A}=\frac{\partial P_b}{\partial\log A}=P_b(1-P_b)x_b}
$$
Sensitivity to $E_{ref}$:
$$
\boxed{S_{E_{\mathrm{ref}}}=\frac{\partial P_b}{\partial\log E_{\mathrm{ref}}}=P_b(1-P_b)x_b\frac{E_b}{E_{\mathrm{ref}}}}
$$

<span style="color:blue">Sensitivity to E_b.d.e.</span>:
$$
\boxed{S_{E_{b.d.e.}}=\frac{\partial P_b}{\partial\log E_{\mathrm{ref}}} = -P_b(1-P_b)x_b\frac{E_b}{E_{ref}}}
$$
Together, we have
$$
\delta P_b= S_{k}\frac{\delta k}{k} + S_A \frac{\delta A}{A} + S_{E_{ref}}\frac{\delta E_{ref}}{E_{ref}} + S_{E_{b.d.e.}}\frac{\delta E_{b.d.e.}}{E_{b.d.e.}}
$$
## Specifically for the `O-O` & `C-C` criteria:
$$
J=\begin{bmatrix}S_{A,\mathrm{O-O}}& S_{E_{\mathrm{ref}},\mathrm{O-O}}& S_{k,\mathrm{O-O}} \\
S_{A,\mathrm{C-C}}&S_{E_{\mathrm{ref}},\mathrm{C-C}}& S_{k,\mathrm{C-C}}
\end{bmatrix}
$$
A simple combined sensitivity is the Frobenius norm:

$$
\boxed{
S_{\mathrm{combined}}=\lVert J\rVert_F=\sqrt{\sum_{b\in\{\mathrm{O-O},\mathrm{C-C}\}}\sum_{\theta\in\{A,E_{\mathrm{ref}},k\}}S_{\theta,b}^2}}
$$
Among parameter triples with comparable target error, a smaller $S_{\mathrm{combined}}$​ indicates a more locally robust triple.

### Including Sensitivity in Parameter Selection

If the desired probabilities only need to be near their targets, define a target-fitting loss such as
$$
L_{\mathrm{target}}=\left(\frac{P_{\mathrm{O-O}}-0.73}{\epsilon_{\mathrm{O-O}}}\right)^2+\left(\frac{P_{\mathrm{C-C}}-0.029}{\epsilon_{\mathrm{C-C}}}\right)^2
$$

where each $\epsilon$ is an acceptable probability deviation.
Sensitivity can then be included as a secondary penalty:
$$
\boxed{L_{\mathrm{selection}}=L_{\mathrm{target}}+\lambda_{\mathrm{sensitivity}}S_{\mathrm{combined}}^2}
$$
The value of $\lambda_{\mathrm{sensitivity}}$ determines the tradeoff:
- a small value prioritizes matching the two probability targets;
- a large value prioritizes robustness;
- an excessively large value may choose a stable curve that no longer matches the desired probabilities adequately.
