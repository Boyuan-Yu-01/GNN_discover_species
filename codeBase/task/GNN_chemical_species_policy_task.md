# Refined Task Description: GNN-Guided Chemical-Species Discovery in a Combustion System

## 1. Project objective

Develop a graph-neural-network reinforcement-learning policy that explores chemically valid graph-edit sequences and discovers molecular species that can occur in a C/H/O combustion system.

One episode begins with an initialized collection of atoms and molecules in a reaction "oven." At every step, the policy selects a bond-order action. The environment applies the action, updates molecule membership, and returns the next graph observation. The episode ends when the policy selects `STOP`, a safety limit is reached, or the search reaches another explicit terminal condition.

The first objective is to recover species present in an existing chemical kinetic mechanism. Once this behavior is reliable, the same framework can be used to propose additional chemically plausible species for further kinetic or quantum-chemical validation.

This is initially a graph-based, steady-state species-discovery problem. Atomic positions and velocities are out of scope. Temperature, pressure, composition or concentration should eventually be included as global context because they influence which reactions and species are relevant even at steady state.

## 2. Files reviewed

The refined design is based on the files referenced by `task.txt`:

| File                                                                                       | Current responsibility                                                       | Important implication for the policy                                                                           |
| ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/codeBase/task.txt`            | Original problem statement and proposed MCTS/Q-learning tools                | The objective and training targets need to be made more explicit.                                              |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/codeBase/src/obj_node.py`     | `atom`, element identity, and `remaining_valence`                            | Supplies four node features for C/H/O atoms.                                                                   |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/codeBase/src/obj_edge.py`     | `Bond`, endpoints, bond order, and one-hot bond features                     | Supplies existing-edge features for message passing.                                                           |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/codeBase/src/obj_subgraph.py` | Connected `molecule` snapshots and 15 subgraph features                      | Supplies molecule-level context and supports hierarchical action scoring.                                      |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/codeBase/src/obj_graph.py`    | `MoleculeEnv`, graph mutations, pair tables, masks, and GNN observations     | Already defines the state transition and most of the concrete action space.                                    |
| `/Users/boyuanyu/Documents/research/GNN/GNN_discover_species/log/mapping_E_to_P.md`        | Heuristic mapping from bond dissociation energy to bond-breaking probability | Useful as a breakage prior or reward feature, but it should not be treated as a complete reaction probability. |

The four Python class files compile successfully in their current form.

## 3. Existing environment formulation

### 3.1 State

The state at step $t$ is the complete oven graph:

$$
s_t = (V, E_t, C_t, X_t),
$$

where:

- $V$ is the fixed atom inventory;
- $E_t$ is the set of current bonds and bond orders;
- $C_t$ is the partition of atoms into connected molecule components;
- $X_t$ contains node, edge, pair, molecule and future reactor-condition features.

`MoleculeEnv.get_gnn_observation()` already returns:

| Key | Shape | Meaning |
|---|---:|---|
| `x` | $(N,4)$ | C/H/O one-hot identity and remaining valence |
| `edge_index` | $(2,2E)$ | Existing bonds represented in both message-passing directions |
| `edge_attr` | $(2E,3)$ | Single/double/triple one-hot bond order |
| `pair_index` | $(2,P)$ | All unordered atom pairs, where $P=N(N-1)/2$ |
| `current_bond_orders` | $(P,)$ | Current bond order for each candidate pair |
| `maximum_bond_orders` | $(P,)$ | Maximum currently reachable order for each pair |
| `formation_mask` | $(P,3)$ | Legal formation changes of $+1$, $+2$, or $+3$ |
| `breakage_mask` | $(P,3)$ | Legal breakage changes of $-1$, $-2$, or $-3$ |
| `molecule_features` | $(M,15)$ | Element counts, total remaining valence, and bond-type counts |
| `component_index` | $(N,)$ | Molecule row containing each atom |

This separation is appropriate: existing bonds are message-passing edges, whereas all candidate pairs are possible decision edges.

### 3.2 Action

The current action is

$$
a_t=(i,j,d,\Delta b),
$$

where:

- $i$ and $j$ are atom indices;
- $d\in\{\text{formation},\text{breakage}\}$;
- $\Delta b\in\{1,2,3\}$ is the bond-order change.

The environment already masks actions that violate its simplified valence and bond-order rules. The policy should produce one logit for every concrete entry of the two $(P,3)$ masks rather than only choosing a pair and assuming a bond change of one.

### 3.3 Transition

The transition is deterministic once an action is selected:

$$
s_{t+1}=T(s_t,a_t).
$$

Formation can strengthen a bond, close a ring, or merge two molecules. Breakage can weaken a bond, remove a non-bridge bond, or split one molecule into two components.

### 3.4 Result

The result of an episode should be a set or multiset of canonical molecular identities, not Python object identities or atom-index assignments:

$$
\mathcal S_{\mathrm{episode}}
=
\left\{\operatorname{canonical\_species}(m):m\in C_t\right\}.
$$

Canonicalization is essential for comparing generated molecules with the kinetic mechanism and for recognizing the same MCTS state reached by different action sequences.

## 4. Recommended policy architecture

The recommended first policy is an AlphaZero-style **policy-value GNN guided by Monte Carlo Tree Search**. This is a better primary choice than vanilla Q-learning for the current problem because the action space is large and state-dependent, rewards are likely to be sparse, and useful planning may require several graph edits.

### 4.1 Shared edge-aware atom encoder

Use an Interaction-Network, GINE, or custom message-passing layer that processes each sender atom and its bond together. For layer $t$:

$$
m_{i\leftarrow j}^{t+1}
=
M_t\left(h_i^t,h_j^t,e_{ij}\right),
$$

$$
m_i^{t+1}
=
\sum_{j\in\mathcal N(i)}m_{i\leftarrow j}^{t+1},
$$

$$
h_i^{t+1}
=
\operatorname{GRU}_t\left(m_i^{t+1},h_i^t\right).
$$

The message function can be an MLP applied to concatenated features:

$$
M_t\left(h_i,h_j,e_{ij}\right)
=
\operatorname{MLP}_t
\left([h_i\Vert h_j\Vert e_{ij}]\right).
$$

This preserves the association between a neighboring atom and the bond attached to it. Three to five message-passing layers are a reasonable starting point. A GRU is optional but useful for controlling how much earlier atom information is retained at each layer.

### 4.2 Molecule embeddings

Pool atom embeddings separately within each connected molecule:

$$
u_m
=
\operatorname{Pool}\left(\{h_i:i\in m\}\right).
$$

Combine the learned pooled representation with the existing 15 fixed molecule features:

$$
c_m
=
\operatorname{MLP}_{\mathrm{mol}}
\left([u_m\Vert f_m]\right).
$$

This solves the problem noted in `task.txt`: a molecule-level score by itself cannot decide which molecule or atoms should be edited. The molecule embedding becomes context for every candidate pair, while the pair head still selects the exact atoms.

For a pair $(i,j)$, include both $c_{m(i)}$ and $c_{m(j)}$. This lets the policy distinguish:

- an edit within one molecule;
- an edit that merges two molecules;
- atoms in different local chemical environments;
- small radicals interacting with a large hydrocarbon.

### 4.3 Global oven embedding

Pool the molecule embeddings to obtain an oven-level vector:

$$
g_s
=
\operatorname{MLP}_{\mathrm{global}}
\left(
\left[
\operatorname{Pool}\left(\{c_m\}\right)
\Vert x_{\mathrm{reactor}}
\right]
\right).
$$

Future reactor context should include, where available:

- temperature;
- pressure;
- mixture composition or concentration/mole fraction;
- equivalence ratio;
- catalyst or surface identity, if relevant;
- normalized episode step.

### 4.4 Pair representation

Because bond actions are undirected, construct a symmetric representation so that swapping $i$ and $j$ does not change the score. One useful form is

$$
z_{ij}
=
\left[
h_i+h_j
\Vert |h_i-h_j|
\Vert h_i\odot h_j
\Vert c_{m(i)}
\Vert c_{m(j)}
\Vert q_{ij}
\Vert g_s
\right],
$$

where $q_{ij}$ contains pair features such as:

- current bond order;
- maximum reachable bond order;
- same-molecule indicator;
- graph distance when the atoms are in the same molecule;
- whether removing the bond would split the molecule;
- estimated BDE or bond-stability feature for an existing bond;
- endpoint remaining valences.

When the two molecule embeddings are concatenated, their representation should also be made symmetric, for example with their sum and absolute difference.

### 4.5 Separate formation and breakage heads

Use the shared encoder with two action heads:

$$
\ell_{ij,\Delta}^{\mathrm{form}}
=
\operatorname{MLP}_{\mathrm{form}}(z_{ij})_\Delta,
$$

$$
\ell_{ij,\Delta}^{\mathrm{break}}
=
\operatorname{MLP}_{\mathrm{break}}(z_{ij})_\Delta.
$$

Each head returns three logits corresponding to $\Delta b=1,2,3$. Illegal entries are set to negative infinity using the masks already supplied by `MoleculeEnv`.

This design respects the desire to learn growth and breakage separately without building two unrelated GNNs. A shared encoder learns chemical representations from both tasks, while the two heads can be pretrained independently and later fine-tuned together.

The simplest combined policy concatenates all legal formation and breakage logits and applies one softmax. If explicit mode separation is desired, add a small mode head:

$$
q(d\mid s)
=
\operatorname{softmax}
\left(\operatorname{MLP}_{\mathrm{mode}}(g_s)\right),
$$

then define

$$
\pi(a\mid s)
=
q(d\mid s)\,
\pi(i,j,\Delta b\mid d,s).
$$

### 4.6 `STOP` and value heads

Add a legal `STOP` action. This allows the model to learn the useful sequence length instead of requiring a manually selected $n$.

Also predict a scalar value:

$$
V_\theta(s)
=
\operatorname{MLP}_{\mathrm{value}}(g_s),
$$

which estimates the expected final discovery reward from the current oven state. MCTS uses both the policy priors and this value.

## 5. How MCTS should use the policy

One MCTS node represents one complete canonical oven state. One tree edge represents one legal bond action or `STOP`.

During selection, use a PUCT score such as

$$
a^*
=
\underset{a}{\operatorname{argmax}}
\left[
Q(s,a)
+
c_{\mathrm{puct}}P_\theta(s,a)
\frac{\sqrt{\sum_bN(s,b)}}{1+N(s,a)}
\right].
$$

Here:

- $P_\theta(s,a)$ is the masked GNN policy prior;
- $Q(s,a)$ is the average backed-up value;
- $N(s,a)$ is the visit count;
- $c_{\mathrm{puct}}$ controls exploration.

After a fixed number of simulations, select the real environment action from the root visit distribution:

$$
\pi_{\mathrm{MCTS}}(a\mid s)
\propto
N(s,a)^{1/\tau}.
$$

The GNN can then be trained on MCTS results:

$$
\mathcal L
=
-\pi_{\mathrm{MCTS}}^\mathsf T\log P_\theta
+
\lambda_V\left(V_\theta-z\right)^2
+
\lambda_R\mathcal L_{\mathrm{regularization}},
$$

where $z$ is the final episode return.

This is not ordinary policy-gradient training. MCTS produces an improved action distribution, and the network learns to imitate that distribution while predicting the final outcome.

## 6. Using the energy-to-probability mapping correctly

The mapping in `mapping_E_to_P.md` captures a useful qualitative rule: weaker bonds should generally be easier to break. It should initially be treated as a **heuristic prior**, not a calibrated physical probability.

### 6.1 Important physical limitations

- Bond dissociation energy is not generally the activation energy of an elementary reaction.
- A reaction probability depends on temperature, time scale, collision frequency, molecular environment and competing pathways.
- The product of independent bond-survival probabilities assumes bond-breaking events are independent, which is usually not true.
- Using $1-P_{\mathrm{break}}$ as a bond-formation probability is not physically justified. Formation and breakage need different barriers and contexts.
- The proposed sigmoid has a nonzero high-energy limit of $1/(1+k)$ and therefore requires calibration.

If a reaction barrier $E_a$ and time interval $\Delta t$ are available, a more physical construction is

$$
k(T)=A\exp\left(-\frac{E_a}{RT}\right),
$$

$$
P(\text{event in }\Delta t)
=
1-\exp[-k(T)\Delta t].
$$

If only BDE is available, use a fitted monotone calibrator such as

$$
p_{\mathrm{BDE}}
=
\sigma\left(\frac{E_0-E_{\mathrm{BDE}}}{s_E}\right),
$$

and learn $E_0$ and $s_E$ from known reactions or sensitivity data.

### 6.2 Recommended integration

Use the BDE mapping as a bias on the learned breakage logits:

$$
\widetilde\ell_{ij,\Delta}^{\mathrm{break}}
=
\ell_{ij,\Delta}^{\mathrm{break}}
+
\beta\log\left(p_{\mathrm{BDE},ij}+\epsilon\right).
$$

The learned head still selects the molecule, bond and bond-order change. The physical prior merely encourages exploration of plausible weak-bond breakages. The coefficient $\beta$ should be validated and may be annealed during training.

For formation, build a separate prior from reaction data, estimated activation barriers, reaction enthalpy, radical compatibility or learned reactivity. Do not use $1-p_{\mathrm{BDE}}$ as the formation prior.

## 7. Reward design

`MoleculeEnv.step()` currently returns `0.0`, so reward evaluation should be implemented as a separate component rather than mixing project-specific learning logic into the graph objects.

### 7.1 Canonical ground truth

Convert every mechanism species into a canonical graph identifier. Ideally this should include:

- element and isotope labels;
- connectivity and bond order;
- formal charge;
- radical/unpaired-electron state;
- stereochemistry where relevant.

A species list provides weak terminal supervision but does not identify the correct action sequence. If reaction equations from the kinetic mechanism are available, they should also be converted into atom-mapped graph edits for supervised pretraining.

### 7.2 Suggested reward components

A first reward can combine:

$$
r_t
=
w_{\mathrm{new}}r_{\mathrm{new\ known\ species}}
+
w_{\mathrm{phys}}r_{\mathrm{physical\ plausibility}}
-
w_{\mathrm{cycle}}r_{\mathrm{repeated\ state}}
-
w_{\mathrm{step}}.
$$

The terminal reward can compare the discovered species set with the reference set using precision, recall or an $F_\beta$ score.

Important cautions:

- Reward each recognized species only the first time it is discovered in an episode.
- Do not reward duplicate copies repeatedly.
- Do not heavily penalize every species absent from the known mechanism, because some may be valid novel discoveries. Instead, evaluate unknown species with physical or quantum-chemical criteria.
- Use a small step cost and a repeated-state penalty to discourage meaningless formation/breakage cycles.
- Keep hard valence constraints in the action mask rather than representing them only as negative reward.

A practical curriculum is to train for recovery of known species first, then test discovery on species or reaction families held out from training.

## 8. How to determine the number of actions

Do not choose one universal value of $n$ as the learned reaction length. Use:

1. a trainable `STOP` action;
2. `max_steps` as a safety cap;
3. an MCTS search-depth cap for computation control;
4. termination for repeated states or other explicit convergence conditions;
5. validation experiments to choose the maximum horizon.

The present environment's `terminated = not bool(valid_actions)` is not sufficient. Whenever a bond exists, a breakage is generally legal, and formation followed by breakage can revisit an earlier state. Therefore, a reversible system may never naturally run out of actions.

Start with short horizons, such as 4–8 edits, and increase them only after the short-horizon policy learns meaningful behavior. Report performance as a function of horizon instead of assuming one chemically correct value.

## 9. Essential environment additions before MCTS

The current objects provide a good one-trajectory environment, but MCTS needs independent hypothetical branches. Add the following before implementing the search:

### 9.1 State cloning or immutable transitions

`MoleculeEnv.step()` mutates the current object. MCTS must be able to apply different actions from the same parent without corrupting sibling branches. Implement one of:

- `env.clone()` plus mutation of the clone; or
- `state_dict()` and `MoleculeEnv.from_state()`; or
- an immutable `next_state(action)` function.

### 9.2 Canonical state key and transposition table

The action space is reversible and different action orders can reach the same graph. Define

```python
state_key = env.canonical_state_key()
```

and let MCTS reuse statistics for equivalent states. The whole-oven key should be an order-independent multiset of canonical molecule keys.

### 9.3 Explicit `STOP`

Include `STOP` in action encoding, masks, `step()`, and episode information.

### 9.4 Flat action encoder

Provide stable conversions between

```python
(pair_row, action_type, bond_change)
```

and one flat action index. This simplifies masked softmax, replay storage and MCTS statistics.

### 9.5 Separate termination and truncation

- `terminated=True`: `STOP`, desired terminal chemistry, or no legal actions.
- `truncated=True`: maximum step/depth/computation limit.

### 9.6 Chemistry-state extensions

Combustion chemistry contains radicals such as H, O, OH, HO2 and CH3. `remaining_valence` alone conflates unused valence with radical electronic state. Before claiming chemical realism, add at least:

- formal charge;
- radical or unpaired-electron count;
- allowed valence states conditioned on charge/radical state.

This is also necessary to represent cases such as the formal C≡O bonding in carbon monoxide consistently. The simplified neutral-valence model can still be used for the first software prototype, but its results should be labeled as graph-valid rather than chemically validated.

## 10. Scalability

The full formation candidate table has

$$
P=\frac{N(N-1)}{2}=O(N^2)
$$

rows. This is manageable for the current 79-atom example ($P=3081$), but it will become expensive for much larger ovens and especially inside MCTS.

Recommended progression:

1. **MVP:** score all pairs with vectorized PyTorch operations.
2. **Immediate pruning:** breakage candidates are only current bonds; discard formation pairs whose endpoints have no compatible capacity.
3. **Hierarchical proposal:** score molecule pairs, then atom pairs only within the top molecule-pair candidates.
4. **Top-$k$ reactivity proposal:** predict reactive atoms first and construct candidate pairs from the top-$k$ endpoints.
5. **Candidate sampling:** retain all known-positive actions during training and sample hard negatives.

The policy should never loop over $P$ pairs in Python during its forward pass. Gather the two endpoint embeddings using `pair_index` and score all candidates as a tensor batch.

## 11. Training plan

### Phase 0: Environment and data foundation

- Add cloning/serialization, canonical species keys, state keys and `STOP`.
- Add tests for independent MCTS branches and equivalent-state detection.
- Parse the kinetic mechanism into canonical species graphs.
- Parse reaction equations and atom mappings if available.
- Define train/validation/test splits by mechanism, reaction family or species family to avoid leakage.

### Phase 1: Policy-network baseline

- Implement the shared edge-aware GNN encoder.
- Implement molecule pooling and the two $(P,3)$ action heads.
- Apply the existing formation and breakage masks.
- Implement a value head and `STOP` head.
- Confirm invariance to atom ordering and endpoint order.

### Phase 2: Separate head pretraining

- Train the formation head on known formation edits while masking breakage actions.
- Train the breakage head on known breakage edits while masking formation actions.
- Keep the encoder shared; either alternate batches or freeze/unfreeze it carefully.
- If only terminal species are available, generate short backward/forward construction tasks as a curriculum, but label this as synthetic supervision.

### Phase 3: Joint policy training

- Combine the heads under one masked distribution or a learned mode gate.
- Add the calibrated breakage-energy prior.
- Fine-tune on mixed formation/breakage trajectories.
- Add cycle penalties and `STOP` supervision.

### Phase 4: MCTS policy improvement

- Implement PUCT search with state cloning and transpositions.
- Store `(state, MCTS visit distribution, terminal return)` in a replay buffer.
- Train policy and value outputs from this buffer.
- Begin with few simulations and short depth; scale after correctness is established.

### Phase 5: Physical validation and discovery

- Re-evaluate high-value unknown species with a stronger chemistry model, kinetics calculation, quantum chemistry or expert rules.
- Feed validated examples back into the reward/value training data.
- Compare newly found species against held-out kinetic mechanisms rather than only the training mechanism.

## 12. Recommended code organization

Keep data objects, environment transitions, learning, search and reward logic separate:

```text
codeBase/
├── src/
│   ├── obj_node.py
│   ├── obj_edge.py
│   ├── obj_subgraph.py
│   ├── obj_graph.py
│   ├── chemistry/
│   │   ├── canonicalize.py
│   │   ├── mechanism_loader.py
│   │   └── energy_prior.py
│   ├── policy/
│   │   ├── message_passing.py
│   │   ├── action_heads.py
│   │   └── policy_value_net.py
│   ├── search/
│   │   ├── mcts_node.py
│   │   └── mcts.py
│   ├── rewards/
│   │   └── species_reward.py
│   └── training/
│       ├── replay_buffer.py
│       ├── pretrain_heads.py
│       └── train_mcts.py
└── tests/
    ├── test_environment_branching.py
    ├── test_canonicalization.py
    ├── test_action_masks.py
    └── test_policy_invariance.py
```

The graph object should continue to enforce graph-state consistency. It should not contain the GNN, optimizer, MCTS statistics or project-specific reward.

## 13. Recommended MVP

The smallest scientifically interpretable experiment is:

1. Use a small C/H/O atom inventory and a maximum horizon of six edits.
2. Add `STOP`, state cloning and canonical graph keys.
3. Use a three-layer edge-aware GNN with 64-dimensional atom embeddings.
4. Pool atom embeddings per molecule and concatenate the 15 current molecule features.
5. Score all concrete formation and breakage actions with separate masked heads.
6. Pretrain from known graph edits if reactions are available.
7. Run 25–100 MCTS simulations per selected environment action.
8. Reward first-time recovery of canonical ground-truth species, apply a small step/cycle cost, and use the BDE model only as a breakage prior.
9. Evaluate recovery rate, validity, search cost and sensitivity to search depth.

## 14. Evaluation metrics

Report at least:

- canonical species precision, recall and $F_1$ against the reference mechanism;
- held-out species/reaction-family recovery;
- fraction of graph-valid and chemically validated species;
- number of unique species discovered per evaluated MCTS node;
- number of expensive energy calculations required;
- mean branching factor before and after pruning;
- repeated-state/transposition rate;
- mean episode length and frequency of `STOP`;
- formation-head and breakage-head top-$k$ accuracy where labeled actions exist;
- ablations without molecule features, without the BDE prior, and without MCTS.

## 15. Key recommendations

1. Use **MCTS plus a policy-value GNN** as the main method; use masked Double DQN only as a baseline.
2. Use **one shared graph encoder with separate formation and breakage heads**, rather than two completely independent models.
3. Let molecule features provide context to every atom-pair score; do not ask one molecule scalar to select the exact action.
4. Add **`STOP`** so the policy can learn sequence length. Keep `max_steps` only as a safety limit.
5. Treat the BDE-to-probability function as a **calibrated breakage prior or reward feature**, not as a complete reaction probability and not as the complement of formation probability.
6. Implement **state cloning and canonical state hashing before MCTS**.
7. Add charge and radical features before interpreting results as realistic combustion chemistry.
8. If possible, train from known reaction transitions, not only the final species list. A species-only target does not uniquely determine the bond-edit path.
9. Consider coordinated reaction macro-actions later. Many elementary reactions break one bond and form another together; forcing every reaction into isolated single edits can create unphysical intermediate states.

## 16. Information needed for the next implementation stage

The next implementation should establish:

- the kinetic-mechanism file format;
- whether reaction equations and atom mappings are available in addition to species names;
- which reactor conditions are known;
- whether hydrogen will remain explicit;
- whether radicals and charged species must be supported in the first model;
- the intended maximum atom inventory and expected MCTS computation budget;
- whether the first target is known-species recovery, genuinely novel discovery, or both.

These choices affect the data loader and reward, but they do not change the recommended policy architecture above.
