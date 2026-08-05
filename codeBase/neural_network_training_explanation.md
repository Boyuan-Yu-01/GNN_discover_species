# How this neural network is trained

This network is not being trained in the usual supervised way with a fixed dataset of correct answers. Instead, it is trained in a reinforcement-learning-style loop where the model tries to build molecules, receives a reward based on whether the final result is chemically valid, and then adjusts its weights so it becomes more likely to make good moves next time.

## 1. The model’s job

The neural network is a graph neural network, and its job is to decide what action to take next in the molecule-building environment.

At each step, it outputs three things:

- `grow_logits`: which atom should grow, and whether it should be H or O
- `connect_logits`: which pair of atoms should be connected, and whether that bond should be single or double
- `term_logit`: whether the episode should stop

So the model is learning a policy:
- “Given the current molecule graph, which action should I take?”

## 2. The training loop begins with an episode

Each epoch is one episode.

Inside the loop:

- a fresh environment is created
- the molecule starts from four atoms: two H and two O
- the model repeatedly chooses actions to grow or connect atoms

This is similar to an agent playing a game, except the “game” is molecular construction.

## 3. The environment is converted into a graph

Before the model can reason about the molecule, the current state is turned into a graph representation:

- node features encode atom types
  - H becomes one feature vector
  - O becomes another
- edges encode bond structure

So the GNN sees the molecule as a graph of atoms and bonds.

This is exactly what `get_pyg_data()` is doing.

## 4. The model makes a forward pass

The network runs:

- graph convolution layers aggregate information from neighboring atoms
- node embeddings are produced
- the model scores possible actions

This is the purpose of `forward()`.

The important idea is:

- each atom does not just look at itself
- it also looks at nearby atoms through bonds
- that lets the network understand local chemical structure

So the model can learn patterns like:
- “this oxygen is connected to a hydrogen”
- “this atom is near a nearly complete valence structure”
- “this fragment looks like a valid molecule”

## 5. Invalid actions are blocked by masks

Before the model is used to choose an action, the code builds masks that say which actions are legal.

For example:
- a self-bond is illegal
- a bond that would exceed valency is illegal
- a growth action from an atom that has no remaining valency is illegal

Those invalid actions are given extremely negative logits.

That means:
- the model can still score them
- but they become effectively impossible to choose

This is important because the network is not learning from arbitrary bad actions; it is learning inside a chemically constrained action space.

## 6. The model chooses an action

The code then uses the model’s scores to choose a move.

It does this with a noisy argmax:

- the model produces logits for all possible actions
- some random noise is added
- the action with the highest value is chosen

That noise is useful because it gives the agent exploration:
- early in training, it should not always pick the same action
- it should try different possibilities to discover useful strategies

So the training process is not just “follow the current best guess”; it is “try, observe, learn”.

## 7. The chosen action is applied to the environment

Once the action is selected, the environment updates:

- new atoms may be added
- bonds may be created or strengthened
- valency counters change
- connected fragments are tracked

This is what `step()` is doing.

The environment is therefore the “world” that the agent interacts with.

## 8. The episode ends and the final molecule is evaluated

After a number of steps, the episode stops.

Then the code checks the final composition by calling `evaluate_inventory()`.

That function asks:

- “Do the final fragments correspond to valid molecules like H2, O2, OH, H2O, etc.?”

If yes, the episode is considered successful.
If no, it is considered a failure.

## 9. The reward is assigned

This is the key learning signal.

The code assigns:

- reward = 1.0 if the final inventory is valid
- reward = -1.0 if it is invalid

That reward is very simple, but it is enough to start training.

The model is effectively learning:

- “actions that helped produce valid molecules are good”
- “actions that led to invalid molecules are bad”

## 10. The loss is computed

There are two kinds of loss here.

### A. Termination loss

The model also has a termination head that predicts whether the episode should stop.

The code compares:
- the model’s predicted stop score
- the true outcome of the episode

If the episode was successful, the target is 1.
If it failed, the target is 0.

This teaches the network:
- “when should I stop building?”

That is what `term_loss` is for.

### B. Policy loss

This is the more important learning signal for action selection.

The code stores the log-probability of the actions that were chosen during the episode.

Then it computes a loss that increases when a chosen action led to a bad outcome and decreases when it led to a good outcome.

So the optimizer pushes the model to:
- increase the probability of actions that led to success
- decrease the probability of actions that led to failure

That is the core of the learning signal.

## 11. Backpropagation updates the weights

Once the losses are computed, the code does:

- `total_loss.backward()`
- `optimizer.step()`

That is the actual training step.

The optimizer adjusts the network weights so that:
- the model becomes better at predicting good actions
- the termination head becomes better at deciding when to stop

## 12. Why this is a kind of policy gradient learning

This is not a normal classification setup with hard labels. It is closer to policy gradient methods:

- the model outputs action probabilities
- the environment gives a reward
- the model updates itself to favor actions that produce good outcomes

So the training is driven by trial and error.

## 13. What the network is really learning

Over many epochs, the network is trying to learn:

- which atoms should be grown
- which atom pairs should be connected
- what kinds of structures are chemically valid
- when to stop constructing

It does not learn from an external teacher saying “this action is correct.”
It learns from the final outcome of its own trajectory.

## 14. The biggest limitation in this implementation

This is a very simple prototype, so the training is somewhat crude.

A few important limitations are:

- the reward is very sparse and binary
- all actions in the episode are updated based on the final outcome
- there is no replay buffer, no baseline, and no advanced RL stabilization
- the model is trained on very small synthetic episodes

So this is more of a toy demonstration of learning through environment interaction than a fully robust RL training setup.

## 15. In one sentence

The network is trained by letting it build molecules step by step, giving it a reward when the final structure is valid, and then using backpropagation to make the actions that led to success more likely next time.
