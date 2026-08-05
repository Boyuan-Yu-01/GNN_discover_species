# Rollouts in Monte Carlo Tree Search

In **Monte Carlo Tree Search (MCTS)**, a **rollout**—also called a **playout** or **simulation**—is the process of estimating the value of a node by simulating what happens after reaching that node until a terminal state or a predefined depth.

A rollout is one of the four fundamental steps of MCTS:

1. **Selection**
2. **Expansion**
3. **Rollout (Simulation)**
4. **Backpropagation**

```text
            Root
             |
      Selection (UCT)
             |
        Selected Node
             |
        Expansion
             |
       New Child Node
             |
         Rollout
             |
      Terminal State
             |
     Backpropagation
             |
      Update Statistics
```

## What is a rollout?

Suppose you are playing chess.

You expand a node corresponding to a particular board position. Instead of calculating the exact value of the position, which is impossible for large games, MCTS asks:

> If I continue playing from here, what tends to happen?

A rollout plays the game forward until:

- a win,
- a loss,
- a draw, or
- a predefined search depth,

using a rollout policy.

For example:

```text
Expanded node
     |
Random move
     |
Random move
     |
Random move
     |
    ...
     |
White wins
```

The rollout then returns a reward such as:

```text
Reward = +1
```

This reward is propagated back up the tree.

## Why are rollouts needed?

When a node is newly expanded, there are no statistics available for it.

Instead of assigning an arbitrary value, MCTS estimates the value of the state using Monte Carlo simulation:

\[
V(s) \approx \frac{1}{N}\sum_{i=1}^{N} R_i
\]

where:

- \(R_i\) is the reward from rollout \(i\),
- \(N\) is the number of rollouts.

Using more rollouts generally reduces the variance of the estimate.

## Example

Consider a Tic-Tac-Toe position with several legal moves. Suppose one move is selected and multiple rollouts are performed from the resulting state.

### Rollout 1

```text
Win
```

Reward:

\[
+1
\]

### Rollout 2

```text
Draw
```

Reward:

\[
0
\]

### Rollout 3

```text
Loss
```

Reward:

\[
-1
\]

The average estimated value is:

\[
\frac{1 + 0 - 1}{3} = 0
\]

Therefore, the node is estimated to have value \(0\).

## Rollout policy

The **rollout policy** determines how actions are selected during the simulation.

### 1. Random rollout

The simplest policy chooses a random legal action at every step.

```text
Choose a random legal move
```

Advantages:

- Very fast
- Easy to implement

Disadvantages:

- High variance
- Often unrealistic in complex games

### 2. Heuristic rollout

A heuristic rollout chooses actions using simple domain knowledge.

Examples in chess:

- Capture valuable pieces
- Avoid losing the queen
- Prefer checks
- Improve king safety

Examples in Go:

- Avoid self-atari
- Prefer local tactical responses
- Favor moves near existing stones

Heuristic rollouts are often more informative than completely random rollouts.

### 3. Learned rollout policy

A learned policy uses a machine-learning model, often a neural network, to estimate an action distribution:

\[
\pi(a \mid s)
\]

The rollout action is sampled from this distribution:

```text
Action ~ pi(a | s)
```

This can produce much stronger simulations than random action selection.

## Rollout depth

Classical MCTS often simulates all the way to a terminal state:

```text
Current node
     |
    ...
     |
Terminal state
```

In many applications, however, reaching a terminal state may be too expensive.

Instead, the rollout may stop after a fixed horizon:

```text
Current node
     |
  20 steps
     |
Value estimate
```

At that point, a heuristic or learned value function estimates the remaining return.

## Computational cost

Rollouts can be the most expensive part of MCTS.

For example, if:

- each rollout contains 200 simulated actions, and
- the algorithm performs 10,000 rollouts,

then the search executes approximately:

```text
2,000,000 simulated actions
```

Efficient rollout policies and fast environment models are therefore important.

## AlphaGo and AlphaZero

### AlphaGo

AlphaGo used:

- a policy network,
- a fast rollout policy,
- a value network.

Its leaf evaluation combined rollout results with a value-network prediction.

```text
Leaf node
    |
 Rollout result
    |
Value-network estimate
    |
Combined evaluation
```

### AlphaZero

AlphaZero removed the traditional rollout-to-terminal step.

Instead, it evaluated a leaf node directly using a neural value function:

```text
Leaf node
    |
Value network
    |
Estimated outcome
```

The network predicts:

\[
V(s)
\]

This avoids simulating a complete game from every newly expanded node.

## Why do rollouts work?

A rollout is a Monte Carlo estimator.

The expected return from a state is:

\[
V(s) = \mathbb{E}[R \mid s]
\]

where \(R\) is the future cumulative reward.

Because this expectation is usually difficult to calculate exactly, MCTS approximates it using sampled returns:

\[
\hat{V}(s) = \frac{1}{N}\sum_{i=1}^{N} R_i
\]

As the number of independent rollouts increases, the estimate tends to approach the expected value.

## Summary

| Aspect | Classical MCTS | AlphaZero-style MCTS |
|---|---|---|
| Leaf evaluation | Rollout to terminal state | Neural value network |
| Simulation policy | Random, heuristic, or learned | No traditional rollout |
| Accuracy | Improves with more simulations but may have high variance | Depends on value-network quality |
| Computational cost | Potentially high | Avoids long terminal simulations |
| Typical applications | Games, planning, robotics | Modern game-playing AI |

A **rollout** is a simulation that starts from a selected or newly expanded node, follows a policy, and produces a reward or estimated return. That result is then **backpropagated** through the search tree to update node statistics and improve future action selection.
