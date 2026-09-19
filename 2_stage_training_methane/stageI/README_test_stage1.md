# How test_stage1.py works

`test_stage1.py` checks whether the **training implementation behaves correctly**. It does not check whether a trained model has learned all molecules.

Its workflow is:

```text
Load the implementation classes without starting training
        ↓
Prepare fresh settings for each test
        ↓
Run eleven tests
        ↓
Report which passed or failed
```

## 1. Load classes without running training

Your `train_stage1.py` ends with direct calls:

```python
trainer = Stage1Trainer(training_config)
trainer.run()
```

Therefore, a normal import would start training.

The test script avoids this by reading the file as a **syntax tree**:

```python
tree = ast.parse(source.read_text())
```

A syntax tree represents the structure of Python code—imports, class definitions, assignments, and function calls.

The test retains only imports and class definitions:

```python
definitions = ast.Module(
    body=[
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef))
    ],
    type_ignores=[],
)
```

It then executes those definitions in a separate dictionary:

```python
scope = {"__file__": str(source), "__name__": "stage1_test_definitions"}
exec(compile(definitions, str(source), "exec"), scope)
```

This makes the implementation classes available without creating a trainer or writing training outputs.

## 2. Prepare each test

The tests belong to:

```python
class Stage1Tests(unittest.TestCase):
```

Python's `unittest` framework automatically recognizes methods whose names begin with `test_`.

Before **each test**, it calls:

```python
def setUp(self):
```

This creates fresh atom mappings, valence limits, an action space, and reference graphs.

```python
self.symbols = {0: "H", 1: "O", 2: "C"}
self.valency = {0: 1, 1: 2, 2: 4}
```

The helper:

```python
def env(self):
    return AdvancedMoleculeEnv(...)
```

creates a fresh empty molecule whenever a test needs one.

## 3. Test all reference construction sequences

`test_all_references_and_enumerated_orders_reconstruct()`

This enumerates and replays every unique construction sequence for all 31 species, checking that no duplicate action sequences remain.

For each trajectory, it starts from an empty molecule and replays the demonstrated actions.

At every step, it checks:

- The current state matches the saved teaching example.
- Every acceptable teaching action is allowed by the action mask.

After the last action, it checks:

- STOP was executed.
- The final graph matches the reference's atom types, connectivity, and bond orders.

**Why this matters:** the model cannot learn correctly if the teaching sequences themselves are wrong.

## 4. Test stopping at OH versus growing into H₂O

`test_stop_for_radicals_and_competing_additions()`

The test builds OH:

```python
env.step(("START", 1))       # Start with O
env.step(("GROW", 0, 0, 1))  # Attach H to O
```

It checks that both choices are legal:

```text
OH
├── STOP
└── Add H to O
```

It then copies the environment:

- One copy receives another H and must match H₂O.
- The original receives STOP and must allow no further actions.

**Why this matters:** the environment must allow radicals to finish without forcing every available valence to be filled. This test checks the available choices, not whether a trained policy chooses correctly.

## 5. Test action masks and bond upgrades

`test_masks_are_generic_and_bonds_are_unique()`

Starting with carbon, the test confirms that adding H, O, or C with a single bond is allowed.

It also checks:

### Invalid hydrogen bonds are blocked

Adding H with a double bond must be forbidden.

### Undirected bonds are counted once

```python
("CONNECT", 0, 1, 3)
```

exists, but the reversed duplicate:

```python
("CONNECT", 1, 0, 3)
```

does not.

### Bond upgrades use the final order

After changing C–C from single to triple:

```python
current_bonds == [3, 3]
```

The result must be 3, not 4. The action means “make the bond triple,” rather than “add three more units.”

## 6. Test the maximum atom limit

`test_size_cap_still_allows_stop()`

This creates an environment with a one-atom limit and adds carbon.

The only remaining action must be:

```python
("STOP",)
```

**Why this matters:** reaching the size limit must not leave the policy with no legal action.

## 7. Test that isomers remain distinguishable

`test_isomers_have_same_condition_but_distinct_graphs()`

It examines:

- `CH3O` versus `CH2OH`.
- `CH2CHO` versus `CH3CO`.

For each pair, it verifies:

```text
Same composition
Different molecular graphs
Correct individual reference names
```

**Why this matters:** a formula-only comparison would incorrectly treat each pair as the same structure.

## 8. Test probabilities, symmetry, gradients, and saved weights

`test_masked_distribution_symmetry_and_finite_gradients()`

This creates a small, randomly initialized GNN. It performs several checks.

### Probabilities form a valid distribution

For each graph:

```text
Sum of probabilities over all actions = 1
```

Illegal actions must have probability zero.

### Equivalent attachment sites receive equal probabilities

For a bare C–C fragment, the two carbons are interchangeable.

Therefore:

```text
Add H to carbon 0
Add H to carbon 1
```

should have the same probability.

### The loss can be differentiated

It computes:

```text
L = -log(P(add H to left) + P(add H to right))
```

Then:

```python
loss.backward()
```

checks that any computed parameter gradients are finite—no `NaN` or infinity.

This is a backward-pass check; it does not perform a training run.

### Saved weights can be restored

The test saves the weights into a temporary directory, loads them into a separate model, and checks that both models produce the same output.

**Why this matters:** a checkpoint is useful only if restoring it preserves the model's predictions.

## 9. Test independence from atom numbering

`test_permuting_nodes_permutates_actions()`

This builds the same C–O structure twice:

```text
Graph A: node 0 = C, node 1 = O
Graph B: node 0 = O, node 1 = C
```

It compares corresponding actions. For example:

```text
Add H to node 0 in A
        ↕ same physical action
Add H to node 1 in B
```

Their scores must agree within numerical tolerance.

**Why this matters:** atom IDs are bookkeeping. Changing them should not change the model's physical interpretation.

## 10. Test rejection of an invalid reference

`test_loader_rejects_overvalence()`

This deliberately creates an H=O double bond in a temporary JSON file.

Because the configured hydrogen valence limit is 1, loading it must raise a `ValueError`.

```python
with self.assertRaisesRegex(ValueError, "valence"):
    ReferenceSpecies(...)
```

Here, **raising the expected error means the test passes**.

## 11. Run and interpret the results

The final line starts the tests:

```python
unittest.main(verbosity=2)
```

Run it from the folder containing the scripts and reference JSON:

```bash
python test_stage1.py
```

A successful run ends with:

```text
Ran 11 tests ...
OK
```

- **`ok`**: the test's assertions passed.
- **`FAIL`**: an expected property was not satisfied.
- **`ERROR`**: an unexpected exception prevented the test from completing.

**Passing these tests establishes specific implementation checks. Whether the model reliably generates the reference molecules is measured separately by the free-generation evaluations in `train_stage1.py`.**

*The way we train the neural network, is that we exhausts all approaches for each species, then we choose $n\times P$ approaches per species to complete 1 epoch of training*
- $P\in[0,1]$ 
- Select without replacement within each epoch
- Draw fresh selection each epoch
- Keep each species equally weighted:
$$
L=\frac{1}{S}\sum_{s=1}^{S} \frac{1}{K_s}\sum_{k=1}^{K_s} \frac{1}{T_{s,k}}\sum_{t=1}^{T_{s,k}}\ell_{s,k,t}
$$
In this equation, $S$ is the total number of species, $K$ is the selected number of approaches to construct a molecule, $T$ is the number of construction steps of each approach

The loss is calculated by:
$$
\ell_{s,k,t}=-\log\left( \sum_{a\in A_{s,k,t}} p_\theta(a\mid \text{partial graph},\text{composition}) \right)
$$

## Exhaustive enumeration and percentage sampling checks

Three additional tests verify:

- An independent enumeration of all 7! atom permutations for CH3CHO finds 600 connected labeled orders and exactly 100 distinct action sequences, matching the production enumerator. Relabeling graph nodes preserves the pool.
- A synthetic three-carbon ring exercises CONNECT actions and reconstructs correctly.
- Percentage selection rounds upward, rejects invalid fractions, samples without replacement, reproduces a seeded selection, varies across epochs, and includes the full pool at 100%. Unequal sequence lengths and pool sizes still give each species equal total loss weight.
