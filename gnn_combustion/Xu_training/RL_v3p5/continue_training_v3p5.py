import os
import runpy


# Build all paths relative to this file so it can be run from any directory.
SCRIPT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# Final neural-network file produced by a successful previous training run.
MODEL_PATH = os.path.join(SCRIPT_FOLDER, "output1", "trained_growth_gnn.pt")

# Original script containing the model, environment, and training loop.
TRAINING_SCRIPT = os.path.join(SCRIPT_FOLDER, "grow_train_animation_v3p5.py")

# Number of additional epochs to run using the saved neural-network weights.
ADDITIONAL_EPOCHS = 50


if __name__ == "__main__":
    # Further training cannot start until a trained model has been saved.
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run grow_train_animation_v3p5.py successfully first."
        )

    # Tell the original training script which model to load and how many
    # additional epochs to run.
    os.environ["GNN_PRETRAINED_MODEL_PATH"] = MODEL_PATH
    os.environ["GNN_ADDITIONAL_EPOCHS"] = str(ADDITIONAL_EPOCHS)

    # Execute the original script as the main program. It loads the saved
    # weights, continues training, and writes a new trained model when done.
    runpy.run_path(TRAINING_SCRIPT, run_name="__main__")
