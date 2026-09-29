"""Run from self_updating/: python run_cycle.py

Reusable classes live in codebase/. This file intentionally uses direct calls.
"""
import os

from codebase.pipeline import SelfUpdatingPipeline
from codebase.validators.reference import ReferenceValidator


# ==========================================
# PARAMETERS — paths relative to your working directory
# ==========================================
# Start from the workbook seed set and no parent checkpoint. The pipeline writes
# normalized JSON copies into output/inputs/ before any training begins.
POSITIVE_SPECIES_PATH = 'input/given_species.xlsx'
NEGATIVE_SPECIES_PATH = None
CHECKPOINT_PATH = None
# To start a new cycle from a prior checkpoint, use a DIFFERENT output folder and
# set CHECKPOINT_PATH to '../previous_output/checkpoints/round_001.pt'.
# Set POSITIVE_SPECIES_PATH = None to inherit that checkpoint's positive/negative sets.
REFERENCE_PATH = 'input/FFCM2_CHO_species_reference.xlsx'
OUTPUT_FOLDER = 'output'
TOTAL_ROUNDS = 5
RESUME = False  # True continues saved training epochs/phases and pending validation.
# 0: no console or log-file output. 1: stage banners and progress bars only.
# 2: detailed progress, metrics, and saved logs (the previous behavior).
LOG_LEVEL = 1
 
# CUDA is for NVIDIA GPUs. Set SELF_UPDATING_GPU_IDS in Slurm to select the
# visible GPUs, for example "0" or "0,1".
TRAINING_DEVICE = 'cuda'
GPU_IDS = [int(index) for index in os.environ.get('SELF_UPDATING_GPU_IDS', '0,1').split(',')]
TOTAL_EPOCHS = 100
LEARNING_RATE = 0.0003
LAMBDA_NEGATIVE = 1.0
# Every enumerated positive construction route is used each epoch.
POSITIVE_TRAJECTORY_FRACTION = 1.00
NEGATIVE_TRAJECTORY_FRACTION = 0.20
# Use the top ceil(HARD_NEGATIVE_PERCENT × eligible negatives), limited by the cap.
HARD_NEGATIVE_PERCENT = 0.15
HARD_NEGATIVE_MAX_COUNT = 50
# Direct counts reflect ordinary policy sampling. PUCT counts contribute less because
# PUCT deliberately explores actions that may have low network probability.
PUCT_HARDNESS_WEIGHT = 0.25
EVALUATE_EVERY = 10
CHECKPOINT_EVERY = 10
INITIAL_TRAINING_EPOCHS = 300
INITIAL_LEARNING_RATE = 0.003
INITIAL_TRAJECTORY_FRACTION = 0.20
INITIAL_EVALUATE_EVERY = 50
HIDDEN_DIMS = [32, 64, 32, 32, 16]
MAX_ATOMS = 10
MAX_STEPS = 24
EXCLUDED_SPECIES = ['CO']
INITIAL_CHECKPOINT_EVERY = 50
EVALUATION_SAMPLES_PER_COMPOSITION = 100
FREE_EVALUATION_ATTEMPTS = 500
ROUTES_PER_BATCH = 8
FORWARD_BATCH_SIZE = 256
REPLAY_CACHE_SIZE = 2048  # Maximum cached CPU construction routes.
TRAINING_SEED = 12345

GENERATION_METHODS = ['puct','direct']
EXPLORATION_RUNS = 100
ATTEMPTS_PER_RUN = 1000  # Per method per round: default 200,000 attempts/round.
PUCT_C = 2.0
PUCT_UNIFORM_FRACTION = 0.25
PUCT_REFERENCE_REWARD = 0.5
PUCT_CANDIDATE_REWARD = 1.0
GENERATION_SEED = 24680
GENERATION_BATCH_SIZE = 100
SAVE_PLOTS = True  # PNG
# CPU graph preparation, route replay, and Stage II exploration threads.
CPU_THREADS = int(os.environ.get('SELF_UPDATING_CPU_THREADS', '32'))
UNMATCHED_AS_NEGATIVE = True  # Explicit current experiment assumption, not chemical proof.


# CREATE OBJECTS AND RUN
validator = ReferenceValidator(REFERENCE_PATH)
pipeline = SelfUpdatingPipeline(
    positive_path=POSITIVE_SPECIES_PATH,
    negative_path=NEGATIVE_SPECIES_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    validator=validator,
    output_folder=OUTPUT_FOLDER,
    generation_config={
        'methods':GENERATION_METHODS,'runs':EXPLORATION_RUNS,'attempts_per_run':ATTEMPTS_PER_RUN,
        'puct_c':PUCT_C,'puct_uniform_fraction':PUCT_UNIFORM_FRACTION,
        'puct_reference_reward':PUCT_REFERENCE_REWARD,'puct_candidate_reward':PUCT_CANDIDATE_REWARD,
        'seed':GENERATION_SEED,'batch_size':GENERATION_BATCH_SIZE,'threads':CPU_THREADS,
        'save_media':SAVE_PLOTS,'device':'cpu',
        'log_level':LOG_LEVEL,
    },
    training_config={
        'device':TRAINING_DEVICE,'epochs':TOTAL_EPOCHS,'learning_rate':LEARNING_RATE,
        'positive_trajectory_fraction':POSITIVE_TRAJECTORY_FRACTION,
        'negative_trajectory_fraction':NEGATIVE_TRAJECTORY_FRACTION,
        'lambda_negative':LAMBDA_NEGATIVE,
        'eval_every':EVALUATE_EVERY,'checkpoint_every':CHECKPOINT_EVERY,'eval_samples':EVALUATION_SAMPLES_PER_COMPOSITION,
        'free_eval_attempts':FREE_EVALUATION_ATTEMPTS,'routes_per_batch':ROUTES_PER_BATCH,
        'forward_batch_size':FORWARD_BATCH_SIZE,'replay_cache_size':REPLAY_CACHE_SIZE,
        'seed':TRAINING_SEED,'threads':CPU_THREADS,'gpu_ids':GPU_IDS,
        'log_level':LOG_LEVEL,
    },
    labeling_config={'unmatched_as_negative':UNMATCHED_AS_NEGATIVE},
    hard_negative_config={'percent':HARD_NEGATIVE_PERCENT,'max_count':HARD_NEGATIVE_MAX_COUNT,
                          'puct_weight':PUCT_HARDNESS_WEIGHT},
    initial_training_config={
        'epochs':INITIAL_TRAINING_EPOCHS,'checkpoint_every':INITIAL_CHECKPOINT_EVERY,
        'learning_rate':INITIAL_LEARNING_RATE,'trajectory_sample_fraction':INITIAL_TRAJECTORY_FRACTION,
        'eval_every':INITIAL_EVALUATE_EVERY,'hidden_dims':HIDDEN_DIMS,
        'max_atoms':MAX_ATOMS,'max_steps':MAX_STEPS,'excluded_species':EXCLUDED_SPECIES,
        'replay_cache_size':REPLAY_CACHE_SIZE,
        'gpu_ids':GPU_IDS,
        'log_level':LOG_LEVEL,
    },
)
pipeline.run(rounds=TOTAL_ROUNDS,resume=RESUME)
