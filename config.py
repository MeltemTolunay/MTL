"""
Config parameters
"""

# Dataloader parameters
BATCH_SIZE = 16

# Training parameters
NUM_EPOCHS = 20

# Optimizer parameters
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
STEP_SIZE = 7
GAMMA = 0.1

# Testing
TEST_BATCH_SIZE = 500  # entire test set in a single batch (372)

# Two-Task related
ALPHA = 1

'''
TODO:
2 task model
sparsity
'''