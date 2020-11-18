SRC_DIR := $$(pwd | sed 's/\(.*\)exp\/.*/\1src/')
TRAIN_PROGRAM = "train_tasks.py"
INFER_PROGRAM = "infer_v2.py"

# Verbose
VERBOSE=0
ifeq ($(VERBOSE), 0)
	EXECUTOR = @
endif

# Debugging
DEBUG=0
ifeq ($(DEBUG), 1)
	EXECUTOR += /usr/bin/env ipython3 --pdb --
else
	EXECUTOR += /usr/bin/env python3 -u
endif

# Configs
TRAIN_CONFIG = './training.json5'
INFER_CONFIG = './inference.json5'
LOG_DIR   = './_logs'
CKPT_DIR  = './_ckpts'
PAUSE_CKPT_DIR = './_pause_ckpts'
OUTPUT_DIR = './_outputs'

# train
TRAIN_FLAGS += --config $(TRAIN_CONFIG)
TRAIN_FLAGS += --log-dir $(LOG_DIR)
TRAIN_FLAGS += --ckpt-dir $(CKPT_DIR)
TRAIN_FLAGS += --pause-ckpt-dir $(PAUSE_CKPT_DIR)

# infer
INFER_FLAGS += --config $(INFER_CONFIG)
INFER_FLAGS += --output $(OUTPUT_DIR)

run: clean train

train:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/$(TRAIN_PROGRAM) $(TRAIN_FLAGS)

test: clean
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/$(TRAIN_PROGRAM) $(TRAIN_FLAGS) --test

infer:
	@echo "===== Inferring... ====="
	$(EXECUTOR) $(SRC_DIR)/$(INFER_PROGRAM) $(INFER_FLAGS)

log:
	@echo "===== Launch Tensorboard ====="
	tensorboard --logdir $(LOG_DIR) > /dev/null 2>&1

clean:
	@echo "===== Cleaning all... ====="
	rm -rvf $(CKPT_DIR) $(LOG_DIR) $(PAUSE_CKPT_DIR) $(OUTPUT_DIR)

clean_outputs:
	@echo "===== Cleaning outputs... ====="
	rm -rvf $(OUTPUT_DIR)
