SRC_DIR := $$(pwd | sed 's/\(.*\)exp\/.*/\1src/')
PROGRAM = "train_stages.py"

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
CONFIG = './training.json5'
LOG_DIR   = './_logs'
CKPT_DIR  = './_ckpts'
PAUSE_CKPT_DIR = './_pause_ckpts'

# train
TRAIN_FLAGS += --config $(CONFIG)
TRAIN_FLAGS += --log-dir $(LOG_DIR)
TRAIN_FLAGS += --ckpt-dir $(CKPT_DIR)
TRAIN_FLAGS += --pause-ckpt-dir $(PAUSE_CKPT_DIR)


run: clean train

train:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/$(PROGRAM) $(TRAIN_FLAGS)

test: clean
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/$(PROGRAM) $(TRAIN_FLAGS) --test

log:
	@echo "===== Launch Tensorboard ====="
	tensorboard --logdir $(TRAIN_LOG_DIR) > /dev/null 2>&1

clean:
	@echo "===== Cleaning... ====="
	rm -rvf $(CKPT_DIR) $(LOG_DIR) $(PAUSE_CKPT_DIR)
