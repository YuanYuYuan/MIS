SRC_DIR := $$(pwd | sed 's/\(.*\)exp\/.*/\1src/')

## Verbose
VERBOSE=0
ifeq ($(VERBOSE), 0)
	EXECUTOR = @
endif

## Debugging
DEBUG=0
ifeq ($(DEBUG), 1)
	EXECUTOR += /usr/bin/env ipython3 --pdb --
else
	EXECUTOR += /usr/bin/env python3 -u
endif

# Configs
TRAINING_CONFIG = './training.yaml'
INFERING_CONFIG = './infering.yaml'
TRAIN_LOG_DIR   = './_logs'
CKPT_DIR  = './_ckpts'
CKPT = 'ckpt.pt'
PAUSE_CKPT = 'pause.pt'

# train
TRAIN_FLAGS += --config $(TRAINING_CONFIG)
TRAIN_FLAGS += --log-dir $(TRAIN_LOG_DIR)
TRAIN_FLAGS += --checkpoint-dir $(CKPT_DIR)
TRAIN_FLAGS += --pause-ckpt $(PAUSE_CKPT)


run: clean train

test:
	@echo "===== Small data test ====="
	$(EXECUTOR) $(SRC_DIR)/train.py $(TRAIN_FLAGS) --test

train:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/train.py $(TRAIN_FLAGS)

retrain:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/train.py $(TRAIN_FLAGS) --checkpoint $(CKPT)

continue:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/train.py $(TRAIN_FLAGS) --checkpoint $(PAUSE_CKPT)

validate:
	@echo "===== Validating... ====="
	$(EXECUTOR) $(SRC_DIR)/train.py $(TRAIN_FLAGS) --checkpoint $(CKPT) --validate-only

infer:
	@echo "===== Infering... ====="
	$(EXECUTOR) $(SRC_DIR)/infer.py --config $(INFERING_CONFIG)

log:
	@echo "===== Launch Tensorboard ====="
	tensorboard --logdir $(TRAIN_LOG_DIR) > /dev/null 2>&1

clean:
	rm -rvf $(CKPT_DIR) $(TRAIN_LOG_DIR) $(PAUSE_CKPT)
