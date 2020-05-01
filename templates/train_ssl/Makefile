SRC_DIR := $$(pwd | sed 's/\(.*\)MIS\/.*/\1MIS\/src/')

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
PAUSE_CKPT = 'pause.pt'

# train
TRAIN_FLAGS += --config $(TRAINING_CONFIG)
TRAIN_FLAGS += --log-dir $(TRAIN_LOG_DIR)
TRAIN_FLAGS += --checkpoint-dir $(CKPT_DIR)
TRAIN_FLAGS += --pause-ckpt $(PAUSE_CKPT)

ifneq ($(SEG_CKPT),)
	TRAIN_FLAGS += --checkpoint $(SEG_CKPT)
endif

ifneq ($(DIS_CKPT),)
	TRAIN_FLAGS += --checkpoint $(DIS_CKPT)
endif


run: clean train

train:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/train_ssl.py $(TRAIN_FLAGS)

continue:
	@echo "===== Training... ====="
	$(EXECUTOR) $(SRC_DIR)/train_ssl.py $(TRAIN_FLAGS) --checkpoint $(PAUSE_CKPT)

log:
	@echo "===== Launch Tensorboard ====="
	tensorboard --logdir $(TRAIN_LOG_DIR) > /dev/null 2>&1

clean:
	rm -rvf $(CKPT_DIR) $(TRAIN_LOG_DIR) $(PAUSE_CKPT)
