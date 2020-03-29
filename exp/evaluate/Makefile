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
CONFIG = './evaluation.yaml'
CKPT = 'ckpt.pt'
LOG_DIR = 'logs'

run: clean evaluate

evaluate:
	@echo "===== Evaluating ====="
	$(EXECUTOR) $(SRC_DIR)/evaluate.py --config $(CONFIG) --checkpoint $(CKPT) --log-dir $(LOG_DIR)


log:
	@echo "===== Launch Tensorboard ====="
	tensorboard --logdir $(LOG_DIR) > /dev/null 2>&1

clean:
	rm -rvf $(LOG_DIR)
