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

evaluate:
	@echo "===== Evaluating ====="
	$(EXECUTOR) $(SRC_DIR)/evaluate.py --config $(CONFIG) --checkpoint $(CKPT)

# clean:
#     rm -rvf $(CKPT_DIR) $(TRAIN_LOG_DIR) $(PAUSE_CKPT)
