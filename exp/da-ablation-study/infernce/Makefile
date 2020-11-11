SRC_DIR := $$(pwd | sed 's/\(.*\)exp\/.*/\1src/')
PROGRAM = "infer_v2.py"

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
CONFIG = './inference.json5'
OUTPUT_DIR = './outputs'

# flags
FLAGS += --config $(CONFIG)
FLAGS += --output $(OUTPUT_DIR)

run: clean infer

infer:
	@echo "===== Inferring... ====="
	$(EXECUTOR) $(SRC_DIR)/$(PROGRAM) $(FLAGS)

test: clean
	@echo "===== Inferring... ====="
	$(EXECUTOR) $(SRC_DIR)/$(PROGRAM) $(FLAGS) --test

clean:
	@echo "===== Cleaning... ====="
	rm -rvf $(OUTPUT_DIR)
