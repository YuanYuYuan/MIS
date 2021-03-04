ROOT_DIR := $$(pwd | sed 's/\(.*\)deployments\/.*/\1/')
SRC_DIR=$(ROOT_DIR)/src
INFER_PROGRAM = "infer_v2.py"
MODEL_ID = "1fsTHjBec4In8rCGpu_MuFep80zItEfHc"
CKPT = 'ckpts.tar.gz'


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
INFER_CONFIG = './inference.json5'
OUTPUT_DIR = './outputs'

# infer
INFER_FLAGS += --config $(INFER_CONFIG)
INFER_FLAGS += --output $(OUTPUT_DIR)


download_model:
	@ echo "===== Downloading model... ====="
	@ $(ROOT_DIR)/utils/download_large_file_from_google_drive.sh $(MODEL_ID) $(CKPT)
	@ tar xvf $(CKPT) && rm $(CKPT)

infer:
	@echo "===== Inferring... ====="
	$(EXECUTOR) $(SRC_DIR)/$(INFER_PROGRAM) $(INFER_FLAGS)
