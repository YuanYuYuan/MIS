MODEL_ID = "1Yt0N2ipPMnOsVuX9s2gDhfJDUC6nAgJx"
CKPT = 'model_checkpoint.pt'

download_model:
	@ echo "===== Downloading model... ====="
	@ ../../utils/download_large_file_from_google_drive.sh $(MODEL_ID) $(CKPT)

evaluate:
	@ cd ./evaluation
	@ make evaluate CKPT=$(CKPT)

train:
	@ cd ./training
	@ make retrain CKPT=$(CKPT)
