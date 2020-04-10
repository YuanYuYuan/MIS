MODEL_ID = "1E2Qbekb1mts_uOPiIA93EtUDYfXzS445"
CKPT = 'model_checkpoint.pt'

download_model:
	@ echo "===== Downloading model... ====="
	@ ../../utils/download_large_file_from_google_drive.sh $(MODEL_ID) $(CKPT)

evaluate:
	@ cd ./evaluation
	@ make evaluate CKPT=$(CKPT)

