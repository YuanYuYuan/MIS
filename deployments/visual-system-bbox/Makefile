MODEL_ID = "1IRJ9E-WP9fm4wWOhxDY0QcEZ_ABQKi16"
CKPT = 'model_checkpoint.pt'

download_model:
	@ echo "===== Downloading model... ====="
	@ ../../utils/download_large_file_from_google_drive.sh $(MODEL_ID) $(CKPT)

evaluate:
	@ make -C ./evaluation evaluate CKPT=../$(CKPT)

predict:
	@ make -C ./evaluation predict CKPT=../$(CKPT)

train:
	@ make -C ./training train CKPT=../$(CKPT)
