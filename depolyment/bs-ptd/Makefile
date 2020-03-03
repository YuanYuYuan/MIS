MODEL_ID = "1EL8h2zfPQa0wOWwWj5H2Fynt20WA4Pe6"

run: download_model infer

download_model:
	@ echo "===== Downloading model... ====="
	@ ../../utils/download_large_file_from_google_drive.sh $(MODEL_ID) model.pt

infer:
	@ echo "===== Infering... ====="
	@ ../src/infer.py --config ./infering.yaml
