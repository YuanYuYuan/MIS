MODEL_ID = "1E2Qbekb1mts_uOPiIA93EtUDYfXzS445"

run: download_model infer

download_model:
	@ echo "===== Downloading model... ====="
	@ ../../utils/download_large_file_from_google_drive.sh $(MODEL_ID) model.pt

infer:
	@ echo "===== Infering... ====="
	@ ../../src/infer.py --config ./infering.yaml
