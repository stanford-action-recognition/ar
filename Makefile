train-c3d:
	python train.py \
		--dataset HMDB51 \
		--dataset_percentage 0.2 \
		--model C3D \
		--lr 0.001 \
		--batch_size 20 \
		--epochs 100 \
		--num_workers 6 \
		--clip_max_norm 0.1 \
		--optimizer Adam \
		--c3d_dropout_rate 0.2 # avoid adding it if you are running other models for better W&B statistics

train-r3d:
	python train.py \
		--dataset HMDB51 \
		--dataset_percentage 0.2 \
		--model R3D \
		--batch_size 15 \
		--lr 0.001 \
		--epochs 300 \
		--num_workers 6 \
		--clip_max_norm 0.1 \
		--optimizer Adam

train-r2plus1d:
	python train.py \
		--dataset HMDB51 \
		--dataset_percentage 0.2 \
		--model R2Plus1D \
		--batch_size 15 \
		--lr 0.001 \
		--epochs 300 \
		--num_workers 6 \
		--clip_max_norm 0.1 \
		--optimizer Adam
