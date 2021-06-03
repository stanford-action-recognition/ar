train-rgb:
	python rgb_train.py \
		--dataset HMDB51 \
		--dataset_percentage 0.2 \
		--model R3D \
		--batch_size 15 \
		--lr 0.001 \
		--epochs 300 \
		--num_workers 6 \
		--clip_max_norm 0.1 \
		--optimizer Adam

train-flow:
	python motion_train.py

train-flow:
	python multi_stream_train.py --streams=rgb,flow
