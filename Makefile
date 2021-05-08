train:
	python train.py --dataset hmdb51 --model C3D --lr 1e-3 --batch_size 64 --epochs 100 --num_workers 8 --clip_max_norm 0.1 --optimizer Adam
