train:
	python train.py --dataset hmdb51 --model C3D --lr 1e-3 --batch_size 32 --epochs 100 --clip_max_norm 0.1 --optimizer Adam
