debug:
	python train.py --dataset hmdb51 --dataset_percentage 0.1 --model C3D --lr 0.001 --batch_size 20 --epochs 100 --num_workers 8 --clip_max_norm 0.1 --optimizer Adam

train:
	python train.py --dataset hmdb51 --dataset_percentage 1.0 --model C3D --lr 0.001 --batch_size 20 --epochs 100 --num_workers 8 --clip_max_norm 0.1 --optimizer Adam
