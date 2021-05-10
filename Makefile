debug:
	python train.py \
      --dataset HMDB51 \
      --dataset_percentage 0.1 \
      --model C3D \
      --lr 0.001 \
      --batch_size 20 \
      --epochs 100 \
      --c3d_dropout_rate 0.2 \
      --num_workers 6 \
      --clip_max_norm 0.1 \
      --optimizer Adam

train:
	python train.py \
      --dataset HMDB51 \
      --dataset_percentage 1.0 \
      --model C3D \
      --lr 0.001 \
      --batch_size 20 \
      --epochs 100 \
      --c3d_dropout_rate 0.2 \
      --num_workers 6 \
      --clip_max_norm 0.1 \
      --optimizer Adam
