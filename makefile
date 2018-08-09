build:
	cd ./lib/OneGan/ && python setup.py install
	cd ./lib/lsun_room_api// && python setup.py install

train: build
	python main.py \
		--phase train \
		--arch resnet --edge_factor 0.2 --l2_factor 0.2 \
		--name experience_v1 \
		--gpu True

eval: build
	python main.py \
		--phase eval \
		--arch resnet --pretrain_path exp/checkpoints/experience_v1_08-09T16-44/net-29.pt \
		--name Test --tri_visual \
		--gpu True

eval_search:
	python main.py \
		--phase eval_search \
		--arch resnet --pretrain_path exp/checkpoint/experience_v1_08-09T16-44/ \
		--gpu True

