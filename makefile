build_external:
	pip install click tqdm

build_onnx:
	rm -rf /onnx/ && git clone https://github.com/onnx/onnx.git /onnx/
	cd /onnx/ \
	&& git submodule update --init --recursive \
	&& python setup.py install
	rm -rf /onnx-tensorflow/ && git clone https://github.com/onnx/onnx-tensorflow /onnx-tensorflow/
	cd /onnx-tensorflow/ \
	&& git submodule update --init --recursive \
	&& python setup.py install

build_local:
	cd ./lib/OneGan/ && python setup.py install
	cd ./lib/lsun_room_api/ && python setup.py install

build: build_external build_onnx build_local
	cd ./onnx-tensorflow/ && python setup.py install

train: build_local
	python main.py \
		--phase train \
		--arch resnet --edge_factor 0.2 --l2_factor 0.2 \
		--name experience_v1 \
		--gpu True

eval: build_local
	python main.py \
		--phase eval \
		--arch resnet --pretrain_path exp/checkpoints/experience_v1_08-09T16-44/net-29.pt \
		--name Test --tri_visual \
		--gpu True

eval_search: build_local
	python main.py \
		--phase eval_search \
		--arch resnet --pretrain_path exp/checkpoint/experience_v1_08-09T16-44/ \
		--gpu True

export: build_local
	python main.py \
		--phase export \
		--arch resnet --pretrain_path exp/checkpoints/experience_v1_08-09T16-44/net-29.pt \
		--name Test --tri_visual \
		--gpu True

export2keras: build_local
	python main.py \
		--phase export2keras \
		--arch resnet --pretrain_path exp/checkpoints/experience_v1_08-09T16-44/net-29.pt \
		--name Test --tri_visual \
		--gpu True

tf_eval:
	python tf_eval.py