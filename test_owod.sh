#!/bin/bash

BENCHMARK=${BENCHMARK:-"M-OWODB"}  # M-OWODB or S-OWODB
PORT=${PORT:-"50210"}
EXP=${EXP:-"exp17"}
# if raise error, change num_gpus to 1
if [ $BENCHMARK == "M-OWODB" ]; then
  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t1 --config-file configs/M-OWODB/t1.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/${EXP}/model_0019999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:50210 --task M-OWODB/t2_ft --config-file configs/M-OWODB/t2_ft.yaml --eval-only MODEL.WEIGHTS output/M-OWODB/${EXP}/model_0049999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0079999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/${EXP}/model_0109999.pth
else
#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t1 --config-file configs/${BENCHMARK}/t1.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0039999.pth

  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t2_ft --config-file configs/${BENCHMARK}/t2_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/exp4/model_0064999.pth

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t3_ft --config-file configs/${BENCHMARK}/t3_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0099999.pth

#  python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:${PORT} --task ${BENCHMARK}/t4_ft --config-file configs/${BENCHMARK}/t4_ft.yaml --eval-only MODEL.WEIGHTS output/${BENCHMARK}/model_0129999.pth
fi