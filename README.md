# GPU_TPU_usage
Torch DDP, TPU, nsml (naver AI research env.),  etc.


This repository is written for the skills that I use. Specifically, this repository aims to guide how to use Multi-GPU (using torch DDP, DP), and TPU (GCP)settings. Also, for the NSML usage.



NSML command:

NSML에 저장된 파일 다운 받기:
  nsml download KR95288/stl10/29 PATH -s '/app/alexnet_dense_stl10_results'

NSML에 모델 또는 데이터 push하기
  
  nsml dataset push -v -f [올릴파일 이름 eg, stl10_jh] ./
 
 
NSML 실행 명령
nsml run -e nsml.py -d stl10 --gpu-driver-version 440.0 -m 'dense_train_2' --gpus 1 --memory 20G --shm-size 20G --cpus 12 --gpu-model P40 -n 1
