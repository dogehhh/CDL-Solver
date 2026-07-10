import subprocess
import time

# 要运行的命令
command = ["llamafactory-cli", "train", "examples/train_full/gpu.yaml"]

def get_gpu_util():
    # 使用 nvidia-smi 查询 GPU 利用率
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    # 可能有多张卡，取第一个
    try:
        util = int(result.stdout.strip().split('\n')[0])
    except Exception as e:
        util = None
    return util

if __name__ == "__main__":
    while True:
        gpu_util = get_gpu_util()
        if gpu_util is not None and gpu_util < 10:
            print(f'GPU利用率: {gpu_util}, 开始SFT')
            subprocess.run(command)
            time.sleep(300)