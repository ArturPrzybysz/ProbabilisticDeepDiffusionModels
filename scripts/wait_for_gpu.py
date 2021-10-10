import subprocess
import json
import sys
import time


def GPUs_free(required_gpu_count=1):
    gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']).decode("utf-8"))
    gpus = gpu_info["gpus"]
    free_gpus = [gpu for gpu in gpus if len(gpu["processes"]) == 0]
    return len(free_gpus) >= required_gpu_count


if __name__ == '__main__':
    seconds_to_wait = 5

    while not GPUs_free():
        time.sleep(seconds_to_wait)

    sys.exit(0)
