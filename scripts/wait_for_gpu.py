import subprocess, json, time


def free_GPUs():
    gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']).decode("utf-8"))
    gpus = gpu_info["gpus"]
    return [gpu["index"] for gpu in gpus if len(gpu["processes"]) == 0]


def wait_and_get_free_GPU_idx():
    wait_seconds = 0.5
    free_gpus_idx = free_GPUs()
    while not any(free_gpus_idx):
        time.sleep(wait_seconds)
        free_gpus_idx = free_GPUs()
    return free_gpus_idx
