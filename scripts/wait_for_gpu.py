import subprocess, json, time


def free_GPUs():
    try:
        gpu_info = json.loads(subprocess.check_output(['gpustat', '--json']).decode("utf-8"))
    except:
        return [""]
    gpus = gpu_info["gpus"]
    return [gpu["index"] for gpu in gpus if len(gpu["processes"]) == 0]


def wait_and_get_free_GPU_idx():
    i = 0
    wait_seconds = 0.5
    free_gpus_idx = free_GPUs()
    while not any(free_gpus_idx):
        if i % 10 == 0:
            print(f'Waiting for GPU: {i * wait_seconds} seconds')
        time.sleep(wait_seconds)
        free_gpus_idx = free_GPUs()
        i += 1
    return free_gpus_idx
