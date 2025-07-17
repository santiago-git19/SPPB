import os
import sys
import time
import psutil
import resource
import argparse

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    try:
        nvmlInit()
        NVML_AVAILABLE = True
    except Exception:
        NVML_AVAILABLE = False
except ImportError:
    NVML_AVAILABLE = False

import torch
from torch2trt import torch2trt

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Monitor and convert TRT_Pose model via torch2trt on Jetson Nano')
parser.add_argument('--checkpoint-path', '-c', required=True,
                    help='Path to the PyTorch .pth checkpoint file')
parser.add_argument('--engine-path', '-e', required=True,
                    help='Output path for the TensorRT .engine file')
parser.add_argument('--model-class', '-m', required=True,
                    help='Python path to model class, e.g. mymodule.ResNet18Pose')
parser.add_argument('--input-shape', nargs=4, type=int, metavar=('N','C','H','W'),
                    default=[1,3,224,224],
                    help='Input tensor shape for the model')
parser.add_argument('--precision', choices=['fp16','fp32'], default='fp16',
                    help='Precision mode for TensorRT engine')
parser.add_argument('--max-workspace-size', type=int, default=1<<28,
                    help='Max workspace size in bytes')
parser.add_argument('--swap-file', default='/swapfile',
                    help='Path to swapfile')
parser.add_argument('--swap-size-gb', type=int, default=2,
                    help='Size of swap in GB')
parser.add_argument('--monitor-interval', type=int, default=5,
                    help='Seconds between resource checks')
parser.add_argument('--memory-margin', type=float, default=0.1,
                    help='Minimum fraction of RAM to keep free')
parser.add_argument('--gpu-memory-margin', type=float, default=0.1,
                    help='Minimum fraction of GPU memory to keep free')
parser.add_argument('--max-ram-limit', type=int, default=None,
                    help='Limit for process memory (bytes), None for no limit')
args = parser.parse_args()

# --- Configuration ---
SWAP_FILE = args.swap_file
SWAP_SIZE_GB = args.swap_size_gb
MEMORY_MARGIN = args.memory_margin
GPU_MEMORY_MARGIN = args.gpu_memory_margin
MONITOR_INTERVAL = args.monitor_interval
MAX_RAM_LIMIT = args.max_ram_limit

# --- Functions ---
def ensure_swap(file_path, size_gb):
    if os.path.exists(file_path): return
    size_bytes = size_gb * (1 << 30)
    os.system(f"sudo fallocate -l {size_bytes} {file_path} && sudo chmod 600 {file_path} && sudo mkswap {file_path} && sudo swapon {file_path}")
    print(f"Swap file '{file_path}' of size {size_gb}GB activated.")


def get_system_usage():
    vm = psutil.virtual_memory()
    usage = {
        'ram_used': vm.used,
        'ram_total': vm.total,
        'ram_free_ratio': vm.available / vm.total
    }
    if NVML_AVAILABLE:
        try:
            handle = nvmlDeviceGetHandleByIndex(0)
            mem = nvmlDeviceGetMemoryInfo(handle)
            usage.update({
                'gpu_used': mem.used,
                'gpu_total': mem.total,
                'gpu_free_ratio': (mem.total-mem.used)/mem.total
            })
        except:
            pass
    return usage


def set_memory_limit(limit_bytes):
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    print(f"Process memory limit set to {limit_bytes/(1<<20):.1f} MB.")

# --- Main flow ---
if __name__ == '__main__':
    ensure_swap(SWAP_FILE, SWAP_SIZE_GB)
    if MAX_RAM_LIMIT:
        set_memory_limit(MAX_RAM_LIMIT)

    # Load model
    module_name, class_name = args.model_class.rsplit('.', 1)
    mod = __import__(module_name, fromlist=[class_name])
    ModelClass = getattr(mod, class_name)
    model = ModelClass()
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval().cuda()

    # Dummy input
    input_shape = tuple(args.input_shape)
    dummy = torch.randn(input_shape).cuda()

    # Conversion with torch2trt
    print("Starting torch2trt conversion...")
    start = time.time()
    if args.precision == 'fp16':
        model_trt = torch2trt(model, [dummy], fp16_mode=True,
                               max_workspace_size=args.max_workspace_size)
    else:
        model_trt = torch2trt(model, [dummy], fp16_mode=False,
                               max_workspace_size=args.max_workspace_size)
    torch.save(model_trt.state_dict(), args.engine_path)
    print(f"Conversion finished in {time.time()-start:.1f}s, saved to {args.engine_path}.")

    # Monitoring loop
    print("Entering resource monitor (Ctrl+C to exit)")
    try:
        while True:
            u = get_system_usage()
            print(f"RAM {u['ram_used']/(1<<30):.2f}/{u['ram_total']/(1<<30):.2f} GB free {u['ram_free_ratio']:.2%}")
            if 'gpu_free_ratio' in u:
                print(f"GPU free {u['gpu_free_ratio']:.2%}")
            if u['ram_free_ratio'] < MEMORY_MARGIN:
                print("[Warning] Low RAM")
            if 'gpu_free_ratio' in u and u['gpu_free_ratio'] < GPU_MEMORY_MARGIN:
                print("[Warning] Low GPU mem")
            time.sleep(MONITOR_INTERVAL)
    except KeyboardInterrupt:
        print("Monitoring stopped.")
