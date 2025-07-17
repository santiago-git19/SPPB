import os
import sys
import subprocess
import time
import psutil
import resource
import argparse

try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Monitor and convert TRT_Pose model on Jetson Nano')
parser.add_argument('--checkpoint-path', '-c', required=True,
                    help='Path to the PyTorch .pth checkpoint file')
parser.add_argument('--engine-path', '-e', required=True,
                    help='Output path for the TensorRT engine file')
parser.add_argument('--model', default='resnet18_baseline_att',
                    help='Model name for trt_pose converter')
parser.add_argument('--input-shape', nargs=4, type=int, metavar=('N','C','H','W'),
                    default=[1,3,224,224],
                    help='Input tensor shape for the model')
parser.add_argument('--precision', choices=['fp16','fp32','int8'], default='fp16',
                    help='Precision mode for TensorRT engine')
parser.add_argument('--max-workspace-size', type=int, default=1<<30,
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

# --- Configuration from args ---
SWAP_FILE = args.swap_file
SWAP_SIZE_GB = args.swap_size_gb
MEMORY_MARGIN = args.memory_margin
GPU_MEMORY_MARGIN = args.gpu_memory_margin
MONITOR_INTERVAL = args.monitor_interval
MAX_RAM_LIMIT = args.max_ram_limit

CONVERT_CMD = [
    sys.executable, 'trt_pose/converter.py',
    '--model', args.model,
    '--input-shape'] + [str(x) for x in args.input_shape] + [
    '--checkpoint-path', args.checkpoint_path,
    '--engine-path', args.engine_path,
    '--precision', args.precision,
    '--max_workspace_size', str(args.max_workspace_size)
]

# --- Functions ---
def ensure_swap(file_path, size_gb):
    if os.path.exists(file_path):
        return
    size_bytes = size_gb * (1 << 30)
    subprocess.run(['sudo', 'fallocate', '-l', str(size_bytes), file_path], check=True)
    subprocess.run(['sudo', 'chmod', '600', file_path], check=True)
    subprocess.run(['sudo', 'mkswap', file_path], check=True)
    subprocess.run(['sudo', 'swapon', file_path], check=True)
    print(f"Swap file '{file_path}' of size {size_gb}GB activated.")


def get_system_usage():
    vm = psutil.virtual_memory()
    ram_used = vm.used
    ram_total = vm.total
    ram_free_ratio = vm.available / ram_total
    usage = {
        'ram_used': ram_used,
        'ram_total': ram_total,
        'ram_free_ratio': ram_free_ratio
    }
    if NVML_AVAILABLE:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        meminfo = nvmlDeviceGetMemoryInfo(handle)
        gpu_used = meminfo.used
        gpu_total = meminfo.total
        gpu_free_ratio = (gpu_total - gpu_used) / gpu_total
        usage.update({
            'gpu_used': gpu_used,
            'gpu_total': gpu_total,
            'gpu_free_ratio': gpu_free_ratio
        })
    return usage


def set_memory_limit(limit_bytes):
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    print(f"Process memory limit set to {limit_bytes/(1<<20):.1f} MB.")

# --- Main ---
if __name__ == '__main__':
    # 1. Create swap if missing
    try:
        ensure_swap(SWAP_FILE, SWAP_SIZE_GB)
    except Exception as e:
        print(f"[Warning] Failed to setup swap: {e}")

    # 2. Apply memory limit if specified
    if MAX_RAM_LIMIT:
        set_memory_limit(MAX_RAM_LIMIT)

    # 3. Start conversion
    print("Starting TensorRT conversion process...")
    proc = subprocess.Popen(CONVERT_CMD, stdout=sys.stdout, stderr=sys.stderr)

    # 4. Monitor until completion
    try:
        while proc.poll() is None:
            usage = get_system_usage()
            print(f"RAM: {usage['ram_used']/(1<<30):.2f}GB / {usage['ram_total']/(1<<30):.2f}GB (free {usage['ram_free_ratio']:.2%})")
            if NVML_AVAILABLE:
                print(f"GPU: {usage['gpu_used']/(1<<30):.2f}GB / {usage['gpu_total']/(1<<30):.2f}GB (free {usage['gpu_free_ratio']:.2%})")
            if usage['ram_free_ratio'] < MEMORY_MARGIN:
                print("[Warning] RAM free below threshold")
            if NVML_AVAILABLE and usage['gpu_free_ratio'] < GPU_MEMORY_MARGIN:
                print("[Warning] GPU free below threshold")
            time.sleep(MONITOR_INTERVAL)
    except KeyboardInterrupt:
        print("Terminating conversion process...")
        proc.terminate()
    print("Conversion process finished.")

