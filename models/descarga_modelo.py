import os
import sys
import subprocess
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

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Convert PyTorch .pth to TensorRT .engine with resource monitoring')
parser.add_argument('--checkpoint-path', '-c', required=True,
                    help='Path to the PyTorch .pth checkpoint file')
parser.add_argument('--engine-path', '-e', required=True,
                    help='Output path for the TensorRT engine file')
parser.add_argument('--converter-script', default='trt_pose/converter.py',
                    help='Path to trt_pose converter.py script')
parser.add_argument('--model', default='resnet18_baseline_att',
                    help='Model name for trt_pose converter')
parser.add_argument('--input-shape', nargs=4, type=int, metavar=('N','C','H','W'),
                    default=[1,3,224,224],
                    help='Input tensor shape')
parser.add_argument('--precision', choices=['fp16','fp32','int8'], default='fp16',
                    help='Precision mode')
parser.add_argument('--max-workspace-size', type=int, default=1<<30,
                    help='Max workspace size (bytes)')
parser.add_argument('--swap-file', default='/swapfile',
                    help='Swap file path')
parser.add_argument('--swap-size-gb', type=int, default=2,
                    help='Swap size (GB)')
parser.add_argument('--monitor-interval', type=int, default=5,
                    help='Seconds between resource checks')
parser.add_argument('--memory-margin', type=float, default=0.1,
                    help='Min free RAM fraction')
parser.add_argument('--gpu-memory-margin', type=float, default=0.1,
                    help='Min free GPU memory fraction')
parser.add_argument('--max-ram-limit', type=int, default=None,
                    help='Max RAM for process (bytes)')
args = parser.parse_args()

# Build conversion command
convert_cmd = [
    sys.executable, args.converter_script,
    '--model', args.model,
    '--input-shape'] + [str(x) for x in args.input_shape] + [
    '--checkpoint-path', args.checkpoint_path,
    '--engine-path', args.engine_path,
    '--precision', args.precision,
    '--max_workspace_size', str(args.max_workspace_size)
]

# --- Functions ---
def ensure_swap(path, gb):
    if os.path.exists(path): return
    size = gb * (1<<30)
    os.system(f"sudo fallocate -l {size} {path} && sudo chmod 600 {path} && sudo mkswap {path} && sudo swapon {path}")
    print(f"Swap {gb}GB activated at {path}")


def get_usage():
    vm = psutil.virtual_memory()
    usage = {'ram_free_ratio': vm.available/vm.total}
    if NVML_AVAILABLE:
        try:
            h = nvmlDeviceGetHandleByIndex(0)
            m = nvmlDeviceGetMemoryInfo(h)
            usage['gpu_free_ratio'] = (m.total-m.used)/m.total
        except:
            pass
    return usage


def set_limit(limit):
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    print(f"RAM limit {limit/(1<<20):.1f}MB")

# --- Main ---
if __name__ == '__main__':
    ensure_swap(args.swap_file, args.swap_size_gb)
    if args.max_ram_limit:
        set_limit(args.max_ram_limit)

    print("Starting conversion...")
    p = subprocess.Popen(convert_cmd)

    try:
        while p.poll() is None:
            u = get_usage()
            print(f"Free RAM {u['ram_free_ratio']:.2%}", end='')
            if 'gpu_free_ratio' in u:
                print(f" | Free GPU {u['gpu_free_ratio']:.2%}")
            else:
                print()
            if u['ram_free_ratio'] < args.memory_margin:
                print("Warn: low RAM")
            if 'gpu_free_ratio' in u and u['gpu_free_ratio'] < args.gpu_memory_margin:
                print("Warn: low GPU")
            time.sleep(args.monitor_interval)
    except KeyboardInterrupt:
        p.terminate()
    print("Done.")
