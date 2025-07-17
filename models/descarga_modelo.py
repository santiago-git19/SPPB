import os
import sys
import subprocess
import time
import psutil
import resource
import argparse
import torch
import trt_pose.models

# Try NVML
try:
    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    try:
        nvmlInit()
        NVML_AVAILABLE = True
    except:
        NVML_AVAILABLE = False
except ImportError:
    NVML_AVAILABLE = False

# --- Argument parsing ---
parser = argparse.ArgumentParser(description='Convert PyTorch .pth to TensorRT .engine with ONNX + trtexec, monitoring resources')
parser.add_argument('--checkpoint-path', '-c', required=True, help='Path to the PyTorch .pth file')
parser.add_argument('--engine-path', '-e', required=True, help='Path where to save the TensorRT engine (.engine)')
parser.add_argument('--onnx-path', default='model_tmp.onnx', help='Temporary ONNX export path')
parser.add_argument('--model', default='resnet18_baseline_att', help='TRT_Pose model name')
parser.add_argument('--num-classes', type=int, required=True, help='Number of keypoint classes')
parser.add_argument('--num-links', type=int, required=True, help='Number of link classes')
parser.add_argument('--input-shape', nargs=4, type=int, default=[1,3,224,224], metavar=('N','C','H','W'), help='Input tensor shape')
parser.add_argument('--precision', choices=['fp16','fp32'], default='fp16', help='Precision for TensorRT')
parser.add_argument('--max-workspace', type=int, default=1<<30, help='Max workspace size in bytes')
parser.add_argument('--swap-file', default='/swapfile', help='Swap file path')
parser.add_argument('--swap-size-gb', type=int, default=2, help='Swap file size GB')
parser.add_argument('--monitor-interval', type=int, default=5, help='Seconds between checks')
parser.add_argument('--memory-margin', type=float, default=0.1, help='Min free RAM fraction')
parser.add_argument('--gpu-memory-margin', type=float, default=0.1, help='Min free GPU mem fraction')
parser.add_argument('--max-ram-limit', type=int, default=None, help='Limit for process memory in bytes')
args = parser.parse_args()

# Functions

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
    print(f"RAM limit set to {limit/(1<<20):.1f} MB")

# Main
if __name__ == '__main__':
    ensure_swap(args.swap_file, args.swap_size_gb)
    if args.max_ram_limit:
        set_limit(args.max_ram_limit)

    # 1. Load and export ONNX
    print("Loading model and exporting to ONNX...")
    # Build model
    model = getattr(trt_pose.models, args.model)(args.num_classes, args.num_links)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval().cuda()

    dummy = torch.randn(*args.input_shape).cuda()
    torch.onnx.export(model, dummy, args.onnx_path,
                      input_names=['input'], output_names=['vector_field','heatmap'],
                      opset_version=11)
    print(f"ONNX saved to {args.onnx_path}")

    # 2. Convert ONNX to TensorRT with trtexec
    print("Converting ONNX to TensorRT engine...")
    trtexec = [
        'trtexec',
        f"--onnx={args.onnx_path}",
        f"--saveEngine={args.engine_path}",
        f"--maxWorkspaceSize={args.max_workspace}",
        '--fp16' if args.precision=='fp16' else '',
        f"--minShapes=input:1x3x{args.input_shape[2]}x{args.input_shape[3]}",
        f"--optShapes=input:1x3x{args.input_shape[2]}x{args.input_shape[3]}",
        f"--maxShapes=input:1x3x{args.input_shape[2]}x{args.input_shape[3]}"
    ]
    # Remove empty strings
    trtexec = [x for x in trtexec if x]
    proc = subprocess.Popen(trtexec, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    print(f"TensorRT engine saved to {args.engine_path}")

    # 3. Monitor resources
    print("Entering resource monitor (Ctrl+C to exit)")
    try:
        while True:
            u = get_usage()
            msg = f"Free RAM {u['ram_free_ratio']:.2%}"
            if 'gpu_free_ratio' in u:
                msg += f" | Free GPU {u['gpu_free_ratio']:.2%}"
            print(msg)
            if u['ram_free_ratio'] < args.memory_margin:
                print("[Warning] Low RAM")
            if 'gpu_free_ratio' in u and u['gpu_free_ratio'] < args.gpu_memory_margin:
                print("[Warning] Low GPU mem")
            time.sleep(args.monitor_interval)
    except KeyboardInterrupt:
        print("Monitor stopped")
    print("Done.")
