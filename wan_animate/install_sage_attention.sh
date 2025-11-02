#!/usr/bin/env bash
set -e
set -o pipefail

# We are applying a patch to setup.py to allow building without a GPU present
# and to customize the CUDA architectures built for each extension.
export SAGE_ATTENTION_ARCHS="8.0+PTX;8.6+PTX;8.9+PTX;9.0+PTX;12.0+PTX"
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
export FORCE_CUDA=1

export EXT_PARALLEL=4
export NVCC_APPEND_FLAGS="--threads 8"
export MAX_JOBS=8

# 1) Clone if missing
if [ ! -d "SageAttention" ]; then
    git clone https://github.com/thu-ml/SageAttention.git
fi
git -C SageAttention checkout 22c68fb5e561b66c0a816eb27f46299b3d2cc8af

# 2) Patch setup.py
# Patch setup.py: read env archs; use per-extension gencodes (sm80:80/86, sm89:89, sm90:90/90a, fused:80/86/89/90/90a/120)
echo "--- Patching SageAttention/setup.py ---"
/app/.venv/bin/python - <<'PY'
from pathlib import Path, re as _re
p = Path('SageAttention/setup.py')
s = p.read_text(encoding='utf-8')

# 1) Allow env-provided arch list (no GPU required)
inj = '''compute_capabilities = set()
# Injected: read archs from env if no GPUs
import os, re as _re
_env_archs = os.getenv("SAGE_ATTENTION_ARCHS") or os.getenv("CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST") or os.getenv("CMAKE_CUDA_ARCHITECTURES")
if _env_archs:
    compute_capabilities.update([a for a in _re.split(r"[;,\\s]+", _env_archs) if a])
'''
s = s.replace('compute_capabilities = set()', inj, 1)

# 2) Per-extension NVCC flags (filter gencodes)
add = r'''
# Injected: filter gencodes per extension to avoid sm90 kernels on sm_120
import re as _re
def _strip_gencodes(flags):
    out=[]; i=0
    while i < len(flags):
        if flags[i] == "-gencode":
            i += 2
        else:
            out.append(flags[i]); i += 1
    return out

def _filter_gencodes(flags, allowed):
    out=[]; i=0
    while i < len(flags):
        if flags[i] == "-gencode" and i+1 < len(flags):
            arg = flags[i+1]
            m = _re.search(r'arch=compute_([0-9]+a?)', arg)
            num = m.group(1) if m else None
            if num in allowed:
                out.extend([flags[i], arg])
            i += 2
        else:
            i += 1
    return out

NVCC_BASE = _strip_gencodes(NVCC_FLAGS)
NVCC_FLAGS_SM80 = NVCC_BASE + _filter_gencodes(NVCC_FLAGS, {"80","86"})
NVCC_FLAGS_SM89 = NVCC_BASE + _filter_gencodes(NVCC_FLAGS, {"89"})
NVCC_FLAGS_SM90 = NVCC_BASE + _filter_gencodes(NVCC_FLAGS, {"90","90a"})
NVCC_FLAGS_FUSED = NVCC_BASE + _filter_gencodes(NVCC_FLAGS, {"80","86","89","90","90a","120"})
'''
s = s.replace('ext_modules = []', add + '\next_modules = []', 1)

# 3) Swap NVCC flags usage per extension (order: sm80, sm89, sm90, fused)
count = {'i':0}
def repl(m):
    count['i'] += 1
    return {
        1: '"nvcc": NVCC_FLAGS_SM80,',
        2: '"nvcc": NVCC_FLAGS_SM89,',
        3: '"nvcc": NVCC_FLAGS_SM90,',
        4: '"nvcc": NVCC_FLAGS_FUSED,',
    }.get(count['i'], m.group(0))
s = _re.sub(r'"nvcc": NVCC_FLAGS,', repl, s)

p.write_text(s, encoding='utf-8')
print("Successfully patched SageAttention/setup.py")
PY


echo "--- Building and installing SageAttention ---"
(cd SageAttention && /app/.venv/bin/python setup.py install)