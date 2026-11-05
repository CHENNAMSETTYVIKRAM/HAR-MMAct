"""
create_kaggle_notebook.py - Generates the complete 15-section Kaggle benchmark notebook for MMAct HAR.

Structure:
Section 1:  Environment & Dependencies Setup (Preserving Kaggle C ABI)
Section 2:  Preflight Hardware & GPU Diagnostics (CUDA, VRAM, CPU, RAM, Disk)
Section 3:  Repository Checkout from master Branch
Section 4:  Dataset Discovery & Multi-Root Staging (Symlink-Safe)
Section 5:  Label Mapping & Pre-Flight Modality Audit (Zero-Leakage Split)
Section 6:  Video & Sensor Preprocessing Cache Warm-Up (/kaggle/working/har_mmact_cache)
Section 7:  Memory-Safe Calibration & OOM Warm-Up
Section 8:  Experiment 1: Improved Custom CNN Baseline (50 Epochs, AMP, Regularization)
Section 9:  Experiment 2: Hyperparameter Tuning (Train/Val only) & Tuned CNN (50 Epochs)
Section 10: Experiment 3: MobileNetV2 Staged Fine-Tuning (50 Epochs, AMP)
Section 11: Experiments 4 & 5: Sensor-Only InceptionTime & Multimodal Late Fusion (50 Epochs)
Section 12: Standalone Evaluation on Held-Out Test Set (All 5 Models)
Section 13: Comprehensive Publication Visualizations & Galleries
Section 14: Sample Multimodal Inference Demonstration
Section 15: Consolidated Summary Table, Manifest, Runtime Report & 1-Click Packaging

SAFETY GUARANTEE:
Target notebook is notebooks/kaggle_mmact_har_benchmark_50ep.ipynb to ensure
any currently active Kaggle notebook session is never disturbed.
"""

import argparse
import json
import sys
from pathlib import Path


def generate_benchmark_cells():
    cells = [
        # --- TITLE & OVERVIEW ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Multimodal Human Activity Recognition on MMAct (Kaggle GPU Pipeline)\n",
                "## 50-Epoch Benchmark: Custom CNN, Tuned CNN, MobileNetV2 Staged Fine-Tuning, Sensor-Only, and Multimodal Fusion\n",
                "\n",
                "This benchmark notebook executes the complete Multimodal Human Activity Recognition (HAR) pipeline on the **MMAct dataset** across 15 structured sections:\n",
                "- **Section 1**: Environment & Dependencies Setup (strictly preserving Kaggle's binary C ABI stack)\n",
                "- **Section 2**: Preflight Hardware & GPU Diagnostics (CUDA device, memory headroom, CPU, RAM, disk)\n",
                "- **Section 3**: Repository Checkout from `master` branch\n",
                "- **Section 4**: Multi-Root Dataset Discovery & Canonical Staging\n",
                "- **Section 5**: Label Mapping & Pre-Flight Modality Audit (Zero-Leakage Split Verification)\n",
                "- **Section 6**: Video & Sensor Preprocessing Cache Warm-Up (`/kaggle/working/har_mmact_cache`)\n",
                "- **Section 7**: Memory-Safe Calibration & OOM Warm-Up\n",
                "- **Section 8**: Experiment 1: Improved Custom CNN Baseline (50 epochs, AMP, spatial dropout, weight decay)\n",
                "- **Section 9**: Experiment 2: Hyperparameter Tuning (train/val only) & Tuned CNN (50 epochs)\n",
                "- **Section 10**: Experiment 3: MobileNetV2 Transfer Learning with Staged Fine-Tuning (50 epochs, AMP)\n",
                "- **Section 11**: Experiments 4 & 5: Sensor-Only InceptionTime & Multimodal Late Fusion (50 epochs each)\n",
                "- **Section 12**: Standalone Evaluation on Held-Out Test Set across all 5 models\n",
                "- **Section 13**: Comprehensive Publication Visualizations & Galleries (annotated curves, ROC, PR, confusion matrices)\n",
                "- **Section 14**: Sample Multimodal Inference Demonstration\n",
                "- **Section 15**: Consolidated Summary Table, Manifest, Runtime Report & 1-Click Packaging (`har_mmact_results.zip`)\n",
                "\n",
                "All experiments evaluate on the **10 verified target activity classes**:\n",
                "`crouching`, `jumping`, `kicking`, `pointing`, `running`, `sitting`, `standing`, `using_phone`, `walking`, `waving_hand`.\n",
                "\n",
                "---\n",
                "### Kaggle Setup Instructions:\n",
                "1. **Enable GPU Accelerator:** Under **Session Options** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.\n",
                "2. **Enable Internet Access:** Under **Session Options** -> **Internet**, toggle to **On** (required for GitHub repo checkout).\n",
                "3. **Attach MMAct Dataset:** Click **+ Add Input** and attach your MMAct dataset (e.g. `mmact-trimmed`)."
            ]
        },

        # --- SECTION 1: SETUP & DEPENDENCIES ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 1: Environment & Dependencies Setup\n",
                "Installs auxiliary dependencies while strictly preserving Kaggle's preinstalled, binary-compatible C ABI scientific stack (NumPy, Pandas, SciPy, PyTorch, Torchvision, Scikit-Learn)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 1: Dependencies Setup with Kaggle C ABI Protection\n",
                "import sys\n",
                "import os\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "PROJECT_ROOT = Path('/kaggle/working/HAR-MMAct')\n",
                "CORE_C_PACKAGES = {'numpy', 'pandas', 'scipy', 'torch', 'torchvision', 'opencv-python', 'pillow', 'scikit-learn', 'matplotlib', 'seaborn', 'pytest'}\n",
                "\n",
                "kaggle_req = PROJECT_ROOT / 'requirements-kaggle.txt'\n",
                "if kaggle_req.exists():\n",
                "    print(f'Installing auxiliary dependencies from {kaggle_req.name}...')\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--prefer-binary', '--no-deps', '-r', str(kaggle_req)], check=False)\n",
                "else:\n",
                "    # Install standard light requirements without overriding core C packages\n",
                "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabulate', 'psutil'], check=False)\n",
                "\n",
                "# Post-install C ABI Verification\n",
                "print('\\n' + '=' * 70)\n",
                "print('POST-INSTALL ABI & IMPORT VERIFICATION')\n",
                "print('=' * 70)\n",
                "abi_failed = False\n",
                "abi_errors = []\n",
                "\n",
                "try:\n",
                "    import numpy as np\n",
                "    print(f'  [PASS] Python      : {sys.version.split()[0]}')\n",
                "    print(f'  [PASS] NumPy       : {np.__version__}')\n",
                "    _arr = np.eye(2, dtype=np.float64)\n",
                "    _ = np.linalg.pinv(_arr)\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'NumPy C-extension check failed: {e}')\n",
                "\n",
                "try:\n",
                "    import pandas as pd\n",
                "    print(f'  [PASS] Pandas      : {pd.__version__}')\n",
                "    _df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})\n",
                "    _ = _df.mean()\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'Pandas C-extension check failed: {e}')\n",
                "\n",
                "try:\n",
                "    import scipy.signal\n",
                "    print(f'  [PASS] SciPy       : {scipy.__version__}')\n",
                "    _ = scipy.signal.resample([1.0, 2.0, 3.0, 4.0], 8)\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'SciPy C-extension check failed: {e}')\n",
                "\n",
                "try:\n",
                "    import cv2\n",
                "    print(f'  [PASS] OpenCV      : {cv2.__version__}')\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'OpenCV check failed: {e}')\n",
                "\n",
                "try:\n",
                "    import torch\n",
                "    print(f'  [PASS] PyTorch     : {torch.__version__} (CUDA: {torch.cuda.is_available()})')\n",
                "    _t = torch.randn(2, 2)\n",
                "    _ = _t.numpy()\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'PyTorch <-> NumPy C ABI bridge check failed: {e}')\n",
                "\n",
                "try:\n",
                "    import sklearn\n",
                "    print(f'  [PASS] Scikit-Learn: {sklearn.__version__}')\n",
                "except Exception as e:\n",
                "    abi_failed = True\n",
                "    abi_errors.append(f'Scikit-Learn check failed: {e}')\n",
                "\n",
                "if abi_failed:\n",
                "    print('\\n' + '!' * 70)\n",
                "    print('CRITICAL ABI BINARY INCOMPATIBILITY DETECTED:')\n",
                "    for err in abi_errors:\n",
                "        print(f'  - {err}')\n",
                "    print('\\nREMEDY: Restart Kaggle Session from top menu, do NOT run pip install numpy manually.')\n",
                "    print('!' * 70)\n",
                "    raise RuntimeError('C ABI binary incompatibility detected:\\n' + '\\n'.join(abi_errors))\n",
                "\n",
                "print('=' * 70)\n",
                "print('>>> C ABI & IMPORT VERIFICATION PASSED: Preinstalled scientific stack is healthy.')\n",
                "print('=' * 70)\n"
            ]
        },

        # --- SECTION 2: HARDWARE & GPU DIAGNOSTICS ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 2: Preflight Hardware & GPU Diagnostics\n",
                "Audits GPU accelerator, CUDA device capability, VRAM headroom, CPU threads, host RAM, and working disk space. Enforces fail-loud check if GPU accelerator is disabled."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 2: Preflight Hardware & GPU Diagnostics\n",
                "import os\n",
                "import shutil\n",
                "import psutil\n",
                "import torch\n",
                "\n",
                "!nvidia-smi\n",
                "\n",
                "print('=' * 78)\n",
                "print(' PREFLIGHT HARDWARE & GPU DIAGNOSTICS')\n",
                "print('=' * 78)\n",
                "\n",
                "# 1. CUDA & GPU Verification\n",
                "cuda_ok = torch.cuda.is_available()\n",
                "print(f'PyTorch Version       : {torch.__version__}')\n",
                "print(f'CUDA Available        : {cuda_ok}')\n",
                "\n",
                "if not cuda_ok:\n",
                "    error_msg = (\n",
                "        'CRITICAL FAILURE: CUDA GPU is not available!\\n'\n",
                "        'This 50-epoch benchmark requires a GPU accelerator.\\n'\n",
                "        'ACTION REQUIRED: In Kaggle right-hand panel -> Session Options -> Accelerator, '\n",
                "        'select GPU T4 x2 or GPU P100, then restart the session.'\n",
                "    )\n",
                "    print('!' * 78)\n",
                "    print(error_msg)\n",
                "    print('!' * 78)\n",
                "    raise RuntimeError(error_msg)\n",
                "\n",
                "gpu_count = torch.cuda.device_count()\n",
                "gpu_name = torch.cuda.get_device_name(0)\n",
                "compute_cap = torch.cuda.get_device_capability(0)\n",
                "props = torch.cuda.get_device_properties(0)\n",
                "total_vram_gb = props.total_memory / (1024**3)\n",
                "alloc_vram_mb = torch.cuda.memory_allocated(0) / (1024**2)\n",
                "res_vram_mb = torch.cuda.memory_reserved(0) / (1024**2)\n",
                "\n",
                "print(f'GPU Device Name       : {gpu_name} (x{gpu_count} detected)')\n",
                "print(f'Compute Capability    : {compute_cap[0]}.{compute_cap[1]}')\n",
                "print(f'Total VRAM Available  : {total_vram_gb:.2f} GB')\n",
                "print(f'Initial Allocated VRAM: {alloc_vram_mb:.2f} MB')\n",
                "print(f'Initial Reserved VRAM : {res_vram_mb:.2f} MB')\n",
                "\n",
                "# 2. Host System Hardware\n",
                "cpu_count = os.cpu_count() or 1\n",
                "vm = psutil.virtual_memory()\n",
                "ram_total_gb = vm.total / (1024**3)\n",
                "ram_avail_gb = vm.available / (1024**3)\n",
                "\n",
                "working_disk = shutil.disk_usage('/kaggle/working')\n",
                "disk_free_gb = working_disk.free / (1024**3)\n",
                "disk_total_gb = working_disk.total / (1024**3)\n",
                "\n",
                "print(f'CPU Logical Cores     : {cpu_count}')\n",
                "print(f'System RAM Total      : {ram_total_gb:.2f} GB (Available: {ram_avail_gb:.2f} GB)')\n",
                "print(f'Working Disk Space    : {disk_free_gb:.2f} GB free of {disk_total_gb:.2f} GB')\n",
                "\n",
                "# 3. Runtime Training Settings\n",
                "print('\\nRuntime Acceleration Configuration:')\n",
                "print('  - Batch Size          : 8 (optimal for 16-frame 112x112 video + 9-channel sensor)')\n",
                "print('  - DataLoader Workers  : 2 (non-blocking pin_memory)')\n",
                "print('  - Automatic Mixed Prec: ENABLED (torch.amp.autocast, float16)')\n",
                "print('  - cuDNN Benchmark     : ENABLED (torch.backends.cudnn.benchmark = True)')\n",
                "print('=' * 78)\n"
            ]
        },

        # --- SECTION 3: REPOSITORY CHECKOUT ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 3: Repository Checkout from master Branch\n",
                "Clones or pulls the latest verified code from the `master` branch of `https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct.git`, configures Python `sys.path`, and verifies project modules."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 3: Repository Checkout from master Branch\n",
                "import sys\n",
                "import os\n",
                "import subprocess\n",
                "from pathlib import Path\n",
                "\n",
                "PROJECT_ROOT = Path('/kaggle/working/HAR-MMAct')\n",
                "REPO_URL = 'https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct.git'\n",
                "BRANCH = 'master'\n",
                "\n",
                "# 1. Clone or update repository into PROJECT_ROOT explicitly on 'master' branch\n",
                "if not PROJECT_ROOT.exists():\n",
                "    print(f'Cloning {REPO_URL} (branch: {BRANCH}) into {PROJECT_ROOT}...')\n",
                "    subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, str(PROJECT_ROOT)], check=True)\n",
                "else:\n",
                "    print(f'Repository already present at {PROJECT_ROOT}. Fetching and checking out {BRANCH}...')\n",
                "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'fetch', 'origin', BRANCH], check=False)\n",
                "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'checkout', BRANCH], check=False)\n",
                "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'pull', 'origin', BRANCH], check=False)\n",
                "\n",
                "# 2. Print active branch and commit hash for diagnostics\n",
                "try:\n",
                "    branch_out = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()\n",
                "    commit_out = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip()\n",
                "    print(f'Active Git Branch : {branch_out}')\n",
                "    print(f'Active Git Commit : {commit_out}')\n",
                "except Exception as e:\n",
                "    print(f'[Warning] Could not inspect git metadata: {e}')\n",
                "    branch_out, commit_out = 'unknown', 'unknown'\n",
                "\n",
                "# 3. Verify expected files exist on master\n",
                "req_files = [\n",
                "    PROJECT_ROOT / 'src' / 'data' / 'ingestion.py',\n",
                "    PROJECT_ROOT / 'src' / 'data' / 'indexer.py',\n",
                "    PROJECT_ROOT / 'src' / 'data' / 'dataset.py',\n",
                "    PROJECT_ROOT / 'scripts' / 'train.py',\n",
                "    PROJECT_ROOT / 'scripts' / 'tune.py',\n",
                "    PROJECT_ROOT / 'scripts' / 'evaluate.py',\n",
                "]\n",
                "missing_files = [str(f.relative_to(PROJECT_ROOT)) for f in req_files if not f.exists()]\n",
                "if missing_files:\n",
                "    raise FileNotFoundError(f'CRITICAL: Missing required files on branch {branch_out}: {missing_files}')\n",
                "\n",
                "print('SUCCESS: Verified presence of core modules on master branch.')\n",
                "\n",
                "# 4. Configure import path before importing project modules\n",
                "sys.path.insert(0, str(PROJECT_ROOT))\n",
                "os.chdir(PROJECT_ROOT)\n",
                "print(f'PROJECT_ROOT configured : {PROJECT_ROOT}')\n",
                "print(f'sys.path[0] set to      : {sys.path[0]}')\n",
                "\n",
                "# 5. Verify project module imports\n",
                "from src.data.ingestion import DatasetIngestionPipeline\n",
                "from src.data.indexer import MMActIndexer\n",
                "from src.data.dataset import MMActMultimodalDataset\n",
                "from src.models.fusion_model import MultimodalHARModel\n",
                "from src.utils.plots import plot_cross_model_comparison, plot_modality_ablation\n",
                "print('SUCCESS: Verified project imports (ingestion, indexer, dataset, models, plots).')\n"
            ]
        },

        # --- SECTION 4: DATASET DISCOVERY & STAGING ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 4: Dataset Discovery & Multi-Root Staging (Symlink-Safe)\n",
                "Crawls `/kaggle/input`, detects all attached MMAct dataset directories and archives, safely unpacks archives if needed, and stages a canonical dataset hierarchy at `/kaggle/working/HAR-MMAct/data/genuine_mmact/`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 4: Dataset Discovery & Multi-Root Staging\n",
                "import os\n",
                "import re\n",
                "import shutil\n",
                "from pathlib import Path\n",
                "from src.data.ingestion import DatasetIngestionPipeline\n",
                "from src.data.indexer import MMActIndexer\n",
                "\n",
                "kaggle_input = Path('/kaggle/input')\n",
                "canonical_data_dir = PROJECT_ROOT / 'data' / 'genuine_mmact'\n",
                "canonical_trimmed = canonical_data_dir / 'trimmed'\n",
                "canonical_video = canonical_trimmed / 'video'\n",
                "canonical_sensors = canonical_trimmed / 'sensors'\n",
                "canonical_splits = canonical_trimmed / 'splits'\n",
                "\n",
                "canonical_video.mkdir(parents=True, exist_ok=True)\n",
                "canonical_sensors.mkdir(parents=True, exist_ok=True)\n",
                "canonical_splits.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print('=' * 78)\n",
                "print('MULTI-DATASET INSPECTION & CANONICAL INGESTION')\n",
                "print('=' * 78)\n",
                "\n",
                "# 1. Inspect all attached dataset directories under /kaggle/input\n",
                "input_roots = sorted([p for p in kaggle_input.iterdir() if p.is_dir()]) if kaggle_input.exists() else []\n",
                "if not input_roots:\n",
                "    local_data = PROJECT_ROOT / 'data'\n",
                "    if local_data.exists():\n",
                "        input_roots = [local_data]\n",
                "\n",
                "print(f'Found {len(input_roots)} top-level dataset roots:')\n",
                "for root in input_roots:\n",
                "    print(f'  - {root.name} ({root})')\n",
                "\n",
                "def find_all_files(root: Path, extensions: set) -> list:\n",
                "    return MMActIndexer.find_files_recursively(root, extensions)\n",
                "\n",
                "def safe_link_or_copy(src: Path, dst: Path):\n",
                "    if dst.exists():\n",
                "        if dst.is_symlink() or dst.is_file():\n",
                "            dst.unlink()\n",
                "        elif dst.is_dir() and not any(dst.iterdir()):\n",
                "            dst.rmdir()\n",
                "        else:\n",
                "            return\n",
                "    try:\n",
                "        os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())\n",
                "        print(f'  [Symlinked] {src.name} -> {dst.relative_to(PROJECT_ROOT)}')\n",
                "    except Exception:\n",
                "        if src.is_dir():\n",
                "            shutil.copytree(src.resolve(), dst, dirs_exist_ok=True)\n",
                "            print(f'  [Copied Dir] {src.name} -> {dst.relative_to(PROJECT_ROOT)}')\n",
                "        else:\n",
                "            shutil.copy2(src.resolve(), dst)\n",
                "            print(f'  [Copied File] {src.name} -> {dst.relative_to(PROJECT_ROOT)}')\n",
                "\n",
                "# 2. Discover and unpack any ZIP / TAR archives\n",
                "all_archives = []\n",
                "for r in input_roots:\n",
                "    for f in find_all_files(r, {'.zip', '.tar.gz', '.tar'}):\n",
                "        all_archives.append(f)\n",
                "\n",
                "if all_archives:\n",
                "    print(f'\\n[Archives] Discovered {len(all_archives)} dataset archive(s):')\n",
                "    dest_downloads = PROJECT_ROOT / 'data' / 'downloads'\n",
                "    dest_downloads.mkdir(parents=True, exist_ok=True)\n",
                "    for arc in all_archives:\n",
                "        print(f'  - Archive: {arc.name} ({arc.stat().st_size / (1024**2):.1f} MB)')\n",
                "        staging_arc = dest_downloads / arc.name\n",
                "        if not staging_arc.exists():\n",
                "            try:\n",
                "                os.symlink(arc.resolve(), staging_arc)\n",
                "            except Exception:\n",
                "                shutil.copy2(arc.resolve(), staging_arc)\n",
                "    ingestion = DatasetIngestionPipeline(incoming_dir=dest_downloads, raw_storage_dir=canonical_trimmed)\n",
                "    records = ingestion.scan_and_ingest()\n",
                "    print(f'  - Ingestion processed {len(records)} archive records.')\n",
                "\n",
                "# 3. Stage video, sensor, and split files from ALL input roots\n",
                "discovered_subjects = set()\n",
                "for r in input_roots:\n",
                "    for mp4 in find_all_files(r, {'.mp4'}):\n",
                "        for parent in mp4.parents:\n",
                "            if re.match(r'^(subject|sub)_?\\d+$', parent.name, re.I):\n",
                "                norm_subj = MMActIndexer.normalize_identifier(parent.name, 'subject')\n",
                "                if norm_subj not in discovered_subjects:\n",
                "                    discovered_subjects.add(norm_subj)\n",
                "                    dest_subj = canonical_video / 'cross_scene' / 'trainval' / norm_subj\n",
                "                    dest_subj.parent.mkdir(parents=True, exist_ok=True)\n",
                "                    safe_link_or_copy(parent, dest_subj)\n",
                "                break\n",
                "\n",
                "for r in input_roots:\n",
                "    for csv in find_all_files(r, {'.csv'}):\n",
                "        for parent in csv.parents:\n",
                "            p_lower = parent.name.lower()\n",
                "            if p_lower in MMActIndexer.SENSOR_NAME_MAP:\n",
                "                canon_mod_name = p_lower if p_lower.endswith('_clip') else f'{p_lower}_clip'\n",
                "                dest_mod = canonical_sensors / canon_mod_name\n",
                "                dest_mod.mkdir(parents=True, exist_ok=True)\n",
                "                for s_child in parent.iterdir():\n",
                "                    if s_child.is_dir() and re.match(r'^(subject|sub)_?\\d+$', s_child.name, re.I):\n",
                "                        norm_s = MMActIndexer.normalize_identifier(s_child.name, 'subject')\n",
                "                        safe_link_or_copy(s_child, dest_mod / norm_s)\n",
                "            elif p_lower in {'sensors', 'sensor', 'trimmed_sensor'}:\n",
                "                for mod_cand in parent.iterdir():\n",
                "                    if mod_cand.is_dir():\n",
                "                        m_lower = mod_cand.name.lower()\n",
                "                        if m_lower in MMActIndexer.SENSOR_NAME_MAP:\n",
                "                            canon_mod_name = m_lower if m_lower.endswith('_clip') else f'{m_lower}_clip'\n",
                "                            dest_mod = canonical_sensors / canon_mod_name\n",
                "                            dest_mod.mkdir(parents=True, exist_ok=True)\n",
                "                            for s_child in mod_cand.iterdir():\n",
                "                                if s_child.is_dir() and re.match(r'^(subject|sub)_?\\d+$', s_child.name, re.I):\n",
                "                                    norm_s = MMActIndexer.normalize_identifier(s_child.name, 'subject')\n",
                "                                    safe_link_or_copy(s_child, dest_mod / norm_s)\n",
                "\n",
                "discovered_splits = set()\n",
                "for r in input_roots:\n",
                "    for txt in find_all_files(r, {'.txt'}):\n",
                "        txt_name_lower = txt.name.lower()\n",
                "        if any(k in txt_name_lower for k in ['cross_scene', 'cross_view', 'train', 'val', 'test']):\n",
                "            if txt.name not in discovered_splits:\n",
                "                discovered_splits.add(txt.name)\n",
                "                safe_link_or_copy(txt, canonical_splits / txt.name)\n",
                "\n",
                "print(f'\\nCanonical Staging Summary:')\n",
                "print(f'  - Staged Video Subjects   : {sorted(list(discovered_subjects))}')\n",
                "print(f'  - Staged Sensor Modalities: {[m.name for m in canonical_sensors.iterdir() if m.is_dir()]}')\n",
                "print(f'  - Staged Split Files       : {sorted(list(discovered_splits))}')\n",
                "print('=' * 78)\n"
            ]
        },

        # --- SECTION 5: AUDIT & ZERO-LEAKAGE SPLIT ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 5: Label Mapping & Pre-Flight Modality Audit (Zero-Leakage Split)\n",
                "Verifies video frame decoding, sensor CSV parsing, index pairing, and enforces strict subject-independent split isolation (train/val/test) across the 10 target activity classes."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 5: Label Mapping & Pre-Flight Modality Audit\n",
                "import cv2\n",
                "import pandas as pd\n",
                "from src.data.indexer import MMActIndexer\n",
                "from src.data.splits import DatasetSplitter, verify_split_isolation\n",
                "from src.config import SplitConfig, VERIFIED_TARGET_CLASSES\n",
                "\n",
                "print('=' * 78)\n",
                "print('PRE-FLIGHT VALIDATION & COMPREHENSIVE MULTI-MODALITY AUDIT')\n",
                "print('=' * 78)\n",
                "print(f'Repository Root : {PROJECT_ROOT}')\n",
                "print(f'Canonical Root  : {canonical_data_dir.resolve()}')\n",
                "\n",
                "# 1. File Inventory\n",
                "total_videos = MMActIndexer.find_files_recursively(canonical_data_dir, {'.mp4'})\n",
                "total_sensors = MMActIndexer.find_files_recursively(canonical_data_dir, {'.csv'})\n",
                "total_splits = [f for f in MMActIndexer.find_files_recursively(canonical_data_dir, {'.txt'}) if any(k in f.name.lower() for k in ['cross_scene', 'cross_view', 'train', 'val', 'test'])]\n",
                "\n",
                "print(f'CANONICAL STAGED TOTALS: {len(total_videos)} Videos | {len(total_sensors)} Sensors | {len(total_splits)} Splits')\n",
                "assert len(total_videos) > 0 and len(total_sensors) > 0, 'CRITICAL: No valid MMAct data files found in canonical directory!'\n",
                "\n",
                "# 2. Modality Readability check\n",
                "sample_video = total_videos[0]\n",
                "cap = cv2.VideoCapture(str(sample_video))\n",
                "ret, frame = cap.read()\n",
                "frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
                "cap.release()\n",
                "assert ret and frame is not None, f'Failed to read frame from {sample_video}'\n",
                "print(f'  [PASS] Video Readability: Decoded frame {frame.shape} from {sample_video.name} ({frame_count} frames)')\n",
                "\n",
                "sample_sensor = total_sensors[0]\n",
                "df_sample = pd.read_csv(sample_sensor)\n",
                "print(f'  [PASS] Sensor Parseability: Parsed CSV {df_sample.shape} from {sample_sensor.name}')\n",
                "\n",
                "# 3. Indexing & Target Class Coverage\n",
                "indexer = MMActIndexer(dataset_root=canonical_data_dir)\n",
                "report = indexer.index()\n",
                "print(f'  - Paired Samples Created: {report.paired_samples_count}')\n",
                "print(f'  Target Class Coverage ({len(report.class_distribution)}/{len(VERIFIED_TARGET_CLASSES)} target classes):')\n",
                "for cls_name in VERIFIED_TARGET_CLASSES:\n",
                "    cnt = report.class_distribution.get(cls_name, 0)\n",
                "    print(f'    * {cls_name:<16}: {cnt} samples')\n",
                "\n",
                "assert report.paired_samples_count > 0, 'Zero paired multimodal samples created!'\n",
                "\n",
                "# 4. Subject-Independent Split Isolation Verification\n",
                "splitter = DatasetSplitter(SplitConfig(split_strategy='subject'))\n",
                "train_samples, val_samples, test_samples = splitter.split(report.paired_samples)\n",
                "isolation = verify_split_isolation(train_samples, val_samples, test_samples, strategy='subject')\n",
                "print(f'  - Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}')\n",
                "for check_k, passed in isolation.items():\n",
                "    print(f'      * {check_k:<28}: {\"PASSED\" if passed else \"FAILED\"}')\n",
                "    assert passed, f'CRITICAL: Split leakage detected on {check_k}!'\n",
                "print('=' * 78)\n",
                "print('>>> PRE-FLIGHT VALIDATION PASSED: Dataset is readable, parseable, and strictly isolated.')\n",
                "print('=' * 78)\n"
            ]
        },

        # --- SECTION 6: CACHE WARM-UP ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 6: Video & Sensor Preprocessing Cache Warm-Up\n",
                "Pre-caches normalized 16-frame 112x112 video tensors and aligned 100-sample 9-axis sensor windows to `/kaggle/working/har_mmact_cache/` so all 5 subsequent experiments execute with maximum speed without repeated OpenCV decoding."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 6: Video & Sensor Preprocessing Cache Warm-Up\n",
                "from src.data.dataset import MMActMultimodalDataset\n",
                "\n",
                "CACHE_DIR = Path('/kaggle/working/har_mmact_cache')\n",
                "CACHE_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'Initializing multimodal datasets with shared cache at {CACHE_DIR}...')\n",
                "train_ds = MMActMultimodalDataset(train_samples, cache_dir=CACHE_DIR / 'train', enable_cache=True)\n",
                "val_ds = MMActMultimodalDataset(val_samples, cache_dir=CACHE_DIR / 'val', enable_cache=True)\n",
                "test_ds = MMActMultimodalDataset(test_samples, cache_dir=CACHE_DIR / 'test', enable_cache=True)\n",
                "\n",
                "# Sequentially precompute and warm up cache\n",
                "print('\\n[Warming Up Training Set Cache]')\n",
                "train_stats = train_ds.warm_up_cache(verbose=True)\n",
                "\n",
                "print('\\n[Warming Up Validation Set Cache]')\n",
                "val_stats = val_ds.warm_up_cache(verbose=True)\n",
                "\n",
                "print('\\n[Warming Up Test Set Cache]')\n",
                "test_stats = test_ds.warm_up_cache(verbose=True)\n",
                "\n",
                "disk_free = shutil.disk_usage(str(CACHE_DIR)).free / (1024**3)\n",
                "print('\\n' + '=' * 78)\n",
                "print(f'CACHE WARM-UP COMPLETED:')\n",
                "print(f'  Train Cache Writes: {train_stats[\"writes\"]} | Hits: {train_stats[\"hits\"]}')\n",
                "print(f'  Val   Cache Writes: {val_stats[\"writes\"]} | Hits: {val_stats[\"hits\"]}')\n",
                "print(f'  Test  Cache Writes: {test_stats[\"writes\"]} | Hits: {test_stats[\"hits\"]}')\n",
                "print(f'  Remaining Disk    : {disk_free:.2f} GB free')\n",
                "print('=' * 78)\n"
            ]
        },

        # --- SECTION 7: MEMORY-SAFE CALIBRATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 7: Memory-Safe Calibration & OOM Warm-Up\n",
                "Executes a 1-batch calibration forward/backward pass with Automatic Mixed Precision (AMP) to verify GPU VRAM safety and establish baseline memory headroom before starting full 50-epoch training."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 7: Memory-Safe Calibration & OOM Warm-Up\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import DataLoader\n",
                "from src.models.fusion_model import MultimodalHARModel\n",
                "\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "torch.cuda.reset_peak_memory_stats(0)\n",
                "\n",
                "print('Running 1-batch GPU memory calibration test with AMP...')\n",
                "calib_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)\n",
                "v_sample, s_sample, l_sample, _ = next(iter(calib_loader))\n",
                "\n",
                "model_calib = MultimodalHARModel(\n",
                "    num_classes=10,\n",
                "    video_backbone='mobilenet_v2',\n",
                "    sensor_backbone='inception_time',\n",
                "    fusion_hidden_dim=128\n",
                ").to(device)\n",
                "\n",
                "optimizer = torch.optim.Adam(model_calib.parameters(), lr=1e-4)\n",
                "criterion = nn.CrossEntropyLoss()\n",
                "scaler = torch.amp.GradScaler('cuda')\n",
                "\n",
                "v_gpu = v_sample.to(device, non_blocking=True)\n",
                "s_gpu = s_sample.to(device, non_blocking=True)\n",
                "l_gpu = l_sample.to(device, non_blocking=True)\n",
                "\n",
                "optimizer.zero_grad(set_to_none=True)\n",
                "with torch.amp.autocast(device_type='cuda', dtype=torch.float16):\n",
                "    out = model_calib(video=v_gpu, sensor=s_gpu, override_mode='multimodal')\n",
                "    loss = criterion(out, l_gpu)\n",
                "\n",
                "scaler.scale(loss).backward()\n",
                "scaler.step(optimizer)\n",
                "scaler.update()\n",
                "\n",
                "peak_calib_mb = torch.cuda.max_memory_allocated(0) / (1024**2)\n",
                "total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)\n",
                "headroom_mb = total_vram_mb - peak_calib_mb\n",
                "\n",
                "print(f'Peak Memory during Multimodal Calibration : {peak_calib_mb:.2f} MB')\n",
                "print(f'Total VRAM Available                       : {total_vram_mb:.2f} MB')\n",
                "print(f'Safe Headroom Remaining                    : {headroom_mb:.2f} MB ({headroom_mb/total_vram_mb*100:.1f}%)')\n",
                "\n",
                "del model_calib, optimizer, criterion, scaler, v_gpu, s_gpu, l_gpu, out, loss\n",
                "torch.cuda.empty_cache()\n",
                "print('SUCCESS: Memory calibration passed. Safe to proceed with 50-epoch training.')\n"
            ]
        },

        # --- SECTION 8: EXPERIMENT 1 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 8: Experiment 1 - Improved Custom CNN Baseline (50 Epochs)\n",
                "Trains the improved custom video CNN baseline for 50 epochs using Spatial Dropout2d, BatchNorm2d, Weight Decay ($10^{-3}$), Cosine Annealing, and Automatic Mixed Precision."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 8: Experiment 1 - Improved Custom CNN Baseline (50 Epochs)\n",
                "EPOCHS = 50\n",
                "out_cnn = 'outputs/baseline_cnn'\n",
                "Path(out_cnn).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'=== Starting Experiment 1: Custom CNN Baseline ({EPOCHS} Epochs) ===')\n",
                "!python scripts/train.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --video_backbone cnn \\\n",
                "    --sensor_backbone inception_time \\\n",
                "    --mode video_only \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --min_epochs 50 \\\n",
                "    --batch_size 8 \\\n",
                "    --lr 0.0001 \\\n",
                "    --weight_decay 0.001 \\\n",
                "    --dropout 0.35 \\\n",
                "    --split_strategy subject \\\n",
                "    --early_stopping_patience 15 \\\n",
                "    --checkpoint_name best_model_cnn.pth \\\n",
                "    --output_dir {out_cnn} \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache \\\n",
                "    --benchmark_warmup\n"
            ]
        },

        # --- SECTION 9: EXPERIMENT 2 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 9: Experiment 2 - Hyperparameter Tuning (Train/Val Only) & Tuned CNN (50 Epochs)\n",
                "Systematically explores learning rate, weight decay, and dropout across 4 trials strictly on train/val splits. The held-out test split is NEVER touched. The best configuration is then trained for 50 epochs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 9: Experiment 2 - Hyperparameter Tuning & Tuned CNN (50 Epochs)\n",
                "print('=== Starting Experiment 2: Hyperparameter Tuning (Train/Val Only) ===')\n",
                "!python scripts/tune.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --split_strategy subject \\\n",
                "    --epochs_per_trial 10 \\\n",
                "    --batch_size 8 \\\n",
                "    --output_dir outputs/tuning \\\n",
                "    --plots_dir outputs/plots \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache\n",
                "\n",
                "# Load best hyperparameters discovered during tuning\n",
                "import json\n",
                "with open('outputs/tuning/best_hyperparameters.json') as f:\n",
                "    best_cfg = json.load(f)\n",
                "print(f'Optimal Hyperparameters Discovered: {best_cfg}')\n",
                "\n",
                "out_tuned = 'outputs/tuned_cnn'\n",
                "Path(out_tuned).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'\\n=== Training Tuned CNN for {EPOCHS} Epochs on Best Configuration ===')\n",
                "!python scripts/train.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --video_backbone cnn \\\n",
                "    --sensor_backbone inception_time \\\n",
                "    --mode video_only \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --min_epochs 50 \\\n",
                "    --batch_size 8 \\\n",
                "    --lr {best_cfg['lr']} \\\n",
                "    --weight_decay {best_cfg['weight_decay']} \\\n",
                "    --dropout {best_cfg['dropout']} \\\n",
                "    --split_strategy subject \\\n",
                "    --early_stopping_patience 15 \\\n",
                "    --checkpoint_name best_model_tuned_cnn.pth \\\n",
                "    --output_dir {out_tuned} \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache \\\n",
                "    --benchmark_warmup\n"
            ]
        },

        # --- SECTION 10: EXPERIMENT 3 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 10: Experiment 3 - MobileNetV2 Staged Fine-Tuning (50 Epochs)\n",
                "Phase 1 (Epochs 1-15): Pretrained MobileNetV2 backbone frozen, training classification head at $10^{-4}$.\n",
                "Phase 2 (Epochs 16-50): Backbone unfrozen with differential learning rate ($10^{-5}$) for gentle fine-tuning."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 10: Experiment 3 - MobileNetV2 Staged Fine-Tuning (50 Epochs)\n",
                "out_mob = 'outputs/mobilenet_v2'\n",
                "Path(out_mob).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'=== Starting Experiment 3: MobileNetV2 Staged Fine-Tuning ({EPOCHS} Epochs) ===')\n",
                "!python scripts/train.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --video_backbone mobilenet_v2 \\\n",
                "    --sensor_backbone inception_time \\\n",
                "    --mode video_only \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --min_epochs 50 \\\n",
                "    --batch_size 8 \\\n",
                "    --lr 0.0001 \\\n",
                "    --weight_decay 0.0005 \\\n",
                "    --dropout 0.35 \\\n",
                "    --staged_finetune \\\n",
                "    --unfreeze_epoch 15 \\\n",
                "    --unfreeze_lr 0.00001 \\\n",
                "    --split_strategy subject \\\n",
                "    --early_stopping_patience 15 \\\n",
                "    --checkpoint_name best_model_mobilenet_v2.pth \\\n",
                "    --output_dir {out_mob} \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache \\\n",
                "    --benchmark_warmup\n"
            ]
        },

        # --- SECTION 11: EXPERIMENTS 4 & 5 ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 11: Experiments 4 & 5 - Sensor-Only InceptionTime & Multimodal Late Fusion (50 Epochs)\n",
                "- **Experiment 4**: InceptionTime 1D convolutional network trained on 9-axis sensor time series.\n",
                "- **Experiment 5**: Multimodal Late Fusion combining MobileNetV2 video representations with InceptionTime sensor features."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 11: Experiments 4 & 5 - Sensor-Only & Multimodal Late Fusion (50 Epochs)\n",
                "out_sensor = 'outputs/sensor_only'\n",
                "out_fusion = 'outputs/multimodal_fusion'\n",
                "Path(out_sensor).mkdir(parents=True, exist_ok=True)\n",
                "Path(out_fusion).mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "print(f'=== Starting Experiment 4: Sensor-Only Baseline ({EPOCHS} Epochs) ===')\n",
                "!python scripts/train.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --mode sensor_only \\\n",
                "    --sensor_backbone inception_time \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --min_epochs 50 \\\n",
                "    --batch_size 8 \\\n",
                "    --lr 0.0005 \\\n",
                "    --weight_decay 0.0001 \\\n",
                "    --split_strategy subject \\\n",
                "    --checkpoint_name best_model_sensor.pth \\\n",
                "    --output_dir {out_sensor} \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache \\\n",
                "    --benchmark_warmup\n",
                "\n",
                "print(f'\\n=== Starting Experiment 5: Multimodal Late Fusion ({EPOCHS} Epochs) ===')\n",
                "!python scripts/train.py \\\n",
                "    --data_root data/genuine_mmact \\\n",
                "    --mode multimodal \\\n",
                "    --video_backbone mobilenet_v2 \\\n",
                "    --sensor_backbone inception_time \\\n",
                "    --epochs {EPOCHS} \\\n",
                "    --min_epochs 50 \\\n",
                "    --batch_size 8 \\\n",
                "    --lr 0.0001 \\\n",
                "    --weight_decay 0.0005 \\\n",
                "    --split_strategy subject \\\n",
                "    --checkpoint_name best_model_multimodal.pth \\\n",
                "    --output_dir {out_fusion} \\\n",
                "    --cache_dir /kaggle/working/har_mmact_cache \\\n",
                "    --benchmark_warmup\n"
            ]
        },

        # --- SECTION 12: HELD-OUT TEST EVALUATION ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 12: Standalone Evaluation on Held-Out Test Set (All 5 Models)\n",
                "Evaluates all 5 saved checkpoints on the strictly isolated, held-out test split using Automatic Mixed Precision."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 12: Standalone Evaluation on Held-Out Test Set (All 5 Models)\n",
                "import json\n",
                "\n",
                "models_to_evaluate = [\n",
                "    ('Baseline Custom CNN', 'outputs/baseline_cnn/checkpoints/best_model_cnn.pth', 'outputs/baseline_cnn'),\n",
                "    ('Tuned Custom CNN', 'outputs/tuned_cnn/checkpoints/best_model_tuned_cnn.pth', 'outputs/tuned_cnn'),\n",
                "    ('MobileNetV2 Fine-Tuned', 'outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth', 'outputs/mobilenet_v2'),\n",
                "    ('InceptionTime (Sensor-Only)', 'outputs/sensor_only/checkpoints/best_model_sensor.pth', 'outputs/sensor_only'),\n",
                "    ('Multimodal Fusion (Video+Sensor)', 'outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth', 'outputs/multimodal_fusion'),\n",
                "]\n",
                "\n",
                "eval_results = {}\n",
                "for name, ckpt, out_d in models_to_evaluate:\n",
                "    print(f'\\nEvaluating {name} on held-out test split...')\n",
                "    !python scripts/evaluate.py \\\n",
                "        --checkpoint {ckpt} \\\n",
                "        --data_root data/genuine_mmact \\\n",
                "        --output_file {out_d}/metrics.json \\\n",
                "        --plots_dir {out_d}/plots \\\n",
                "        --cache_dir /kaggle/working/har_mmact_cache\n",
                "    with open(f'{out_d}/metrics.json') as f:\n",
                "        m_data = json.load(f)\n",
                "    mode_key = 'multimodal' if 'multimodal' in m_data else 'video_only'\n",
                "    if 'sensor_only' in out_d:\n",
                "        mode_key = 'sensor_only'\n",
                "    eval_results[name] = m_data[mode_key]\n"
            ]
        },

        # --- SECTION 13: PUBLICATION VISUALIZATIONS ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 13: Comprehensive Publication Visualizations & Galleries\n",
                "Generates cross-model benchmark comparisons, modality ablation bar charts, annotated training trajectories, confusion matrices, ROC curves, and PR curves."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 13: Comprehensive Publication Visualizations & Galleries\n",
                "import json\n",
                "from pathlib import Path\n",
                "from IPython.display import Image, display\n",
                "from src.utils.plots import (\n",
                "    plot_cross_model_comparison,\n",
                "    plot_modality_ablation,\n",
                ")\n",
                "\n",
                "plots_output = Path('outputs/plots')\n",
                "plots_output.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# 1. Cross-Model Comparison Bar Chart\n",
                "plot_cross_model_comparison(eval_results, plots_output / 'cross_model_comparison.png', split_name='Held-Out Test Split')\n",
                "print('Generated outputs/plots/cross_model_comparison.png')\n",
                "\n",
                "# 2. Modality Ablation Bar Chart\n",
                "ablation_summary = {\n",
                "    'Video-Only (MobileNetV2)': eval_results.get('MobileNetV2 Fine-Tuned', {}),\n",
                "    'Sensor-Only (InceptionTime)': eval_results.get('InceptionTime (Sensor-Only)', {}),\n",
                "    'Multimodal Fusion': eval_results.get('Multimodal Fusion (Video+Sensor)', {}),\n",
                "}\n",
                "plot_modality_ablation(ablation_summary, plots_output / 'modality_ablation_comparison.png', split_name='Held-Out Test Split')\n",
                "print('Generated outputs/plots/modality_ablation_comparison.png')\n",
                "\n",
                "# Display publication gallery\n",
                "gallery_plots = [\n",
                "    ('Cross-Model Benchmark Comparison (All 5 Models)', 'outputs/plots/cross_model_comparison.png'),\n",
                "    ('Modality Ablation (Video vs Sensor vs Multimodal Fusion)', 'outputs/plots/modality_ablation_comparison.png'),\n",
                "    ('Hyperparameter Tuning Trials Comparison', 'outputs/plots/hyperparameter_trials.png'),\n",
                "    ('MobileNetV2: Train vs Val Accuracy (Annotated 50 Epochs)', 'outputs/mobilenet_v2/plots/train_vs_val_accuracy_annotated_mobilenet_v2.png'),\n",
                "    ('MobileNetV2: Train vs Val Loss (Annotated 50 Epochs)', 'outputs/mobilenet_v2/plots/train_vs_val_loss_annotated_mobilenet_v2.png'),\n",
                "    ('MobileNetV2: Test Confusion Matrix', 'outputs/mobilenet_v2/plots/test_confusion_matrix_mobilenet_v2.png'),\n",
                "    ('MobileNetV2: Standalone Per-Class F1-Score', 'outputs/mobilenet_v2/plots/test_f1_bars_mobilenet_v2.png'),\n",
                "    ('MobileNetV2: Multiclass OvR ROC Curves', 'outputs/mobilenet_v2/plots/test_roc_ovr_mobilenet_v2.png'),\n",
                "    ('MobileNetV2: Multiclass Precision-Recall Curves', 'outputs/mobilenet_v2/plots/test_pr_curves_mobilenet_v2.png'),\n",
                "    ('Custom CNN: Train vs Val Accuracy (Annotated)', 'outputs/baseline_cnn/plots/train_vs_val_accuracy_annotated_cnn.png'),\n",
                "    ('Multimodal Fusion: Test Confusion Matrix', 'outputs/multimodal_fusion/plots/test_confusion_matrix_mobilenet_v2.png'),\n",
                "]\n",
                "\n",
                "for title, p in gallery_plots:\n",
                "    if Path(p).exists():\n",
                "        print('\\n' + '=' * 75)\n",
                "        print(title)\n",
                "        print('=' * 75)\n",
                "        display(Image(filename=p))\n"
            ]
        },

        # --- SECTION 14: MULTIMODAL INFERENCE DEMO ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 14: Sample Multimodal Inference Demonstration\n",
                "Runs standalone single-sample multimodal inference using the trained fusion model and prints predicted class probabilities."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 14: Sample Multimodal Inference Demonstration\n",
                "sample_video = 'data/genuine_mmact/trimmed/video/cross_scene/trainval/subject16/cam1/scene1/session1/crouching.mp4'\n",
                "sample_phone = 'data/genuine_mmact/trimmed/sensors/acc_phone_clip/subject16/scene1/session1/crouching.csv'\n",
                "sample_watch = 'data/genuine_mmact/trimmed/sensors/acc_watch_clip/subject16/scene1/session1/crouching.csv'\n",
                "sample_gyro  = 'data/genuine_mmact/trimmed/sensors/gyro_clip/subject16/scene1/session1/crouching.csv'\n",
                "\n",
                "print('Running Multimodal Inference Demo on Test Sample...')\n",
                "!python scripts/infer.py \\\n",
                "    --video {sample_video} \\\n",
                "    --sensor_phone {sample_phone} \\\n",
                "    --sensor_watch {sample_watch} \\\n",
                "    --sensor_gyro {sample_gyro} \\\n",
                "    --checkpoint outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth\n"
            ]
        },

        # --- SECTION 15: SUMMARY, MANIFEST & PACKAGING ---
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Section 15: Consolidated Summary Table, Manifest, Runtime Report & 1-Click Packaging\n",
                "Aggregates experiment manifests, runtime reports, and performance metrics into a consolidated summary table and packages `/kaggle/working/har_mmact_results.zip` for 1-click download."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Section 15: Consolidated Summary Table, Manifest, Runtime Report & 1-Click Packaging\n",
                "import shutil\n",
                "import pandas as pd\n",
                "from pathlib import Path\n",
                "\n",
                "print('=' * 80)\n",
                "print(' CONSOLIDATED HAR_MMACT BENCHMARK SUMMARY TABLE')\n",
                "print('=' * 80)\n",
                "rows = []\n",
                "for model_name, m in eval_results.items():\n",
                "    rows.append({\n",
                "        'Model Architecture': model_name,\n",
                "        'Accuracy (%)': round(m.get('accuracy', 0.0) * 100.0, 2),\n",
                "        'Macro F1': round(m.get('macro_f1', 0.0), 4),\n",
                "        'Macro Precision': round(m.get('macro_precision', 0.0), 4),\n",
                "        'Macro Recall': round(m.get('macro_recall', 0.0), 4),\n",
                "    })\n",
                "df_results = pd.DataFrame(rows)\n",
                "print(df_results.to_markdown(index=False))\n",
                "df_results.to_csv('outputs/benchmark_summary.csv', index=False)\n",
                "print('Saved outputs/benchmark_summary.csv')\n",
                "\n",
                "# Organize structured results into /kaggle/working/har_mmact_results\n",
                "results_dir = Path('/kaggle/working/har_mmact_results')\n",
                "results_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "for sub in ['baseline_cnn', 'tuned_cnn', 'mobilenet_v2', 'sensor_only', 'multimodal_fusion', 'tuning', 'plots']:\n",
                "    src_p = Path('outputs') / sub\n",
                "    if src_p.exists():\n",
                "        shutil.copytree(src_p, results_dir / sub, dirs_exist_ok=True)\n",
                "\n",
                "for f in ['benchmark_summary.csv']:\n",
                "    src_f = Path('outputs') / f\n",
                "    if src_f.exists():\n",
                "        shutil.copy2(src_f, results_dir / f)\n",
                "\n",
                "# Package into zip archive\n",
                "archive_path = '/kaggle/working/har_mmact_results'\n",
                "shutil.make_archive(archive_path, 'zip', results_dir)\n",
                "print('\\n' + '=' * 80)\n",
                "print('SUCCESS: All benchmark experiments, checkpoints, and visualizations packaged!')\n",
                "print(f'Download File: {archive_path}.zip')\n",
                "print('Available under Kaggle Output tab in the right-hand panel.')\n",
                "print('=' * 78)\n"
            ]
        }
    ]
    return cells


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle Benchmark Notebook")
    parser.add_argument(
        "--output",
        type=str,
        default="notebooks/kaggle_mmact_har_benchmark_50ep.ipynb",
        help="Target notebook file path (default: notebooks/kaggle_mmact_har_benchmark_50ep.ipynb)",
    )
    parser.add_argument(
        "--update_legacy",
        action="store_true",
        help="Update legacy notebooks/kaggle_mmact_har.ipynb (disabled by default to prevent disturbing active sessions)",
    )
    args = parser.parse_args()

    cells = generate_benchmark_cells()

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    target_path = Path(args.output)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated benchmark notebook at: {target_path}")

    if args.update_legacy:
        legacy_path = Path("notebooks/kaggle_mmact_har.ipynb")
        with open(legacy_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"Updated legacy notebook at: {legacy_path}")


if __name__ == "__main__":
    main()
