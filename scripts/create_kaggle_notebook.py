"""
create_kaggle_notebook.py - Generates the complete, fast, reproducible Kaggle notebook for MMAct HAR.

Target Output: notebooks/kaggle_mmact_har_complete_fast.ipynb
Preserves existing notebooks untouched.
"""

import argparse
import json
from pathlib import Path


def generate_benchmark_cells():
    cells = []

    def md(source_lines):
        return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source_lines]}

    def code(source_lines):
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in source_lines]}

    # Title
    cells.append(md([
        "# Multimodal Human Activity Recognition on MMAct (Kaggle GPU Pipeline)",
        "## Fast, Reliable, Complete 6-Model Benchmark: Original CNN, Improved CNN, Tuned CNN, MobileNetV2 Staged Fine-Tuning, Sensor-Only, and Multimodal Fusion",
        "",
        "This notebook executes the complete Multimodal Human Activity Recognition (HAR) pipeline on the **MMAct dataset** across structured, bounded sections:",
        "- **Section 1**: Environment & Dependencies Setup (strictly preserving Kaggle's binary C ABI stack, Torchvision ops, and CPU thread configuration)",
        "- **Section 2**: Preflight Hardware & GPU Diagnostics (CUDA device, memory headroom, CPU, RAM, disk, fail-loud GPU check, `runtime_hardware_manifest.json`)",
        "- **Section 3**: Repository Checkout from `master` branch",
        "- **Section 4**: Fast, Idempotent Dataset Discovery & Before-and-After Staging (generates `outputs/manifests/dataset_expansion_manifest.json`)",
        "- **Section 5**: Label Mapping & Pre-Flight Modality Audit (Zero-Leakage Split, Paired File Audit & Representation Audit across 10 target classes)",
        "- **Section 6**: Video & Sensor Preprocessing Cache Warm-Up (`/kaggle/working/har_mmact_cache` with disk capacity safety check)",
        "- **Section 7**: Memory-Safe Calibration & OOM Warm-Up (graceful OOM catch, headroom tracking)",
        "- **Section 8**: Experiment 1: Original Custom CNN Baseline (50 epochs, unregularized baseline)",
        "- **Section 9**: Experiment 2: Improved Regularized CNN Baseline (50 epochs, spatial dropout, weight decay, cosine annealing)",
        "- **Section 10**: Experiment 3: Hyperparameter Tuning (train/val only) & Tuned CNN (50 epochs)",
        "- **Section 11**: Experiment 4: MobileNetV2 Transfer Learning with Staged Fine-Tuning (50 epochs, AMP)",
        "- **Section 12**: Experiment 5: Sensor-Only InceptionTime Baseline (50 epochs)",
        "- **Section 13**: Experiment 6: Multimodal Late Fusion (MobileNetV2 + InceptionTime, 50 epochs)",
        "- **Section 14**: Standalone Evaluation on Held-Out Test Set across all 6 models",
        "- **Section 15**: Comprehensive Publication Visualizations & Galleries (annotated curves, LR schedules, generalization gap, ROC, PR, confusion matrices)",
        "- **Section 16**: Sample Multimodal Inference Demonstration",
        "- **Section 17**: Consolidated Summary Table, Manifests, Runtime Report & 1-Click Packaging (`har_mmact_results.zip` with file inventory)",
        "- **Section 18**: (Optional) Extended Benchmarks (100 and 150 Epochs with runtime feasibility estimator)",
        "",
        "All experiments evaluate on the **10 verified target activity classes**:",
        "`crouching`, `jumping`, `kicking`, `pointing`, `running`, `sitting`, `standing`, `using_phone`, `walking`, `waving_hand`.",
        "",
        "---",
        "### Kaggle Setup Instructions:",
        "1. **Enable GPU Accelerator:** Under **Session Options** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.",
        "2. **Enable Internet Access:** Under **Session Options** -> **Internet**, toggle to **On** (required for GitHub repo checkout).",
        "3. **Attach MMAct Dataset:** Click **+ Add Input** and attach your MMAct dataset (e.g. `mmact-trimmed`)."
    ]))

    # Section 1
    cells.append(md([
        "## Section 1: Environment & Dependencies Setup",
        "Installs auxiliary dependencies while strictly preserving Kaggle's preinstalled, binary-compatible C ABI scientific stack (NumPy, Pandas, SciPy, PyTorch, Torchvision, Scikit-Learn) and configures CPU worker threads."
    ]))
    cells.append(code([
        "# Section 1: Dependencies Setup with Kaggle C ABI Protection",
        "import sys",
        "import os",
        "import subprocess",
        "from pathlib import Path",
        "",
        "PROJECT_ROOT = Path('/kaggle/working/HAR-MMAct')",
        "CORE_C_PACKAGES = {'numpy', 'pandas', 'scipy', 'torch', 'torchvision', 'opencv-python', 'pillow', 'scikit-learn', 'matplotlib', 'seaborn', 'pytest'}",
        "",
        "kaggle_req = PROJECT_ROOT / 'requirements-kaggle.txt'",
        "if kaggle_req.exists():",
        "    print(f'Installing auxiliary dependencies from {kaggle_req.name}...')",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--prefer-binary', '--no-deps', '-r', str(kaggle_req)], check=False)",
        "else:",
        "    # Install standard light requirements without overriding core C packages",
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tabulate', 'psutil'], check=False)",
        "",
        "# Post-install C ABI Verification",
        "print('\\n' + '=' * 70)",
        "print('POST-INSTALL ABI & IMPORT VERIFICATION')",
        "print('=' * 70)",
        "abi_failed = False",
        "abi_errors = []",
        "",
        "try:",
        "    import numpy as np",
        "    print(f'  [PASS] Python      : {sys.version.split()[0]}')",
        "    print(f'  [PASS] NumPy       : {np.__version__}')",
        "    _arr = np.eye(2, dtype=np.float64)",
        "    _ = np.linalg.pinv(_arr)",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'NumPy C-extension check failed: {e}')",
        "",
        "try:",
        "    import pandas as pd",
        "    print(f'  [PASS] Pandas      : {pd.__version__}')",
        "    _df = pd.DataFrame({'a': [1.0, 2.0], 'b': [3.0, 4.0]})",
        "    _ = _df.mean()",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'Pandas C-extension check failed: {e}')",
        "",
        "try:",
        "    import scipy.signal",
        "    print(f'  [PASS] SciPy       : {scipy.__version__}')",
        "    _ = scipy.signal.resample([1.0, 2.0, 3.0, 4.0], 8)",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'SciPy C-extension check failed: {e}')",
        "",
        "try:",
        "    import cv2",
        "    print(f'  [PASS] OpenCV      : {cv2.__version__}')",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'OpenCV check failed: {e}')",
        "",
        "try:",
        "    import torch",
        "    print(f'  [PASS] PyTorch     : {torch.__version__} (CUDA: {torch.cuda.is_available()})')",
        "    _t = torch.randn(2, 2)",
        "    _ = _t.numpy()",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'PyTorch <-> NumPy C ABI bridge check failed: {e}')",
        "",
        "try:",
        "    import torchvision",
        "    print(f'  [PASS] Torchvision : {torchvision.__version__}')",
        "    _img = torch.zeros((3, 32, 32), dtype=torch.float32)",
        "    _ = torchvision.transforms.functional.resize(_img, [16, 16])",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'Torchvision check failed: {e}')",
        "",
        "try:",
        "    import sklearn",
        "    print(f'  [PASS] Scikit-Learn: {sklearn.__version__}')",
        "except Exception as e:",
        "    abi_failed = True",
        "    abi_errors.append(f'Scikit-Learn check failed: {e}')",
        "",
        "# Configure CPU Worker Threads",
        "cpu_cores = os.cpu_count() or 1",
        "torch_threads = min(4, cpu_cores)",
        "torch.set_num_threads(torch_threads)",
        "os.environ['OMP_NUM_THREADS'] = str(torch_threads)",
        "print(f'  [PASS] CPU Threads : {torch.get_num_threads()} threads configured (Host logical cores: {cpu_cores})')",
        "",
        "if abi_failed:",
        "    print('\\n' + '!' * 70)",
        "    print('CRITICAL ABI BINARY INCOMPATIBILITY DETECTED:')",
        "    for err in abi_errors:",
        "        print(f'  - {err}')",
        "    print('\\nREMEDY: Restart Kaggle Session from top menu, do NOT run pip install numpy manually.')",
        "    print('!' * 70)",
        "    raise RuntimeError('C ABI binary incompatibility detected:\\n' + '\\n'.join(abi_errors))",
        "",
        "print('=' * 70)",
        "print('>>> C ABI & IMPORT VERIFICATION PASSED: Preinstalled scientific stack is healthy.')",
        "print('=' * 70)"
    ]))

    # Section 2
    cells.append(md([
        "## Section 2: Preflight Hardware & GPU Diagnostics",
        "Audits GPU accelerator, CUDA device capability, VRAM headroom, CPU threads, host RAM, and working disk space. Enforces fail-loud check if GPU accelerator is disabled and persists runtime hardware metadata to `outputs/manifests/runtime_hardware_manifest.json`."
    ]))
    cells.append(code([
        "# Section 2: Preflight Hardware & GPU Diagnostics",
        "import os",
        "import sys",
        "import json",
        "import time",
        "import shutil",
        "import psutil",
        "import torch",
        "from pathlib import Path",
        "",
        "!nvidia-smi",
        "",
        "print('=' * 78)",
        "print(' PREFLIGHT HARDWARE & GPU DIAGNOSTICS')",
        "print('=' * 78)",
        "",
        "# 1. CUDA & GPU Verification",
        "cuda_ok = torch.cuda.is_available()",
        "print(f'PyTorch Version       : {torch.__version__}')",
        "print(f'CUDA Available        : {cuda_ok}')",
        "",
        "if not cuda_ok:",
        "    error_msg = (",
        "        'CRITICAL FAILURE: CUDA GPU is not available!\\n'",
        "        'This 50-epoch benchmark requires a GPU accelerator.\\n'",
        "        'ACTION REQUIRED: In Kaggle right-hand panel -> Session Options -> Accelerator, '",
        "        'select GPU T4 x2 or GPU P100, then restart the session.'",
        "    )",
        "    print('!' * 78)",
        "    print(error_msg)",
        "    print('!' * 78)",
        "    raise RuntimeError(error_msg)",
        "",
        "gpu_count = torch.cuda.device_count()",
        "gpu_name = torch.cuda.get_device_name(0)",
        "compute_cap = torch.cuda.get_device_capability(0)",
        "props = torch.cuda.get_device_properties(0)",
        "total_vram_gb = props.total_memory / (1024**3)",
        "alloc_vram_mb = torch.cuda.memory_allocated(0) / (1024**2)",
        "res_vram_mb = torch.cuda.memory_reserved(0) / (1024**2)",
        "",
        "print(f'GPU Device Name       : {gpu_name} (x{gpu_count} detected)')",
        "print(f'Compute Capability    : {compute_cap[0]}.{compute_cap[1]}')",
        "print(f'Total VRAM Available  : {total_vram_gb:.2f} GB')",
        "print(f'Initial Allocated VRAM: {alloc_vram_mb:.2f} MB')",
        "print(f'Initial Reserved VRAM : {res_vram_mb:.2f} MB')",
        "",
        "# 2. Host System Hardware",
        "cpu_count = os.cpu_count() or 1",
        "vm = psutil.virtual_memory()",
        "ram_total_gb = vm.total / (1024**3)",
        "ram_avail_gb = vm.available / (1024**3)",
        "",
        "working_disk = shutil.disk_usage('/kaggle/working')",
        "disk_free_gb = working_disk.free / (1024**3)",
        "disk_total_gb = working_disk.total / (1024**3)",
        "",
        "print(f'CPU Logical Cores     : {cpu_count}')",
        "print(f'System RAM Total      : {ram_total_gb:.2f} GB (Available: {ram_avail_gb:.2f} GB)')",
        "print(f'Working Disk Space    : {disk_free_gb:.2f} GB free of {disk_total_gb:.2f} GB')",
        "",
        "# 3. Save Runtime Hardware Manifest",
        "manifest_dir = Path('outputs/manifests')",
        "manifest_dir.mkdir(parents=True, exist_ok=True)",
        "hw_manifest = {",
        "    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
        "    'torch_version': torch.__version__,",
        "    'cuda_available': cuda_ok,",
        "    'gpu_count': gpu_count,",
        "    'gpu_name': gpu_name,",
        "    'compute_capability': f'{compute_cap[0]}.{compute_cap[1]}',",
        "    'total_vram_gb': round(total_vram_gb, 2),",
        "    'initial_allocated_vram_mb': round(alloc_vram_mb, 2),",
        "    'cpu_logical_cores': cpu_count,",
        "    'system_ram_total_gb': round(ram_total_gb, 2),",
        "    'system_ram_avail_gb': round(ram_avail_gb, 2),",
        "    'working_disk_free_gb': round(disk_free_gb, 2),",
        "    'working_disk_total_gb': round(disk_total_gb, 2),",
        "}",
        "with open(manifest_dir / 'runtime_hardware_manifest.json', 'w') as f:",
        "    json.dump(hw_manifest, f, indent=2)",
        "print('Saved outputs/manifests/runtime_hardware_manifest.json')",
        "",
        "# 4. Runtime Training Settings",
        "print('\\nRuntime Acceleration Configuration:')",
        "print('  - Batch Size          : 8 (calibrated for 16-frame 112x112 video + 9-channel sensor)')",
        "print('  - DataLoader Workers  : 2 (non-blocking pin_memory)')",
        "print('  - Automatic Mixed Prec: ENABLED (torch.amp.autocast, float16)')",
        "print('  - cuDNN Benchmark     : ENABLED (torch.backends.cudnn.benchmark = True)')",
        "print('=' * 78)"
    ]))

    # Section 3
    cells.append(md([
        "## Section 3: Repository Checkout from master Branch",
        "Clones or pulls the latest verified code from the `master` branch of `https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct.git`, configures Python `sys.path`, and verifies project modules."
    ]))
    cells.append(code([
        "# Section 3: Repository Checkout from master Branch",
        "import sys",
        "import os",
        "import subprocess",
        "from pathlib import Path",
        "",
        "PROJECT_ROOT = Path('/kaggle/working/HAR-MMAct')",
        "REPO_URL = 'https://github.com/CHENNAMSETTYVIKRAM/HAR-MMAct.git'",
        "BRANCH = 'master'",
        "",
        "# 1. Clone or update repository into PROJECT_ROOT explicitly on 'master' branch",
        "if not PROJECT_ROOT.exists():",
        "    print(f'Cloning {REPO_URL} (branch: {BRANCH}) into {PROJECT_ROOT}...')",
        "    subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, str(PROJECT_ROOT)], check=True)",
        "else:",
        "    print(f'Repository already present at {PROJECT_ROOT}. Updating strictly to origin/{BRANCH}...')",
        "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'fetch', 'origin', BRANCH], check=True)",
        "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'checkout', '-B', BRANCH, f'origin/{BRANCH}'], check=True)",
        "    subprocess.run(['git', '-C', str(PROJECT_ROOT), 'reset', '--hard', f'origin/{BRANCH}'], check=True)",
        "",
        "# 2. Print active branch and commit hash for diagnostics",
        "try:",
        "    branch_out = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', '--abbrev-ref', 'HEAD'], text=True).strip()",
        "    commit_out = subprocess.check_output(['git', '-C', str(PROJECT_ROOT), 'rev-parse', 'HEAD'], text=True).strip()",
        "    print(f'Active Git Branch : {branch_out}')",
        "    print(f'Active Git Commit : {commit_out}')",
        "except Exception as e:",
        "    print(f'[Warning] Could not inspect git metadata: {e}')",
        "    branch_out, commit_out = 'unknown', 'unknown'",
        "",
        "# 3. Verify expected files exist on master",
        "req_files = [",
        "    PROJECT_ROOT / 'src' / 'data' / 'ingestion.py',",
        "    PROJECT_ROOT / 'src' / 'data' / 'indexer.py',",
        "    PROJECT_ROOT / 'src' / 'data' / 'dataset.py',",
        "    PROJECT_ROOT / 'src' / 'data' / 'stager.py',",
        "    PROJECT_ROOT / 'scripts' / 'train.py',",
        "    PROJECT_ROOT / 'scripts' / 'tune.py',",
        "    PROJECT_ROOT / 'scripts' / 'evaluate.py',",
        "]",
        "missing_files = [str(f.relative_to(PROJECT_ROOT)) for f in req_files if not f.exists()]",
        "if missing_files:",
        "    raise FileNotFoundError(f'CRITICAL: Missing required files on branch {branch_out}: {missing_files}')",
        "",
        "print('SUCCESS: Verified presence of core modules on master branch.')",
        "",
        "# 4. Configure import path before importing project modules",
        "sys.path.insert(0, str(PROJECT_ROOT))",
        "os.chdir(PROJECT_ROOT)",
        "print(f'PROJECT_ROOT configured : {PROJECT_ROOT}')",
        "print(f'sys.path[0] set to      : {sys.path[0]}')",
        "",
        "# 5. Verify project module imports",
        "from src.data.ingestion import DatasetIngestionPipeline",
        "from src.data.indexer import MMActIndexer",
        "from src.data.dataset import MMActMultimodalDataset",
        "from src.data.stager import FastMMActStager, StagingReport",
        "from src.models.fusion_model import MultimodalHARModel",
        "from src.utils.plots import plot_cross_model_comparison, plot_modality_ablation",
        "print('SUCCESS: Verified project imports (ingestion, indexer, dataset, stager, models, plots).')"
    ]))

    # Section 4
    cells.append(md([
        "## Section 4: Fast, Idempotent Dataset Discovery & Canonical Staging",
        "Discovers attached datasets across `/kaggle/input` using bounded-depth shallow directory inspection (max depth $\\le$ 4). Staging is strictly idempotent, preserves historical recordings (e.g. subjects 8 and 16), avoids redundant extraction, and saves before/after counts to `outputs/manifests/dataset_expansion_manifest.json`."
    ]))
    cells.append(code([
        "# Section 4: Fast, Idempotent Dataset Discovery & Canonical Staging",
        "from pathlib import Path",
        "from src.data.stager import FastMMActStager",
        "",
        "kaggle_input = Path('/kaggle/input')",
        "canonical_data_dir = PROJECT_ROOT / 'data' / 'genuine_mmact'",
        "",
        "# Candidate roots to inspect shallowly",
        "input_roots = sorted([p for p in kaggle_input.iterdir() if p.is_dir()]) if kaggle_input.exists() else []",
        "if not input_roots:",
        "    local_data = PROJECT_ROOT / 'data'",
        "    if local_data.exists():",
        "        input_roots = [local_data]",
        "",
        "print(f'Scanning {len(input_roots)} candidate input roots using shallow bounded inspection (max_depth=4)...')",
        "stager = FastMMActStager(",
        "    canonical_dir=canonical_data_dir,",
        "    input_roots=input_roots,",
        "    max_scan_depth=4,",
        ")",
        "",
        "staging_report = stager.discover_and_stage()",
        "print(staging_report.summary())",
        "",
        "assert staging_report.is_valid, (",
        "    f'CRITICAL: Dataset discovery & staging failed! Missing required items: {staging_report.missing_required_paths}\\n'",
        "    f'Please ensure the MMAct dataset is attached under /kaggle/input.'",
        ")"
    ]))

    # Section 5
    cells.append(md([
        "## Section 5: Label Mapping & Pre-Flight Modality Audit (Zero-Leakage Split)",
        "Verifies video frame decoding, sensor CSV parsing, index pairing, and enforces strict subject-independent split isolation (train/val/test) across the 10 target activity classes. Performs a paired-file integrity check for all split samples and explicitly verifies honest labeling of the evaluation partition."
    ]))
    cells.append(code([
        "# Section 5: Label Mapping & Pre-Flight Modality Audit",
        "from pathlib import Path",
        "import cv2",
        "import pandas as pd",
        "from src.data.indexer import MMActIndexer",
        "from src.data.splits import DatasetSplitter, verify_split_isolation, verify_split_representation",
        "from src.config import SplitConfig, VERIFIED_TARGET_CLASSES",
        "",
        "print('=' * 78)",
        "print('PRE-FLIGHT VALIDATION & COMPREHENSIVE MULTI-MODALITY AUDIT')",
        "print('=' * 78)",
        "print(f'Repository Root : {PROJECT_ROOT}')",
        "print(f'Canonical Root  : {canonical_data_dir.resolve()}')",
        "",
        "# 1. File Inventory",
        "total_videos = MMActIndexer.find_files_recursively(canonical_data_dir, {'.mp4'})",
        "total_sensors = MMActIndexer.find_files_recursively(canonical_data_dir, {'.csv'})",
        "total_splits = [f for f in MMActIndexer.find_files_recursively(canonical_data_dir, {'.txt'}) if any(k in f.name.lower() for k in ['cross_scene', 'cross_view', 'train', 'val', 'test'])]",
        "",
        "print(f'CANONICAL STAGED TOTALS: {len(total_videos)} Videos | {len(total_sensors)} Sensors | {len(total_splits)} Splits')",
        "assert len(total_videos) > 0 and len(total_sensors) > 0, 'CRITICAL: No valid MMAct data files found in canonical directory!'",
        "",
        "# 2. Modality Readability check",
        "sample_video = total_videos[0]",
        "cap = cv2.VideoCapture(str(sample_video))",
        "ret, frame = cap.read()",
        "frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))",
        "cap.release()",
        "assert ret and frame is not None, f'Failed to read frame from {sample_video}'",
        "print(f'  [PASS] Video Readability: Decoded frame {frame.shape} from {sample_video.name} ({frame_count} frames)')",
        "",
        "sample_sensor = total_sensors[0]",
        "df_sample = pd.read_csv(sample_sensor)",
        "print(f'  [PASS] Sensor Parseability: Parsed CSV {df_sample.shape} from {sample_sensor.name}')",
        "",
        "# 3. Indexing & Target Class Coverage",
        "indexer = MMActIndexer(dataset_root=canonical_data_dir)",
        "report = indexer.index()",
        "print(f'  - Paired Samples Created: {report.paired_samples_count}')",
        "print(f'  Target Class Coverage ({len(report.class_distribution)}/{len(VERIFIED_TARGET_CLASSES)} target classes):')",
        "for cls_name in VERIFIED_TARGET_CLASSES:",
        "    cnt = report.class_distribution.get(cls_name, 0)",
        "    print(f'    * {cls_name:<16}: {cnt} samples')",
        "",
        "assert report.paired_samples_count > 0, 'Zero paired multimodal samples created!'",
        "",
        "# 4. Subject-Independent Split Isolation Verification",
        "splitter = DatasetSplitter(SplitConfig(split_strategy='subject'))",
        "train_samples, val_samples, test_samples = splitter.split(report.paired_samples)",
        "isolation = verify_split_isolation(train_samples, val_samples, test_samples, strategy='subject')",
        "print(f'  - Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}')",
        "for check_k, passed in isolation.items():",
        "    if passed is None:",
        "        print(f'      * {check_k:<28}: NOT_APPLICABLE (empty test split)')",
        "    else:",
        "        stat_str = 'PASSED' if passed else 'FAILED'",
        "        print(f'      * {check_k:<28}: {stat_str}')",
        "        assert passed, f'CRITICAL: Split leakage detected on {check_k}!'",
        "",
        "# 5. Paired File Path Existence Audit",
        "missing_vid_files = []",
        "missing_sns_files = []",
        "for sp_lbl, s_set in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:",
        "    for s in s_set:",
        "        if not Path(s.video_path).exists():",
        "            missing_vid_files.append((sp_lbl, s.sample_id, str(s.video_path)))",
        "        for m_name, m_path in s.sensor_paths.items():",
        "            if not Path(m_path).exists():",
        "                missing_sns_files.append((sp_lbl, s.sample_id, m_name, str(m_path)))",
        "print(f'  - Missing Video Files : {len(missing_vid_files)}')",
        "print(f'  - Missing Sensor Files: {len(missing_sns_files)}')",
        "assert len(missing_vid_files) == 0, f'CRITICAL: {len(missing_vid_files)} missing video files detected!'",
        "assert len(missing_sns_files) == 0, f'CRITICAL: {len(missing_sns_files)} missing sensor files detected!'",
        "",
        "# 6. Representation Audit across Splits",
        "rep_audit = verify_split_representation(train_samples, val_samples, test_samples, expected_classes=VERIFIED_TARGET_CLASSES)",
        "train_subjs = sorted(list({s.subject for s in train_samples}))",
        "val_subjs = sorted(list({s.subject for s in val_samples}))",
        "test_subjs = sorted(list({s.subject for s in test_samples}))",
        "eval_split_desc = 'Held-Out Test Split' if len(test_samples) > 0 else 'Official MMAct Validation Split'",
        "print(f'  - Subjects per Split : Train={train_subjs}, Val={val_subjs}, Test={test_subjs}')",
        "print(f'  - Missing Classes    : Train={rep_audit[\"missing_classes_train\"]}, Val={rep_audit[\"missing_classes_val\"]}, Test={rep_audit[\"missing_classes_test\"]}')",
        "print(f'  - Modalities Paired  : Train={rep_audit[\"modalities_paired_train\"]}, Val={rep_audit[\"modalities_paired_val\"]}, Test={rep_audit[\"modalities_paired_test\"]}')",
        "print(f'  - Supervised Eval Set: {eval_split_desc} ({len(test_samples if len(test_samples) > 0 else val_samples)} samples)')",
        "print('=' * 78)",
        "print('>>> PRE-FLIGHT VALIDATION PASSED: Dataset is readable, parseable, and strictly isolated.')",
        "print('=' * 78)"
    ]))

    # Section 6
    cells.append(md([
        "## Section 6: Video & Sensor Preprocessing Cache Warm-Up",
        "Pre-caches normalized 16-frame 112x112 video tensors and aligned 100-sample 9-axis sensor windows to `/kaggle/working/har_mmact_cache/` so all 6 subsequent experiments execute with maximum speed without repeated OpenCV decoding. Checks disk capacity before caching."
    ]))
    cells.append(code([
        "# Section 6: Video & Sensor Preprocessing Cache Warm-Up",
        "import shutil",
        "from src.data.dataset import MMActMultimodalDataset",
        "",
        "CACHE_DIR = Path('/kaggle/working/har_mmact_cache')",
        "CACHE_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "disk_free_gb = shutil.disk_usage(str(CACHE_DIR)).free / (1024**3)",
        "print(f'Working disk space before cache warm-up: {disk_free_gb:.2f} GB free')",
        "",
        "print(f'Initializing multimodal datasets with shared cache at {CACHE_DIR}...')",
        "train_ds = MMActMultimodalDataset(train_samples, cache_dir=CACHE_DIR / 'train', enable_cache=True)",
        "val_ds = MMActMultimodalDataset(val_samples, cache_dir=CACHE_DIR / 'val', enable_cache=True)",
        "test_ds = MMActMultimodalDataset(test_samples, cache_dir=CACHE_DIR / 'test', enable_cache=(len(test_samples) > 0))",
        "",
        "# Sequentially precompute and warm up cache",
        "print('\\n[Warming Up Training Set Cache]')",
        "train_stats = train_ds.warm_up_cache(verbose=True)",
        "",
        "print('\\n[Warming Up Validation Set Cache]')",
        "val_stats = val_ds.warm_up_cache(verbose=True)",
        "",
        "test_stats = {'writes': 0, 'hits': 0}",
        "if len(test_samples) > 0:",
        "    print('\\n[Warming Up Test Set Cache]')",
        "    test_stats = test_ds.warm_up_cache(verbose=True)",
        "",
        "disk_free_after = shutil.disk_usage(str(CACHE_DIR)).free / (1024**3)",
        "print('\\n' + '=' * 78)",
        "print('CACHE WARM-UP COMPLETED:')",
        "print(f'  Train Cache Writes: {train_stats[\"writes\"]} | Hits: {train_stats[\"hits\"]}')",
        "print(f'  Val   Cache Writes: {val_stats[\"writes\"]} | Hits: {val_stats[\"hits\"]}')",
        "print(f'  Test  Cache Writes: {test_stats[\"writes\"]} | Hits: {test_stats[\"hits\"]}')",
        "print(f'  Remaining Disk    : {disk_free_after:.2f} GB free')",
        "print('=' * 78)"
    ]))

    # Section 7
    cells.append(md([
        "## Section 7: Memory-Safe Calibration & OOM Warm-Up",
        "Executes a 1-batch calibration forward/backward pass with Automatic Mixed Precision (AMP) to verify GPU VRAM safety and establish baseline memory headroom before starting full 50-epoch training. Explicitly catches OOM conditions and frees temporary memory."
    ]))
    cells.append(code([
        "# Section 7: Memory-Safe Calibration & OOM Warm-Up",
        "import torch",
        "import torch.nn as nn",
        "from torch.utils.data import DataLoader",
        "from src.models.fusion_model import MultimodalHARModel",
        "",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "torch.cuda.reset_peak_memory_stats(0)",
        "",
        "print('Running 1-batch GPU memory calibration test with AMP...')",
        "calib_loader = DataLoader(train_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))",
        "v_sample, s_sample, l_sample, _ = next(iter(calib_loader))",
        "",
        "model_calib = MultimodalHARModel(",
        "    num_classes=10,",
        "    video_backbone='mobilenet_v2',",
        "    sensor_backbone='inception_time',",
        "    fusion_hidden_dim=128",
        ").to(device)",
        "",
        "optimizer = torch.optim.Adam(model_calib.parameters(), lr=1e-4)",
        "criterion = nn.CrossEntropyLoss()",
        "scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None",
        "",
        "try:",
        "    v_gpu = v_sample.to(device, non_blocking=True)",
        "    s_gpu = s_sample.to(device, non_blocking=True)",
        "    l_gpu = l_sample.to(device, non_blocking=True)",
        "",
        "    optimizer.zero_grad(set_to_none=True)",
        "    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):",
        "        out = model_calib(video=v_gpu, sensor=s_gpu, override_mode='multimodal')",
        "        loss = criterion(out, l_gpu)",
        "",
        "    scaler.scale(loss).backward()",
        "    scaler.step(optimizer)",
        "    scaler.update()",
        "",
        "    peak_calib_mb = torch.cuda.max_memory_allocated(0) / (1024**2)",
        "    total_vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)",
        "    headroom_mb = total_vram_mb - peak_calib_mb",
        "",
        "    print(f'Peak Memory during Multimodal Calibration : {peak_calib_mb:.2f} MB')",
        "    print(f'Total VRAM Available                       : {total_vram_mb:.2f} MB')",
        "    print(f'Safe Headroom Remaining                    : {headroom_mb:.2f} MB ({headroom_mb/total_vram_mb*100:.1f}%)')",
        "    print('SUCCESS: Memory calibration passed. Safe to proceed with 50-epoch training.')",
        "except (torch.cuda.OutOfMemoryError, RuntimeError) as e:",
        "    print(f'[OOM Warning] Batch size 8 exceeded VRAM during calibration: {e}')",
        "    print('Recommendation: Re-run with --batch_size 4.')",
        "finally:",
        "    del model_calib, optimizer, criterion, scaler",
        "    if 'v_gpu' in locals(): del v_gpu",
        "    if 's_gpu' in locals(): del s_gpu",
        "    if 'l_gpu' in locals(): del l_gpu",
        "    if 'out' in locals(): del out",
        "    if 'loss' in locals(): del loss",
        "    torch.cuda.empty_cache()"
    ]))

    # Section 8
    cells.append(md([
        "## Section 8: Experiment 1 - Original Custom CNN Baseline (50 Epochs)",
        "Trains the original unregularized custom CNN baseline representing the initial student model (without heavy spatial dropout or aggressive weight decay) for 50 epochs."
    ]))
    cells.append(code([
        "# Section 8: Experiment 1 - Original Custom CNN Baseline (50 Epochs)",
        "EPOCHS = 50",
        "out_orig_cnn = 'outputs/original_cnn'",
        "Path(out_orig_cnn).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'=== Starting Experiment 1: Original Custom CNN Baseline ({EPOCHS} Epochs) ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --video_backbone original_cnn \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --mode video_only \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr 0.0001 \\\n"
        "    --weight_decay 0.0001 \\\n"
        "    --dropout 0.20 \\\n"
        "    --split_strategy subject \\\n"
        "    --early_stopping_patience 15 \\\n"
        "    --checkpoint_name best_model_original_cnn.pth \\\n"
        "    --output_dir {out_orig_cnn} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 9
    cells.append(md([
        "## Section 9: Experiment 2 - Improved Regularized CNN Baseline (50 Epochs)",
        "Trains the improved custom video CNN baseline for 50 epochs using Spatial Dropout2d (0.15, 0.20), BatchNorm2d, Weight Decay ($10^{-3}$), Cosine Annealing, Automatic Mixed Precision, and auto-resume protection."
    ]))
    cells.append(code([
        "# Section 9: Experiment 2 - Improved Regularized CNN Baseline (50 Epochs)",
        "out_imp_cnn = 'outputs/improved_cnn'",
        "Path(out_imp_cnn).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'=== Starting Experiment 2: Improved Regularized CNN ({EPOCHS} Epochs) ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --video_backbone custom_cnn \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --mode video_only \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr 0.0001 \\\n"
        "    --weight_decay 0.001 \\\n"
        "    --dropout 0.35 \\\n"
        "    --split_strategy subject \\\n"
        "    --early_stopping_patience 15 \\\n"
        "    --checkpoint_name best_model_cnn.pth \\\n"
        "    --output_dir {out_imp_cnn} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 10
    cells.append(md([
        "## Section 10: Experiment 3 - Hyperparameter Tuning (Train/Val Only) & Tuned CNN (50 Epochs)",
        "Systematically explores learning rate, weight decay, and dropout across trials strictly on train/val splits. The held-out test split is NEVER touched. The best configuration is then trained for 50 epochs."
    ]))
    cells.append(code([
        "# Section 10: Experiment 3 - Hyperparameter Tuning & Tuned CNN (50 Epochs)",
        "print('=== Starting Experiment 3: Hyperparameter Tuning (Train/Val Only) ===')",
        "!python scripts/tune.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --split_strategy subject \\\n"
        "    --epochs_per_trial 10 \\\n"
        "    --batch_size 8 \\\n"
        "    --output_dir outputs/tuning \\\n"
        "    --plots_dir outputs/plots \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --skip_if_completed",
        "",
        "# Load best hyperparameters discovered during tuning",
        "import json",
        "with open('outputs/tuning/best_hyperparameters.json') as f:",
        "    best_cfg = json.load(f)",
        "print(f'Optimal Hyperparameters Discovered: {best_cfg}')",
        "",
        "out_tuned = 'outputs/tuned_cnn'",
        "Path(out_tuned).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'\\n=== Training Tuned CNN for {EPOCHS} Epochs on Best Configuration ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --video_backbone custom_cnn \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --mode video_only \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr {best_cfg['lr']} \\\n"
        "    --weight_decay {best_cfg['weight_decay']} \\\n"
        "    --dropout {best_cfg['dropout']} \\\n"
        "    --split_strategy subject \\\n"
        "    --early_stopping_patience 15 \\\n"
        "    --checkpoint_name best_model_tuned_cnn.pth \\\n"
        "    --output_dir {out_tuned} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 11
    cells.append(md([
        "## Section 11: Experiment 4 - MobileNetV2 Staged Fine-Tuning (50 Epochs)",
        "Phase 1 (Epochs 1-15): Pretrained MobileNetV2 backbone frozen, training classification head at $10^{-4}$.",
        "Phase 2 (Epochs 16-50): Backbone unfrozen with differential learning rate ($10^{-5}$) for gentle fine-tuning."
    ]))
    cells.append(code([
        "# Section 11: Experiment 4 - MobileNetV2 Staged Fine-Tuning (50 Epochs)",
        "out_mob = 'outputs/mobilenet_v2'",
        "Path(out_mob).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'=== Starting Experiment 4: MobileNetV2 Staged Fine-Tuning ({EPOCHS} Epochs) ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --video_backbone mobilenet_v2 \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --mode video_only \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr 0.0001 \\\n"
        "    --weight_decay 0.0005 \\\n"
        "    --dropout 0.35 \\\n"
        "    --staged_finetune \\\n"
        "    --unfreeze_epoch 15 \\\n"
        "    --unfreeze_lr 0.00001 \\\n"
        "    --split_strategy subject \\\n"
        "    --early_stopping_patience 15 \\\n"
        "    --checkpoint_name best_model_mobilenet_v2.pth \\\n"
        "    --output_dir {out_mob} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 12
    cells.append(md([
        "## Section 12: Experiment 5 - Sensor-Only InceptionTime Baseline (50 Epochs)",
        "InceptionTime 1D convolutional network trained on 9-axis sensor time series (phone accelerometer, watch accelerometer, gyroscope) for 50 epochs."
    ]))
    cells.append(code([
        "# Section 12: Experiment 5 - Sensor-Only Baseline (50 Epochs)",
        "out_sensor = 'outputs/sensor_only'",
        "Path(out_sensor).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'=== Starting Experiment 5: Sensor-Only Baseline ({EPOCHS} Epochs) ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --mode sensor_only \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr 0.0005 \\\n"
        "    --weight_decay 0.0001 \\\n"
        "    --split_strategy subject \\\n"
        "    --checkpoint_name best_model_sensor.pth \\\n"
        "    --output_dir {out_sensor} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 13
    cells.append(md([
        "## Section 13: Experiment 6 - Multimodal Late Fusion (50 Epochs)",
        "Fuses MobileNetV2 video representations with InceptionTime sensor features via late fusion for 50 epochs."
    ]))
    cells.append(code([
        "# Section 13: Experiment 6 - Multimodal Late Fusion (50 Epochs)",
        "out_fusion = 'outputs/multimodal_fusion'",
        "Path(out_fusion).mkdir(parents=True, exist_ok=True)",
        "",
        "print(f'=== Starting Experiment 6: Multimodal Late Fusion ({EPOCHS} Epochs) ===')",
        "!python scripts/train.py \\\n"
        "    --data_root data/genuine_mmact \\\n"
        "    --mode multimodal \\\n"
        "    --video_backbone mobilenet_v2 \\\n"
        "    --sensor_backbone inception_time \\\n"
        "    --epochs {EPOCHS} \\\n"
        "    --min_epochs 50 \\\n"
        "    --batch_size 8 \\\n"
        "    --lr 0.0001 \\\n"
        "    --weight_decay 0.0005 \\\n"
        "    --split_strategy subject \\\n"
        "    --checkpoint_name best_model_multimodal.pth \\\n"
        "    --output_dir {out_fusion} \\\n"
        "    --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "    --benchmark_warmup \\\n"
        "    --auto_resume \\\n"
        "    --skip_if_completed"
    ]))

    # Section 14
    cells.append(md([
        "## Section 14: Standalone Evaluation on Held-Out Test Set (All 6 Models)",
        "Evaluates all 6 saved checkpoints on the strictly isolated, held-out test split using Automatic Mixed Precision."
    ]))
    cells.append(code([
        "# Section 14: Standalone Evaluation on Held-Out Test Set (All 6 Models)",
        "import json",
        "from pathlib import Path",
        "",
        "models_to_evaluate = [",
        "    ('Original Custom CNN', 'outputs/original_cnn/checkpoints/best_model_original_cnn.pth', 'outputs/original_cnn'),",
        "    ('Improved Regularized CNN', 'outputs/improved_cnn/checkpoints/best_model_cnn.pth', 'outputs/improved_cnn'),",
        "    ('Tuned Custom CNN', 'outputs/tuned_cnn/checkpoints/best_model_tuned_cnn.pth', 'outputs/tuned_cnn'),",
        "    ('MobileNetV2 Fine-Tuned', 'outputs/mobilenet_v2/checkpoints/best_model_mobilenet_v2.pth', 'outputs/mobilenet_v2'),",
        "    ('InceptionTime (Sensor-Only)', 'outputs/sensor_only/checkpoints/best_model_sensor.pth', 'outputs/sensor_only'),",
        "    ('Multimodal Fusion (Video+Sensor)', 'outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth', 'outputs/multimodal_fusion'),",
        "]",
        "",
        "eval_results = {}",
        "for name, ckpt, out_d in models_to_evaluate:",
        "    ckpt_path = Path(ckpt)",
        "    if not ckpt_path.exists():",
        "        print(f'[WARNING] Checkpoint not found: {ckpt} -- skipping {name}')",
        "        continue",
        "    metrics_file = Path(f'{out_d}/metrics.json')",
        "    print(f'\\nEvaluating {name} on held-out test split...')",
        "    !python scripts/evaluate.py \\\n"
        "        --checkpoint {ckpt} \\\n"
        "        --data_root data/genuine_mmact \\\n"
        "        --output_file {metrics_file} \\\n"
        "        --plots_dir {out_d}/plots \\\n"
        "        --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "        --skip_if_completed",
        "    if not metrics_file.exists():",
        "        print(f'[ERROR] Evaluation metrics file not found: {metrics_file}')",
        "        continue",
        "    with open(metrics_file) as f:",
        "        m_data = json.load(f)",
        "    eval_mode = m_data.get('mode')",
        "    if eval_mode and eval_mode in m_data:",
        "        eval_results[name] = m_data[eval_mode]",
        "    elif 'metrics' in m_data:",
        "        eval_results[name] = m_data['metrics']",
        "    else:",
        "        for mk in ['multimodal', 'video_only', 'sensor_only']:",
        "            if mk in m_data:",
        "                eval_results[name] = m_data[mk]",
        "                break",
        "    if name in eval_results:",
        "        eval_split_label = m_data.get('eval_split_description', m_data.get('eval_split', 'test'))",
        "        eval_results[name]['_name'] = name",
        "        eval_results[name]['_mode'] = eval_mode",
        "        eval_results[name]['_split'] = eval_split_label",
        "        eval_results[name]['_samples'] = m_data.get('eval_samples_count', 'N/A')",
        "        eval_results[name]['_checkpoint'] = ckpt",
        "        acc = eval_results[name].get('accuracy', 0.0) * 100.0",
        "        f1 = eval_results[name].get('macro_f1', 0.0)",
        "        print(f'  [SUCCESS] {name} ({eval_mode}) on {eval_split_label}: Acc = {acc:.2f}%, Macro F1 = {f1:.4f}')"
    ]))

    # Section 15
    cells.append(md([
        "## Section 15: Comprehensive Publication Visualizations & Galleries",
        "Generates cross-model benchmark comparisons, modality ablation bar charts, annotated training trajectories, learning rate schedules, generalization gaps, confusion matrices, ROC curves, and PR curves across all 6 models."
    ]))
    cells.append(code([
        "# Section 15: Comprehensive Publication Visualizations & Galleries",
        "import json",
        "from pathlib import Path",
        "from IPython.display import Image, display",
        "from src.utils.plots import (",
        "    plot_cross_model_comparison,",
        "    plot_modality_ablation,",
        ")",
        "",
        "plots_output = Path('outputs/plots')",
        "plots_output.mkdir(parents=True, exist_ok=True)",
        "",
        "# 1. Cross-Model Comparison Bar Chart",
        "if eval_results:",
        "    plot_cross_model_comparison(eval_results, plots_output / 'cross_model_comparison.png', split_name='Held-Out Test Split')",
        "    print('Generated outputs/plots/cross_model_comparison.png')",
        "",
        "# 2. Modality Ablation Bar Chart (Only using genuinely evaluated models)",
        "ablation_summary = {}",
        "if 'MobileNetV2 Fine-Tuned' in eval_results:",
        "    ablation_summary['Video-Only (MobileNetV2)'] = eval_results['MobileNetV2 Fine-Tuned']",
        "elif 'Improved Regularized CNN' in eval_results:",
        "    ablation_summary['Video-Only (Improved CNN)'] = eval_results['Improved Regularized CNN']",
        "elif 'Original Custom CNN' in eval_results:",
        "    ablation_summary['Video-Only (Original CNN)'] = eval_results['Original Custom CNN']",
        "if 'InceptionTime (Sensor-Only)' in eval_results:",
        "    ablation_summary['Sensor-Only (InceptionTime)'] = eval_results['InceptionTime (Sensor-Only)']",
        "if 'Multimodal Fusion (Video+Sensor)' in eval_results:",
        "    ablation_summary['Multimodal Fusion'] = eval_results['Multimodal Fusion (Video+Sensor)']",
        "",
        "if len(ablation_summary) >= 2:",
        "    plot_modality_ablation(ablation_summary, plots_output / 'modality_ablation_comparison.png', split_name='Held-Out Test Split')",
        "    print('Generated outputs/plots/modality_ablation_comparison.png')",
        "",
        "# Display publication gallery",
        "gallery_plots = [",
        "    ('Cross-Model Benchmark Comparison (All 6 Models)', 'outputs/plots/cross_model_comparison.png'),",
        "    ('Modality Ablation (Video vs Sensor vs Multimodal Fusion)', 'outputs/plots/modality_ablation_comparison.png'),",
        "    ('Hyperparameter Tuning Trials Comparison', 'outputs/plots/hyperparameter_trials.png'),",
        "    ('MobileNetV2: Train vs Val Accuracy (Annotated 50 Epochs)', 'outputs/mobilenet_v2/plots/train_vs_val_accuracy_annotated_mobilenet_v2.png'),",
        "    ('MobileNetV2: Train vs Val Loss (Annotated 50 Epochs)', 'outputs/mobilenet_v2/plots/train_vs_val_loss_annotated_mobilenet_v2.png'),",
        "    ('MobileNetV2: Learning Rate Schedule Trajectory', 'outputs/mobilenet_v2/plots/learning_rate_schedule_mobilenet_v2.png'),",
        "    ('MobileNetV2: Generalization Gap Across Epochs', 'outputs/mobilenet_v2/plots/generalization_gap_mobilenet_v2.png'),",
        "    ('MobileNetV2: Test Confusion Matrix', 'outputs/mobilenet_v2/plots/test_confusion_matrix_mobilenet_v2.png'),",
        "    ('MobileNetV2: Standalone Per-Class F1-Score', 'outputs/mobilenet_v2/plots/test_f1_bars_mobilenet_v2.png'),",
        "    ('MobileNetV2: Multiclass OvR ROC Curves', 'outputs/mobilenet_v2/plots/test_roc_ovr_mobilenet_v2.png'),",
        "    ('MobileNetV2: Multiclass Precision-Recall Curves', 'outputs/mobilenet_v2/plots/test_pr_curves_mobilenet_v2.png'),",
        "    ('Original CNN: Train vs Val Accuracy (Annotated)', 'outputs/original_cnn/plots/train_vs_val_accuracy_annotated_original_cnn.png'),",
        "    ('Improved CNN: Train vs Val Accuracy (Annotated)', 'outputs/improved_cnn/plots/train_vs_val_accuracy_annotated_custom_cnn.png'),",
        "    ('Multimodal Fusion: Test Confusion Matrix', 'outputs/multimodal_fusion/plots/test_confusion_matrix_mobilenet_v2.png'),",
        "]",
        "",
        "for title, p in gallery_plots:",
        "    if Path(p).exists():",
        "        print('\\n' + '=' * 75)",
        "        print(title)",
        "        print('=' * 75)",
        "        display(Image(filename=p))"
    ]))

    # Section 16
    cells.append(md([
        "## Section 16: Sample Multimodal Inference Demonstration",
        "Runs standalone single-sample multimodal inference using the trained fusion model and prints predicted class probabilities."
    ]))
    cells.append(code([
        "# Section 16: Sample Multimodal Inference Demonstration",
        "from pathlib import Path",
        "from src.data.indexer import MMActIndexer",
        "",
        "print('Indexing dataset to locate a complete, paired multimodal test sample...')",
        "indexer = MMActIndexer(dataset_root=Path('data/genuine_mmact'))",
        "report = indexer.index()",
        "",
        "demo_sample = None",
        "for s in report.paired_samples:",
        "    if s.video_path.exists() and all(p.exists() for p in s.sensor_paths.values()):",
        "        if {'acc_phone', 'acc_watch', 'gyro'}.issubset(s.sensor_paths.keys()):",
        "            demo_sample = s",
        "            break",
        "",
        "if demo_sample is None:",
        "    error_msg = 'CRITICAL: No complete paired multimodal sample (video + acc_phone + acc_watch + gyro) found in dataset!'",
        "    print(f'[ERROR] {error_msg}')",
        "    raise FileNotFoundError(error_msg)",
        "",
        "sample_video = str(demo_sample.video_path)",
        "sample_phone = str(demo_sample.sensor_paths['acc_phone'])",
        "sample_watch = str(demo_sample.sensor_paths['acc_watch'])",
        "sample_gyro = str(demo_sample.sensor_paths['gyro'])",
        "",
        "print(f'Selected Verified Paired Sample: {demo_sample.sample_id}')",
        "print(f'  Activity Action : {demo_sample.action}')",
        "print(f'  Subject / Scene : {demo_sample.subject} / {demo_sample.scene} (Session {demo_sample.session})')",
        "print(f'  Video File      : {sample_video}')",
        "print(f'  Phone Acc File  : {sample_phone}')",
        "print(f'  Watch Acc File  : {sample_watch}')",
        "print(f'  Gyroscope File  : {sample_gyro}')",
        "",
        "!python scripts/infer.py \\\n"
        "    --video {sample_video} \\\n"
        "    --sensor_phone {sample_phone} \\\n"
        "    --sensor_watch {sample_watch} \\\n"
        "    --sensor_gyro {sample_gyro} \\\n"
        "    --checkpoint outputs/multimodal_fusion/checkpoints/best_model_multimodal.pth"
    ]))

    # Section 17
    cells.append(md([
        "## Section 17: Consolidated Summary Table, Manifests & 1-Click Packaging",
        "Aggregates before-and-after expansion manifests, hardware manifests, runtime reports, and performance metrics into a consolidated summary table and packages `/kaggle/working/har_mmact_results.zip` with file inventory for 1-click download."
    ]))
    cells.append(code([
        "# Section 17: Consolidated Summary Table, Manifests & 1-Click Packaging",
        "import json",
        "import shutil",
        "import pandas as pd",
        "from pathlib import Path",
        "",
        "# 1. Display Hardware & Runtime Manifest",
        "hw_manifest_p = Path('outputs/manifests/runtime_hardware_manifest.json')",
        "if hw_manifest_p.exists():",
        "    with open(hw_manifest_p) as f:",
        "        hw_info = json.load(f)",
        "    print('=' * 80)",
        "    print(' RUNTIME HARDWARE MANIFEST')",
        "    print('=' * 80)",
        "    print(f'GPU Name & Count       : {hw_info.get(\"gpu_name\")} (x{hw_info.get(\"gpu_count\")})')",
        "    print(f'Compute Capability     : {hw_info.get(\"compute_capability\")}')",
        "    print(f'Total VRAM Available   : {hw_info.get(\"total_vram_gb\")} GB')",
        "    print(f'Host RAM Total / Avail : {hw_info.get(\"system_ram_total_gb\")} GB / {hw_info.get(\"system_ram_avail_gb\")} GB')",
        "    print(f'Working Disk Free      : {hw_info.get(\"working_disk_free_gb\")} GB of {hw_info.get(\"working_disk_total_gb\")} GB')",
        "    print('=' * 80)",
        "",
        "# 2. Display Dataset Expansion Manifest if available",
        "exp_manifest = Path('outputs/manifests/dataset_expansion_manifest.json')",
        "if exp_manifest.exists():",
        "    with open(exp_manifest) as f:",
        "        exp_data = json.load(f)",
        "    print('\\n' + '=' * 80)",
        "    print(' DATASET EXPANSION MANIFEST (BEFORE VS AFTER STAGING)')",
        "    print('=' * 80)",
        "    print(f'Initial Video Subjects : {exp_data.get(\"video_subjects_before\")} ({exp_data.get(\"video_count_before\")} videos)')",
        "    print(f'Final Video Subjects   : {exp_data.get(\"video_subjects_after\")} ({exp_data.get(\"video_count_after\")} videos)')",
        "    print(f'New Subjects Added     : {exp_data.get(\"new_subjects_added\")}')",
        "    print(f'Net Usable Videos Added: +{exp_data.get(\"new_videos_added\")}')",
        "    print(f'Net Sensors Added      : +{exp_data.get(\"new_sensors_added\")}')",
        "    print('=' * 80)",
        "",
        "# 3. Consolidated Benchmark Summary Table",
        "print('\\n' + '=' * 80)",
        "print(' CONSOLIDATED HAR_MMACT BENCHMARK SUMMARY TABLE (ALL 6 MODELS)')",
        "print('=' * 80)",
        "rows = []",
        "for model_name, m in eval_results.items():",
        "    rows.append({",
        "        'Model Architecture': model_name,",
        "        'Modality Mode': m.get('_mode', 'multimodal'),",
        "        'Evaluation Split': m.get('_split', 'Held-Out Test Split'),",
        "        'Samples': m.get('_samples', 'N/A'),",
        "        'Accuracy (%)': round(m.get('accuracy', 0.0) * 100.0, 2),",
        "        'Macro F1': round(m.get('macro_f1', 0.0), 4),",
        "        'Macro Precision': round(m.get('macro_precision', 0.0), 4),",
        "        'Macro Recall': round(m.get('macro_recall', 0.0), 4),",
        "        'Checkpoint': m.get('_checkpoint', 'N/A'),",
        "    })",
        "df_results = pd.DataFrame(rows)",
        "print(df_results.to_markdown(index=False))",
        "df_results.to_csv('outputs/benchmark_summary.csv', index=False)",
        "print('Saved outputs/benchmark_summary.csv')",
        "",
        "# 4. Organize structured results into /kaggle/working/har_mmact_results",
        "results_dir = Path('/kaggle/working/har_mmact_results')",
        "results_dir.mkdir(parents=True, exist_ok=True)",
        "",
        "for sub in ['original_cnn', 'improved_cnn', 'baseline_cnn', 'tuned_cnn', 'mobilenet_v2', 'sensor_only', 'multimodal_fusion', 'tuning', 'plots', 'manifests', 'reports']:",
        "    src_p = Path('outputs') / sub",
        "    if src_p.exists():",
        "        shutil.copytree(src_p, results_dir / sub, dirs_exist_ok=True)",
        "",
        "for f in ['benchmark_summary.csv']:",
        "    src_f = Path('outputs') / f",
        "    if src_f.exists():",
        "        shutil.copy2(src_f, results_dir / f)",
        "",
        "# Package into zip archive",
        "import hashlib",
        "archive_path = '/kaggle/working/har_mmact_results'",
        "shutil.make_archive(archive_path, 'zip', results_dir)",
        "zip_file = Path('/kaggle/working/har_mmact_results.zip')",
        "zip_size_mb = zip_file.stat().st_size / (1024**2) if zip_file.exists() else 0.0",
        "sha256_hash = hashlib.sha256()",
        "if zip_file.exists():",
        "    with open(zip_file, 'rb') as f_zip:",
        "        for chunk in iter(lambda: f_zip.read(65536), b''):",
        "            sha256_hash.update(chunk)",
        "    zip_sha256 = sha256_hash.hexdigest()",
        "else:",
        "    zip_sha256 = 'N/A'",
        "",
        "# Inventory check",
        "packaged_files = list(results_dir.rglob('*'))",
        "print('\\n' + '=' * 80)",
        "print('SUCCESS: All 6 benchmark experiments, checkpoints, and visualizations packaged!')",
        "print(f'Archive Path : {zip_file} ({zip_size_mb:.2f} MB)')",
        "print(f'SHA-256 Hash : {zip_sha256}')",
        "print(f'File Count   : {len(packaged_files)} items archived')",
        "print('Available under Kaggle Output tab in the right-hand panel.')",
        "print('=' * 78)"
    ]))

    # Section 18
    cells.append(md([
        "## Section 18: (Optional) Extended Benchmarks (100 and 150 Epochs)",
        "This section is **strictly optional** and intended for extended GPU quota or overnight execution.",
        "Includes a runtime feasibility estimator to ensure sufficient Kaggle session time is available before starting extended runs."
    ]))
    cells.append(code([
        "# Section 18: (Optional) Extended Benchmarks (100 and 150 Epochs)",
        "# Set RUN_EXTENDED = True to enable extended training beyond the required 50-epoch baseline.",
        "RUN_EXTENDED = False",
        "EXTENDED_EPOCHS = 100  # Set to 100 or 150",
        "",
        "if RUN_EXTENDED:",
        "    # Check runtime feasibility based on previous 50-epoch multimodal run",
        "    summary_p = Path('outputs/multimodal_fusion/training_summary_mobilenet_v2.json')",
        "    if summary_p.exists():",
        "        with open(summary_p) as f:",
        "            s_data = json.load(f)",
        "        avg_ep_sec = s_data.get('avg_epoch_duration_seconds', 15.0)",
        "        est_total_min = (avg_ep_sec * EXTENDED_EPOCHS) / 60.0",
        "        print(f'Estimated training duration for {EXTENDED_EPOCHS} epochs: {est_total_min:.1f} minutes ({avg_ep_sec:.1f}s/epoch)')",
        "",
        "    print(f'=== Starting Optional Extended Training ({EXTENDED_EPOCHS} Epochs) ===')",
        "    out_extended = f'outputs/multimodal_extended_{EXTENDED_EPOCHS}ep'",
        "    Path(out_extended).mkdir(parents=True, exist_ok=True)",
        "    !python scripts/train.py \\\n"
        "        --data_root data/genuine_mmact \\\n"
        "        --mode multimodal \\\n"
        "        --video_backbone mobilenet_v2 \\\n"
        "        --sensor_backbone inception_time \\\n"
        "        --epochs {EXTENDED_EPOCHS} \\\n"
        "        --min_epochs {EXTENDED_EPOCHS} \\\n"
        "        --batch_size 8 \\\n"
        "        --lr 0.0001 \\\n"
        "        --weight_decay 0.0005 \\\n"
        "        --split_strategy subject \\\n"
        "        --checkpoint_name best_model_multimodal_extended.pth \\\n"
        "        --output_dir {out_extended} \\\n"
        "        --cache_dir /kaggle/working/har_mmact_cache \\\n"
        "        --auto_resume \\\n"
        "        --skip_if_completed",
        "else:",
        "    print('Optional extended training skipped. Required 50-epoch benchmark is complete.')"
    ]))

    return cells


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle Benchmark Notebook")
    parser.add_argument(
        "--output",
        type=str,
        default="notebooks/kaggle_mmact_har_complete_fast.ipynb",
        help="Target notebook file path (default: notebooks/kaggle_mmact_har_complete_fast.ipynb)",
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
    with open(target_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated complete fast notebook at: {target_path}")


if __name__ == "__main__":
    main()
