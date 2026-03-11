import sys
import os


# Runs in a separate process via multiprocessing.Process (NOT subprocess.Popen).
# Using multiprocessing.Process is the correct PyInstaller-compatible approach:
# PyInstaller's freeze_support() hooks handle the re-execution of the frozen
# binary so that the full import machinery (including torch.__spec__) is intact.
#
# This function is defined at module level so multiprocessing can pickle it.
# It communicates with the parent via three multiprocessing.Queue objects
# passed as arguments (task_q, result_q, log_q).
# ─────────────────────────────────────────────────────────────────────────────

def _ai_worker_process(task_q, result_q, log_q):
    """AI worker — runs in a child process, owns faster-whisper + ctranslate2."""
    import os, sys, traceback

    def _log(msg):
        try:
            log_q.put_nowait(msg)
        except Exception:
            pass

    # ── Env vars — set BEFORE any C-extension import ─────────────────────────
    # CUDA_VISIBLE_DEVICES=-1 is the most important one: it tells torch AND
    # ctranslate2 to not initialize any CUDA context on import. Without it,
    # `import faster_whisper` triggers torch CUDA DLL loading, which SEH-crashes
    # in the frozen app if cuDNN/cuBLAS DLLs are present but broken/mismatched.
    # We remove this env var only when actually attempting a CUDA model load.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # CT2_USE_MKL=0: Force oneDNN backend instead of Intel MKL.
    # ctranslate2 auto-selects MKL on Intel CPUs and oneDNN on AMD CPUs, but
    # the auto-detection can misfire in frozen apps. oneDNN (dnnl.dll) is the
    # correct backend for AMD hardware and avoids missing MKL DLL errors.
    os.environ["CT2_USE_MKL"] = "0"  # Force oneDNN, not MKL (correct for AMD CPUs)

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Hard-set: must override any inherited value
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")
    os.environ["CT2_VERBOSE"] = "1"  # Enable backend selection logging
    os.environ.setdefault("PYTORCH_JIT", "0")

    # ── DLL search paths ──────────────────────────────────────────────────────
    # CRITICAL: os.add_dll_directory() is per-process. The main process set it up
    # but that does NOT inherit to this child. We must repeat the setup here.
    if os.name == 'nt':
        _base = os.path.dirname(sys.executable)
        _internal = os.path.join(_base, '_internal')

        # 1) Add _internal/ and all its subdirectories (recursive 2 levels)
        for _root in [_base, _internal]:
            if os.path.isdir(_root):
                try:
                    os.add_dll_directory(_root)
                    os.environ['PATH'] = _root + os.pathsep + os.environ.get('PATH', '')
                except Exception:
                    pass
                try:
                    for _e in os.scandir(_root):
                        if _e.is_dir():
                            try:
                                os.add_dll_directory(_e.path)
                                os.environ['PATH'] = _e.path + os.pathsep + os.environ.get('PATH', '')
                            except Exception:
                                pass
                            # 2nd level — ctranslate2 nests cuda DLLs deep
                            try:
                                for _ee in os.scandir(_e.path):
                                    if _ee.is_dir():
                                        try:
                                            os.add_dll_directory(_ee.path)
                                            os.environ['PATH'] = _ee.path + os.pathsep + os.environ.get('PATH', '')
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                except Exception:
                    pass

        # 2) site-packages/nvidia/*/bin — where `pip install nvidia-cudnn-cu12` puts DLLs
        try:
            import site
            for _sp in site.getsitepackages():
                for _lib in ("cudnn", "cublas", "cuda_runtime", "cufft",
                             "curand", "cusolver", "cusparse", "nvrtc", "nvjitlink"):
                    _p = os.path.join(_sp, "nvidia", _lib, "bin")
                    if os.path.isdir(_p):
                        try:
                            os.add_dll_directory(_p)
                            os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')
                        except Exception:
                            pass
        except Exception:
            pass

        # 3) ctranslate2 package directory — where dnnl.dll and CUDA DLLs live
        try:
            import ctranslate2 as _ct2_path
            _ct2_dir = os.path.dirname(_ct2_path.__file__)
            if os.path.isdir(_ct2_dir):
                os.add_dll_directory(_ct2_dir)
                os.environ['PATH'] = _ct2_dir + os.pathsep + os.environ.get('PATH', '')
        except Exception:
            pass

        # 4) System CUDA toolkit paths (common install locations)
        for _cuda_base in (
            os.environ.get("CUDA_PATH", ""),
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        ):
            if _cuda_base and os.path.isdir(_cuda_base):
                try:
                    os.add_dll_directory(_cuda_base)
                    os.environ['PATH'] = _cuda_base + os.pathsep + os.environ.get('PATH', '')
                except Exception:
                    pass

    _log(f"AI worker process started (pid={os.getpid()})")

    # Enable faulthandler so crashes write a traceback to stderr
    # (visible in the log if stderr is captured, otherwise helps with debugging)
    try:
        import faulthandler
        import tempfile
        _fh_path = os.path.join(tempfile.gettempdir(), f'whisperr_crash_{os.getpid()}.txt')
        _fh_file = open(_fh_path, 'w')
        faulthandler.enable(file=_fh_file)
        _log(f"  Crash log: {_fh_path}")
    except Exception as _fe:
        _log(f"  faulthandler setup failed: {_fe}")

    # ── Pre-load ctranslate2.dll and CUDA DLLs at worker startup ─────────────
    # ctranslate2 uses CUDA_DYNAMIC_LOADING=ON — it calls LoadLibrary() for
    # CUDA DLLs from inside its C++ constructor. In a frozen spawned subprocess
    # os.add_dll_directory() from the parent doesn't carry over. We pre-load
    # everything via ctypes so DLLs are pinned before any WhisperModel() call.
    if os.name == 'nt':
        import ctypes, shutil as _shutil
        _base_exe = os.path.dirname(sys.executable)
        _internal = os.path.join(_base_exe, '_internal')

        # -- ctranslate2.dll --
        try:
            _ct2_pkg = None
            try:
                import ctranslate2 as _ct2_tmp
                _ct2_pkg = os.path.dirname(_ct2_tmp.__file__)
            except Exception:
                pass
            if _ct2_pkg:
                _ct2_dll = os.path.join(_ct2_pkg, 'ctranslate2.dll')
                if os.path.isfile(_ct2_dll):
                    ctypes.CDLL(_ct2_dll)
                    _log(f"  Pre-loaded ctranslate2.dll from {_ct2_pkg}")
        except Exception as _e:
            _log(f"  ctranslate2.dll pre-load skipped: {_e}")

        # -- CUDA DLLs: cudart64_12, cublas64_12, cublasLt64_12 --
        # Search: _internal/ (bundled by spec), nvidia pip package bin dirs,
        # known fallback locations. Copy into _internal/ on first find so all
        # future worker spawns find them immediately without searching.
        _cuda_search = []
        if os.path.isdir(_internal):
            _cuda_search.append(_internal)
            try: os.add_dll_directory(_internal)
            except Exception: pass
        for _pyroot in [
            os.path.dirname(os.path.dirname(sys.executable)),
            r'C:\Python312', r'C:\Python311', r'C:\Python310',
            os.environ.get('PYTHONHOME', ''),
        ]:
            if not _pyroot: continue
            _nv = os.path.join(_pyroot, 'Lib', 'site-packages', 'nvidia')
            if os.path.isdir(_nv):
                for _nvpkg in os.listdir(_nv):
                    _bin = os.path.join(_nv, _nvpkg, 'bin')
                    if os.path.isdir(_bin):
                        _cuda_search.append(_bin)
                        try: os.add_dll_directory(_bin)
                        except Exception: pass
        for _fb in [
            r'C:\Users\koura\AppData\Local\Programs\Ollama\lib\ollama',
            r'C:\Ducklord\Faster-Whisper-XXL\Faster-Whisper-XXL\_xxl_data\torch\lib',
        ]:
            if os.path.isdir(_fb):
                _cuda_search.append(_fb)
                try: os.add_dll_directory(_fb)
                except Exception: pass

        # cudnn64_9.dll is a stub that loads the actual compute DLLs.
        # In a frozen subprocess they must be pre-loaded explicitly.
        # Note: engines_precompiled, engines_runtime_compiled, graph, and
        # heuristic are training/graph-API DLLs — not needed for Whisper inference.
        _cuda_needed = [
            'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
            'cudnn64_9.dll',
            'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll', 'cudnn_adv64_9.dll',
        ]
        _cuda_loaded, _cuda_missing = [], []
        for _dll in _cuda_needed:
            _ok = False
            try:
                ctypes.CDLL(_dll)
                _cuda_loaded.append(_dll)
                _ok = True
            except Exception:
                pass
            if not _ok:
                for _d in _cuda_search:
                    _fp = os.path.join(_d, _dll)
                    if os.path.isfile(_fp):
                        try:
                            ctypes.CDLL(_fp)
                            _dst = os.path.join(_internal, _dll)
                            if os.path.isdir(_internal) and not os.path.isfile(_dst):
                                try: _shutil.copy2(_fp, _dst)
                                except Exception: pass
                            _cuda_loaded.append(_dll)
                            _ok = True
                            break
                        except Exception:
                            pass
            if not _ok:
                _cuda_missing.append(_dll)
        if _cuda_loaded:
            _log(f"  CUDA pre-loaded: {', '.join(_cuda_loaded)}")
        if _cuda_missing:
            _log(f"  CUDA missing: {', '.join(_cuda_missing)}")

    # ── DLL fix: copy ctranslate2 DLLs from package dir into _internal/ ─────
    # PyInstaller collects ctranslate2's .pyd but misses the sibling DLLs
    # (dnnl.dll, cublas64_12.dll, etc.) that ctranslate2 loads via LoadLibrary
    # at runtime. We find ctranslate2's package dir and add it to the DLL
    # search path, and also copy any missing DLLs into _internal/.
    if os.name == 'nt':
        _base_d = os.path.dirname(sys.executable)
        _internal_d = os.path.join(_base_d, '_internal')

        # Find ctranslate2's package directory — that's where dnnl.dll lives
        _ct2_pkg_dir = None
        try:
            import ctranslate2 as _ct2_tmp
            _ct2_pkg_dir = os.path.dirname(_ct2_tmp.__file__)
        except Exception:
            pass

        # If not importable yet, try finding it via sys.path
        if not _ct2_pkg_dir:
            for _sp in sys.path:
                _candidate = os.path.join(_sp, 'ctranslate2')
                if os.path.isdir(_candidate):
                    _ct2_pkg_dir = _candidate
                    break

        if _ct2_pkg_dir and os.path.isdir(_ct2_pkg_dir):
            # Add ctranslate2's package dir to DLL search so Windows finds the DLLs
            try:
                os.add_dll_directory(_ct2_pkg_dir)
                os.environ['PATH'] = _ct2_pkg_dir + os.pathsep + os.environ.get('PATH', '')
            except Exception:
                pass

            # Copy any missing critical DLLs into _internal/ (permanent fix for this run)
            _critical_dlls = [
                # CUDA DLLs shipped inside ctranslate2's package dir.
                "cublas64_12.dll", "cublasLt64_12.dll",
                "cudnn64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
                "cudnn_adv64_9.dll", "cudnn_engines_precompiled64_9.dll",
                "cudnn_engines_runtime_compiled64_9.dll", "cudnn_graph64_9.dll",
                "cudnn_heuristic64_9.dll",
            ]
            # ctranslate2.dll lives in _internal/ctranslate2/ but CUDA/cuDNN DLLs
            # are bundled into _internal/. When ctranslate2 lazily loads sub-DLLs
            # during inference (e.g. cudnn_ops64_9.dll for the first conv op),
            # Windows searches relative to ctranslate2.dll's own directory first
            # and can't find them one level up — causing CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED.
            # Fix: copy all CUDA + cuDNN DLLs from _internal/ into _internal/ctranslate2/.
            import shutil as _shutil_dll
            _cuda_copy_list = [
                'libiomp5md.dll',
                'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll',
                'cudnn64_9.dll', 'cudnn_ops64_9.dll', 'cudnn_cnn64_9.dll', 'cudnn_adv64_9.dll',
                # engines_runtime_compiled is required by cuDNN 9 for inference (JIT kernel compiler)
                'cudnn_engines_runtime_compiled64_9.dll',
                # precompiled is optional (large cache), include if present
                'cudnn_engines_precompiled64_9.dll',
                'cudnn_graph64_9.dll', 'cudnn_heuristic64_9.dll',
            ]
            _cuda_copied = []
            for _cdll in _cuda_copy_list:
                _csrc = os.path.join(_internal_d, _cdll)
                _cdst = os.path.join(_ct2_pkg_dir, _cdll)
                if os.path.isfile(_csrc) and not os.path.isfile(_cdst):
                    try:
                        _shutil_dll.copy2(_csrc, _cdst)
                        _cuda_copied.append(_cdll)
                    except Exception as _ce:
                        _log(f"  Could not copy {_cdll} to ct2 dir: {_ce}")
            if _cuda_copied:
                _log(f"  Copied to ct2 dir: {', '.join(_cuda_copied)}")
            _copied = []
            _found_in_pkg = []
            for _dll in _critical_dlls:
                _src = os.path.join(_ct2_pkg_dir, _dll)
                if os.path.isfile(_src):
                    _found_in_pkg.append(_dll)
                    _dst = os.path.join(_internal_d, _dll)
                    if not os.path.isfile(_dst):
                        try:
                            import shutil
                            shutil.copy2(_src, _dst)
                            _copied.append(_dll)
                        except Exception as _ce:
                            _log(f"  Could not copy {_dll}: {_ce}")
            if _found_in_pkg:
                _log(f"  Found in ct2 pkg: {', '.join(_found_in_pkg)}")
            if _copied:
                _log(f"  Copied to _internal/: {', '.join(_copied)}")

        # Log every DLL in ctranslate2's package dir (full inventory)
        try:
            _all_dlls = [f for f in os.listdir(_ct2_pkg_dir) if f.lower().endswith('.dll')]
            _log(f"  ct2 pkg DLLs: {', '.join(sorted(_all_dlls))}")
        except Exception:
            pass

        # Log DLL presence status for diagnostics
        _ct2_dlls_check = ["dnnl.dll", "mkldnn.dll", "mkl_core.dll",
                           "mkl_rt.dll", "mkl_intel_thread.dll",
                           "iomp5md.dll", "libiomp5md.dll", "ctranslate2.dll"]
        _ok, _miss = [], []
        for _dll in _ct2_dlls_check:
            _found_dll = False
            for _search in ([_ct2_pkg_dir] if _ct2_pkg_dir else []) + [_base_d, _internal_d]:
                if _search and os.path.isfile(os.path.join(_search, _dll)):
                    _found_dll = True
                    break
            (_ok if _found_dll else _miss).append(_dll)
        if _ok:
            _log(f"  DLL OK: {', '.join(_ok)}")
        if _miss:
            _log(f"  DLL MISSING: {', '.join(_miss)}")

    model = None
    current_model_name = None
    current_language = None
    WhisperModel = None

    _vad_broken = False  # once VAD fails in this worker session, disable it permanently

    while True:
        try:
            msg = task_q.get(timeout=1.0)
        except Exception:
            continue

        if msg == '__STOP__':
            _log("AI worker: received stop signal, exiting")
            break

        model_name, lang_code, compute_pref, audio_data, src, translate, use_vad, prompt = msg

        # If VAD failed in a previous task this session, honour that for all future tasks.
        if _vad_broken and use_vad:
            use_vad = False

        # Import on first task — by the time we get here, multiprocessing has
        # fully initialized the frozen interpreter so torch.__spec__ is valid.
        if WhisperModel is None:
            try:
                _log("Importing ctranslate2...")
                import ctranslate2 as _ct2
                _log(f"ctranslate2 {_ct2.__version__} imported OK")
                _log("Importing faster_whisper...")
                from faster_whisper import WhisperModel
                _log("faster_whisper imported OK")
            except Exception as e:
                _log(f"Import error: {type(e).__name__}: {e}")
                result_q.put(('status', False))
                continue

        # Load model if needed
        need_load = (
            model is None or
            current_model_name != model_name or
            current_language != lang_code
        )
        if need_load and model_name:
            _log(f"Loading {model_name} (pref={compute_pref})...")
            loaded = False

            # ── Resolve model path from HF cache (before CUDA or CPU attempt) ─
            # In frozen apps huggingface_hub's internal resolution can fail
            # (missing constants module). We locate the cached snapshot ourselves
            # and pass an absolute path, bypassing all hub download logic.
            _model_path = model_name  # fallback → name triggers normal download
            try:
                _hf_home = os.environ.get(
                    "HF_HOME",
                    os.environ.get(
                        "HUGGINGFACE_HUB_CACHE",
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                    )
                )
                _log(f"  HF cache: {_hf_home}")
                _repo_dir = os.path.join(_hf_home, f"models--Systran--faster-whisper-{model_name}")
                if os.path.isdir(_repo_dir):
                    _snaps_dir = os.path.join(_repo_dir, "snapshots")
                    _snaps = [s for s in os.listdir(_snaps_dir)
                              if os.path.isdir(os.path.join(_snaps_dir, s))]
                    if _snaps:
                        _snap_path = os.path.join(_snaps_dir, sorted(_snaps)[-1])
                        _has_weights = any(f.endswith(('.bin', '.ct2'))
                                           for f in os.listdir(_snap_path))
                        if _has_weights:
                            _model_path = _snap_path
                            _log(f"  Resolved: {_model_path}")
                        else:
                            _log(f"  Snapshot empty, using name")
                    else:
                        _log(f"  No snapshots, using name")
                else:
                    _log(f"  Not cached, will download")
            except Exception as _hfe:
                _log(f"  Path resolve failed: {_hfe}")

            if compute_pref in ('cuda', 'auto'):
                # CUDA DLLs were pre-loaded at worker startup (see above).
                # _cuda_missing tells us if cudart was found.
                _critical = {'cudart64_12.dll', 'cublas64_12.dll', 'cudnn_ops64_9.dll'}
                _skip_cuda = bool(_critical & set(_cuda_missing)) if os.name == 'nt' else False
                if _skip_cuda:
                    _log("  cudart64_12.dll not loaded — skipping CUDA, using CPU")
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

                cuda_device_count = 0
                if not _skip_cuda:
                    try:
                        import ctranslate2 as _ct2_check
                        cuda_device_count = _ct2_check.get_cuda_device_count()
                        _log(f"  CUDA probe: {cuda_device_count} device(s) found")
                    except Exception as _pe:
                        _log(f"  CUDA probe error: {_pe}")
                        cuda_device_count = 0

                if cuda_device_count > 0:
                    for ctype in ('float16', 'int8_float16'):
                        try:
                            _log(f"  Trying CUDA {ctype}...")
                            model = WhisperModel(
                                _model_path, device="cuda",
                                compute_type=ctype,
                                cpu_threads=4,
                                num_workers=1,
                                download_root=None,
                                local_files_only=True,
                            )
                            current_model_name = model_name
                            current_language = lang_code
                            _log(f"✓ {model_name} loaded on GPU ({ctype})")
                            loaded = True
                            break
                        except Exception as e:
                            _log(f"  CUDA {ctype} failed: {type(e).__name__}: {e}")
                else:
                    if not _skip_cuda:
                        _log("  GPU unavailable — falling back to CPU")

                if not loaded:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

            if not loaded:
                # ── CPU path ──────────────────────────────────────────────────
                # CUDA_VISIBLE_DEVICES=-1 is already set (at startup, or just
                # re-set above after a failed CUDA attempt). This prevents
                # ctranslate2 from enumerating/touching any CUDA DLLs during
                # CPU model load — the main cause of SEH crashes in frozen apps.
                # Resolve the model path from HF cache before calling WhisperModel.
                # In frozen apps, huggingface_hub's internal path resolution can
                # fail or return a bad path — we find the cached model directory
                # ourselves and pass the absolute path directly to WhisperModel,
                # bypassing any hub download logic entirely.
                # Force offline mode — prevents any HF hub network calls or
                # symlink resolution inside WhisperModel.__init__ that crash
                # when huggingface_hub is partially initialised in frozen apps.
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_DATASETS_OFFLINE"] = "1"

                _files_in_model = sorted(os.listdir(_model_path)) if os.path.isdir(_model_path) else []
                _log(f"  model files: {_files_in_model}")

                for ctype in ('float32', 'int8'):
                    try:
                        _log(f"  Trying CPU {ctype}...")
                        model = WhisperModel(
                            _model_path, device="cpu",
                            compute_type=ctype,
                            cpu_threads=4,
                            num_workers=1,
                            download_root=None,
                            local_files_only=True,
                        )
                        current_model_name = model_name
                        current_language = lang_code
                        _log(f"✓ {model_name} loaded on CPU ({ctype})")
                        loaded = True
                        break
                    except Exception as e:
                        _log(f"  CPU {ctype} failed: {type(e).__name__}: {e}")

            if not loaded:
                _log(f"All load attempts failed for {model_name}")
                result_q.put(('status', False))
                continue

        # src=None → preload sentinel
        if src is None:
            result_q.put(('status', False))
            continue

        # Transcribe
        result_q.put(('status', True))
        try:
            import numpy as np
            audio_np = np.frombuffer(audio_data, dtype=np.float32)

            # onnxruntime (used by VAD/Silero) crashes in frozen apps when it tries
            # to initialise its CUDA execution provider alongside ctranslate2's
            # already-loaded CUDA DLLs (access violation in onnxruntime_pybind11_state.dll).
            # Force CPU-only ORT providers via env var before the first import.
            # This must be set before faster_whisper touches onnxruntime.
            if use_vad:
                import os as _os
                _os.environ.setdefault('ORT_LOGGING_LEVEL', '3')           # suppress ORT spam
                _os.environ.setdefault('ORTEP_DISABLE_PROVIDERS', 'CUDA')  # no CUDA provider
                # Belt-and-suspenders: also set the disable-all-non-CPU flag
                _os.environ.setdefault('ORT_DISABLE_CUDA', '1')

            segments, _ = model.transcribe(
                audio_np,
                language=lang_code if lang_code != 'auto' else None,
                task='translate' if translate else 'transcribe',
                vad_filter=use_vad,
                initial_prompt=prompt or None,
            )
            text = ' '.join(s.text.strip() for s in segments).strip()
            result_q.put(('text', text, src))
        except RuntimeError as e:
            if 'onnxruntime' in str(e).lower() or 'vad' in str(e).lower():
                # VAD failed — retry without it, disable for this session, warn ONCE
                _log(f"VAD unavailable ({e}), retrying without VAD filter")
                if not _vad_broken:
                    _vad_broken = True
                    result_q.put(('warn', 'VAD filter unavailable — disabled for this session. (onnxruntime DLL conflict in frozen app)'))
                try:
                    segments, _ = model.transcribe(
                        audio_np,
                        language=lang_code if lang_code != 'auto' else None,
                        task='translate' if translate else 'transcribe',
                        vad_filter=False,
                        initial_prompt=prompt or None,
                    )
                    text = ' '.join(s.text.strip() for s in segments).strip()
                    result_q.put(('text', text, src))
                except Exception as e2:
                    _log(f"Transcription error (no-VAD retry): {e2}\n{traceback.format_exc()}")
            else:
                _log(f"Transcription error: {e}\n{traceback.format_exc()}")
        except Exception as e:
            _log(f"Transcription error: {e}\n{traceback.format_exc()}")
        result_q.put(('status', False))




import json
import time
import threading
import queue
import subprocess
import shutil
import wave
import zipfile
import urllib.request
import traceback
import logging
from pathlib import Path
from datetime import datetime

# Application version
__version__ = "2.0.6"
APP_NAME = "WhisperR"

# --- 1. GLOBAL CRASH LOGGING ---
def crash_logger(etype, value, tb):
    try:
        with open("CRASH_LOG.txt", "w") as f:
            f.write(f"--- CRASH REPORT {datetime.now()} ---\n")
            f.write(f"{APP_NAME} v{__version__}\n\n")
            traceback.print_exception(etype, value, tb, file=f)
    except: pass
sys.excepthook = crash_logger

# --- 2. DLL & ENVIRONMENT HARDENING ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
    LIB_DIR = os.path.join(BASE_DIR, "_internal")
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LIB_DIR = BASE_DIR

# Critical: Prevent torch from loading problematic DLLs in frozen mode
# This fixes the "Invalid access to memory location" error with shm.dll
os.environ["PYTORCH_JIT"] = "0"
os.environ["PYTORCH_JIT_USE_NNC_NOT_NVFUSER"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["QT_PA_PLATFORM"] = "windows:dpiawareness=0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Force ctranslate2 to use GENERIC CPU ISA (no AVX/AVX2 dispatch) in worker subprocess.
# AVX-dispatched code can crash in a freshly-spawned frozen subprocess on some Windows
# configs. This is inherited by the child via child_env = os.environ.copy().
os.environ.setdefault("CT2_FORCE_CPU_ISA", "GENERIC")

import multiprocessing
import multiprocessing.connection

# Additional fix: Disable torch multiprocessing which uses shared memory
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Prevent torch from loading its C extensions (shm.dll, _C.pyd etc.) in the
# parent process before the UI is up.  The worker subprocess has its own more
# complete stub; here we only need to suppress shm.dll in the frozen parent.
if getattr(sys, 'frozen', False):
    import warnings
    warnings.filterwarnings('ignore')

    class DummySHM:
        """Robust stub for torch.multiprocessing — handles calls, iteration, etc."""
        def __call__(self, *a, **k):  return DummySHM()
        def __iter__(self):           return iter([])
        def __len__(self):            return 0
        def __bool__(self):           return False
        def __getitem__(self, k):     return DummySHM()
        def __enter__(self):          return self
        def __exit__(self, *a):       return False
        def __getattr__(self, name):  return DummySHM()

    sys.modules['torch.multiprocessing'] = DummySHM()
    sys.modules['torch.multiprocessing.reductions'] = DummySHM()

if os.name == 'nt':
    # Ensure dependencies like zlibwapi.dll and torch/ctranslate libs are found
    dll_search_path = [BASE_DIR, LIB_DIR]
    try:
        import site
        for sp in site.getsitepackages():
            for lib in ["cudnn", "cublas", "cuda_runtime", "ctranslate2"]:
                p = os.path.join(sp, "nvidia", lib, "bin")
                if os.path.exists(p): dll_search_path.append(p)
                p_internal = os.path.join(LIB_DIR, lib)
                if os.path.exists(p_internal): dll_search_path.append(p_internal)
    except: pass

    for p in set(dll_search_path):
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
                os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
            except: pass

import pyaudio
import numpy as np
import pyautogui
import pyperclip
from pynput import keyboard
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QComboBox, QLabel, QFileDialog, QTabWidget, QCheckBox, 
    QDoubleSpinBox, QProgressBar, QFormLayout, QLineEdit, QGroupBox, QSpinBox, 
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog, QMessageBox,
    QSystemTrayIcon, QMenu
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QObject, QEvent
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon, QAction, QKeyEvent, QPixmap, QPen

# --- 3. CONSTANTS ---
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANG_MAP = {"Auto": None, "English": "en", "Greek": "el", "German": "de", "French": "fr", "Spanish": "es"}
HALLUCINATIONS = ["thank you.", "thanks for watching.", "god bless.", "god bless you.", "subtitles by", "Thank you for watching, and I'll see you in the next video"]

DARK_STYLE = """
QMainWindow, QDialog, QScrollArea, QTabWidget { background-color: #121212; }
QWidget { color: #e0e0e0; font-family: 'Segoe UI'; font-size: 9pt; }
QTextEdit { background-color: #1e1e1e; border: 1px solid #333; color: #fff; border-radius: 4px; }
QPushButton { background-color: #2a2a2a; border: 1px solid #444; padding: 6px; border-radius: 4px; }
QPushButton:hover { background-color: #353535; border: 1px solid #0078d7; }
QGroupBox { border: 1px solid #333; margin-top: 12px; font-weight: bold; padding: 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QProgressBar { background: #1e1e1e; border: 1px solid #333; text-align: center; height: 12px; border-radius: 6px; }
QProgressBar::chunk { background-color: #0078d7; border-radius: 6px; }
QHeaderView::section { background-color: #252525; color: white; padding: 4px; border: 1px solid #333; }
QTableWidget { background-color: #1e1e1e; gridline-color: #333; }
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit { background-color: #2a2a2a; border: 1px solid #444; padding: 4px; }
QSpinBox, QDoubleSpinBox { padding-right: 24px; min-height: 24px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 22px; border-left: 1px solid #555; border-bottom: 1px solid #555; background-color: #3a3a3a; border-top-right-radius: 3px; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 22px; border-left: 1px solid #555; border-top: 1px solid #555; background-color: #3a3a3a; border-bottom-right-radius: 3px; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { width: 10px; height: 10px; image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-bottom: 6px solid #cccccc; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { width: 10px; height: 10px; image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 6px solid #cccccc; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover { background-color: #0078d7; }
QSpinBox::up-button:hover QSpinBox::up-arrow, QDoubleSpinBox::up-button:hover QDoubleSpinBox::up-arrow { border-bottom-color: #ffffff; }
QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed { background-color: #005fa3; }
"""

# --- 4. LOGGING SETUP ---
class _FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes to disk after every record.
    This ensures log lines are never lost if the process hard-crashes (C++ level)."""
    def emit(self, record):
        super().emit(record)
        self.flush()

class AppLogger:
    def __init__(self):
        self.log_path = os.path.join(BASE_DIR, "app_log.txt")
        self.level = logging.INFO
        self.logger = logging.getLogger(APP_NAME)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler — flushes after every line so hard C++ crashes don't eat logs
        fh = _FlushingFileHandler(self.log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        
        # Console handler for debugging
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)
    
    def set_level(self, level_name):
        levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
        self.level = levels.get(level_name, logging.INFO)
        self.logger.setLevel(self.level)
    
    def debug(self, msg, exc_info=False): 
        self.logger.debug(msg, exc_info=exc_info)
    
    def info(self, msg, exc_info=False): 
        self.logger.info(msg, exc_info=exc_info)
    
    def warning(self, msg, exc_info=False): 
        self.logger.warning(msg, exc_info=exc_info)
    
    def error(self, msg, exc_info=False): 
        self.logger.error(msg, exc_info=exc_info)

app_logger = AppLogger()

# --- 5. CONFIGURATION ---
class AppConfig:
    def __init__(self):
        self.path = os.path.join(BASE_DIR, "config.json")
        self.settings = {
            "model": "base", "lang_name": "English", "lang_code": "en", 
            "translate": False, "timestamps": False, "initial_prompt": "",
            "audio_folder": str(Path.home() / "WhisperR_Recordings"),
            "mon_folder": str(Path.home() / "WhisperR_Watch"),
            "clear_exit": False, "save_to_disk": False, "auto_space": True,
            "min_to_tray": False, "input_device_name": "", "input_device_index": None, "paste_delay": 0.5, 
            "hotkey": "ctrl+alt+r", "ptt_key": "f8", 
            "visibility_hotkey": "ctrl+shift+w", "rollback_hotkey": "ctrl+shift+z", "live_mode": "Simple", 
            "dict_mode": "Continuous", "auto_pause_sec": 1.5, "noise_floor": 200, 
            "speech_vol": 1500, "commands": {"Launch Notepad": "notepad.exe"},
            "ind_show": True, "ind_type": "Both", "ind_pos": "Top-Right", 
            "ind_size": 32, "ind_off": 20, "bar_edge": "Top", "bar_size": 5,
            "bar_thickness": 5, "ind_opacity": 255, "bar_opacity": 255, "ind_hide_idle": True,
            "log_level": "INFO", "use_vad": False
        }
        self.load()
        app_logger.set_level(self.settings["log_level"])

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
                app_logger.info("Configuration loaded successfully")
            except Exception as e:
                app_logger.error(f"Failed to load config: {e}")
                app_logger.warning("Using default configuration")
        
        # Validate and fix paths
        for k in ["audio_folder", "mon_folder"]:
            path_str = self.settings[k]
            
            # Check for invalid characters like \x01
            try:
                # Try to create Path - this will fail if path is invalid
                test_path = Path(path_str)
                
                # Check if path contains invalid characters
                if '\x00' in str(test_path) or '\x01' in str(test_path):
                    raise ValueError(f"Invalid characters in path: {path_str}")
                
                # Try to resolve - catches things like invalid drive letters
                if not test_path.is_absolute():
                    raise ValueError(f"Path is not absolute: {path_str}")
                
                # Try to create the directory
                test_path.mkdir(parents=True, exist_ok=True)
                app_logger.debug(f"Path validated: {k} = {path_str}")
                
            except Exception as e:
                # Path is corrupted or invalid - reset to default
                app_logger.error(f"Invalid path for {k}: {path_str} - Error: {e}")
                
                if k == "audio_folder":
                    default_path = str(Path.home() / "WhisperR_Recordings")
                else:
                    default_path = str(Path.home() / "WhisperR_Watch")
                
                app_logger.warning(f"Resetting {k} to default: {default_path}")
                self.settings[k] = default_path
                
                # Try to create default path
                try:
                    Path(default_path).mkdir(parents=True, exist_ok=True)
                except Exception as e2:
                    app_logger.error(f"Failed to create default path: {e2}")
                    # Last resort - use temp directory
                    import tempfile
                    temp_path = os.path.join(tempfile.gettempdir(), f"WhisperR_{k}")
                    self.settings[k] = temp_path
                    Path(temp_path).mkdir(parents=True, exist_ok=True)
                    app_logger.warning(f"Using temp path: {temp_path}")

    def save(self):
        try:
            # Create backup of existing config
            backup_path = self.path + ".backup"
            if os.path.exists(self.path):
                try:
                    shutil.copy2(self.path, backup_path)
                except:
                    pass  # Backup failed, but continue anyway
            
            # Write new config
            with open(self.path, 'w', encoding='utf-8') as f: 
                json.dump(self.settings, f, indent=4)
            
            app_logger.info("Configuration saved successfully")
            
            # Remove backup after successful save
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except:
                    pass
                    
        except PermissionError as e:
            app_logger.error(f"Permission denied saving config: {e}")
            raise Exception(f"Cannot save settings - permission denied. Try running as administrator.")
        except Exception as e:
            app_logger.error(f"Failed to save config: {e}", exc_info=True)
            
            # Try to restore backup
            backup_path = self.path + ".backup"
            if os.path.exists(backup_path):
                try:
                    shutil.copy2(backup_path, self.path)
                    app_logger.info("Restored config from backup")
                except:
                    pass
            
            raise Exception(f"Failed to save settings: {e}")

# --- 6. WORKERS ---
# ─────────────────────────────────────────────────────────────────
# AI WORKER — runs in a completely separate process
# This is the ONLY way to prevent CTranslate2 (ctranslate2 C++ engine)
# from hard-crashing the entire app when it does AVX2/CUDA initialisation
# inside a PyInstaller frozen app.  If the worker process crashes, it
# crashes alone; the UI process survives and can restart it.
# ─────────────────────────────────────────────────────────────────

# (AI worker function _ai_worker_process is defined at the top of this file)

class CalibrationWorker(QThread):
    progress = pyqtSignal(int)
    status_msg = pyqtSignal(str)
    finished = pyqtSignal(int, int)
    
    def __init__(self, dev_idx): 
        super().__init__()
        self.dev_idx = dev_idx
        app_logger.info(f"Calibration worker initialized for device {dev_idx}")
    
    def run(self):
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            # Get device info safely
            try:
                dev_info = p.get_device_info_by_index(self.dev_idx)
                rate = int(dev_info['defaultSampleRate'])
                device_name = dev_info.get('name', f'Device {self.dev_idx}')
                app_logger.info(f"Calibration starting: device={device_name}, rate={rate}")
            except Exception as e:
                app_logger.error(f"Failed to get device info for index {self.dev_idx}: {e}")
                self.status_msg.emit(f"Error: Invalid device")
                return
            
            # Verify device has input channels
            if dev_info.get('maxInputChannels', 0) <= 0:
                app_logger.error(f"Device {self.dev_idx} has no input channels")
                self.status_msg.emit("Error: Device has no input")
                return
            
            # Open stream
            try:
                stream = p.open(
                    format=pyaudio.paInt16, 
                    channels=1, 
                    rate=rate, 
                    input=True, 
                    input_device_index=self.dev_idx,
                    frames_per_buffer=1024
                )
            except Exception as e:
                app_logger.error(f"Failed to open calibration stream: {e}")
                self.status_msg.emit(f"Error opening device")
                return
            
            n, s = [], []
            
            # Noise calibration phase
            self.status_msg.emit("Stay SILENT (Noise detection)...")
            try:
                for i in range(100):
                    d = stream.read(1024, exception_on_overflow=False)
                    n.append(np.sqrt(np.mean(np.frombuffer(d, dtype=np.int16).astype(np.float64)**2)))
                    self.progress.emit(i+1)
                    time.sleep(0.04)
            except Exception as e:
                app_logger.error(f"Error during noise calibration: {e}")
                self.status_msg.emit("Error during calibration")
                if stream:
                    stream.stop_stream()
                    stream.close()
                return
            
            noise_level = int(np.percentile(n, 90))
            app_logger.info(f"Noise level calibrated: {noise_level}")
            
            # Speech calibration phase — reset bar to 0 so pass 2 visually restarts
            self.progress.emit(0)
            self.status_msg.emit("SPEAK normally (Voice level)...")
            try:
                for i in range(100):
                    d = stream.read(1024, exception_on_overflow=False)
                    s.append(np.sqrt(np.mean(np.frombuffer(d, dtype=np.int16).astype(np.float64)**2)))
                    self.progress.emit(i+1)
                    time.sleep(0.04)
            except Exception as e:
                app_logger.error(f"Error during speech calibration: {e}")
                self.status_msg.emit("Error during calibration")
                if stream:
                    stream.stop_stream()
                    stream.close()
                return
            
            speech_level = int(np.percentile(s, 90))
            app_logger.info(f"Speech level calibrated: {speech_level}")
            
            self.finished.emit(noise_level, speech_level)
            
            if stream:
                stream.stop_stream()
                stream.close()
                
        except Exception as e:
            app_logger.error(f"Calibration error: {e}")
            self.status_msg.emit(f"Calibration failed")
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
        finally:
            p.terminate()

class TranscriberWorker(QThread):
    """Manages a child AI-worker process via multiprocessing.Process.

    The child process owns faster-whisper + ctranslate2. If it crashes,
    this thread detects it and can restart it. IPC via multiprocessing.Queue.
    """
    finished_text  = pyqtSignal(str, str)
    status_changed = pyqtSignal(bool)
    log_msg        = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config  = config
        self.running = True
        self._proc   = None
        self._task_q   = None
        self._result_q = None
        self._log_q    = None
        self._pending     = queue.Queue()
        self._cuda_failed  = False
        self._crash_count  = 0
        app_logger.info("Transcriber worker initialized")

    # ── public API ────────────────────────────────────────────────

    def preload_model(self):
        """Ask the worker to pre-warm the model (runs before first recording)."""
        cfg = self.config.settings
        compute = 'cpu' if self._cuda_failed else cfg.get('compute_pref', 'auto')
        task = (
            cfg['model'], cfg['lang_code'], compute,
            None, None,           # audio_data=None, src=None → preload sentinel
            False, False, '',     # translate, use_vad, prompt
        )
        app_logger.info(f"TranscriberWorker.preload_model: queuing preload for model={cfg['model']} compute={compute}")
        self._pending.put(task)

    def reload_model(self):
        """Force model reload (kills and restarts the worker process)."""
        app_logger.info("Model reload requested — restarting worker")
        self._stop_worker()
        self._start_worker()

    def submit(self, audio_data, src):
        """Queue audio data for transcription."""
        cfg = self.config.settings
        compute = 'cpu' if self._cuda_failed else cfg.get('compute_pref', 'auto')
        task = (
            cfg['model'], cfg['lang_code'], compute,
            audio_data, src,
            cfg.get('translate', False),
            cfg.get('use_vad', False),
            cfg.get('initial_prompt', ''),
        )
        self._pending.put(task)

    # ── subprocess lifecycle ───────────────────────────────────────

    def _start_worker(self):
        """Spawn (or re-spawn) the AI worker as a multiprocessing.Process.

        Uses multiprocessing.Process so PyInstaller's freeze_support() hooks
        handle the frozen-binary re-execution correctly — torch.__spec__ and
        all import machinery are intact in the child, unlike subprocess.Popen.

        IPC uses three multiprocessing.Queue objects (task, result, log).
        """
        try:
            ctx = multiprocessing.get_context('spawn')
            self._task_q   = ctx.Queue()
            self._result_q = ctx.Queue()
            self._log_q    = ctx.Queue()

            self._proc = ctx.Process(
                target=_ai_worker_process,
                args=(self._task_q, self._result_q, self._log_q),
                daemon=True,
                name='WhisperR-AI-Worker',
            )
            self._proc.start()
            app_logger.info(f"AI worker process started (pid={self._proc.pid})")
        except Exception as e:
            app_logger.error(f"Failed to start AI worker: {e}", exc_info=True)
            self._proc = None

    def _stop_worker(self):
        """Gracefully stop the worker process."""
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.is_alive():
                try:
                    self._task_q.put('__STOP__')
                except Exception:
                    pass
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=1)
        except Exception as e:
            app_logger.warning(f"_stop_worker: {e}")
        # Close queues
        for q in (self._task_q, self._result_q, self._log_q):
            try:
                if q is not None:
                    q.close()
                    q.join_thread()
            except Exception:
                pass
        self._task_q = self._result_q = self._log_q = None

    def _worker_alive(self):
        return self._proc is not None and self._proc.is_alive()

    def _drain_log(self):
        """Read any pending log messages from the worker and emit them."""
        try:
            while self._log_q and not self._log_q.empty():
                msg = self._log_q.get_nowait()
                app_logger.info(f"[worker] {msg}")
                self.log_msg.emit(msg)
        except Exception:
            pass

    # ── main thread loop ──────────────────────────────────────────

    def run(self):
        app_logger.info("TranscriberWorker.run: thread started")
        self._start_worker()

        while self.running:
            # Drain log messages
            self._drain_log()

            # Check for a pending task
            try:
                task = self._pending.get(timeout=0.2)
            except queue.Empty:
                continue

            # Restart dead worker
            if not self._worker_alive():
                app_logger.warning("AI worker died — restarting...")
                self.log_msg.emit("⚠ AI worker restarted (crashed?)")
                self._start_worker()
                if not self._worker_alive():
                    app_logger.error("Could not restart AI worker")
                    self.log_msg.emit("AI worker failed to restart — check logs")
                    continue

            # Send task to worker
            try:
                self._task_q.put(task)
            except Exception as e:
                app_logger.error(f"Failed to send task to worker: {e}")
                continue

            # Read results until we get the final status=False.
            # We poll in 1s increments so we can:
            #   • drain log messages continuously (worker progress visible)
            #   • detect a dead process quickly
            #   • apply a short timeout only for active transcription,
            #     but allow unlimited time for model loading
            worker_crashed = False
            transcription_started = False  # True after status=True received
            idle_secs = 0                  # seconds with no message and worker alive
            TRANSCRIPTION_TIMEOUT = 120    # seconds — applies only after status=True

            while True:
                self._drain_log()

                # Dead process check
                if not self._worker_alive():
                    if self._result_q.empty():
                        ec = getattr(self._proc, "exitcode", "?")
                        app_logger.warning(f"Worker died mid-task (exitcode={ec})")
                        worker_crashed = True
                        break
                    # else: process exited cleanly, drain remaining results

                try:
                    msg = self._result_q.get(timeout=1.0)
                    idle_secs = 0  # got a message — reset idle counter
                except Exception:
                    # No message this second
                    if not self._worker_alive():
                        ec = getattr(self._proc, "exitcode", "?")
                        app_logger.warning(f"Worker died mid-task (exitcode={ec})")
                        worker_crashed = True
                        break
                    idle_secs += 1
                    # Only enforce a timeout after transcription has started
                    if transcription_started and idle_secs >= TRANSCRIPTION_TIMEOUT:
                        app_logger.warning("Transcription timeout — restarting worker")
                        self.log_msg.emit("⚠ Transcription timeout — restarting worker")
                        self._stop_worker()
                        self._start_worker()
                        worker_crashed = True
                        break
                    # Model loading: no timeout — just keep waiting
                    continue

                if msg[0] == 'status':
                    if msg[1]:
                        transcription_started = True  # model loaded, transcription running
                    self.status_changed.emit(msg[1])
                    if not msg[1]:
                        self._crash_count = 0  # successful completion → reset circuit breaker
                        break  # Final status=False means task complete
                elif msg[0] == 'text':
                    _, text, src = msg
                    HALLUCINATIONS_LOCAL = set(h.lower() for h in HALLUCINATIONS) if 'HALLUCINATIONS' in dir() else set()
                    if text and text.lower() not in HALLUCINATIONS_LOCAL:
                        app_logger.info(f"Transcription: '{text[:50]}...'")
                        self.finished_text.emit(text, src)
                    else:
                        app_logger.debug("No valid speech / hallucination filtered")
                elif msg[0] == 'warn':
                    # Non-fatal worker warning (e.g. VAD unavailable) — show in log panel
                    app_logger.warning(f"Worker warning: {msg[1]}")
                    self.log_msg.emit(f"⚠ {msg[1]}")

            # Worker crashed mid-task: unblock the UI, circuit-break after 3 failures
            if worker_crashed:
                self.status_changed.emit(False)
                app_logger.warning("Worker crashed — emitting status=False to unblock UI")
                self._cuda_failed  = True
                self._crash_count += 1
                if self._crash_count >= 3:
                    app_logger.error(f"Worker crashed {self._crash_count} times — giving up")
                    self.log_msg.emit(
                        "✗ AI worker crashed repeatedly. "
                        "Transcription is unavailable. "
                        "Check the Setup tab for GPU/CPU requirements."
                    )
                    while not self._pending.empty():
                        try: self._pending.get_nowait()
                        except: break
                else:
                    app_logger.info(f"Restarting worker (attempt {self._crash_count}/3, CPU fallback)...")
                    self.log_msg.emit(f"⚠ AI worker crashed — retrying on CPU (attempt {self._crash_count}/3)")
                    self._stop_worker()
                    self._start_worker()

        # Cleanup
        self._stop_worker()
        app_logger.info("TranscriberWorker.run: exiting")


class AudioRecorder(QThread):
    data_ready = pyqtSignal(object)
    speech_active = pyqtSignal(bool)
    volume_out = pyqtSignal(int)
    
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.active = False
        self.ptt_pressed = False
        self.ptt_flush   = False   # set True by poll to trigger immediate dispatch
        app_logger.info("Audio recorder initialized")
    
    def run(self):
        app_logger.info("AudioRecorder.run: starting")
        app_logger.info(f"AudioRecorder.run: frozen={getattr(sys, 'frozen', False)}")

        # Reinitialize PyAudio fresh in the recording thread (same pattern as the
        # working version which called sd._terminate()/_initialize() before each session).
        # In frozen mode the initial PyAudio instance can have stale PortAudio state.
        p = pyaudio.PyAudio()
        try:
            p.terminate()
        except Exception:
            pass
        p = pyaudio.PyAudio()
        app_logger.info(f"AudioRecorder.run: PyAudio reinitialized, device_count={p.get_device_count()}")

        idx = None

        saved_name = self.config.settings.get("input_device_name", "")
        saved_idx  = self.config.settings.get("input_device_index", None)
        app_logger.info(f"AudioRecorder.run: saved_name='{saved_name}', saved_idx={saved_idx}")

        # Log all input devices for diagnostics
        for i in range(p.get_device_count()):
            try:
                d = p.get_device_info_by_index(i)
                if d["maxInputChannels"] > 0:
                    app_logger.info(f"  input device {i}: name='{d['name']}' ch={d['maxInputChannels']} rate={d['defaultSampleRate']}")
            except Exception:
                pass

        # Determine device index from saved settings
        idx = None
        if saved_idx is not None:
            try:
                d = p.get_device_info_by_index(int(saved_idx))
                if d["maxInputChannels"] > 0:
                    idx = int(saved_idx)
                    app_logger.info(f"AudioRecorder.run: using saved index {idx}: '{d['name']}'")
            except Exception as e:
                app_logger.warning(f"AudioRecorder.run: saved index {saved_idx} invalid: {e}")

        if idx is None and saved_name:
            for i in range(p.get_device_count()):
                try:
                    d = p.get_device_info_by_index(i)
                    if d["maxInputChannels"] <= 0:
                        continue
                    if saved_name == d["name"] or d["name"] in saved_name or saved_name in d["name"]:
                        idx = i
                        app_logger.info(f"AudioRecorder.run: name-matched device {idx}: '{d['name']}'")
                        break
                except Exception:
                    continue

        if idx is None:
            try:
                default = p.get_default_input_device_info()
                idx = int(default["index"])
                app_logger.warning(f"AudioRecorder.run: using system default {idx}: '{default['name']}'")
            except Exception:
                idx = 0

        # Read the device's native sample rate AND channel count.
        # WASAPI shared mode requires matching the device's native format exactly —
        # opening with channels=1 on a stereo device succeeds but returns zeroed buffers.
        capture_rate = 16000  # fallback
        capture_channels = 1  # fallback
        if idx is not None:
            try:
                dev_info = p.get_device_info_by_index(idx)
                capture_rate = int(dev_info["defaultSampleRate"])
                if capture_rate <= 0:
                    capture_rate = 44100
                capture_channels = max(1, int(dev_info.get("maxInputChannels", 1)))
            except Exception:
                capture_rate = 44100
                capture_channels = 1
        app_logger.info(f"AudioRecorder.run: device idx={idx}, native rate={capture_rate}, channels={capture_channels}")

        # Open stream — use native rate + native channels, fall back gracefully.
        # We downmix to mono in dispatch() after reading, so Whisper always gets mono.
        stream = None
        actual_rate     = capture_rate
        actual_channels = capture_channels
        for attempt_idx, label in [(idx, "selected"), (None, "system default"), (0, "device 0")]:
            for try_rate in ([capture_rate] if attempt_idx == idx else [capture_rate, 44100, 48000]):
                for try_ch in ([capture_channels, 1] if capture_channels > 1 else [1]):
                    try:
                        kwargs = dict(format=pyaudio.paInt16, channels=try_ch, rate=try_rate,
                                      input=True, frames_per_buffer=2048)
                        if attempt_idx is not None:
                            kwargs["input_device_index"] = attempt_idx
                        app_logger.info(f"AudioRecorder.run: opening stream ({label} @ {try_rate}Hz ch={try_ch}): {kwargs}")
                        stream = p.open(**kwargs)
                        actual_rate     = try_rate
                        actual_channels = try_ch
                        if attempt_idx is not None:
                            idx = attempt_idx
                        app_logger.info(f"AudioRecorder.run: stream opened OK ({label} @ {actual_rate}Hz ch={actual_channels})")
                        break
                    except Exception as e:
                        app_logger.warning(f"AudioRecorder.run: stream open failed ({label} @ {try_rate}Hz ch={try_ch}): {e}")
                if stream is not None:
                    break
            if stream is not None:
                break

        # Use actual_rate / actual_channels everywhere instead of hardcoded constants
        FIXED_RATE     = actual_rate
        FIXED_CHANNELS = actual_channels

        if stream is None:
            app_logger.error("AudioRecorder.run: all stream attempts failed, aborting")
            p.terminate()
            return

        # Test read before main loop
        try:
            test_data = stream.read(1024, exception_on_overflow=False)
            test_rms = int(np.sqrt(np.mean(np.frombuffer(test_data, dtype=np.int16).astype(np.float64)**2)))
            app_logger.info(f"AudioRecorder.run: test read OK, RMS={test_rms}")
        except Exception as e:
            app_logger.error(f"AudioRecorder.run: test read failed: {e} — aborting")
            try:
                stream.stop_stream(); stream.close()
            except Exception:
                pass
            p.terminate()
            return

        frames = []
        last_speech = time.time()
        threshold = (self.config.settings["noise_floor"] + self.config.settings["speech_vol"]) / 2
        app_logger.info(f"AudioRecorder.run: loop starting, threshold={threshold:.1f}")

        self.active = True
        _last_rms_log   = time.time()
        _speech_state   = False
        _last_state_chg = 0.0
        _DEBOUNCE_SEC   = 0.3

        _ptt_was_pressed = False  # track transition for flush-on-release

        while self.active:
            if self.config.settings["live_mode"] == "Push-To-Talk":
                # Detect release edge: was held, now released → flush immediately
                if _ptt_was_pressed and not self.ptt_pressed:
                    _ptt_was_pressed = False
                    if len(frames) > 5:
                        app_logger.debug(f"PTT released — dispatching {len(frames)} frames")
                        self.speech_active.emit(False)
                        self.dispatch(frames, FIXED_RATE, FIXED_CHANNELS)
                        frames = []
                        last_speech = time.time()
                elif self.ptt_pressed:
                    _ptt_was_pressed = True
                if not self.ptt_pressed:
                    time.sleep(0.05)
                    continue

            try:
                data = stream.read(1024, exception_on_overflow=False)
                raw  = np.frombuffer(data, dtype=np.int16).astype(np.float64)
                # Downmix to mono for RMS measurement (handles stereo WASAPI devices)
                if FIXED_CHANNELS > 1:
                    raw = raw.reshape(-1, FIXED_CHANNELS).mean(axis=1)
                rms  = int(np.sqrt(np.mean(raw**2)))
                self.volume_out.emit(rms)

                now = time.time()
                if now - _last_rms_log > 2.0:
                    app_logger.debug(f"AudioRecorder: RMS={rms}, thr={threshold:.1f}, frames={len(frames)}, speech={_speech_state}")
                    _last_rms_log = now

                is_speech = rms > threshold
                if is_speech != _speech_state and (now - _last_state_chg) >= _DEBOUNCE_SEC:
                    _speech_state   = is_speech
                    _last_state_chg = now
                    app_logger.debug(f"AudioRecorder: speech_active -> {is_speech} (RMS={rms})")
                    self.speech_active.emit(is_speech)

                # In Simple mode, collect every frame — the user controls start/stop
                # explicitly and wants everything they said, regardless of RMS level.
                # In Auto-Pause mode, only collect frames during/after detected speech
                # so silence gaps don't pad out the audio unnecessarily.
                if self.config.settings["dict_mode"] == "Auto-Pause":
                    if is_speech:
                        frames.append(data)
                        last_speech = now
                    elif frames:
                        frames.append(data)  # trailing silence after speech
                else:
                    # Simple / Continuous: always collect
                    frames.append(data)
                    if is_speech:
                        last_speech = now

                if self.config.settings["dict_mode"] == "Auto-Pause":
                    silence_dur = now - last_speech
                    if silence_dur > self.config.settings["auto_pause_sec"] and len(frames) > 20:
                        app_logger.debug(f"AudioRecorder: auto-pause dispatch, {len(frames)} frames")
                        self.speech_active.emit(False)
                        _speech_state   = False
                        _last_state_chg = now
                        self.dispatch(frames, FIXED_RATE, FIXED_CHANNELS)
                        frames      = []
                        last_speech = now

            except Exception as e:
                app_logger.error(f"AudioRecorder: read error: {e}", exc_info=True)
                break

        if len(frames) > 20:
            self.dispatch(frames, FIXED_RATE, FIXED_CHANNELS)

        self.speech_active.emit(False)
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()
        app_logger.info("AudioRecorder.run: finished")

    
    def dispatch(self, frames, rate, channels=1):
        raw_np = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
        # Downmix multi-channel (e.g. stereo WASAPI) to mono before resampling
        if channels > 1:
            raw_np = raw_np.reshape(-1, channels).mean(axis=1).astype(np.float32)

        if rate != 16000:
            audio_16k = np.interp(
                np.linspace(0, 1, int(len(raw_np)*16000/rate)), 
                np.linspace(0, 1, len(raw_np)), 
                raw_np
            ).astype(np.float32)
        else:
            audio_16k = raw_np
        
        if self.config.settings["save_to_disk"]:
            path = os.path.join(self.config.settings["audio_folder"], f"rec_{int(time.time()*1000)}.wav")
            try:
                with wave.open(path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes((audio_16k * 32767).astype(np.int16).tobytes())
                app_logger.debug(f"Audio saved to: {path}")
                self.data_ready.emit(os.path.abspath(path))
            except Exception as e:
                app_logger.error(f"Failed to save audio: {e}")
        else:
            self.data_ready.emit(audio_16k)

# --- 7. UI UTILS ---
class HotkeyCaptureDialog(QDialog):
    key_captured = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hotkey Recorder")
        self.setFixedSize(350, 180)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Press your desired key combination.\n\n"
            "Supported modifiers: Ctrl, Shift, Alt\n"
            "Supported keys: A-Z, 0-9, F1-F12, and more\n\n"
            "Press Escape to cancel."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        self.status_label = QLabel("Waiting for input...")
        self.status_label.setStyleSheet("color: #0078d7; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        self.pressed_keys = set()
        self.main_key = None
        
        app_logger.debug("Hotkey capture dialog opened")
    
    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            app_logger.debug("Hotkey capture cancelled")
            self.reject()
            return
        
        # Handle modifiers
        if e.key() in [Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt]:
            mod_name = {
                Qt.Key.Key_Control: "ctrl",
                Qt.Key.Key_Shift: "shift",
                Qt.Key.Key_Alt: "alt"
            }.get(e.key())
            self.pressed_keys.add(mod_name)
            self.update_status()
        else:
            # Handle main key
            key_text = self.get_key_name(e.key())
            if key_text:
                self.main_key = key_text
                self.update_status()
                # Auto-accept after main key is pressed
                QTimer.singleShot(200, self.accept_hotkey)
    
    def get_key_name(self, key):
        """Convert Qt key code to readable key name"""
        # Function keys
        if Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
            return f"f{key - Qt.Key.Key_F1 + 1}"
        
        # Number keys
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)
        
        # Letter keys
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return chr(key).lower()
        
        # Special keys
        special_keys = {
            Qt.Key.Key_Space: "space",
            Qt.Key.Key_Return: "return",
            Qt.Key.Key_Enter: "enter",
            Qt.Key.Key_Tab: "tab",
            Qt.Key.Key_Backspace: "backspace",
            Qt.Key.Key_Delete: "delete",
            Qt.Key.Key_Insert: "insert",
            Qt.Key.Key_Home: "home",
            Qt.Key.Key_End: "end",
            Qt.Key.Key_PageUp: "page_up",
            Qt.Key.Key_PageDown: "page_down",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
        }
        
        return special_keys.get(key, None)
    
    def update_status(self):
        parts = []
        if "ctrl" in self.pressed_keys:
            parts.append("Ctrl")
        if "shift" in self.pressed_keys:
            parts.append("Shift")
        if "alt" in self.pressed_keys:
            parts.append("Alt")
        if self.main_key:
            parts.append(self.main_key.upper())
        
        if parts:
            self.status_label.setText(" + ".join(parts))
    
    def accept_hotkey(self):
        if self.main_key:
            parts = []
            if "ctrl" in self.pressed_keys:
                parts.append("ctrl")
            if "shift" in self.pressed_keys:
                parts.append("shift")
            if "alt" in self.pressed_keys:
                parts.append("alt")
            parts.append(self.main_key)
            
            hotkey_str = "+".join(parts)
            app_logger.info(f"Hotkey captured: {hotkey_str}")
            self.key_captured.emit(hotkey_str)
            self.accept()

# --- 8. MAIN APP ---
class WhisperRApp(QMainWindow):
    sig_toggle_vis = pyqtSignal()
    sig_toggle_rec = pyqtSignal()

    def __init__(self):
        super().__init__()
        app_logger.info(f"=== {APP_NAME} Application Starting ===")
        app_logger.info(f"Base directory: {BASE_DIR}")
        app_logger.info(f"Frozen: {getattr(sys, 'frozen', False)}")
        
        try:
            app_logger.debug("Initializing configuration...")
            self.config = AppConfig()
            self.recorder = None

            # Build UI FIRST so self.scratchpad exists before any worker thread
            # can emit log_msg and try to write to it.
            app_logger.debug("→ Building UI (setup_ui)...")
            self.setup_ui()
            app_logger.debug("✓ setup_ui() complete")

            app_logger.debug("→ Creating TranscriberWorker...")
            self.transcriber = TranscriberWorker(self.config)
            app_logger.debug("✓ TranscriberWorker created, connecting signals...")
            self.transcriber.finished_text.connect(self.on_text)
            self.transcriber.status_changed.connect(self.on_trans_status)
            self.transcriber.log_msg.connect(self._on_transcriber_log)
            app_logger.debug("→ Starting TranscriberWorker thread...")
            self.transcriber.start()
            app_logger.debug("✓ TranscriberWorker thread started")
            
            app_logger.debug("→ Creating StatusOverlay...")
            self.indicator = StatusOverlay(self.config)
            app_logger.debug("✓ StatusOverlay created")
            
            app_logger.debug("→ Connecting app-level signals...")
            self.sig_toggle_vis.connect(self.toggle_visibility_safe)
            self.sig_toggle_rec.connect(self.toggle_rec)
            app_logger.debug("✓ App-level signals connected")
            
            app_logger.debug("Setting up icons...")
            icon_path = os.path.join(BASE_DIR, "icon.png")
            app_logger.debug(f"  __init__: Looking for icon at: {icon_path}")
            app_logger.debug(f"  __init__: icon.png exists: {os.path.exists(icon_path)}")
            if os.path.exists(icon_path):
                # Build a multi-resolution QIcon so Windows taskbar, tray, and
                # alt-tab all use our image at the right size.
                app_icon = QIcon()
                base_pix = QPixmap(icon_path)
                for sz in (16, 24, 32, 48, 64, 128, 256):
                    app_icon.addPixmap(
                        base_pix.scaled(sz, sz,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                    )
                # Apply to this window AND the QApplication so the taskbar picks it up
                self.setWindowIcon(app_icon)
                QApplication.instance().setWindowIcon(app_icon)
                app_logger.info(f"Loaded icon from: {icon_path}")
            else:
                app_icon = QIcon()
                app_logger.warning("icon.png not found — no custom icon set")
            
            app_logger.debug("  __init__: Creating QSystemTrayIcon...")
            self.tray = QSystemTrayIcon(self)
            app_logger.debug(f"  __init__: QSystemTrayIcon created (id={id(self.tray)})")
            app_logger.debug("  __init__: Setting tray icon...")
            self.tray.setIcon(app_icon)
            app_logger.debug(f"  __init__: Tray icon set (isNull={self.tray.icon().isNull()})")
            
            app_logger.debug("  __init__: Creating tray context menu...")
            tm = QMenu()
            tm.addAction("Show/Restore", self.toggle_visibility_safe)
            tm.addAction("Quit", QApplication.instance().quit)
            self.tray.setContextMenu(tm)
            app_logger.debug("  __init__: Tray context menu set")
            
            app_logger.debug("  __init__: Calling tray.show()...")
            self.tray.show()
            app_logger.debug(f"  __init__: tray.show() called, tray.isVisible={self.tray.isVisible()}")
            
            # Double-click or single-click tray icon → restore window
            self.tray.activated.connect(self._on_tray_activated)
            
            app_logger.debug("→ Setting up hotkeys and listeners (setup_logic)...")
            self.setup_logic()
            app_logger.debug("✓ setup_logic() complete")
            
            app_logger.debug("→ Starting folder monitor timer...")
            self.m_timer = QTimer()
            self.m_timer.timeout.connect(self.monitor_dirs)
            self.m_timer.start(5000)
            app_logger.debug("✓ Folder monitor timer started")
            
            # No standalone pa_sys / meter_timer.
            # The live meter is driven purely by AudioRecorder.volume_out signal
            # (connected in toggle_rec). This avoids running a second PyAudio
            # instance alongside AudioRecorder's, which crashes in frozen mode.
            self.meter_stream = None  # kept for compat refs in toggle_rec cleanup
            
            app_logger.info("Application initialized successfully")
            self._model_loading   = True
            self._is_listening    = False
            self._speech_active   = False
            self._is_transcribing = False
            self._ptt_held: set   = set()  # tracks currently held keys for combo PTT
            self._last_paste_text: str = ""   # last pasted text (for rollback)
            self._rollback_pending: bool = False  # lowercase first result of next session
            self._rollback_armed:   bool = False  # set by rollback hotkey, consumed by toggle_rec
            self._ptt_starting: bool = False  # guard: recorder start in progress
            app_logger.info("Queueing initial model preload...")
            self.transcriber.preload_model()
            
        except Exception as e:
            app_logger.error(f"Critical error during initialization: {e}", exc_info=True)
            
            # Try to show error message to user
            try:
                QMessageBox.critical(
                    None,
                    "Initialization Error",
                    f"WhisperR failed to start:\n\n{e}\n\nCheck app_log.txt for details."
                )
            except:
                pass
            
            raise

    def toggle_visibility_safe(self):
        # Read from the live checkbox so unsaved changes are respected.
        if hasattr(self, 'cfg_tray'):
            min_to_tray = self.cfg_tray.isChecked()
        else:
            min_to_tray = self.config.settings.get("min_to_tray", False)
        app_logger.debug(f"toggle_visibility_safe: isHidden={self.isHidden()}, isMinimized={self.isMinimized()}, min_to_tray={min_to_tray}")
        if self.isHidden() or self.isMinimized():
            self.showNormal()
            self.activateWindow()
            self.raise_()
            app_logger.debug("toggle_visibility_safe: window restored")
        else:
            if min_to_tray:
                self.hide()
                app_logger.debug("toggle_visibility_safe: window hidden to tray")
            else:
                self.showMinimized()
                app_logger.debug("toggle_visibility_safe: window minimized to taskbar")

    def _on_tray_activated(self, reason):
        # Restore on double-click or single-click (Trigger)
        if reason in (QSystemTrayIcon.ActivationReason.DoubleClick,
                      QSystemTrayIcon.ActivationReason.Trigger):
            app_logger.debug(f"_on_tray_activated: reason={reason}, restoring window")
            self.showNormal()
            self.activateWindow()
            self.raise_()
    
    def showEvent(self, event):
        """Override showEvent to log when window is shown"""
        app_logger.debug("→ showEvent: Window about to be shown")
        try:
            super().showEvent(event)
            app_logger.debug("✓ showEvent: Window shown successfully")
        except Exception as e:
            app_logger.error(f"✗ showEvent crashed: {e}", exc_info=True)
            raise
    
    def paintEvent(self, event):
        """Thin override retained for crash safety only (debug logging removed)."""
        try:
            super().paintEvent(event)
        except Exception as e:
            app_logger.error(f"✗ paintEvent crashed: {e}", exc_info=True)
            raise
    
    def resizeEvent(self, event):
        """Override resizeEvent to log resizing"""
        app_logger.debug(f"→ resizeEvent: Resizing to {event.size().width()}x{event.size().height()}")
        try:
            super().resizeEvent(event)
            app_logger.debug("✓ resizeEvent: Resize complete")
        except Exception as e:
            app_logger.error(f"✗ resizeEvent crashed: {e}", exc_info=True)
            raise

    def setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(820, 650)
        
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        
        # ===== MAIN TAB =====
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        
        # Compact label with minimal space
        label_layout = QVBoxLayout()
        label_layout.setSpacing(0)
        label_layout.setContentsMargins(0, 0, 0, 20)  # 20px bottom margin
        logs_label = QLabel("Logs & Results:")
        logs_label.setStyleSheet("font-weight: bold;")
        label_layout.addWidget(logs_label)
        l1.addLayout(label_layout)
        
        # Textarea takes rest of space
        self.scratchpad = QTextEdit()
        self.scratchpad.setFont(QFont("Consolas", 9))
        l1.addWidget(self.scratchpad)  # No max height - takes all available space
        
        # ── App state indicator ──────────────────────────────────────
        # A simple coloured dot + label embedded in the main window.
        # Zero floating-window complexity — works perfectly in frozen mode.
        self.app_state_label = QLabel("● Idle")
        self.app_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.app_state_label.setFixedHeight(26)
        self.app_state_label.setStyleSheet(
            "color: #888888; font-size: 13px; font-weight: bold; "
            "background: #1e1e1e; border-radius: 4px; padding: 2px 8px;"
        )
        l1.addWidget(self.app_state_label)

        hb = QHBoxLayout()
        self.btn_toggle = QPushButton("Start Dictation")
        self.btn_toggle.setFixedHeight(40)
        self.btn_toggle.clicked.connect(self.toggle_rec)
        
        self.btn_import = QPushButton("Import Audio Files")
        self.btn_import.setFixedHeight(40)
        self.btn_import.clicked.connect(self.import_files)
        
        hb.addWidget(self.btn_toggle)
        hb.addWidget(self.btn_import)
        l1.addLayout(hb)
        
        self.tabs.addTab(t1, "Main")
        
        # ===== PROMPT TAB =====
        tp = QWidget()
        lp = QVBoxLayout(tp)
        
        lp.addWidget(QLabel("Whisper Steering Prompt (helps guide transcription):"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setText(self.config.settings["initial_prompt"])
        self.prompt_edit.setMaximumHeight(250)
        lp.addWidget(self.prompt_edit)
        
        hbp = QHBoxLayout()
        bi = QPushButton("Import .txt")
        bi.clicked.connect(self.import_p)
        be = QPushButton("Export .txt")
        be.clicked.connect(self.export_p)
        hbp.addWidget(bi)
        hbp.addWidget(be)
        lp.addLayout(hbp)
        lp.addStretch()
        
        self.tabs.addTab(tp, "AI Prompt")
        
        # ===== COMMANDS TAB =====
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        
        l2.addWidget(QLabel("Voice Commands (phrase detection → action):"))
        
        self.cmd_table = QTableWidget(0, 2)
        self.cmd_table.setHorizontalHeaderLabels(["Phrase to Detect", "Command to Execute"])
        self.cmd_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for k, v in self.config.settings["commands"].items():
            r = self.cmd_table.rowCount()
            self.cmd_table.insertRow(r)
            self.cmd_table.setItem(r, 0, QTableWidgetItem(k))
            self.cmd_table.setItem(r, 1, QTableWidgetItem(v))
        
        l2.addWidget(self.cmd_table)
        
        btn_row = QHBoxLayout()
        ba = QPushButton("Add Row")
        ba.clicked.connect(lambda: self.cmd_table.insertRow(self.cmd_table.rowCount()))
        
        bd = QPushButton("Delete Selected Row")
        bd.clicked.connect(self.delete_command_row)
        
        btn_row.addWidget(ba)
        btn_row.addWidget(bd)
        l2.addLayout(btn_row)
        
        self.tabs.addTab(t2, "Commands")
        
        # ===== SETTINGS TAB =====
        sc = QScrollArea()
        cw = QWidget()
        main_layout = QVBoxLayout(cw)
        
        # --- AI Model Settings ---
        ai_group = QGroupBox("AI Model Settings")
        ai_layout = QFormLayout()
        
        self.cfg_model = QComboBox()
        self.cfg_model.addItems(WHISPER_MODELS)
        self.cfg_model.setCurrentText(self.config.settings["model"])
        self.cfg_model.currentTextChanged.connect(self._on_model_changed)
        ai_layout.addRow("Whisper Model:", self.cfg_model)
        
        self.cfg_lang = QComboBox()
        self.cfg_lang.addItems(list(LANG_MAP.keys()))
        self.cfg_lang.setCurrentText(self.config.settings["lang_name"])
        ai_layout.addRow("Language:", self.cfg_lang)
        
        self.cfg_ts = QCheckBox("Include timestamps")
        self.cfg_ts.setChecked(self.config.settings["timestamps"])
        ai_layout.addRow(self.cfg_ts)
        
        self.cfg_trans = QCheckBox("Translation mode (to English)")
        self.cfg_trans.setChecked(self.config.settings["translate"])
        ai_layout.addRow(self.cfg_trans)
        
        ai_group.setLayout(ai_layout)
        main_layout.addWidget(ai_group)
        
        # --- Audio Input Settings ---
        audio_group = QGroupBox("Audio Input Settings")
        audio_layout = QFormLayout()
        
        self.cfg_mic = QComboBox()
        self.cfg_mic.setPlaceholderText("— Select microphone —")
        self.pop_mics()
        # Connect signal to reset meter when device changes
        self.cfg_mic.currentIndexChanged.connect(self.on_mic_changed)
        audio_layout.addRow("Microphone:", self.cfg_mic)
        
        self.cfg_dict_m = QComboBox()
        self.cfg_dict_m.addItems(["Continuous", "Auto-Pause"])
        self.cfg_dict_m.setCurrentText(self.config.settings["dict_mode"])
        audio_layout.addRow("Detection Mode:", self.cfg_dict_m)
        
        self.cfg_p_sec = QDoubleSpinBox()
        self.cfg_p_sec.setRange(0.1, 5.0)
        self.cfg_p_sec.setValue(self.config.settings["auto_pause_sec"])
        self.cfg_p_sec.setSuffix(" sec")
        audio_layout.addRow("Silence Threshold:", self.cfg_p_sec)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # --- Microphone Calibration ---
        cal_group = QGroupBox("Microphone Calibration")
        cal_layout = QVBoxLayout()
        
        cal_layout.addWidget(QLabel("Live Input Level:"))
        self.live_meter = QProgressBar()
        self.live_meter.setRange(0, 5000)
        cal_layout.addWidget(self.live_meter)
        
        self.btn_cal = QPushButton("Run Auto-Calibration")
        self.btn_cal.clicked.connect(self.start_cal)
        cal_layout.addWidget(self.btn_cal)
        
        self.cal_prog = QProgressBar()
        cal_layout.addWidget(self.cal_prog)
        
        self.lbl_cal = QLabel("Idle")
        cal_layout.addWidget(self.lbl_cal)
        
        levels_layout = QHBoxLayout()
        levels_layout.addWidget(QLabel("Noise Floor:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(0, 8000)
        self.n_spin.setValue(self.config.settings["noise_floor"])
        levels_layout.addWidget(self.n_spin)
        
        levels_layout.addWidget(QLabel("Speech Level:"))
        self.s_spin = QSpinBox()
        self.s_spin.setRange(0, 8000)
        self.s_spin.setValue(self.config.settings["speech_vol"])
        levels_layout.addWidget(self.s_spin)
        
        cal_layout.addLayout(levels_layout)
        cal_group.setLayout(cal_layout)
        main_layout.addWidget(cal_group)
        
        # --- Hotkeys ---
        hotkey_group = QGroupBox("Keyboard Shortcuts")
        hotkey_layout = QFormLayout()
        
        self.btn_hk1 = QPushButton(self.config.settings["hotkey"])
        self.btn_hk1.clicked.connect(lambda: self.cap_hk(self.btn_hk1, "hotkey"))
        hotkey_layout.addRow("Toggle Dictation:", self.btn_hk1)
        
        # Note about PTT
        ptt_info = QLabel("Note: PTT key will also function normally in other apps")
        ptt_info.setStyleSheet("color: #888; font-size: 8pt; font-style: italic;")
        ptt_info.setWordWrap(True)
        hotkey_layout.addRow(ptt_info)
        
        self.btn_hk_vis = QPushButton(self.config.settings["visibility_hotkey"])
        self.btn_hk_vis.clicked.connect(lambda: self.cap_hk(self.btn_hk_vis, "visibility_hotkey"))
        hotkey_layout.addRow("Show/Hide Window:", self.btn_hk_vis)

        self.btn_hk_rollback = QPushButton(self.config.settings.get("rollback_hotkey", "ctrl+shift+z"))
        self.btn_hk_rollback.clicked.connect(lambda: self.cap_hk(self.btn_hk_rollback, "rollback_hotkey"))
        self.btn_hk_rollback.setToolTip(
            "Erase trailing punctuation/fragment from the last transcription\n"
            "and position the cursor for seamless continuation."
        )
        hotkey_layout.addRow("Resume Transcription:", self.btn_hk_rollback)

        self.cfg_live_mode = QComboBox()
        self.cfg_live_mode.addItems(["Simple", "Push-To-Talk"])
        self.cfg_live_mode.setCurrentText(self.config.settings.get("live_mode", "Simple"))
        self.cfg_live_mode.setToolTip(
            "Simple: dictation starts/stops with the Toggle Dictation hotkey.\n"
            "Push-To-Talk: hold the PTT key to record; release to transcribe."
        )
        hotkey_layout.addRow("Dictation Mode:", self.cfg_live_mode)

        self.btn_hk2 = QPushButton(self.config.settings["ptt_key"])
        self.btn_hk2.clicked.connect(lambda: self.cap_hk(self.btn_hk2, "ptt_key"))
        hotkey_layout.addRow("Push-to-Talk Key:", self.btn_hk2)
        
        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)
        
        # --- Output & Behavior ---
        output_group = QGroupBox("Output & Behavior")
        output_layout = QFormLayout()
        
        self.cfg_p_win = QDoubleSpinBox()
        self.cfg_p_win.setRange(0.1, 5.0)
        self.cfg_p_win.setValue(self.config.settings["paste_delay"])
        self.cfg_p_win.setSuffix(" sec")
        output_layout.addRow("Paste Delay:", self.cfg_p_win)
        
        self.cfg_space = QCheckBox("Auto-append space after paste")
        self.cfg_space.setChecked(self.config.settings["auto_space"])
        output_layout.addRow(self.cfg_space)
        
        self.cfg_tray = QCheckBox("Minimize to system tray")
        self.cfg_tray.setChecked(self.config.settings["min_to_tray"])
        output_layout.addRow(self.cfg_tray)
        
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # --- File Storage ---
        storage_group = QGroupBox("File Storage")
        storage_layout = QFormLayout()
        
        rec_row = QHBoxLayout()
        self.cfg_folder = QLineEdit(self.config.settings["audio_folder"])
        b_f = QPushButton("Browse")
        b_f.clicked.connect(lambda: self.browse_f(self.cfg_folder))
        rec_row.addWidget(self.cfg_folder)
        rec_row.addWidget(b_f)
        storage_layout.addRow("Recordings Folder:", rec_row)
        
        mon_row = QHBoxLayout()
        self.cfg_mon = QLineEdit(self.config.settings["mon_folder"])
        b_m = QPushButton("Browse")
        b_m.clicked.connect(lambda: self.browse_f(self.cfg_mon))
        mon_row.addWidget(self.cfg_mon)
        mon_row.addWidget(b_m)
        storage_layout.addRow("Monitor Folder:", mon_row)
        
        self.cfg_ram = QCheckBox("RAM-only mode (no disk writes)")
        self.cfg_ram.setChecked(not self.config.settings["save_to_disk"])
        storage_layout.addRow(self.cfg_ram)
        
        self.cfg_clear = QCheckBox("Clear recordings on exit")
        self.cfg_clear.setChecked(self.config.settings["clear_exit"])
        storage_layout.addRow(self.cfg_clear)
        
        storage_group.setLayout(storage_layout)
        main_layout.addWidget(storage_group)
        
        # --- Visual Indicators ---
        visual_group = QGroupBox("Visual Indicators")
        visual_layout = QFormLayout()
        
        self.cfg_ind_show = QCheckBox("Enable status indicators")
        self.cfg_ind_show.setChecked(self.config.settings["ind_show"])
        visual_layout.addRow(self.cfg_ind_show)
        
        self.cfg_ind_type = QComboBox()
        self.cfg_ind_type.addItems(["Icons", "Bar", "Both"])
        self.cfg_ind_type.setCurrentText(self.config.settings["ind_type"])
        visual_layout.addRow("Indicator Type:", self.cfg_ind_type)
        
        self.cfg_ind_pos = QComboBox()
        self.cfg_ind_pos.addItems(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"])
        self.cfg_ind_pos.setCurrentText(self.config.settings["ind_pos"])
        visual_layout.addRow("Icon Position:", self.cfg_ind_pos)
        
        self.cfg_bar_edge = QComboBox()
        self.cfg_bar_edge.addItems(["Top", "Bottom", "Left", "Right"])
        self.cfg_bar_edge.setCurrentText(self.config.settings["bar_edge"])
        visual_layout.addRow("Bar Edge:", self.cfg_bar_edge)
        
        self.cfg_bar_thickness = QSpinBox()
        self.cfg_bar_thickness.setRange(1, 50)
        self.cfg_bar_thickness.setValue(self.config.settings.get("bar_thickness", 5))
        self.cfg_bar_thickness.setSuffix(" px")
        self.cfg_bar_thickness.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # Ensure it gets focus
        visual_layout.addRow("Bar Thickness:", self.cfg_bar_thickness)
        
        self.cfg_ind_sz = QSpinBox()
        self.cfg_ind_sz.setRange(16, 256)
        self.cfg_ind_sz.setValue(self.config.settings["ind_size"])
        self.cfg_ind_sz.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        visual_layout.addRow("Icon Size:", self.cfg_ind_sz)
        
        self.cfg_ind_off = QSpinBox()
        self.cfg_ind_off.setRange(0, 256)
        self.cfg_ind_off.setValue(self.config.settings["ind_off"])
        self.cfg_ind_off.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        visual_layout.addRow("Corner Offset:", self.cfg_ind_off)
        
        self.cfg_ind_hide_idle = QCheckBox("Hide indicators when idle (dictation off)")
        self.cfg_ind_hide_idle.setChecked(self.config.settings.get("ind_hide_idle", True))
        visual_layout.addRow(self.cfg_ind_hide_idle)
        
        visual_group.setLayout(visual_layout)
        main_layout.addWidget(visual_group)
        
        # --- Advanced ---
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout()
        
        self.cfg_log_level = QComboBox()
        self.cfg_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.cfg_log_level.setCurrentText(self.config.settings["log_level"])
        advanced_layout.addRow("Logging Level:", self.cfg_log_level)
        
        self.cfg_use_vad = QCheckBox("Use VAD (Voice Activity Detection)")
        self.cfg_use_vad.setChecked(self.config.settings.get("use_vad", False))
        self.cfg_use_vad.setToolTip("Filters out non-speech segments before transcription.\nReduces hallucinations on silence. Recommended for push-to-talk.")
        advanced_layout.addRow(self.cfg_use_vad)
        
        self.btn_setup = QPushButton("GPU Acceleration Setup Guide")
        self.btn_setup.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_setup.clicked.connect(self.setup_deps)
        advanced_layout.addRow(self.btn_setup)
        
        btn_open_log = QPushButton("Open Log File")
        btn_open_log.clicked.connect(self.open_log_file)
        advanced_layout.addRow(btn_open_log)
        
        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)
        
        # Save button
        btn_s = QPushButton("💾 SAVE ALL SETTINGS")
        btn_s.setFixedHeight(40)
        btn_s.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        btn_s.clicked.connect(self.save_cfg)
        main_layout.addWidget(btn_s)
        
        # Version label at bottom
        version_label = QLabel(f"{APP_NAME} v{__version__}")
        version_label.setStyleSheet("color: #666; font-size: 8pt; padding: 5px;")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(version_label)
        
        main_layout.addStretch()
        
        sc.setWidget(cw)
        sc.setWidgetResizable(True)
        self.tabs.addTab(sc, "Settings")

    def delete_command_row(self):
        current_row = self.cmd_table.currentRow()
        if current_row >= 0:
            self.cmd_table.removeRow(current_row)
            app_logger.debug(f"Deleted command row {current_row}")
        else:
            QMessageBox.warning(self, "No Selection", "Please select a row to delete.")

    def pop_mics(self):
        """Populate microphone dropdown with better device matching"""
        p = pyaudio.PyAudio()
        self.cfg_mic.clear()
        sel = -1  # -1 shows placeholder until a real device is matched
        saved_device = self.config.settings.get("input_device_name", "")

        app_logger.debug("Scanning audio input devices:")

        try:
            device_count = p.get_device_count()
        except Exception as e:
            app_logger.error(f"Failed to get device count: {e}")
            p.terminate()
            return

        # First pass: collect all valid input devices grouped by host API
        from collections import defaultdict
        groups = defaultdict(list)   # api_name -> [(pyaudio_idx, device_name)]
        seen_names = {}              # device_name -> api_name (for dedup)

        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                if info["maxInputChannels"] <= 0:
                    continue
                try:
                    h = p.get_host_api_info_by_index(info["hostApi"])["name"]
                except Exception:
                    h = "Unknown"
                device_name = info["name"]
                # Prefer WASAPI/MME over DirectSound for the same device
                if device_name in seen_names:
                    if "DirectSound" in h:
                        app_logger.debug(f"  Skipping duplicate DirectSound: {device_name}")
                        continue
                    # Replace the existing entry if the new one is WASAPI
                    prev_api = seen_names[device_name]
                    if "DirectSound" in prev_api:
                        groups[prev_api] = [(idx, n) for idx, n in groups[prev_api] if n != device_name]
                seen_names[device_name] = h
                groups[h].append((i, device_name))
                app_logger.debug(f"  Device {i}: {device_name} ({h})")
            except Exception as e:
                app_logger.warning(f"Error reading device {i}: {e}")

        # Second pass: populate combo with group headers + items
        # Sort groups: WASAPI first, then MME, then others, then DirectSound last
        def _api_sort_key(api):
            order = {"Windows WASAPI": 0, "MME": 1, "Windows DirectSound": 99}
            return order.get(api, 50)

        for api_name in sorted(groups.keys(), key=_api_sort_key):
            devices = groups[api_name]
            if not devices:
                continue
            # Add a disabled header item for this group
            header_idx = self.cfg_mic.count()
            self.cfg_mic.addItem(f"── {api_name} ──", -1)
            header_item = self.cfg_mic.model().item(header_idx)
            header_item.setEnabled(False)
            from PyQt6.QtGui import QFont
            f = header_item.font()
            f.setBold(True)
            header_item.setFont(f)

            for pyaudio_idx, device_name in devices:
                full_name = f"{device_name} ({api_name})"
                self.cfg_mic.addItem(f"  {device_name}", pyaudio_idx)
                app_logger.debug(f"  Added: {device_name} [{api_name}]")
                # Selection matching uses full_name for backward compat with saved settings
                if saved_device:
                    if saved_device == full_name or saved_device == f"  {device_name}":
                        sel = self.cfg_mic.count() - 1
                        app_logger.debug(f"  ✓ Exact match for saved device")
                    elif device_name in saved_device or saved_device in device_name:
                        sel = self.cfg_mic.count() - 1
                        app_logger.debug(f"  ✓ Partial match for saved device")
        
        if self.cfg_mic.count() == 0:
            app_logger.error("No input devices found!")
            self.cfg_mic.addItem("No microphone detected", -1)
        else:
            self.cfg_mic.setCurrentIndex(sel)  # -1 shows placeholder if no match
            app_logger.info(f"Selected device index: {sel}, name: {self.cfg_mic.currentText()}")
        
        p.terminate()

    def update_meter(self):
        """Meter is now driven by AudioRecorder.volume_out signal — no-op here."""
        pass


    def save_cfg(self):
        # Get reference to save button before any operations
        save_button = self.sender()
        
        # Visual feedback - change button temporarily
        save_button.setEnabled(False)
        save_button.setText("💾 Saving...")
        QApplication.processEvents()  # Force UI update
        
        # Collect commands from table
        cmds = {}
        for r in range(self.cmd_table.rowCount()):
            phrase_item = self.cmd_table.item(r, 0)
            cmd_item = self.cmd_table.item(r, 1)
            if phrase_item and cmd_item:
                phrase = phrase_item.text().strip()
                cmd = cmd_item.text().strip()
                if phrase and cmd:
                    cmds[phrase] = cmd
        
        # Update all settings
        self.config.settings.update({
            "model": self.cfg_model.currentText(),
            "lang_name": self.cfg_lang.currentText(),
            "lang_code": LANG_MAP[self.cfg_lang.currentText()],
            "audio_folder": self.cfg_folder.text(),
            "mon_folder": self.cfg_mon.text(),
            "clear_exit": self.cfg_clear.isChecked(),
            "save_to_disk": not self.cfg_ram.isChecked(),
            "input_device_name": self.cfg_mic.currentText().strip(),
            "input_device_index": self.cfg_mic.currentData() if self.cfg_mic.currentData() != -1 else None,
            "dict_mode": self.cfg_dict_m.currentText(),
            "auto_pause_sec": self.cfg_p_sec.value(),
            "paste_delay": self.cfg_p_win.value(),
            "hotkey": self.btn_hk1.text(),
            "ptt_key": self.btn_hk2.text(),
            "live_mode": self.cfg_live_mode.currentText(),
            "rollback_hotkey": self.btn_hk_rollback.text(),
            "visibility_hotkey": self.btn_hk_vis.text(),
            "noise_floor": self.n_spin.value(),
            "speech_vol": self.s_spin.value(),
            "commands": cmds,
            "initial_prompt": self.prompt_edit.toPlainText(),
            "min_to_tray": self.cfg_tray.isChecked(),
            "auto_space": self.cfg_space.isChecked(),
            "ind_show": self.cfg_ind_show.isChecked(),
            "ind_type": self.cfg_ind_type.currentText(),
            "ind_pos": self.cfg_ind_pos.currentText(),
            "bar_edge": self.cfg_bar_edge.currentText(),
            "ind_size": self.cfg_ind_sz.value(),
            "ind_off": self.cfg_ind_off.value(),
            "bar_thickness": self.cfg_bar_thickness.value(),
            "ind_hide_idle": self.cfg_ind_hide_idle.isChecked(),
            "ind_opacity": 220,
            "bar_opacity": 220,
            "timestamps": self.cfg_ts.isChecked(),
            "translate": self.cfg_trans.isChecked(),
            "log_level": self.cfg_log_level.currentText(),
            "use_vad": self.cfg_use_vad.isChecked()
        })
        
        try:
            self.config.save()
            app_logger.set_level(self.config.settings["log_level"])
            self.scratchpad.append("✓ Settings saved successfully")
            
            # Visual feedback - success
            save_button.setText("✓ SAVED!")
            save_button.setStyleSheet("background-color: #27ae60; color: white; font-weight: bold;")
            save_button.setEnabled(True)
            
            # Reset after delay
            QTimer.singleShot(1500, lambda: self.reset_save_button(save_button))
            
            # Restart hotkey listeners with new keys
            self.setup_logic()
            
        except Exception as e:
            app_logger.error(f"Failed to save settings: {e}", exc_info=True)
            self.scratchpad.append(f"✗ Failed to save settings: {e}")
            
            # Visual feedback - error
            save_button.setText("✗ SAVE FAILED")
            save_button.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
            save_button.setEnabled(True)
            
            # Reset after delay
            QTimer.singleShot(2000, lambda: self.reset_save_button(save_button))
    
    def reset_save_button(self, button):
        """Reset save button to original state"""
        try:
            button.setText("💾 SAVE ALL SETTINGS")
            button.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold;")
        except RuntimeError:
            # Widget was deleted, ignore
            pass

    def start_cal(self):
        if self.recorder and self.recorder.active:
            QMessageBox.warning(self, "Recording Active", "Stop dictation before calibrating.")
            return
        
        # meter_stream is always None (meter driven by volume_out signal)
        self.meter_stream = None
        
        self.btn_cal.setEnabled(False)
        self.cal_w = CalibrationWorker(self.cfg_mic.currentData())
        self.cal_w.progress.connect(self.cal_prog.setValue)
        self.cal_w.status_msg.connect(self.lbl_cal.setText)
        self.cal_w.finished.connect(self.on_calibration_finished)
        self.cal_w.start()
    
    def on_calibration_finished(self, noise, speech):
        self.n_spin.setValue(noise)
        self.s_spin.setValue(speech)
        self.btn_cal.setEnabled(True)
        self.lbl_cal.setText("✓ Calibration complete")
        app_logger.info(f"Calibration complete: noise={noise}, speech={speech}")
    
    def on_mic_changed(self):
        """Handle microphone device change - reset meter stream"""
        # Skip group header items (they have pyaudio_idx == -1)
        if self.cfg_mic.currentData() == -1:
            # Jump forward to the next real device
            next_idx = self.cfg_mic.currentIndex() + 1
            if next_idx < self.cfg_mic.count():
                self.cfg_mic.setCurrentIndex(next_idx)
            return
        app_logger.info(f"Microphone changed to: {self.cfg_mic.currentText()}")
        
        # Close existing meter stream
        if self.meter_stream:
            try:
                self.meter_stream.close()
                app_logger.debug("Closed old meter stream")
            except Exception as e:
                app_logger.debug(f"Error closing meter stream: {e}")
            finally:
                self.meter_stream = None

    def toggle_rec(self):
        app_logger.debug(f"→ toggle_rec: recorder={self.recorder}, recorder.active={self.recorder.active if self.recorder else 'N/A'}")
        if self.recorder and self.recorder.active:
            app_logger.info("toggle_rec: Stopping dictation")
            self.recorder.active = False
            self.btn_toggle.setText("Start Dictation")
            self._is_listening  = False
            self._speech_active = False
            self._update_app_state()
            app_logger.info("Dictation stopped")
        else:
            app_logger.info("toggle_rec: Starting dictation")
            # Rollback handshake:
            # rollback_transcription() sets _rollback_armed (survives this block).
            # We consume it here and set _rollback_pending for this session only.
            # Any stale _rollback_pending from a previous no-speech session is cleared.
            self._rollback_pending = bool(getattr(self, '_rollback_armed', False))
            self._rollback_armed   = False
            if self._rollback_pending:
                app_logger.debug("toggle_rec: rollback armed — will lowercase first result")
            # Always kill any existing recorder first, even if it claims inactive.
            # A PTT recorder-storm can leave orphaned recorders with active=False
            # that are still mid-startup — stopping them prevents resource leaks
            # and the "stuck on listening" state caused by multiple simultaneous streams.
            if self.recorder is not None:
                try:
                    self.recorder.active = False
                except Exception:
                    pass
                self.recorder = None
            if self.meter_stream:
                app_logger.debug("toggle_rec: Closing meter stream before starting recorder")
                self.meter_stream.close()
                self.meter_stream = None
            
            self.recorder = AudioRecorder(self.config)
            app_logger.debug(f"toggle_rec: AudioRecorder created, id={id(self.recorder)}")
            
            self.recorder.data_ready.connect(lambda d: self.transcriber.submit(d, "live"))
            app_logger.debug("toggle_rec: data_ready signal connected")
            
            # FIX: Debounce speech_active signal to prevent rapid mic icon flickering.
            # The AudioRecorder emits speech_active on EVERY audio chunk that crosses the
            # RMS threshold, which can fire 10-20x per second. Instead of directly setting
            # is_rec, we route it through _on_speech_active which only repaints when the
            # state actually changes, eliminating the rapid icon flicker.
            self.recorder.speech_active.connect(self._on_speech_active)
            app_logger.debug("toggle_rec: speech_active signal connected to debounced handler")
            
            self.recorder.volume_out.connect(self.live_meter.setValue)
            app_logger.debug("toggle_rec: volume_out signal connected")
            
            self.recorder.start()
            app_logger.debug("toggle_rec: AudioRecorder thread started")
            
            self.btn_toggle.setText("⏹ STOP DICTATION")
            self._is_listening = True
            self._update_app_state()
            app_logger.info("Dictation started")
        app_logger.debug("✓ toggle_rec: Complete")
    
    def _on_speech_active(self, active):
        app_logger.debug(f"→ _on_speech_active: active={active}")
        if active != self._speech_active:
            self._speech_active = active
            self._update_app_state()
        app_logger.debug("✓ _on_speech_active: Complete")

    # ── PTT polling (GetAsyncKeyState) ──────────────────────────────────────
    # pynput keyboard.Listener uses SetWindowsHookEx which silently fails in
    # frozen --noconsole apps (hook thread has no Win32 message pump).
    # GetAsyncKeyState polling on a QTimer is simpler and works everywhere.

    # Map canonical key names to Windows Virtual Key codes
    _VK_MAP = {
        'ctrl': 0x11, 'shift': 0x10, 'alt': 0x12,
        'space': 0x20, 'enter': 0x0D, 'tab': 0x09,
        'esc': 0x1B, 'backspace': 0x08, 'delete': 0x2E,
        'f1': 0x70, 'f2': 0x71, 'f3': 0x72, 'f4': 0x73,
        'f5': 0x74, 'f6': 0x75, 'f7': 0x76, 'f8': 0x77,
        'f9': 0x78, 'f10': 0x79, 'f11': 0x7A, 'f12': 0x7B,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
        'home': 0x24, 'end': 0x23, 'page_up': 0x21, 'page_down': 0x22,
        'insert': 0x2D, 'cmd': 0x5B,
        # A-Z
        **{chr(c): ord(chr(c).upper()) for c in range(ord('a'), ord('z')+1)},
        # 0-9
        **{str(d): ord(str(d)) for d in range(10)},
    }

    def _setup_ptt_poll(self):
        """Create and start the QTimer that polls PTT key state."""
        if hasattr(self, '_ptt_timer') and self._ptt_timer is not None:
            self._ptt_timer.stop()
        self._ptt_timer = QTimer(self)
        self._ptt_timer.setInterval(30)  # 30ms poll = ~33 checks/sec, negligible CPU
        self._ptt_timer.timeout.connect(self._poll_ptt)
        self._ptt_timer.start()
        self.ptt_l = self._ptt_timer  # keep ptt_l set so setup_logic doesn't re-run
        app_logger.debug(f"  PTT poll timer started (30ms interval)")

    def _poll_ptt(self):
        """Poll Win32 GetAsyncKeyState for the PTT combo. Called by QTimer."""
        try:
            if self.config.settings.get("live_mode") != "Push-To-Talk":
                return
            ptt_key = self.config.settings.get("ptt_key", "")
            if not ptt_key:
                return
            parts = [p.strip().lower() for p in ptt_key.split('+')]
            import ctypes
            user32 = ctypes.windll.user32
            combo_held = all(
                user32.GetAsyncKeyState(self._VK_MAP.get(p, 0)) & 0x8000
                for p in parts if p in self._VK_MAP
            )
            if combo_held:
                # Clear the starting guard once the recorder is confirmed running.
                # recorder.active is set to True inside AudioRecorder.run() which
                # runs on a separate thread — it won't be True yet on the very next
                # poll tick after toggle_rec() returns, hence the guard.
                if self._ptt_starting and self.recorder and self.recorder.active:
                    self._ptt_starting = False

                # Only auto-start if not already running AND not mid-start.
                # Without this guard, every 30ms poll tick would spawn a new
                # AudioRecorder thread (causing a recorder storm that crashes the app).
                if not self._ptt_starting and (not self.recorder or not self.recorder.active):
                    app_logger.debug("PTT held — auto-starting recorder session")
                    self._ptt_starting = True
                    self.toggle_rec()

                if self.recorder and not self.recorder.ptt_pressed:
                    self.recorder.ptt_pressed = True
                    app_logger.debug("PTT activated (poll)")
            else:
                # PTT released — clear guard so next press can start fresh
                self._ptt_starting = False
                if self.recorder and self.recorder.ptt_pressed:
                    self.recorder.ptt_pressed = False
                    app_logger.debug("PTT deactivated (poll)")
        except Exception as e:
            app_logger.error(f"PTT poll error: {e}", exc_info=True)

    def rollback_transcription(self):
        """Strip trailing punctuation/ellipsis from the last pasted transcription
        so the user can speak again and continue the sentence seamlessly.

        Use case: Whisper outputs "I was going to..." → user presses rollback →
        app sends 4 backspaces (erasing "...") leaving "I was going to " →
        user speaks "finish the thought" → gets pasted as "finish the thought"
        (lowercased) → final text: "I was going to finish the thought".

        Only trailing punctuation and whitespace are erased.  If the transcription
        ended cleanly (no trailing punct), the trailing space is still removed so
        the next paste joins without a double-space — but NO words are deleted.
        """
        import re, time as _time
        txt = self._last_paste_text
        if not txt:
            app_logger.debug("rollback: nothing to roll back")
            return

        # Strip only trailing whitespace + punctuation/ellipsis.
        # We NEVER delete words — if there is no trailing junk, we just
        # remove the trailing space so the next paste joins cleanly.
        stripped = txt.rstrip()                                    # drop trailing whitespace
        stripped = re.sub(r'[.,;:!?…\-]+$', '', stripped)         # drop trailing punctuation
        stripped = stripped.rstrip()                               # drop any whitespace before punct

        # chars to delete = everything we pasted after the clean word boundary
        chars_to_delete = len(txt) - len(stripped)

        # Always delete at least the trailing space even if no punct was found,
        # so the next dictation doesn't double-space. But never delete words.
        if chars_to_delete == 0:
            # txt had no trailing punct and no trailing space (shouldn't happen,
            # but guard anyway — nothing useful to do)
            app_logger.debug("rollback: no trailing junk to erase — nothing to do")
            return

        app_logger.info(f"rollback: erasing {chars_to_delete} chars — '{txt[-chars_to_delete:]!r}'")
        try:
            import pyautogui as _pag
            import ctypes as _ct
            # The rollback hotkey (e.g. ctrl+shift+z) fires this callback while
            # ctrl and shift are still physically held down by the user.
            # Sending bare backspaces in that state causes apps to interpret them
            # as ctrl+backspace (delete whole word) or ctrl+shift+backspace
            # (delete line / undo), wiping far more than intended.
            #
            # Fix: release all modifier keys via Win32 keybd_event before sending
            # backspaces, then give the target window a moment to settle.
            _VK_CONTROL = 0x11
            _VK_SHIFT   = 0x10
            _VK_MENU    = 0x12   # Alt
            _KEYEVENTF_KEYUP = 0x0002
            for _vk in (_VK_CONTROL, _VK_SHIFT, _VK_MENU):
                _ct.windll.user32.keybd_event(_vk, 0, _KEYEVENTF_KEYUP, 0)
            _time.sleep(0.08)   # let the target window process the key-ups

            for _ in range(chars_to_delete):
                _pag.press('backspace')
            # Leave one space so the next paste joins as "word nextword"
            _pag.press('space')
            self._last_paste_text = stripped + " "
            # Arm the lowercase flag for the next recording session.
            self._rollback_armed = True
            self._rollback_pending = False  # consumed by toggle_rec on session start
            self.scratchpad.append(
                f"[Rollback] Removed trailing {chars_to_delete} char(s): '{txt[-chars_to_delete:]!r}'"
            )
        except Exception as e:
            app_logger.error(f"rollback error: {e}", exc_info=True)

    def key_to_string(self, key):
        """Convert pynput key to string format (kept for hotkey capture)."""
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        elif hasattr(key, 'name'):
            return key.name.lower()
        return str(key).lower()
    
    # ── State indicator helpers ──────────────────────────────────────────────

    def _on_transcriber_log(self, msg: str):
        """Route TranscriberWorker log messages to the scratchpad and update
        the loading indicator."""
        self.scratchpad.append(f"[System] {msg}")
        loading_words = ("Loading ", "GPU unavailable", "loading on CPU",
                         "Trying CPU", "Trying CUDA", "Importing")
        done_words    = ("✓", "ready", "failed", "error", "Error", "Import error")
        crash_restart = "restarted after crash"

        if crash_restart in msg:
            # Worker restarted after a CUDA crash — re-queue preload on CPU
            self._model_loading = True
            self._update_app_state()
            self.transcriber.preload_model()
        elif any(w in msg for w in loading_words):
            self._model_loading = True
            self._update_app_state()
        elif any(w in msg for w in done_words):
            self._model_loading = False
            self._update_app_state()

    def _on_model_changed(self, model_name: str):
        """Called when the model dropdown selection changes.
        Saves new selection and pre-warms the model immediately."""
        if not hasattr(self, 'transcriber'):
            return
        app_logger.info(f"Model changed to: {model_name} — preloading")
        self._model_loading = True
        self._update_app_state()
        self.scratchpad.append(f"[System] Model changed to {model_name} — loading in background...")
        self.transcriber.preload_model()

    def _make_tray_icon(self, state: str) -> QIcon:
        """Draw a coloured circle QIcon for the system tray.
        Pure QPainter — no external files, works in frozen mode.
        
        state: 'idle' | 'loading' | 'recording' | 'transcribing' | 'both'
        """
        COLORS = {
            'idle':         (100, 100, 100),   # grey
            'loading':      (200, 140,  20),   # amber
            'recording':    (220,  40,  40),   # red
            'transcribing': ( 40, 100, 220),   # blue
            'both':         (160,  40, 190),   # purple
        }
        r, g, b = COLORS.get(state, COLORS['idle'])
        sz = 64
        pix = QPixmap(sz, sz)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor(r, g, b))
        p.setPen(QPen(QColor(255, 255, 255, 160), 3))
        p.drawEllipse(4, 4, sz - 8, sz - 8)
        p.end()
        return QIcon(pix)

    def _update_app_state(self):
        """Refresh the in-window state label and tray icon to reflect current state.
        
        Priority: loading > recording+transcribing > recording > transcribing > idle
        Called from on_trans_status, _on_speech_active, toggle_rec, _on_transcriber_log.
        """
        loading  = getattr(self, '_model_loading',  False)
        is_list  = getattr(self, '_is_listening',   False)  # dictation armed
        is_rec   = getattr(self, '_speech_active',  False)  # currently speaking
        is_trans = getattr(self, '_is_transcribing', False) # processing audio

        if loading:
            state = 'loading'
            dot   = '⏳'
            text  = 'Loading model...'
            color = '#ffaa00'
            tip   = 'WhisperR — Loading model'
        elif is_rec and is_trans:
            state = 'both'
            dot   = '●'
            text  = 'Recording + Transcribing'
            color = '#cc44ff'
            tip   = 'WhisperR — Recording + Transcribing'
        elif is_rec:
            state = 'recording'
            dot   = '●'
            text  = 'Recording'
            color = '#ff4444'
            tip   = 'WhisperR — Recording'
        elif is_trans:
            state = 'transcribing'
            dot   = '●'
            text  = 'Transcribing...'
            color = '#4488ff'
            tip   = 'WhisperR — Transcribing'
        elif is_list:
            state = 'listening'
            dot   = '●'
            text  = 'Listening...'
            color = '#28b450'
            tip   = 'WhisperR — Listening'
        else:
            state = 'idle'
            dot   = '●'
            text  = 'Idle'
            color = '#666666'
            tip   = 'WhisperR — Idle'

        # In-window label (always visible when window is open)
        if hasattr(self, 'app_state_label'):
            self.app_state_label.setText(f"{dot} {text}")
            self.app_state_label.setStyleSheet(
                f"color: {color}; font-size: 13px; font-weight: bold; "
                "background: #1e1e1e; border-radius: 4px; padding: 2px 8px;"
            )

        # Bar + icon overlay — mirrors the window label exactly
        if hasattr(self, 'indicator'):
            self.indicator.set_state(state)

        # Tray icon (visible even when window is hidden / minimised to tray)
        if hasattr(self, 'tray'):
            self.tray.setIcon(self._make_tray_icon(state))
            self.tray.setToolTip(tip)

    def on_trans_status(self, active):
        app_logger.debug(f"→ on_trans_status: active={active}")
        self._model_loading   = False
        self._is_transcribing = active
        self._update_app_state()
        app_logger.debug("✓ on_trans_status: done")
    
    def on_text(self, text, src):
        timestamp = datetime.now().strftime('%H:%M:%S')
        app_logger.debug(f"→ on_text: src='{src}', text length={len(text)}, text='{text[:60]}{'...' if len(text)>60 else ''}'")
        self.scratchpad.append(f"[{timestamp}] {text}")
        
        if src == "live":
            # ── Command detection (runs BEFORE paste) ────────────────────────
            # If the transcribed text matches a voice command, execute it and
            # do NOT paste the text — the spoken phrase was a command, not prose.
            _cmd_fired = False
            app_logger.debug(f"  on_text: Checking {len(self.config.settings['commands'])} voice commands...")
            for phrase, cmd in self.config.settings["commands"].items():
                if phrase.lower() in text.lower():
                    app_logger.debug(f"  on_text: Command matched: '{phrase}' → '{cmd}'")
                    try:
                        subprocess.Popen(cmd, shell=True)
                        app_logger.info(f"Command executed: {cmd}")
                        self.scratchpad.append(f"[Command] {phrase} → {cmd}")
                        _cmd_fired = True
                    except Exception as e:
                        app_logger.error(f"Command execution failed: {e}", exc_info=True)

            if _cmd_fired:
                app_logger.debug("  on_text: Command fired — skipping paste")
            else:
                # ── Rollback buffer ───────────────────────────────────────────
                # Store the last transcription so the resume-transcription hotkey
                # can send backspaces to erase it and rejoin the previous sentence.
                auto_space = self.config.settings["auto_space"]
                # If a rollback just happened, the next transcription needs its
                # first letter lowercased — Whisper always capitalises new sentences.
                if self._rollback_pending and text:
                    text = text[0].lower() + text[1:]
                    self._rollback_pending = False
                p_text = text + " " if auto_space else text
                # Strip trailing punctuation/spaces to find the last real word,
                # then record how many characters were actually output.
                self._last_paste_len = len(p_text)
                self._last_paste_text = p_text

                paste_delay = self.config.settings["paste_delay"]
                app_logger.debug(f"  on_text: auto_space={auto_space}, paste_delay={paste_delay}s, p_text length={len(p_text)}")
                try:
                    pyperclip.copy(p_text)
                    time.sleep(paste_delay)
                    pyautogui.hotkey('ctrl', 'v')
                    app_logger.info(f"Text pasted: '{text[:30]}{'...' if len(text)>30 else ''}'")
                except Exception as e:
                    app_logger.error(f"Paste error: {e}", exc_info=True)
        
        app_logger.debug("✓ on_text: Complete")

    def setup_logic(self):
        app_logger.debug(f"→ setup_logic: Starting. has hk_l={hasattr(self, 'hk_l')}, has ptt_l={hasattr(self, 'ptt_l')}")
        
        # Stop existing listeners
        if hasattr(self, 'hk_l'):
            try:
                app_logger.debug(f"  setup_logic: Stopping existing hk_l (id={id(self.hk_l)})...")
                self.hk_l.stop()
                app_logger.debug("  setup_logic: Stopped old hotkey listener")
            except Exception as e:
                app_logger.debug(f"  setup_logic: Error stopping hotkey listener: {e}")
        
        if hasattr(self, 'ptt_l'):
            try:
                app_logger.debug(f"  setup_logic: Stopping existing ptt_l (id={id(self.ptt_l)})...")
                self.ptt_l.stop()
                app_logger.debug("  setup_logic: Stopped old PTT listener")
            except Exception as e:
                app_logger.debug(f"  setup_logic: Error stopping PTT listener: {e}")
        
        # Create hotkey mapping (for toggle dictation and show/hide)
        hotkey_map = {}
        
        try:
            raw_toggle = self.config.settings["hotkey"]
            raw_vis = self.config.settings["visibility_hotkey"]
            app_logger.debug(f"  setup_logic: Raw hotkeys: toggle='{raw_toggle}', visibility='{raw_vis}'")
            
            toggle_hotkey     = self.normalize_hotkey(raw_toggle)
            visibility_hotkey = self.normalize_hotkey(raw_vis)
            raw_rollback      = self.config.settings.get("rollback_hotkey", "")
            rollback_hotkey   = self.normalize_hotkey(raw_rollback) if raw_rollback else None
            app_logger.debug(f"  setup_logic: Normalized hotkeys: toggle='{toggle_hotkey}', visibility='{visibility_hotkey}', rollback='{rollback_hotkey}'")
            
            # Store hotkeys for conflict checking
            self.toggle_hotkey_normalized = toggle_hotkey
            self.visibility_hotkey_normalized = visibility_hotkey
            app_logger.debug(f"  setup_logic: Stored normalized hotkeys on self")
            
            hotkey_map[toggle_hotkey]     = self.on_toggle_hotkey
            hotkey_map[visibility_hotkey] = self.on_visibility_hotkey
            if rollback_hotkey:
                hotkey_map[rollback_hotkey] = self.rollback_transcription
            app_logger.debug(f"  setup_logic: hotkey_map = {list(hotkey_map.keys())}")
            
            app_logger.debug("  setup_logic: Creating GlobalHotKeys listener...")
            self.hk_l = keyboard.GlobalHotKeys(hotkey_map)
            app_logger.debug(f"  setup_logic: GlobalHotKeys created (id={id(self.hk_l)}), starting...")
            self.hk_l.start()
            app_logger.info(f"Hotkeys registered: {list(hotkey_map.keys())}")
        except Exception as e:
            app_logger.error(f"Hotkey registration failed: {e}", exc_info=True)
            QMessageBox.warning(
                self, 
                "Hotkey Error", 
                f"Failed to register hotkeys:\n{e}\n\nPlease check your hotkey settings."
            )
        
        # PTT polling via GetAsyncKeyState (Win32) or key state query.
        # We deliberately avoid pynput keyboard.Listener here because it uses
        # SetWindowsHookEx which requires the hook thread to run its own Win32
        # message pump — this silently fails in frozen (--noconsole) apps.
        # GetAsyncKeyState polling on a QTimer runs on the Qt main thread and
        # works reliably in all deployment modes.
        try:
            ptt_key_setting = self.config.settings.get('ptt_key', 'NOT SET')
            app_logger.debug(f"  setup_logic: Setting up PTT polling, ptt_key='{ptt_key_setting}'")
            self._ptt_held.clear()
            self._setup_ptt_poll()
            app_logger.info("PTT listener started successfully")
        except Exception as e:
            app_logger.error(f"PTT listener failed to start: {e}", exc_info=True)
            self.ptt_l = None
            app_logger.warning("Continuing without PTT listener")
        
        app_logger.debug("✓ setup_logic: Complete")
    
    def on_toggle_hotkey(self):
        """Handler for toggle dictation hotkey - prevents subset conflicts"""
        app_logger.debug("Toggle dictation hotkey triggered (exact match)")
        self.sig_toggle_rec.emit()
    
    def on_visibility_hotkey(self):
        """Handler for visibility hotkey - prevents subset conflicts"""
        app_logger.debug("Visibility hotkey triggered (exact match)")
        self.sig_toggle_vis.emit()
    
    def normalize_hotkey(self, hotkey_str):
        """Convert our hotkey format to pynput format"""
        # Our format: "ctrl+shift+w"
        # pynput format: "<ctrl>+<shift>+w"
        
        parts = hotkey_str.lower().split('+')
        normalized = []
        
        for part in parts:
            part = part.strip()
            if part in ['ctrl', 'shift', 'alt', 'cmd', 'win']:
                normalized.append(f'<{part}>')
            else:
                # Check if it's a function key
                if part.startswith('f') and len(part) > 1 and part[1:].isdigit():
                    normalized.append(f'<{part}>')
                else:
                    normalized.append(part)
        
        result = '+'.join(normalized)
        app_logger.debug(f"Normalized hotkey '{hotkey_str}' to '{result}'")
        return result

    def monitor_dirs(self):
        root = Path(self.config.settings["mon_folder"])
        
        if not root.exists():
            return
        
        proc_dir = root / "Processed"
        proc_dir.mkdir(exist_ok=True)
        
        for f in root.glob("*.*"):
            if f.suffix.lower() in ['.wav', '.mp3', '.m4a'] and f.parent != proc_dir:
                try:
                    target = proc_dir / f.name
                    shutil.move(str(f), str(target))
                    self.transcriber.submit(str(target.absolute()), "file")
                    app_logger.info(f"File moved to processing: {f.name}")
                except Exception as e:
                    app_logger.error(f"Failed to process file {f.name}: {e}")
    
    def setup_deps(self):
        """Show guide for downloading GPU dependencies instead of auto-download (URLs keep breaking)"""
        guide_text = """
<b>GPU Acceleration Setup Guide</b>

<b>For NVIDIA GPU users:</b>

The app needs CUDA libraries for GPU acceleration. Unfortunately, download URLs keep changing, so here's how to get them manually:

<b>Option 1: Download cuDNN (Recommended)</b>
1. Visit: <a href='https://developer.nvidia.com/cudnn-downloads'>https://developer.nvidia.com/cudnn-downloads</a>
2. Download cuDNN for Windows
3. Extract the ZIP file
4. Copy ALL .dll files to the WhisperR folder (same folder as WhisperR.exe)
5. Restart WhisperR

<b>Option 2: Use CPU Mode</b>
CPU mode works perfectly fine - just slower:
- No setup needed
- Works on all systems
- Good for "base" and "small" models

<b>AMD GPU Users:</b>
AMD GPU support requires ROCm (complex setup). Recommend using CPU mode instead.

<b>Already have CUDA installed?</b>
If you have CUDA toolkit or other GPU apps, WhisperR might already work with GPU!

<b>How to verify GPU is working:</b>
After placing DLL files, check the logs when transcribing:
• "Model loaded successfully on GPU" = GPU working! ✓
• "GPU Failed... Using CPU" = CPU mode (still works fine)

<b>Note:</b> The "tiny" and "base" models work great on CPU for real-time dictation!
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("GPU Acceleration Setup")
        msg_box.setTextFormat(Qt.TextFormat.RichText)
        msg_box.setText(guide_text)
        msg_box.setIcon(QMessageBox.Icon.Information)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        msg_box.exec()
        
        app_logger.info("GPU setup guide displayed")
    
    def open_log_file(self):
        try:
            if os.path.exists(app_logger.log_path):
                if os.name == 'nt':
                    os.startfile(app_logger.log_path)
                else:
                    subprocess.Popen(['xdg-open', app_logger.log_path])
            else:
                QMessageBox.information(self, "No Log File", "Log file not found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open log file:\n{e}")

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Audio Files", 
            "", 
            "Audio Files (*.wav *.mp3 *.m4a *.mp4)"
        )
        
        for p in paths:
            self.transcriber.submit(os.path.abspath(p), "file")
            app_logger.info(f"File imported for transcription: {p}")

    def browse_f(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit.setText(path)

    def cap_hk(self, button, config_key):
        dialog = HotkeyCaptureDialog(self)
        dialog.key_captured.connect(button.setText)
        dialog.exec()

    def import_p(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Prompt", "", "Text Files (*.txt)")
        if path:
            try:
                self.prompt_edit.setText(Path(path).read_text(encoding='utf-8'))
                app_logger.info(f"Prompt imported from {path}")
            except Exception as e:
                app_logger.error(f"Failed to import prompt: {e}")

    def export_p(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Prompt", "", "Text Files (*.txt)")
        if path:
            try:
                Path(path).write_text(self.prompt_edit.toPlainText(), encoding='utf-8')
                app_logger.info(f"Prompt exported to {path}")
            except Exception as e:
                app_logger.error(f"Failed to export prompt: {e}")
    
    def closeEvent(self, event):
        app_logger.info("closeEvent: X button pressed — shutting down application")
        
        # Clean up recordings if requested
        if self.config.settings["clear_exit"]:
            try:
                folder = Path(self.config.settings["audio_folder"])
                if folder.exists():
                    for f in folder.glob("*.wav"):
                        f.unlink()
                    app_logger.info("Recordings cleared on exit")
            except Exception as e:
                app_logger.error(f"Failed to clear recordings: {e}")
        
        # Stop workers
        if self.recorder:
            self.recorder.active = False
            app_logger.debug("closeEvent: recorder stopped")
        
        self.transcriber.running = False
        self.transcriber._stop_worker()
        app_logger.debug("closeEvent: transcriber stopped")
        
        # Hide tray icon so it doesn't linger in the system tray after exit
        if hasattr(self, 'tray'):
            self.tray.hide()
            app_logger.debug("closeEvent: tray icon hidden")
        
        # Destroy the overlay widget so it doesn't outlive the main window
        if hasattr(self, 'indicator'):
            self.indicator.hide_all()
            self.indicator.deleteLater()
            app_logger.debug("closeEvent: overlay hidden and scheduled for deletion")
        
        app_logger.info("closeEvent: accepting — application will exit")
        event.accept()
        
        # Force Qt to quit the event loop so the process actually exits
        QApplication.instance().quit()
    
    def changeEvent(self, event):
        if event.type() == QEvent.Type.WindowStateChange:
            was_minimized = bool(event.oldState() & Qt.WindowState.WindowMinimized)
            is_minimized  = bool(self.windowState() & Qt.WindowState.WindowMinimized)
            # Read from the live checkbox so unsaved changes are respected.
            # Fall back to config dict if the UI hasn't been built yet.
            if hasattr(self, 'cfg_tray'):
                min_to_tray = self.cfg_tray.isChecked()
            else:
                min_to_tray = self.config.settings.get("min_to_tray", False)
            app_logger.debug(f"changeEvent: was_minimized={was_minimized}, is_minimized={is_minimized}, min_to_tray={min_to_tray}")
            if not was_minimized and is_minimized:
                if min_to_tray:
                    self.hide()
                    return
        super().changeEvent(event)


class StatusOverlay(QWidget):
    """On-screen status indicator — shown at a screen edge, stays on top.

    Design: TWO completely separate opaque windows, styled with CSS background-color.
    - A thin BAR along one edge of the screen (configurable: Top/Bottom/Left/Right).
    - A small circular ICON in one corner.
    
    No compositor tricks, no WA_TranslucentBackground, no ctypes SetLayeredWindowAttributes.
    Just plain colored rectangles.  This works reliably in frozen apps because:
      - No DWM composition involved
      - No magenta color-key transparency (which crashed previous attempts)
      - The windows simply have a solid background and sit on top of everything
      - They do NOT steal focus (WA_ShowWithoutActivating + Tool flag)
    
    The bar/icon are hidden when the app is idle (configurable) so they don't
    distract during normal use.
    """

    # State → CSS color string (must match _update_app_state states)
    COLORS = {
        'idle':         'rgba(80, 80, 80, 180)',
        'listening':    'rgba(40, 180, 80, 220)',
        'loading':      'rgba(200, 140, 20, 220)',
        'recording':    'rgba(220, 40, 40, 230)',
        'transcribing': 'rgba(40, 100, 220, 230)',
        'both':         'rgba(150, 40, 190, 230)',
    }

    _COMMON_FLAGS = (
        Qt.WindowType.FramelessWindowHint |
        Qt.WindowType.WindowStaysOnTopHint |
        Qt.WindowType.Tool |
        Qt.WindowType.WindowDoesNotAcceptFocus
    )

    def __init__(self, config):
        # StatusOverlay is NOT a real widget itself — it just coordinates two child windows.
        # It holds NO state of its own. set_state(state) is the only way to update it,
        # called directly from _update_app_state() so bar/icon always match the window label.
        super().__init__(None)
        self.hide()  # Never show the parent

        self.config = config
        self._current_state = 'idle'

        # Create the two visible child windows
        self._bar  = self._make_panel()
        self._icon = self._make_panel()

        # Reposition timer — updates geometry every second in case screen changes
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._reposition)
        self._timer.start(1000)

        QTimer.singleShot(100, self._reposition)
        app_logger.debug("StatusOverlay: initialised (stateless display)")

    def _make_panel(self):
        """Create a single borderless always-on-top opaque panel window."""
        w = QWidget(None)
        w.setWindowFlags(self._COMMON_FLAGS)
        w.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        w.hide()
        return w

    def set_state(self, state: str):
        """Mirror the window label state directly. Called from _update_app_state()."""
        self._current_state = state
        cfg  = self.config.settings

        if not cfg.get("ind_show", True):
            self._bar.hide()
            self._icon.hide()
            return

        # Hide when idle if configured
        if state == 'idle':
            if cfg.get("ind_hide_idle", True):
                self._bar.hide()
                self._icon.hide()
                return

        color_css = self.COLORS.get(state, self.COLORS['idle'])
        kind = cfg.get("ind_type", "Both")

        # Bar
        if "Bar" in kind or "Both" in kind:
            self._bar.setStyleSheet(f"background-color: {color_css}; border: none;")
            self._bar.show()
            self._bar.raise_()
        else:
            self._bar.hide()

        # Icon (circle)
        if "Icon" in kind or "Both" in kind:
            size = cfg.get("ind_size", 28)
            r    = size // 2
            self._icon.setStyleSheet(
                f"background-color: {color_css}; border: none; border-radius: {r}px;"
            )
            self._icon.show()
            self._icon.raise_()
        else:
            self._icon.hide()

    def _reposition(self):
        """Recalculate geometry for bar and icon (called every second and on state change)."""
        try:
            screen = QApplication.primaryScreen().geometry()
            cfg    = self.config.settings

            if not cfg.get("ind_show", True):
                self._bar.hide()
                self._icon.hide()
                return

            kind   = cfg.get("ind_type",     "Both")
            pos    = cfg.get("ind_pos",       "Top-Right")
            edge   = cfg.get("bar_edge",      "Top")
            thick  = cfg.get("bar_thickness", 5)
            size   = cfg.get("ind_size",      28)
            offset = cfg.get("ind_off",       20)
            sw, sh = screen.width(), screen.height()

            if "Bar" in kind or "Both" in kind:
                if edge == "Top":
                    self._bar.setGeometry(screen.x(), screen.y(), sw, thick)
                elif edge == "Bottom":
                    self._bar.setGeometry(screen.x(), screen.y() + sh - thick, sw, thick)
                elif edge == "Left":
                    self._bar.setGeometry(screen.x(), screen.y(), thick, sh)
                else:
                    self._bar.setGeometry(screen.x() + sw - thick, screen.y(), thick, sh)

            if "Icon" in kind or "Both" in kind:
                ix = screen.x() + (offset if "Left" in pos else sw - size - offset)
                iy = screen.y() + (offset if "Top"  in pos else sh - size - offset)
                self._icon.setGeometry(ix, iy, size, size)

            # Re-apply current state so visibility/color stay correct after resize
            self.set_state(self._current_state)

        except Exception as e:
            app_logger.error(f"StatusOverlay._reposition error: {e}", exc_info=True)

    def hide_all(self):
        self._bar.hide()
        self._icon.hide()

    def deleteLater(self):
        self._bar.deleteLater()
        self._icon.deleteLater()
        super().deleteLater()


if __name__ == "__main__":
    # PyInstaller + multiprocessing on Windows: must call freeze_support()
    # before anything else so worker subprocesses are handled correctly.
    import multiprocessing
    multiprocessing.freeze_support()
    app_logger.info("="*60)
    app_logger.info(f"{APP_NAME} v{__version__} - Starting")
    app_logger.info(f"Python: {sys.version}")
    app_logger.info(f"Platform: {sys.platform}\"")
    app_logger.info("="*60)
    
    try:
        app_logger.debug("→ Creating QApplication instance...")
        app = QApplication(sys.argv)
        app_logger.debug("✓ QApplication created")
        
        # Don't quit when the main window is hidden (e.g. minimized to tray)
        app.setQuitOnLastWindowClosed(False)
        app_logger.debug("✓ setQuitOnLastWindowClosed(False) set")
        
        app_logger.debug("→ Applying dark stylesheet...")
        app.setStyleSheet(DARK_STYLE)
        app_logger.debug("✓ Stylesheet applied")
        
        app_logger.debug("→ Creating WhisperRApp instance...")
        window = WhisperRApp()
        app_logger.debug("✓ WhisperRApp instance created")
        
        app_logger.debug("→ Calling window.show()...")
        window.show()
        app_logger.debug("✓ window.show() returned — window is now visible")
        app_logger.info(f"  window.isVisible={window.isVisible()}, geometry={window.geometry()}")
        
        app_logger.debug("→ Processing pending Qt events (processEvents)...")
        app.processEvents()
        app_logger.debug("✓ processEvents() returned — about to enter event loop")
        
        app_logger.debug("→ Starting Qt event loop (app.exec())...")
        exit_code = app.exec()
        app_logger.info(f"Application exited with code {exit_code}")
        sys.exit(exit_code)
        
    except Exception as e:
        app_logger.error("="*60)
        app_logger.error(f"CRASH IN MAIN: {e}")
        app_logger.error("="*60)
        app_logger.error("Full traceback:", exc_info=True)
        
        # Write crash details to separate file
        with open(os.path.join(BASE_DIR, "MAIN_CRASH.txt"), "w") as f:
            f.write(f"Crash in main at {datetime.now()}\n")
            f.write(f"Error: {e}\n\n")
            traceback.print_exc(file=f)
        
        raise
