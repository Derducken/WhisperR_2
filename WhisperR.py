import sys
import os
import multiprocessing

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
    # Force onnxruntime (used by VAD/Silero) to CPU-only mode before any import.
    # Without this, onnxruntime_pybind11_state.dll tries to init its CUDA execution
    # provider which conflicts with ctranslate2's already-loaded CUDA DLLs and causes
    # an access violation.  These env vars must be set before onnxruntime is ever imported.
    os.environ["ORT_DISABLE_CUDA"]        = "1"
    os.environ["ORT_LOGGING_LEVEL"]       = "3"   # suppress ORT verbosity
    os.environ["ORTEP_DISABLE_PROVIDERS"] = "CUDA"

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

    _vad_broken = False  # set True on first VAD failure; disables VAD for rest of session

    while True:
        try:
            msg = task_q.get(timeout=1.0)
        except Exception:
            continue

        if msg == '__STOP__':
            _log("AI worker: received stop signal, exiting")
            break

        model_name, lang_code, compute_pref, audio_data, src, translate, use_vad, prompt, min_confidence = msg

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

        # ── VAD safety: block onnxruntime DLL load in frozen app ───────────────
        # onnxruntime_pybind11_state.dll crashes in DllMain (access violation)
        # when loaded alongside ctranslate2's CUDA DLLs in a PyInstaller child.
        # env vars like ORT_DISABLE_CUDA are set AFTER DllMain runs — too late.
        # Fix: in frozen mode, monkey-patch faster_whisper.vad so get_vad_model()
        # raises RuntimeError immediately, before `import onnxruntime` is reached.
        # This is done once per worker lifetime and is idempotent.
        if getattr(sys, 'frozen', False) and use_vad and not _vad_broken:
            try:
                import faster_whisper.vad as _fwvad
                def _safe_get_vad_model():
                    raise RuntimeError(
                        "onnxruntime blocked in frozen app to prevent DLL crash")
                _fwvad.get_vad_model = _safe_get_vad_model
                _vad_broken = True
                result_q.put(('warn',
                    'VAD disabled — onnxruntime crashes in frozen mode. '
                    'Transcription will work normally without VAD.'))
                _log('VAD monkey-patched out (frozen app protection).')
                use_vad = False
            except Exception as _vp:
                _log(f'VAD patch failed ({_vp}), forcing vad_filter=False anyway.')
                _vad_broken = True
                use_vad = False

        # Transcribe
        result_q.put(('status', True))
        try:
            import numpy as np
            if isinstance(audio_data, str):
                # File path submitted via import_files() — decode with faster-whisper's
                # own audio loading (which uses ffmpeg/av under the hood).
                from faster_whisper.audio import decode_audio
                audio_np = decode_audio(audio_data, sampling_rate=16000)
            else:
                audio_np = np.frombuffer(audio_data, dtype=np.float32)

            # _conf_ok maps min_confidence (0-1 UI slider) to avg_logprob threshold.
            # avg_logprob is negative: 0.0 = perfect, -1.0 = poor.
            # We scale the UI value so that:
            #   0.0 → threshold = -inf (keep everything)
            #   0.5 → threshold = -0.5  (reasonable default)
            #   0.9 → threshold = -0.1  (strict but achievable)
            #   1.0 → threshold =  0.0  (only perfect segments)
            # Formula: threshold = -(1.0 - min_confidence)
            _logprob_threshold = -(1.0 - min_confidence) if min_confidence > 0.0 else float('-inf')

            def _conf_ok(s):
                ok = s.avg_logprob >= _logprob_threshold
                if not ok:
                    _log(f"  segment filtered (logprob={s.avg_logprob:.3f} < {_logprob_threshold:.3f}): {s.text!r}")
                return ok

            segments, _ = model.transcribe(
                audio_np,
                language=lang_code if lang_code != 'auto' else None,
                task='translate' if translate else 'transcribe',
                vad_filter=use_vad,
                initial_prompt=prompt or None,
            )
            seg_list = list(segments)  # materialise — generator exhausts on iteration
            if seg_list:
                _log(f"  segments={len(seg_list)}, logprobs={[round(s.avg_logprob,3) for s in seg_list]}, threshold={_logprob_threshold:.3f}")
            text = ' '.join(s.text.strip() for s in seg_list if _conf_ok(s)).strip()
            result_q.put(('text', text, src))
        except Exception as e:
            _log(f"Transcription error: {e}\n{traceback.format_exc()}")
        result_q.put(('status', False))




# ── PyInstaller freeze_support — MUST be after _ai_worker_process is defined ─
# On Windows spawn, PyInstaller re-runs the .exe to create worker subprocesses.
# freeze_support() intercepts that re-run, looks up _ai_worker_process by name,
# calls it, then exits — so the full Qt/UI code never runs in the worker.
# Placing this AFTER the function definition ensures it can be found.
multiprocessing.freeze_support()

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
    QSystemTrayIcon, QMenu, QSlider, QListWidget, QAbstractItemView, QSplitter
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
QComboBox, QLineEdit { background-color: #2a2a2a; border: 1px solid #444; padding: 4px 28px 4px 6px; min-height: 22px; }
QSpinBox, QDoubleSpinBox { background-color: #2a2a2a; border: 1px solid #444; padding: 4px 6px 4px 6px; min-height: 22px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { width: 0; border: none; }
QSpinBox::down-button, QDoubleSpinBox::down-button { width: 0; border: none; }
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
        self._file_handler = None
        self._disabled = False
        self.logger = logging.getLogger(APP_NAME)
        self.logger.setLevel(logging.DEBUG)
        self._attach_file_handler()
        # Console handler for debugging (always low-traffic — warnings/errors only)
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self.logger.addHandler(ch)

    def _attach_file_handler(self):
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)
            self._file_handler.close()
            self._file_handler = None
        fh = _FlushingFileHandler(self.log_path, mode='w', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        self._file_handler = fh

    def set_level(self, level_name):
        if level_name == "NONE":
            # Disable file logging entirely — no log file created/written
            if self._file_handler:
                self.logger.removeHandler(self._file_handler)
                self._file_handler.close()
                self._file_handler = None
            # Delete any existing log file so we start clean
            try:
                if os.path.exists(self.log_path):
                    os.remove(self.log_path)
            except Exception:
                pass
            self._disabled = True
            self.logger.setLevel(logging.CRITICAL + 1)  # suppress everything
        else:
            levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO,
                      "WARNING": logging.WARNING, "ERROR": logging.ERROR}
            self._disabled = False
            if not self._file_handler:
                self._attach_file_handler()  # re-attach if previously disabled
            self.level = levels.get(level_name, logging.INFO)
            self.logger.setLevel(self.level)

    def debug(self, msg, exc_info=False):
        if not self._disabled: self.logger.debug(msg, exc_info=exc_info)
    def info(self, msg, exc_info=False):
        if not self._disabled: self.logger.info(msg, exc_info=exc_info)
    def warning(self, msg, exc_info=False):
        if not self._disabled: self.logger.warning(msg, exc_info=exc_info)
    def error(self, msg, exc_info=False):
        if not self._disabled: self.logger.error(msg, exc_info=exc_info)

app_logger = AppLogger()

# --- 5. CONFIGURATION ---
class AppConfig:
    def __init__(self):
        self.path = os.path.join(BASE_DIR, "config.json")
        self.settings = {
            "model": "large-v3", "lang_name": "English", "lang_code": "en",
            "translate": False, "timestamps": False,
            "initial_prompt": "An article for a data recovery company's site. The article uses a lot of storage-related terminology, with app and service names like Disk Drill, Recuva, CHKDSK, Windows File History, Windows File Explorer, Google Drive (GDrive), Microsoft OneDrive, Dropbox, companies like CleverFiles, file-system-related words like RAW, FAT8, FAT16, FAT32, NTFS, EXT2, EXT3, EXT4, EXT2/3/4, ReiserFS, ReFS, XFS, JFS, file formats like AVI, MKV, MP4, MOV, ARI, BRAW, R3D, FLV, OSes like Linux, Windows 95/98/NT/2000/XP/Vista/7/8/10/11, OS virtual folders like This PC, Quick Access, Recent Files, Recycle Bin, Trashcan, Libraries, technologies like S.M.A.R.T., Hard Disk Drives (HDDs), Solid State Drives (SSDs), M.2 drives, external drives, internal drives, USB drives, SD cards, TRIM, USB Type-A, USB Type-C, USB-A, USB-C, FireWire, card readers, cameras like GoPro, GoPro HERO, GoPro HERO 13 Black, GoPro MAX2, and more. Extra note: when the user says okay, parse it as OK.",
            "audio_folder": str(Path.home() / "WhisperR_Recordings"),
            "mon_folder": str(Path.home() / "WhisperR_Watch"),
            "clear_exit": True, "save_to_disk": False, "auto_space": True,
            "min_to_tray": True,
            "input_device_name": "Microphone (Sound Blaster AE-7)",
            "input_device_index": 1,
            "paste_delay": 0.5,
            "hotkey": "<ctrl>+<alt>+z",
            "ptt_key": "ctrl+shift+space",
            "visibility_hotkey": "ctrl+shift+alt+z",
            "rollback_hotkey": "ctrl+shift+z",
            "live_mode": "Auto-Pause",
            "dict_mode": "Auto-Pause", "auto_pause_sec": 2.0,
            "noise_floor": 50, "speech_vol": 211,
            "commands": {"Launch Notepad": "notepad.exe"},
            "terms": {"hexagon software": "Hexagon Software"},
            "ind_show": True, "ind_type": "Both", "ind_pos": "Top-Left",
            "ind_size": 32, "ind_off": 5, "bar_edge": "Top", "bar_size": 5,
            "bar_thickness": 3, "ind_opacity": 220, "bar_opacity": 220,
            "ind_hide_idle": True,
            "log_level": "NONE", "use_vad": True,
            "ft_output_folder": str(Path.home() / "WhisperR_Output"),
            "ft_mon_folder": str(Path.home() / "WhisperR_Watch"),
            "ft_mon_enabled": False,
            "use_confidence": True, "min_confidence": 0.9,
            "sendkeys_trigger":   "sendkeys",
            "select_trigger":     "whisperselect",
            "move_trigger":       "whispermove",
            "movebefore_trigger": "whisperbefore",
            "moveafter_trigger":  "whisperafter",
            "fuzzy_threshold":    0.75
        }
        self._first_run = not os.path.exists(self.path)
        self.load()
        # On first run, write defaults immediately so they are persisted
        if self._first_run:
            try:
                self.save()
                app_logger.info("First run — default config.json created.")
            except Exception as _e:
                app_logger.warning(f"First run: could not write config: {_e}")
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
            0.0,                  # min_confidence (unused on preload)
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
            cfg.get('min_confidence', 0.0) if cfg.get('use_confidence', False) else 0.0,
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
                    # PTT not held — if we were recording, fall through to dispatch
                    # (handled by the release-edge block above). Otherwise idle-sleep.
                    if not _ptt_was_pressed:
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


class SpinWidget(QWidget):
    """Spin field with [−] [+] buttons on the right.

    Layout:  [──── spin field ────] [−] [+]

    Optional slider to the left of the spin field (use_slider=True):
    Layout:  [──── slider 70% ────] [ spin ] [−] [+]

    The ± buttons are fixed-width with always-visible glyphs, guaranteed
    by an explicit per-widget stylesheet that is not affected by the global
    dark theme rules.
    """
    _BTN_CSS = (
        "QPushButton { background:#3a3a3a; border:1px solid #666; "
        "color:#e0e0e0; font-size:15px; font-weight:bold; "
        "min-width:26px; max-width:26px; min-height:26px; max-height:26px; "
        "padding:0; border-radius:3px; }"
        "QPushButton:hover  { background:#0078d7; color:#fff; border-color:#0078d7; }"
        "QPushButton:pressed{ background:#005fa3; color:#fff; }"
    )

    def __init__(self, parent=None, *, is_double=False,
                 min_v=0, max_v=100, step=1, value=0,
                 decimals=2, use_slider=False, spin_width=90):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)

        if is_double:
            self.spin = QDoubleSpinBox()
            self.spin.setDecimals(decimals)
            self.spin.setSingleStep(step)
        else:
            self.spin = QSpinBox()
            self.spin.setSingleStep(int(step))
        self.spin.setRange(min_v, max_v)
        self.spin.setValue(value)
        self.spin.setMinimumWidth(spin_width)

        self._slider = None
        if use_slider:
            self._slider = QSlider(Qt.Orientation.Horizontal)
            self._slider.setRange(int(min_v * 100 if is_double else min_v),
                                  int(max_v * 100 if is_double else max_v))
            self._slider.setValue(int(value * 100 if is_double else value))
            self._slider.setTickInterval(
                int((max_v - min_v) * (10 if is_double else 1)))
            self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            # Bidirectional sync
            if is_double:
                self._slider.valueChanged.connect(
                    lambda v: (self.spin.blockSignals(True),
                               self.spin.setValue(v / 100.0),
                               self.spin.blockSignals(False)))
                self.spin.valueChanged.connect(
                    lambda v: (self._slider.blockSignals(True),
                               self._slider.setValue(int(v * 100)),
                               self._slider.blockSignals(False)))
            else:
                self._slider.valueChanged.connect(
                    lambda v: (self.spin.blockSignals(True),
                               self.spin.setValue(v),
                               self.spin.blockSignals(False)))
                self.spin.valueChanged.connect(
                    lambda v: (self._slider.blockSignals(True),
                               self._slider.setValue(v),
                               self._slider.blockSignals(False)))
            lay.addWidget(self._slider, stretch=7)   # ~70 % width
            lay.addWidget(self.spin,    stretch=2)   # ~20 %
        else:
            lay.addWidget(self.spin, stretch=1)

        btn_minus = QPushButton("−")
        btn_plus  = QPushButton("+")
        for b in (btn_minus, btn_plus):
            b.setStyleSheet(self._BTN_CSS)
            b.setSizePolicy(b.sizePolicy().horizontalPolicy(),
                            b.sizePolicy().verticalPolicy())

        btn_minus.clicked.connect(self.spin.stepDown)
        btn_plus.clicked.connect(self.spin.stepUp)
        lay.addWidget(btn_minus)
        lay.addWidget(btn_plus)

    # ── Proxy API ────────────────────────────────────────────────────────────
    def value(self):          return self.spin.value()
    def setValue(self, v):    self.spin.setValue(v)
    def setRange(self, a, b): self.spin.setRange(a, b)
    def setToolTip(self, t):
        super().setToolTip(t)
        self.spin.setToolTip(t)
        if self._slider:
            self._slider.setToolTip(t)
    @property
    def valueChanged(self):   return self.spin.valueChanged

def _build_dark_style():
    # Generate tiny triangle PNG files for spinbox up/down arrows.
    # Qt ignores CSS border-triangle tricks for ::up-arrow/::down-arrow --
    # it only accepts image: url() paths there.
    import tempfile
    from PyQt6.QtGui import QPixmap, QPainter, QColor, QPolygon
    from PyQt6.QtCore import QPoint as QP
    tmpdir = tempfile.gettempdir()

    def _arrow_png(fname, pts_fn, sz=14):
        path = os.path.join(tmpdir, fname)
        pix = QPixmap(sz, sz)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor("#cccccc"))
        p.setPen(QColor(0, 0, 0, 0))
        p.drawPolygon(QPolygon(pts_fn(sz)))
        p.end()
        pix.save(path, "PNG")
        return path.replace("\\", "/")

    up   = _arrow_png("whr_up.png",
               lambda s: [QP(s//2, 2), QP(s-2, s-2), QP(2, s-2)])
    down = _arrow_png("whr_dn.png",
               lambda s: [QP(2, 2), QP(s-2, 2), QP(s//2, s-2)])
    return DARK_STYLE.replace("{UP_ARROW}", up).replace("{DOWN_ARROW}", down)


# ── Sendkeys helper ──────────────────────────────────────────────────────────
# Parses strings like "Hello <ENTER>World<CTRL+C>" and sends them via pyautogui.
# Supported tag formats:
#   <KEY>          single key press           e.g. <ENTER>, <TAB>, <ESC>, <F5>
#   <MOD+KEY>      modifier + key             e.g. <CTRL+C>, <ALT+F4>, <SHIFT+HOME>
#   <MOD+MOD+KEY>  multiple modifiers         e.g. <CTRL+SHIFT+Z>
#   <WINKEY>       Windows key (special case)
# Literal text between tags is typed via pyautogui.typewrite or pyautogui.write.

_SENDKEYS_MAP = {
    # Navigation
    "ENTER": "enter", "RETURN": "enter", "TAB": "tab", "ESC": "escape",
    "ESCAPE": "escape", "SPACE": "space", "BACKSPACE": "backspace", "DELETE": "delete",
    "DEL": "delete", "INSERT": "insert", "INS": "insert",
    "HOME": "home", "END": "end", "PGUP": "pageup", "PAGEUP": "pageup",
    "PGDN": "pagedown", "PAGEDOWN": "pagedown",
    "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right",
    # Function keys
    "F1":"f1","F2":"f2","F3":"f3","F4":"f4","F5":"f5","F6":"f6",
    "F7":"f7","F8":"f8","F9":"f9","F10":"f10","F11":"f11","F12":"f12",
    # Modifiers (used in combos)
    "CTRL":"ctrl","CONTROL":"ctrl","LCTRL":"ctrlleft","RCTRL":"ctrlright",
    "ALT":"alt","LALT":"altleft","RALT":"altright",
    "SHIFT":"shift","LSHIFT":"shiftleft","RSHIFT":"shiftright",
    "WIN":"winleft","WINKEY":"winleft","LWIN":"winleft","RWIN":"winright",
    "WINDOWS":"winleft",
    # Numpad
    "NUM0":"num0","NUM1":"num1","NUM2":"num2","NUM3":"num3","NUM4":"num4",
    "NUM5":"num5","NUM6":"num6","NUM7":"num7","NUM8":"num8","NUM9":"num9",
    "NUMMINUS":"subtract","NUMPLUS":"add","NUMMUL":"multiply","NUMDIV":"divide",
    "NUMDECIMAL":"decimal","NUMENTER":"enter",
    # Media / misc
    "CAPSLOCK":"capslock","NUMLOCK":"numlock","SCROLLLOCK":"scrolllock",
    "PRINTSCREEN":"printscreen","PAUSE":"pause","BREAK":"pause",
    "VOLUMEUP":"volumeup","VOLUMEDOWN":"volumedown","VOLUMEMUTE":"volumemute",
    "MEDIAPLAYPAUSE":"playpause","MEDIANEXT":"nexttrack","MEDIAPREV":"prevtrack",
}

def _resolve_key(raw):
    """Map a tag name to a pyautogui key string, or return lowercased name as fallback."""
    return _SENDKEYS_MAP.get(raw.upper(), raw.lower())

def _send_keys_sequence(text, paste_delay=0.0):
    """Parse and send a string containing <KEY> tags and literal text.
    
    Examples:
        _send_keys_sequence("Best regards,<ENTER>John")
        _send_keys_sequence("<CTRL+A><DEL>new content")
        _send_keys_sequence("<WINKEY+R>notepad<ENTER>")
    """
    import re as _re
    import pyautogui as _pag
    _pag.FAILSAFE = False

    # Tokenise: split on <...> tags preserving the tags
    tokens = _re.split(r'(<[^>]+>)', text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith('<') and tok.endswith('>'):
            inner = tok[1:-1].strip()
            parts = [p.strip() for p in inner.split('+')]
            keys  = [_resolve_key(p) for p in parts]
            if len(keys) == 1:
                try:
                    _pag.press(keys[0])
                except Exception as e:
                    app_logger.warning(f"sendkeys: press({keys[0]}) failed: {e}")
            else:
                try:
                    _pag.hotkey(*keys)
                except Exception as e:
                    app_logger.warning(f"sendkeys: hotkey({keys}) failed: {e}")
        else:
            # Literal text — type it character by character to handle Unicode
            try:
                _pag.write(tok, interval=0.02)
            except Exception:
                # Fallback: clipboard paste for non-ASCII characters
                import pyperclip
                pyperclip.copy(tok)
                time.sleep(max(paste_delay, 0.05))
                _pag.hotkey('ctrl', 'v')


# ── Fuzzy trigger matching ────────────────────────────────────────────────────
# Whisper often mishears single-word trigger commands because they are not
# natural English (e.g. "whisperselect" → "whispers elect", "whisper select",
# "whisper's elect").  We use difflib.SequenceMatcher to compare the spoken
# text's first N words (where N = number of words in the trigger) against the
# trigger, and accept if similarity ≥ threshold.
#
# Returns (matched: bool, remainder: str)
#   matched   — True if the spoken prefix fuzzy-matches the trigger
#   remainder — everything after the matched prefix (the target argument)

def _fuzzy_trigger_match(spoken_text, trigger, threshold=0.75):
    """Check if `spoken_text` starts with something that fuzzy-matches `trigger`.

    Compares the first len(trigger.split()) words of the spoken text against
    the trigger as a whole string, plus tries a few common Whisper
    mis-segmentations (joined / space-split variants).

    Returns (matched, remainder_text).
    """
    from difflib import SequenceMatcher

    spoken_lower  = spoken_text.lower().strip()
    trigger_lower = trigger.lower().strip()

    if not trigger_lower:
        return False, spoken_text

    trigger_words = trigger_lower.split()
    n = len(trigger_words)

    spoken_words = spoken_lower.split()

    # Build candidate prefixes to test:
    #  1. First n spoken words joined (handles "whispers elect" → "whispers elect")
    #  2. First n+1 spoken words joined (trigger may split into one extra word)
    #  3. Exact prefix chars (same length as trigger) — handles run-together speech
    candidates = []
    if len(spoken_words) >= n:
        candidates.append((' '.join(spoken_words[:n]),   n))
    if len(spoken_words) >= n + 1:
        candidates.append((' '.join(spoken_words[:n+1]), n + 1))
    # char-length match
    if len(spoken_lower) >= len(trigger_lower):
        candidates.append((spoken_lower[:len(trigger_lower)], None))

    best_ratio = 0.0
    best_words_consumed = n

    for candidate, words_consumed in candidates:
        ratio = SequenceMatcher(None, trigger_lower, candidate).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            if words_consumed is not None:
                best_words_consumed = words_consumed

    if best_ratio >= threshold:
        # Reconstruct remainder from original (preserve original casing)
        orig_words = spoken_text.split()
        remainder  = ' '.join(orig_words[best_words_consumed:]).strip()
        return True, remainder

    return False, spoken_text

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
            self.pop_mics()
            app_logger.debug("✓ pop_mics() complete")

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
            # Set the Windows AppUserModelID — without this, Windows groups the
            # window under the Python interpreter's taskbar entry instead of our exe.
            try:
                import ctypes as _c
                _c.windll.shell32.SetCurrentProcessExplicitAppUserModelID("WhisperR.App.1")
            except Exception:
                pass
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
            self._app_icon = app_icon  # keep reference for tray idle state
            
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
            # QFileSystemWatcher for real-time folder monitoring
            self._setup_ft_watcher()
            
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
            self._session_buffer:  str = ""   # running concat of all pastes this session (for select/move)
            self._cursor_ops_pending: bool = False  # next transcription should be lowercased (select/move used)
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
        l1.setSpacing(0)
        l1.setContentsMargins(4, 4, 4, 4)

        # ── Splitter with two collapsible panes ──────────────────────────────
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)
        self._main_splitter.setHandleWidth(6)
        self._main_splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a2a; border-top: 1px solid #444; }"
            "QSplitter::handle:hover { background: #0078d7; }"
        )

        # Helper: build one collapsible pane
        def _make_pane(title, placeholder):
            outer = QWidget()
            vl = QVBoxLayout(outer)
            vl.setContentsMargins(0, 0, 0, 0)
            vl.setSpacing(2)

            # Header row: label + collapse button
            hdr = QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            lbl = QLabel(title)
            lbl.setStyleSheet("font-weight: bold; color: #aaa; font-size: 8pt; padding: 2px 4px;")
            btn = QPushButton("▲")
            btn.setFixedSize(22, 18)
            btn.setStyleSheet(
                "QPushButton { background: #252525; border: 1px solid #444; color: #aaa; "
                "font-size: 9pt; padding: 0; border-radius: 3px; }"
                "QPushButton:hover { background: #0078d7; color: #fff; }"
            )
            btn.setToolTip("Collapse / expand this pane")
            hdr.addWidget(lbl)
            hdr.addStretch()
            hdr.addWidget(btn)
            vl.addLayout(hdr)

            ta = QTextEdit()
            ta.setFont(QFont("Consolas", 9))
            ta.setPlaceholderText(placeholder)
            ta.setReadOnly(False)
            vl.addWidget(ta)

            # Collapse/expand toggle
            def _toggle(checked=None, _ta=ta, _btn=btn, _outer=outer):
                if _ta.isVisible():
                    _ta.hide()
                    _btn.setText("▼")
                    _outer.setMinimumHeight(24)
                    _outer.setMaximumHeight(24)
                    # Give all space to the other pane
                    idx = self._main_splitter.indexOf(_outer)
                    other = 1 - idx
                    sizes = list(self._main_splitter.sizes())
                    total = sum(sizes)
                    sizes[idx] = 0
                    sizes[other] = total
                    self._main_splitter.setSizes(sizes)
                else:
                    _ta.show()
                    _btn.setText("▲")
                    _outer.setMinimumHeight(60)
                    _outer.setMaximumHeight(16777215)
                    idx = self._main_splitter.indexOf(_outer)
                    other = 1 - idx
                    sizes = list(self._main_splitter.sizes())
                    total = sum(sizes)
                    half = total // 2
                    sizes[idx] = half
                    sizes[other] = total - half
                    self._main_splitter.setSizes(sizes)

            btn.clicked.connect(_toggle)
            return outer, ta

        results_pane, self.results_area = _make_pane(
            "Results  (transcription output)",
            "Transcription results will appear here...")
        log_pane,     self.log_area     = _make_pane(
            "Log  (system messages)",
            "System messages will appear here...")

        self._main_splitter.addWidget(results_pane)
        self._main_splitter.addWidget(log_pane)
        self._main_splitter.setSizes([320, 160])
        self._main_splitter.setCollapsible(0, False)
        self._main_splitter.setCollapsible(1, False)

        # Keep self.scratchpad as alias → all existing .append() calls go to log_area
        self.scratchpad = self.log_area

        l1.addWidget(self._main_splitter)
        
        # ── Live volume meter (driven by AudioRecorder.volume_out) ────
        self.live_meter = QProgressBar()
        self.live_meter.setRange(0, 8000)
        self.live_meter.setValue(0)
        self.live_meter.setTextVisible(False)
        self.live_meter.setFixedHeight(6)
        self.live_meter.setStyleSheet(
            "QProgressBar { background:#1a1a1a; border:none; border-radius:3px; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "  stop:0 #28b450, stop:0.6 #f0a000, stop:1.0 #e74c3c); border-radius:3px; }"
        )
        l1.addWidget(self.live_meter)

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
        hb.addWidget(self.btn_toggle)
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
        bi_cmd = QPushButton("Import .txt")
        bi_cmd.clicked.connect(self._import_commands)
        be_cmd = QPushButton("Export .txt")
        be_cmd.clicked.connect(self._export_commands)
        btn_row.addWidget(ba)
        btn_row.addWidget(bd)
        btn_row.addWidget(bi_cmd)
        btn_row.addWidget(be_cmd)
        l2.addLayout(btn_row)

        # ===== TERMS TAB =====
        t_terms = QWidget()
        l_terms = QVBoxLayout(t_terms)
        lbl_terms = QLabel(
            "Text Replacements \u2014 applied after transcription, before pasting.\n"
            "Left: phrase Whisper says (matched case-insensitively).\n"
            "Right: replacement text \u2014 may include <KEY> tags for special keys.\n"
            "Examples:   sign off \u2192 Best regards,<ENTER>John       bold it \u2192 <CTRL+B>"
        )
        lbl_terms.setWordWrap(True)
        l_terms.addWidget(lbl_terms)
        self.terms_table = QTableWidget(0, 2)
        self.terms_table.setHorizontalHeaderLabels(["Recognised Phrase", "Replacement Text"])
        self.terms_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for k, v in self.config.settings.get("terms", {}).items():
            r = self.terms_table.rowCount()
            self.terms_table.insertRow(r)
            self.terms_table.setItem(r, 0, QTableWidgetItem(k))
            self.terms_table.setItem(r, 1, QTableWidgetItem(v))
        l_terms.addWidget(self.terms_table)
        terms_btn_row = QHBoxLayout()
        ta = QPushButton("Add Row")
        ta.clicked.connect(lambda: self.terms_table.insertRow(self.terms_table.rowCount()))
        td = QPushButton("Delete Selected Row")
        td.clicked.connect(self._delete_terms_row)
        ti = QPushButton("Import .txt")
        ti.clicked.connect(self._import_terms)
        te = QPushButton("Export .txt")
        te.clicked.connect(self._export_terms)
        terms_btn_row.addWidget(ta)
        terms_btn_row.addWidget(td)
        terms_btn_row.addWidget(ti)
        terms_btn_row.addWidget(te)
        l_terms.addLayout(terms_btn_row)
        self.tabs.addTab(t_terms, "Terms")

        self.tabs.addTab(t2, "Commands")

        # ===== FILE TRANSCRIPTION TAB =====
        t_ft = QWidget()
        l_ft = QVBoxLayout(t_ft)

        # ── Queue list ──────────────────────────────────────────────────
        queue_lbl = QLabel("Files to Transcribe  (drag & drop audio files here):")
        queue_lbl.setStyleSheet("font-weight: bold;")
        l_ft.addWidget(queue_lbl)

        self.ft_list = QListWidget()
        self.ft_list.setAcceptDrops(True)
        self.ft_list.setDragDropMode(QListWidget.DragDropMode.DropOnly)
        self.ft_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.ft_list.setMinimumHeight(140)
        # Enable drop events by subclassing with a local event-filter
        self.ft_list.installEventFilter(self)
        l_ft.addWidget(self.ft_list)

        ft_list_btns = QHBoxLayout()
        ft_add_btn = QPushButton("Add Files...")
        ft_add_btn.clicked.connect(self._ft_add_files)
        ft_del_btn = QPushButton("Remove Selected")
        ft_del_btn.clicked.connect(self._ft_remove_selected)
        ft_clear_btn = QPushButton("Clear All")
        ft_clear_btn.clicked.connect(self.ft_list.clear)
        ft_list_btns.addWidget(ft_add_btn)
        ft_list_btns.addWidget(ft_del_btn)
        ft_list_btns.addWidget(ft_clear_btn)
        l_ft.addLayout(ft_list_btns)

        # ── Output folder ────────────────────────────────────────────────
        ft_out_group = QGroupBox("Output Folder")
        ft_out_layout = QFormLayout()
        ft_out_row = QHBoxLayout()
        self.ft_output_folder = QLineEdit(self.config.settings.get("ft_output_folder", ""))
        ft_browse_out = QPushButton("Browse")
        ft_browse_out.clicked.connect(lambda: self.browse_f(self.ft_output_folder))
        ft_out_row.addWidget(self.ft_output_folder)
        ft_out_row.addWidget(ft_browse_out)
        ft_out_layout.addRow("Save transcriptions to:", ft_out_row)
        ft_out_group.setLayout(ft_out_layout)
        l_ft.addWidget(ft_out_group)

        # ── Monitor folder ────────────────────────────────────────────────
        ft_mon_group = QGroupBox("Folder Monitor")
        ft_mon_layout = QFormLayout()
        ft_mon_row = QHBoxLayout()
        self.ft_mon_folder = QLineEdit(self.config.settings.get("ft_mon_folder", ""))
        ft_browse_mon = QPushButton("Browse")
        ft_browse_mon.clicked.connect(lambda: self.browse_f(self.ft_mon_folder))
        ft_mon_row.addWidget(self.ft_mon_folder)
        ft_mon_row.addWidget(ft_browse_mon)
        ft_mon_layout.addRow("Watch folder:", ft_mon_row)
        self.ft_mon_enabled = QCheckBox("Enable folder monitoring (new audio files auto-added to queue)")
        self.ft_mon_enabled.setChecked(self.config.settings.get("ft_mon_enabled", False))
        self.ft_mon_enabled.toggled.connect(self._ft_mon_toggled)
        ft_mon_layout.addRow(self.ft_mon_enabled)
        ft_mon_group.setLayout(ft_mon_layout)
        l_ft.addWidget(ft_mon_group)

        # ── Transcribe button ─────────────────────────────────────────────
        self.ft_start_btn = QPushButton("Transcribe All Queued Files")
        self.ft_start_btn.setFixedHeight(40)
        self.ft_start_btn.setStyleSheet("font-weight: bold; background-color: #1a5c2a; color: white;")
        self.ft_start_btn.clicked.connect(self._ft_start_transcription)
        l_ft.addWidget(self.ft_start_btn)

        self.ft_status_lbl = QLabel("")
        self.ft_status_lbl.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        l_ft.addWidget(self.ft_status_lbl)

        l_ft.addStretch()
        self.tabs.addTab(t_ft, "File Transcription")


        # ===== SETTINGS TAB =====
        sc = QScrollArea()
        cw = QWidget()
        main_layout = QVBoxLayout(cw)
        
        # --- AI Model Settings ---
        ai_group = QGroupBox("AI Model Settings")
        ai_layout = QFormLayout()
        
        self.cfg_model = QComboBox()
        self.cfg_model.addItems(WHISPER_MODELS)
        self.cfg_model.setToolTip(
            "Larger models are more accurate but slower and use more RAM/VRAM.\n"
            "tiny/base: fast, low accuracy, good for live dictation on weaker hardware.\n"
            "small/medium: balanced. large-v3: highest accuracy, needs a GPU."
        )
        self.cfg_model.setCurrentText(self.config.settings["model"])
        self.cfg_model.currentTextChanged.connect(self._on_model_changed)
        ai_layout.addRow("Whisper Model:", self.cfg_model)
        
        self.cfg_lang = QComboBox()
        self.cfg_lang.addItems(list(LANG_MAP.keys()))
        self.cfg_lang.setToolTip(
            "Set to the language you will be speaking.\n"
            "Auto lets Whisper detect the language each time (slightly slower)."
        )
        self.cfg_lang.setCurrentText(self.config.settings["lang_name"])
        ai_layout.addRow("Language:", self.cfg_lang)
        
        self.cfg_ts = QCheckBox("Include timestamps")
        self.cfg_ts.setToolTip(
            "Prepend [HH:MM:SS] timestamps to each transcribed segment.\n"
            "Useful for audio files. Usually unwanted for live dictation."
        )
        self.cfg_ts.setChecked(self.config.settings["timestamps"])
        ai_layout.addRow(self.cfg_ts)
        
        self.cfg_trans = QCheckBox("Translation mode (to English)")
        self.cfg_trans.setToolTip(
            "When enabled, Whisper translates speech to English regardless of source language.\n"
            "Disable this if you want the transcription in the original spoken language."
        )
        self.cfg_trans.setChecked(self.config.settings["translate"])
        ai_layout.addRow(self.cfg_trans)

        # -- Confidence filtering (under AI Model since it directly affects transcription output) --
        self.cfg_use_confidence = QCheckBox("Enable confidence filtering")
        self.cfg_use_confidence.setChecked(self.config.settings.get("use_confidence", False))
        self.cfg_use_confidence.setToolTip(
            "When enabled, Whisper segments whose confidence is below the threshold\n"
            "are silently dropped before pasting.\n\n"
            "Higher value (slider right) = stricter: drops MORE uncertain segments.\n"
            "Lower value (slider left)  = lenient: keeps even uncertain segments.\n\n"
            "Start around 0.50 and raise if you notice hallucinated words."
        )
        ai_layout.addRow(self.cfg_use_confidence)

        self.cfg_conf_spin = SpinWidget(
            is_double=True, min_v=0.0, max_v=1.0, step=0.05,
            value=self.config.settings.get("min_confidence", 0.5), decimals=2,
            use_slider=True, spin_width=70)
        self.cfg_conf_spin.setToolTip(
            "Filters out low-confidence segments based on Whisper's avg_logprob score.\n"
            "Formula: segment kept if avg_logprob >= -(1 - min_confidence)\n"
            "  0.0 = keep everything   0.5 = threshold -0.5   0.9 = threshold -0.1\n"
            "Real speech typically scores between -0.1 (clear) and -0.6 (noisy).\n"
            "Recommended: 0.4-0.6 for most use cases. 0.9 is very strict."
        )
        ai_layout.addRow("Min. Confidence (0-1):", self.cfg_conf_spin)

        ai_group.setLayout(ai_layout)
        main_layout.addWidget(ai_group)

        # --- Audio Input Settings ---
        audio_group = QGroupBox("Audio Input Settings")
        audio_layout = QFormLayout()

        mic_row = QHBoxLayout()
        self.cfg_mic = QComboBox()
        self.cfg_mic.setToolTip(
            "Select the microphone to use for dictation.\n"
            "The app shows all input devices found by PyAudio.\n"
            "If your mic is missing, check Windows Sound settings."
        )
        mic_row.addWidget(self.cfg_mic, stretch=1)
        audio_layout.addRow("Microphone:", mic_row)

        levels_layout = QHBoxLayout()
        self.n_spin = SpinWidget(min_v=0, max_v=8000, step=10,
                               value=self.config.settings["noise_floor"])
        self.n_spin.setToolTip(
            "RMS level below which audio is considered silence.\n"
            "Use Calibrate to set automatically."
        )
        self.s_spin = SpinWidget(min_v=0, max_v=8000, step=10,
                               value=self.config.settings["speech_vol"])
        self.s_spin.setToolTip(
            "RMS level above which audio is considered speech.\n"
            "Use Calibrate to set automatically."
        )
        audio_layout.addRow("Noise Floor:", self.n_spin)
        audio_layout.addRow("Speech Vol:", self.s_spin)

        cal_row = QHBoxLayout()
        self.btn_cal = QPushButton("Calibrate Mic Levels")
        self.btn_cal.clicked.connect(self.start_cal)
        self.btn_cal.setToolTip(
            "Records ~10 s of silence then ~10 s of speech to set Noise Floor and Speech Vol."
        )
        self.cal_prog = QProgressBar()
        self.cal_prog.setRange(0, 100)
        self.cal_prog.setFixedHeight(14)
        self.lbl_cal = QLabel("")
        cal_row.addWidget(self.btn_cal)
        cal_row.addWidget(self.cal_prog)
        cal_row.addWidget(self.lbl_cal)
        audio_layout.addRow(cal_row)

        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)

        # --- Dictation Settings ---
        dict_group = QGroupBox("Dictation Settings")
        dict_layout = QFormLayout()

        self.cfg_dict_m = QComboBox()
        self.cfg_dict_m.addItems(["Simple", "Auto-Pause", "Continuous"])
        self.cfg_dict_m.setCurrentText(self.config.settings.get("dict_mode", "Simple"))
        self.cfg_dict_m.setToolTip(
            "Simple: records until you stop manually, then transcribes.\n"
            "Auto-Pause: detects silence and auto-stops to transcribe.\n"
            "Continuous: transcribes in a rolling loop while active."
        )
        dict_layout.addRow("Dictation Mode:", self.cfg_dict_m)

        self.cfg_live_mode = QComboBox()
        self.cfg_live_mode.addItems(["Simple", "Auto-Pause", "Continuous"])
        self.cfg_live_mode.setCurrentText(self.config.settings.get("live_mode", "Simple"))
        self.cfg_live_mode.setToolTip("Mode used when dictation is triggered via the hotkey.")
        dict_layout.addRow("Live Mode:", self.cfg_live_mode)

        self.cfg_p_sec = SpinWidget(is_double=True, min_v=0.1, max_v=10.0,
                               step=0.1, decimals=1,
                               value=self.config.settings.get("auto_pause_sec", 1.5))
        self.cfg_p_sec.setToolTip(
            "Auto-Pause: seconds of silence before transcription triggers.\n"
            "1.0-2.0 s works well for most people."
        )
        dict_layout.addRow("Auto-Pause Silence (s):", self.cfg_p_sec)

        self.cfg_p_win = SpinWidget(is_double=True, min_v=0.0, max_v=3.0,
                               step=0.05, decimals=2,
                               value=self.config.settings.get("paste_delay", 0.5))
        self.cfg_p_win.setToolTip(
            "Seconds to wait after copying to clipboard before sending Ctrl+V.\n"
            "Increase if text sometimes fails to paste. 0.1-0.5 s is usual."
        )
        dict_layout.addRow("Paste Delay (s):", self.cfg_p_win)

        self.cfg_space = QCheckBox("Auto-add space after each transcription")
        self.cfg_space.setChecked(self.config.settings.get("auto_space", True))
        self.cfg_space.setToolTip(
            "Appends a trailing space after every pasted segment so the\n"
            "next dictation starts a new word automatically."
        )
        dict_layout.addRow(self.cfg_space)

        dict_group.setLayout(dict_layout)
        main_layout.addWidget(dict_group)

        # --- Hotkeys ---
        hotkey_group = QGroupBox("Hotkeys  (click a button to re-assign)")
        hotkey_layout = QFormLayout()

        self.btn_hk1 = QPushButton(self.config.settings["hotkey"])
        self.btn_hk1.clicked.connect(lambda: self.cap_hk(self.btn_hk1, "hotkey"))
        self.btn_hk1.setToolTip("Toggle dictation on/off.")
        hotkey_layout.addRow("Toggle Dictation:", self.btn_hk1)

        self.btn_hk_vis = QPushButton(self.config.settings["visibility_hotkey"])
        self.btn_hk_vis.clicked.connect(lambda: self.cap_hk(self.btn_hk_vis, "visibility_hotkey"))
        self.btn_hk_vis.setToolTip("Show or hide the main window.")
        hotkey_layout.addRow("Show/Hide Window:", self.btn_hk_vis)

        self.btn_hk_rollback = QPushButton(self.config.settings["rollback_hotkey"])
        self.btn_hk_rollback.clicked.connect(lambda: self.cap_hk(self.btn_hk_rollback, "rollback_hotkey"))
        self.btn_hk_rollback.setToolTip(
            "Erases the last pasted segment and lowercases the first\n"
            "letter of the next transcription (for smooth sentence joining)."
        )
        hotkey_layout.addRow("Rollback Last:", self.btn_hk_rollback)

        self.btn_hk2 = QPushButton(self.config.settings["ptt_key"])
        self.btn_hk2.clicked.connect(lambda: self.cap_hk(self.btn_hk2, "ptt_key"))
        self.btn_hk2.setToolTip(
            "Hold this key/combo to record (Push-To-Talk).\n"
            "Works regardless of Dictation Mode. Keys are suppressed\n"
            "while held so they don't reach other apps."
        )
        hotkey_layout.addRow("Push-to-Talk:", self.btn_hk2)

        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)

        # --- Visual Indicators ---
        visual_group = QGroupBox("Visual Indicators")
        visual_layout = QFormLayout()

        self.cfg_ind_show = QCheckBox("Show status indicator overlay")
        self.cfg_ind_show.setChecked(self.config.settings.get("ind_show", True))
        self.cfg_ind_show.setToolTip("Show/hide the floating dot or bar that shows recording state.")
        visual_layout.addRow(self.cfg_ind_show)

        self.cfg_ind_type = QComboBox()
        self.cfg_ind_type.addItems(["Dot", "Bar", "Both"])
        self.cfg_ind_type.setCurrentText(self.config.settings.get("ind_type", "Both"))
        self.cfg_ind_type.setToolTip("Dot: small circle. Bar: thin edge bar. Both: show both.")
        visual_layout.addRow("Indicator Type:", self.cfg_ind_type)

        self.cfg_ind_pos = QComboBox()
        self.cfg_ind_pos.addItems(["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"])
        self.cfg_ind_pos.setCurrentText(self.config.settings.get("ind_pos", "Top-Right"))
        visual_layout.addRow("Dot Position:", self.cfg_ind_pos)

        self.cfg_bar_edge = QComboBox()
        self.cfg_bar_edge.addItems(["Top", "Bottom", "Left", "Right"])
        self.cfg_bar_edge.setCurrentText(self.config.settings.get("bar_edge", "Top"))
        visual_layout.addRow("Bar Edge:", self.cfg_bar_edge)

        self.cfg_ind_sz = SpinWidget(min_v=8, max_v=128, step=2,
                               value=self.config.settings.get("ind_size", 32))
        self.cfg_ind_sz.setToolTip("Size of the dot indicator in pixels.")
        visual_layout.addRow("Dot Size (px):", self.cfg_ind_sz)

        self.cfg_ind_off = SpinWidget(min_v=0, max_v=200, step=2,
                               value=self.config.settings.get("ind_off", 20))
        self.cfg_ind_off.setToolTip("Pixel offset from the screen edge.")
        visual_layout.addRow("Dot Offset (px):", self.cfg_ind_off)

        self.cfg_bar_thickness = SpinWidget(min_v=1, max_v=30, step=1,
                               value=self.config.settings.get("bar_thickness", 5))
        visual_layout.addRow("Bar Thickness (px):", self.cfg_bar_thickness)

        self.cfg_ind_hide_idle = QCheckBox("Hide indicator when idle")
        self.cfg_ind_hide_idle.setChecked(self.config.settings.get("ind_hide_idle", True))
        self.cfg_ind_hide_idle.setToolTip("Auto-hides the overlay when the app is not recording.")
        visual_layout.addRow(self.cfg_ind_hide_idle)

        visual_group.setLayout(visual_layout)
        main_layout.addWidget(visual_group)

        # --- File Storage ---
        storage_group = QGroupBox("File Storage")
        storage_layout = QFormLayout()

        rec_row = QHBoxLayout()
        self.cfg_folder = QLineEdit(self.config.settings["audio_folder"])
        self.cfg_folder.setToolTip("Where live dictation WAV recordings are saved.")
        b_f = QPushButton("Browse")
        b_f.clicked.connect(lambda: self.browse_f(self.cfg_folder))
        rec_row.addWidget(self.cfg_folder)
        rec_row.addWidget(b_f)
        storage_layout.addRow("Recordings Folder:", rec_row)

        self.cfg_ram = QCheckBox("RAM-only mode (no disk writes for recordings)")
        self.cfg_ram.setChecked(not self.config.settings["save_to_disk"])
        self.cfg_ram.setToolTip(
            "Keeps recordings in memory only — never written to disk.\n"
            "Reduces SSD wear. Transcription still works normally."
        )
        storage_layout.addRow(self.cfg_ram)

        self.cfg_clear = QCheckBox("Clear recordings on exit")
        self.cfg_clear.setChecked(self.config.settings["clear_exit"])
        self.cfg_clear.setToolTip("Deletes all WAV files from the recordings folder when the app closes.")
        storage_layout.addRow(self.cfg_clear)

        self.cfg_tray = QCheckBox("Minimize to system tray")
        self.cfg_tray.setChecked(self.config.settings["min_to_tray"])
        self.cfg_tray.setToolTip(
            "Closing the window hides to tray instead of quitting.\n"
            "Right-click the tray icon to quit."
        )
        storage_layout.addRow(self.cfg_tray)

        storage_group.setLayout(storage_layout)
        main_layout.addWidget(storage_group)

        # --- Advanced ---
        advanced_group = QGroupBox("Advanced")
        advanced_layout = QFormLayout()

        sk_row = QHBoxLayout()
        self.cfg_sk_trigger = QLineEdit(self.config.settings.get("sendkeys_trigger", "sendkeys"))
        self.cfg_sk_trigger.setToolTip(
            "When this word appears in a Command phrase, the action field\n"
            "is sent as a key sequence instead of launching a program.\n"
            "Example — Phrase: \"sendkeys undo\"  Action: \"<CTRL+Z>\""
        )
        sk_row.addWidget(self.cfg_sk_trigger)
        advanced_layout.addRow("Sendkeys Trigger Word:", sk_row)

        # ── WhisperNavigate trigger words ─────────────────────────────────
        _nav_help = (
            "Say this word followed by any text from the last dictation to\n"
            "move the cursor or select that text in the active window.\n"
            "The app navigates by counting characters in the session buffer\n"
            "and sending arrow / Shift key presses.\n"
            "Example: \"WhisperSelect Microsoft Corporation\"\n"
            "The next dictated word will be lowercased automatically."
        )
        self.cfg_sel_trigger = QLineEdit(
            self.config.settings.get("select_trigger", "whisperselect"))
        self.cfg_sel_trigger.setToolTip(
            "Select trigger — navigates to the target text and selects it.\n" + _nav_help)
        advanced_layout.addRow("WhisperSelect Trigger:", self.cfg_sel_trigger)

        self.cfg_move_trigger = QLineEdit(
            self.config.settings.get("move_trigger", "whispermove"))
        self.cfg_move_trigger.setToolTip(
            "Move-before trigger (synonym for WhisperBefore).\n" + _nav_help)
        advanced_layout.addRow("WhisperMove Trigger:", self.cfg_move_trigger)

        self.cfg_movebefore_trigger = QLineEdit(
            self.config.settings.get("movebefore_trigger", "whisperbefore"))
        self.cfg_movebefore_trigger.setToolTip(
            "Move-before trigger — places cursor immediately BEFORE the target.\n" + _nav_help)
        advanced_layout.addRow("WhisperBefore Trigger:", self.cfg_movebefore_trigger)

        self.cfg_moveafter_trigger = QLineEdit(
            self.config.settings.get("moveafter_trigger", "whisperafter"))
        self.cfg_moveafter_trigger.setToolTip(
            "Move-after trigger — places cursor immediately AFTER the target.\n" + _nav_help)
        advanced_layout.addRow("WhisperAfter Trigger:", self.cfg_moveafter_trigger)

        self.cfg_fuzzy_threshold = SpinWidget(
            is_double=True, min_v=0.0, max_v=1.0, step=0.05,
            value=self.config.settings.get("fuzzy_threshold", 0.75),
            decimals=2, use_slider=True, spin_width=70)
        self.cfg_fuzzy_threshold.setToolTip(
            "How closely a spoken word must match a trigger/command phrase\n"
            "to be recognised (uses difflib SequenceMatcher ratio).\n"
            "1.0 = exact match only\n"
            "0.75 = default — handles most Whisper mis-segmentations\n"
            "0.5 = very loose — may fire on unrelated speech\n"
            "0.0 = disabled (exact match only, same as 1.0)"
        )
        advanced_layout.addRow("Trigger Fuzzy Match:", self.cfg_fuzzy_threshold)

        self.cfg_log_level = QComboBox()
        self.cfg_log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "NONE"])
        self.cfg_log_level.setCurrentText(self.config.settings["log_level"])
        self.cfg_log_level.setToolTip(
            "Controls how much is written to app_log.txt.\n"
            "DEBUG: everything (large file, for troubleshooting).\n"
            "INFO: normal operation messages.\n"
            "WARNING/ERROR: only problems.\n"
            "NONE: no log file written at all (best for SSD longevity)."
        )
        advanced_layout.addRow("Logging Level:", self.cfg_log_level)

        self.cfg_use_vad = QCheckBox("Use VAD (Voice Activity Detection)")
        self.cfg_use_vad.setChecked(self.config.settings.get("use_vad", False))
        self.cfg_use_vad.setToolTip(
            "Pre-filters audio to remove silence before passing it to Whisper.\n"
            "Reduces hallucinated words during quiet moments.\n"
            "NOTE: may cause a DLL conflict in the frozen app — disable if crashes occur."
        )
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

    def _delete_terms_row(self):
        row = self.terms_table.currentRow()
        if row >= 0:
            self.terms_table.removeRow(row)

    # ── Terms import/export ──────────────────────────────────────────────────
    # File format: one entry per line, key and value separated by " = "
    # e.g.:  hexagon software = Hexagon Software
    def _import_terms(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Terms", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.terms_table.setRowCount(0)  # replace, not append
            for line in lines:
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    r = self.terms_table.rowCount()
                    self.terms_table.insertRow(r)
                    self.terms_table.setItem(r, 0, QTableWidgetItem(k.strip()))
                    self.terms_table.setItem(r, 1, QTableWidgetItem(v.strip()))
            app_logger.info(f"Terms imported from {path}")
        except Exception as e:
            app_logger.error(f"Failed to import terms: {e}")

    def _export_terms(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Terms", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = []
            for r in range(self.terms_table.rowCount()):
                k = self.terms_table.item(r, 0)
                v = self.terms_table.item(r, 1)
                if k and v:
                    lines.append(f"{k.text()} = {v.text()}")
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            app_logger.info(f"Terms exported to {path}")
        except Exception as e:
            app_logger.error(f"Failed to export terms: {e}")

    # ── Commands import/export ───────────────────────────────────────────────
    def _import_commands(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Commands", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            self.cmd_table.setRowCount(0)  # replace, not append
            for line in lines:
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    r = self.cmd_table.rowCount()
                    self.cmd_table.insertRow(r)
                    self.cmd_table.setItem(r, 0, QTableWidgetItem(k.strip()))
                    self.cmd_table.setItem(r, 1, QTableWidgetItem(v.strip()))
            app_logger.info(f"Commands imported from {path}")
        except Exception as e:
            app_logger.error(f"Failed to import commands: {e}")

    def _export_commands(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Commands", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = []
            for r in range(self.cmd_table.rowCount()):
                k = self.cmd_table.item(r, 0)
                v = self.cmd_table.item(r, 1)
                if k and v:
                    lines.append(f"{k.text()} = {v.text()}")
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            app_logger.info(f"Commands exported to {path}")
        except Exception as e:
            app_logger.error(f"Failed to export commands: {e}")

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
            "terms": {
                self.terms_table.item(r, 0).text(): self.terms_table.item(r, 1).text()
                for r in range(self.terms_table.rowCount())
                if self.terms_table.item(r, 0) and self.terms_table.item(r, 1)
                and self.terms_table.item(r, 0).text()
            },
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
            "use_confidence": self.cfg_use_confidence.isChecked(),
            "min_confidence": round(self.cfg_conf_spin.value(), 2),
            "ft_output_folder": self.ft_output_folder.text(),
            "ft_mon_folder": self.ft_mon_folder.text(),
            "ft_mon_enabled": self.ft_mon_enabled.isChecked(),
            "use_vad": self.cfg_use_vad.isChecked(),
            "sendkeys_trigger":   self.cfg_sk_trigger.text().strip() or "sendkeys",
            "select_trigger":     self.cfg_sel_trigger.text().strip() or "whisperselect",
            "move_trigger":       self.cfg_move_trigger.text().strip() or "whispermove",
            "movebefore_trigger": self.cfg_movebefore_trigger.text().strip() or "whisperbefore",
            "moveafter_trigger":  self.cfg_moveafter_trigger.text().strip() or "whisperafter",
            "fuzzy_threshold":    self.cfg_fuzzy_threshold.value()
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
            # Reset session buffer and cursor-ops flag for the new session
            self._session_buffer = ""
            self._cursor_ops_pending = False
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

    def _install_ptt_hook(self, parts):
        """Install a WH_KEYBOARD_LL hook that consumes the PTT combo keys
        so they are never forwarded to the active application."""
        import ctypes, ctypes.wintypes, threading
        if getattr(self, '_ptt_hook_installed', False):
            return
        ptt_vks = {self._VK_MAP[p] for p in parts if p in self._VK_MAP}

        WH_KEYBOARD_LL = 13
        WM_KEYDOWN     = 0x0100
        WM_SYSKEYDOWN  = 0x0104
        WM_KEYUP       = 0x0101
        WM_SYSKEYUP    = 0x0105

        HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_int,
                                      ctypes.wintypes.WPARAM, ctypes.wintypes.LPARAM)

        class KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [("vkCode",      ctypes.wintypes.DWORD),
                        ("scanCode",    ctypes.wintypes.DWORD),
                        ("flags",       ctypes.wintypes.DWORD),
                        ("time",        ctypes.wintypes.DWORD),
                        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        def hook_proc(nCode, wParam, lParam):
            if nCode >= 0 and wParam in (WM_KEYDOWN, WM_SYSKEYDOWN, WM_KEYUP, WM_SYSKEYUP):
                kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if kb.vkCode in ptt_vks:
                    return 1  # consume — do not forward to application
            return user32.CallNextHookEx(None, nCode, wParam, lParam)

        self._ptt_hook_cb  = HOOKPROC(hook_proc)
        self._ptt_hook_handle = None

        def pump():
            hmod = kernel32.GetModuleHandleW(None)
            h = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._ptt_hook_cb, hmod, 0)
            self._ptt_hook_handle = h
            if not h:
                app_logger.warning("PTT hook: SetWindowsHookExW failed")
                return
            msg = ctypes.wintypes.MSG()
            while getattr(self, '_ptt_hook_installed', False):
                r = user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1)
                if r:
                    user32.TranslateMessage(ctypes.byref(msg))
                    user32.DispatchMessageW(ctypes.byref(msg))
                else:
                    import time as _t; _t.sleep(0.005)
            if h:
                user32.UnhookWindowsHookEx(h)
            self._ptt_hook_handle = None

        self._ptt_hook_installed = True
        t = threading.Thread(target=pump, daemon=True, name="ptt-hook")
        t.start()
        self._ptt_hook_thread = t
        app_logger.debug("PTT suppression hook installed")

    def _remove_ptt_hook(self):
        """Signal the hook pump thread to stop and uninstall the hook."""
        self._ptt_hook_installed = False
        # The pump loop will call UnhookWindowsHookEx and exit on its own
        app_logger.debug("PTT suppression hook removed")

    def _poll_ptt(self):
        """Poll Win32 GetAsyncKeyState for the PTT combo. Called by QTimer.
        PTT is always available — no mode switch required.
        Keys are suppressed via a low-level hook while the combo is held."""
        try:
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
                if self._ptt_starting and self.recorder and self.recorder.active:
                    self._ptt_starting = False

                # Install key-suppression hook on first combo detection so the
                # PTT keys are eaten and never reach the foreground application.
                if not getattr(self, '_ptt_hook_installed', False):
                    self._install_ptt_hook(parts)

                # Only auto-start if not already running AND not mid-start.
                if not self._ptt_starting and (not self.recorder or not self.recorder.active):
                    app_logger.debug("PTT held — auto-starting recorder session")
                    self._ptt_starting = True
                    self.toggle_rec()

                if self.recorder and not self.recorder.ptt_pressed:
                    self.recorder.ptt_pressed = True
                    app_logger.debug("PTT activated (poll)")
            else:
                # PTT released — remove suppression hook, stop recording, clear guard
                if getattr(self, '_ptt_hook_installed', False):
                    self._remove_ptt_hook()
                self._ptt_starting = False
                if self.recorder and self.recorder.ptt_pressed:
                    self.recorder.ptt_pressed = False
                    app_logger.debug("PTT deactivated (poll)")
                    # Stop the recorder: PTT is hold-to-talk, not a toggle.
                    # Stopping via toggle_rec must happen on the Qt main thread.
                    if self.recorder and self.recorder.active:
                        self.sig_toggle_rec.emit()
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


    def _whisper_navigate(self, operation, target_text):
        """Perform a cursor navigation / selection operation inside the active window.

        The app cannot read the active window's content, so it works purely from
        the session buffer (all text pasted since the current dictation session
        started).  It navigates by counting characters and sending arrow / shift
        key presses — identical to what a user would do manually.

        Parameters
        ----------
        operation : str
            One of: "select" | "move" | "movebefore" | "moveafter"
        target_text : str
            The substring to locate in the session buffer.

        Cursor movement strategy
        ------------------------
        The cursor is assumed to be AT THE END of the session buffer.

        1.  Find the last occurrence of target_text in the session buffer
            (case-insensitive).
        2.  Compute:
              chars_from_end = len(session_buffer) - match_end_index
              (how many Left presses to reach just after the match)
        3.  Depending on operation:
              "movebefore" / "move":
                    Left × (chars_from_end + len(target_text))
                    → cursor lands immediately BEFORE the target
              "moveafter":
                    Left × chars_from_end
                    → cursor lands immediately AFTER the target
              "select":
                    Left × chars_from_end                    (no shift)
                    then Shift+Left × len(target_text)        (select backwards)
                    … actually: navigate to just before target, then
                    Shift+Right × len(target_text)
        """
        import re as _re
        import pyautogui as _pag
        import ctypes as _ct
        import time as _time

        buf   = self._session_buffer
        clean = target_text.strip()
        if not clean:
            self.scratchpad.append("[Navigate] Empty target — nothing to do.")
            return
        if not buf:
            self.scratchpad.append("[Navigate] Session buffer is empty — start dictation first.")
            return

        # Case-insensitive search — find the LAST occurrence
        pattern = _re.compile(_re.escape(clean), _re.IGNORECASE)
        matches = list(pattern.finditer(buf))
        if not matches:
            self.scratchpad.append(f"[Navigate] '{clean}' not found in session buffer.")
            return

        m           = matches[-1]           # last occurrence
        match_start = m.start()
        match_end   = m.end()
        buf_len     = len(buf)

        # ── Release any held modifier keys before sending arrows ────────────
        _VK_CONTROL = 0x11; _VK_SHIFT = 0x10; _VK_MENU = 0x12
        _KEYUP = 0x0002
        for _vk in (_VK_CONTROL, _VK_SHIFT, _VK_MENU):
            _ct.windll.user32.keybd_event(_vk, 0, _KEYUP, 0)
        _time.sleep(0.08)

        paste_delay = self.config.settings.get("paste_delay", 0.5)

        # Chars from current cursor position (end of buffer) to end of match
        chars_to_match_end   = buf_len - match_end    # Left presses to reach just after match
        chars_in_match       = match_end - match_start

        try:
            if operation in ("move", "movebefore"):
                # Move cursor to just BEFORE the target text
                lefts = chars_to_match_end + chars_in_match
                app_logger.info(f"Navigate movebefore '{clean}': {lefts} Left presses")
                for _ in range(lefts):
                    _pag.press("left")
                self.scratchpad.append(
                    f"[Navigate] Cursor moved before '{clean}' ({lefts} ◀ presses)")

            elif operation == "moveafter":
                # Move cursor to just AFTER the target text
                lefts = chars_to_match_end
                app_logger.info(f"Navigate moveafter '{clean}': {lefts} Left presses")
                for _ in range(lefts):
                    _pag.press("left")
                self.scratchpad.append(
                    f"[Navigate] Cursor moved after '{clean}' ({lefts} ◀ presses)")

            elif operation == "select":
                # Move to just before target, then Shift+Right to select it
                lefts = chars_to_match_end + chars_in_match
                app_logger.info(
                    f"Navigate select '{clean}': {lefts} Left, then {chars_in_match} Shift+Right")
                for _ in range(lefts):
                    _pag.press("left")
                # Now select the text forward
                for _ in range(chars_in_match):
                    _pag.keyDown("shift")
                _pag.keyDown("shift")   # make sure shift is held
                for _ in range(chars_in_match):
                    _pag.press("right")
                _pag.keyUp("shift")
                self.scratchpad.append(
                    f"[Navigate] Selected '{clean}' "
                    f"({lefts} ◀ then {chars_in_match} ⇧▶)")

            # Arm lowercase flag so the next dictated segment joins cleanly
            self._cursor_ops_pending = True

        except Exception as e:
            app_logger.error(f"Navigate error: {e}", exc_info=True)
            self.scratchpad.append(f"[Navigate] Error: {e}")

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

        # Tray icon — when idle use the app icon if available, otherwise a dot
        if hasattr(self, 'tray'):
            if state == 'idle' and hasattr(self, '_app_icon') and not self._app_icon.isNull():
                self.tray.setIcon(self._app_icon)
            else:
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
        # Route transcript text to Results pane; system messages go to Log pane (scratchpad)
        _results = getattr(self, "results_area", self.scratchpad)
        _results.append(f"[{timestamp}] {text}")

        if src != "live":
            # ── File transcription result ─────────────────────────────────────
            # Always copy to clipboard so user can paste anywhere.
            try:
                pyperclip.copy(text)
                app_logger.info("File transcription copied to clipboard")
            except Exception as e:
                app_logger.error(f"Clipboard copy failed: {e}")
            # Auto-save to output folder
            # Read directly from the UI widget so path is current even if user
            # hasn't clicked Save yet.
            _ft_out_raw = (self.ft_output_folder.text().strip()
                           if hasattr(self, 'ft_output_folder') else
                           self.config.settings.get('ft_output_folder', ''))
            out_dir = Path(_ft_out_raw).expanduser() if _ft_out_raw else None
            if out_dir:
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    import re as _re2
                    _words = [_re2.sub(r'[^A-Za-z0-9]', '', w) for w in text.split()[:4]]
                    _words = [w for w in _words if w]  # drop empty after stripping
                    first_words = '-'.join(_words) if _words else 'transcription'
                    fname = datetime.now().strftime('%Y-%m-%d-%H-%M') + f"-{first_words}.txt"
                    out_path = out_dir / fname
                    out_path.write_text(text, encoding='utf-8')
                    src_name = os.path.basename(src) if src else "?"
                    _results = getattr(self, "results_area", self.scratchpad)
                    _results.append(f"[File] {src_name}  \u2192  {out_path}")
                    app_logger.info(f"File transcription saved: {out_path}")
                    # Remove from the queue list if it's still there
                    if hasattr(self, 'ft_list'):
                        for i in range(self.ft_list.count()):
                            item = self.ft_list.item(i)
                            if item and os.path.basename(item.text()) in text or True:
                                pass  # best-effort; remove by tracking in _ft_pending
                except Exception as e:
                    app_logger.error(f"Failed to save transcription: {e}", exc_info=True)
            # Track completed count and update status label
            self._ft_done  = getattr(self, "_ft_done",  0) + 1
            self._ft_total = getattr(self, "_ft_total", self._ft_done)
            if hasattr(self, "ft_status_lbl"):
                if self._ft_done >= self._ft_total:
                    self.ft_status_lbl.setText(
                        f"Done: {self._ft_done} file(s) transcribed successfully.")
                    self._ft_done  = 0  # reset after full batch
                    self._ft_total = 0
                else:
                    self.ft_status_lbl.setText(
                        f"Transcribing... {self._ft_done} / {self._ft_total} complete")
            return

        if src == "live":
            # ── Command detection (runs BEFORE paste) ────────────────────────
            # ── WhisperNavigate: select / move cursor ──────────────────────
            # Check BEFORE command and paste handling — navigation is its own
            # operation; we do not paste the trigger phrase itself.
            _cfg      = self.config.settings
            _fuzz_thr = float(_cfg.get("fuzzy_threshold", 0.75))
            _nav_triggers = [
                ("select",     _cfg.get("select_trigger",     "whisperselect").lower()),
                ("move",       _cfg.get("move_trigger",       "whispermove").lower()),
                ("movebefore", _cfg.get("movebefore_trigger", "whisperbefore").lower()),
                ("moveafter",  _cfg.get("moveafter_trigger",  "whisperafter").lower()),
            ]
            _nav_fired = False
            for _nav_op, _nav_kw in _nav_triggers:
                if not _nav_kw:
                    continue
                _matched, _nav_target = _fuzzy_trigger_match(text, _nav_kw, _fuzz_thr)
                if _matched:
                    # Strip leading filler words Whisper loves to add
                    import re as _re_nav
                    _nav_target = _re_nav.sub(
                        r"^(to|before|after|the|for)\s+", "",
                        _nav_target, flags=_re_nav.IGNORECASE).strip()
                    app_logger.info(
                        f"WhisperNavigate (fuzzy): op={_nav_op!r}, target={_nav_target!r}")
                    self._whisper_navigate(_nav_op, _nav_target)
                    _nav_fired = True
                    break
            if _nav_fired:
                app_logger.debug("  on_text: navigate fired — skipping paste")
                return

            # If the transcribed text matches a voice command, execute it and
            # do NOT paste the text — the spoken phrase was a command, not prose.
            _cmd_fired = False
            app_logger.debug(f"  on_text: Checking {len(self.config.settings['commands'])} voice commands...")
            sk_trigger = self.config.settings.get("sendkeys_trigger", "sendkeys").lower().strip()
            for phrase, cmd in self.config.settings["commands"].items():
                # Exact match first; fall back to fuzzy if threshold > 0
                _phrase_l = phrase.lower()
                _exact    = _phrase_l in text.lower()
                _fuzz_cmd = False
                if not _exact and _fuzz_thr > 0:
                    # Slide a window of len(phrase.split()) words across the spoken text
                    from difflib import SequenceMatcher as _SM
                    _sw = text.lower().split()
                    _pw = _phrase_l.split()
                    _wn = len(_pw)
                    for _wi in range(max(1, len(_sw) - _wn + 1)):
                        _window = " ".join(_sw[_wi:_wi + _wn])
                        if _SM(None, _phrase_l, _window).ratio() >= _fuzz_thr:
                            _fuzz_cmd = True
                            app_logger.debug(
                                f"  on_text: fuzzy command match '{phrase}' ~ '{_window}'")
                            break
                if _exact or _fuzz_cmd:
                    app_logger.debug(f"  on_text: Command matched: '{phrase}' → '{cmd}'")
                    try:
                        # If the trigger phrase appears (exact or fuzzy) in the phrase,
                        # treat the action field as a key sequence rather than a program.
                        _sk_exact = sk_trigger and sk_trigger in phrase.lower()
                        _sk_fuzzy = (not _sk_exact and sk_trigger and
                                     _fuzzy_trigger_match(phrase.lower(), sk_trigger, _fuzz_thr)[0])
                        if _sk_exact or _sk_fuzzy:
                            _send_keys_sequence(
                                cmd,
                                paste_delay=self.config.settings.get("paste_delay", 0.5))
                            app_logger.info(f"Sendkeys sequence sent: {cmd}")
                            self.scratchpad.append(f"[Sendkeys] {phrase} → {cmd}")
                        else:
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
                # _rollback_pending is set by toggle_rec consuming _rollback_armed,
                # but rollback can also fire mid-session (recording already running),
                # in which case toggle_rec never fires and we must consume _rollback_armed here.
                if not self._rollback_pending and getattr(self, '_rollback_armed', False):
                    self._rollback_pending = True
                    self._rollback_armed = False
                if (self._rollback_pending or self._cursor_ops_pending) and text:
                    text = text[0].lower() + text[1:]
                    self._rollback_pending = False
                    self._cursor_ops_pending = False
                # Apply term replacements (case-insensitive, exact output preserved)
                import re as _re
                _key_tag_sequences = []  # term replacements that contain <KEY> tags
                for phrase, replacement in self.config.settings.get("terms", {}).items():
                    if phrase.strip() and _re.search(_re.escape(phrase), text, _re.IGNORECASE):
                        if "<" in replacement and ">" in replacement:
                            # Contains special key tags — collect for sendkeys after paste
                            _key_tag_sequences.append(replacement)
                            # Remove the matched phrase from text so it isn't pasted literally
                            text = _re.sub(_re.escape(phrase), "", text, flags=_re.IGNORECASE).strip()
                        else:
                            text = _re.sub(_re.escape(phrase), replacement,
                                           text, flags=_re.IGNORECASE)
                p_text = text + " " if auto_space else text
                # Strip trailing punctuation/spaces to find the last real word,
                # then record how many characters were actually output.
                self._last_paste_len = len(p_text)
                self._last_paste_text = p_text
                self._session_buffer += p_text  # accumulate for select/move

                paste_delay = self.config.settings["paste_delay"]
                app_logger.debug(f"  on_text: auto_space={auto_space}, paste_delay={paste_delay}s, p_text length={len(p_text)}")
                try:
                    pyperclip.copy(p_text)
                    time.sleep(paste_delay)
                    pyautogui.hotkey('ctrl', 'v')
                    app_logger.info(f"Text pasted: '{text[:30]}{'...' if len(text)>30 else ''}'")
                    # Fire any term sendkeys sequences now that the text is pasted
                    if _key_tag_sequences:
                        time.sleep(max(paste_delay, 0.1))
                        for _seq in _key_tag_sequences:
                            _send_keys_sequence(_seq, paste_delay=paste_delay)
                            app_logger.info(f"Term sendkeys: {_seq}")
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
        """Fallback timer-based monitor — catches anything QFileSystemWatcher misses."""
        if not self.config.settings.get("ft_mon_enabled", False):
            return
        mon_path = self.config.settings.get("ft_mon_folder", "").strip()
        if not mon_path:
            return
        root = Path(mon_path)
        if not root.exists():
            return
        # Reuse the same scan logic as the watcher callback
        self._ft_mon_scan(str(root))
    
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
        """Legacy shim — redirects to the File Transcription tab helper."""
        self._ft_add_files()

    # ── File Transcription tab helpers ──────────────────────────────────────

    _FT_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.mp4', '.ogg', '.flac', '.aac', '.wma'}

    def _ft_add_files(self):
        """Open file picker and add selected audio files to the queue list."""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "Audio Files (*.wav *.mp3 *.m4a *.mp4 *.ogg *.flac *.aac *.wma)"
        )
        for p in paths:
            p = os.path.abspath(p)
            # Avoid duplicates
            existing = [self.ft_list.item(i).text()
                        for i in range(self.ft_list.count())]
            if p not in existing:
                self.ft_list.addItem(p)
        app_logger.info(f"Added {len(paths)} files to transcription queue")

    def _ft_remove_selected(self):
        for item in self.ft_list.selectedItems():
            self.ft_list.takeItem(self.ft_list.row(item))

    def _ft_start_transcription(self):
        """Submit all queued files to the transcriber."""
        n = self.ft_list.count()
        app_logger.debug(f"_ft_start_transcription: ft_list.count()={n}")
        paths = [self.ft_list.item(i).text() for i in range(n)]
        if not paths:
            self.ft_status_lbl.setText("Queue is empty — use Add Files... to add audio files.")
            self.scratchpad.append("[File] Transcribe button pressed but queue is empty.")
            return
        self._ft_pending = list(paths)
        self._ft_total   = len(paths)
        self.ft_status_lbl.setText(f"Transcribing {self._ft_total} file(s)...")
        self.scratchpad.append(f"[File] Starting transcription of {self._ft_total} file(s).")
        # Submit all, then clear the list
        for p in paths:
            if not os.path.isfile(p):
                self.scratchpad.append(f"[File] Skipped (not found): {p}")
                app_logger.warning(f"_ft_start_transcription: file not found: {p}")
                continue
            self.transcriber.submit(p, p)  # src=path so on_text knows filename
            app_logger.info(f"_ft_start_transcription: submitted {p}")
        self.ft_list.clear()

    def _ft_mon_toggled(self, enabled):
        """Start or stop the QFileSystemWatcher when the monitor toggle changes."""
        self.config.settings["ft_mon_enabled"] = enabled
        self.config.settings["ft_mon_folder"]  = self.ft_mon_folder.text()
        self._setup_ft_watcher()

    def _setup_ft_watcher(self):
        """Install or remove a QFileSystemWatcher on the monitor folder."""
        from PyQt6.QtCore import QFileSystemWatcher
        if not hasattr(self, '_ft_watcher'):
            self._ft_watcher = QFileSystemWatcher(self)
            self._ft_watcher.directoryChanged.connect(self._ft_mon_dir_changed)
        # Remove all existing watched paths
        if self._ft_watcher.directories():
            self._ft_watcher.removePaths(self._ft_watcher.directories())
        if self.config.settings.get("ft_mon_enabled") and self.ft_mon_folder.text().strip():
            mon = self.ft_mon_folder.text().strip()
            Path(mon).mkdir(parents=True, exist_ok=True)
            self._ft_watcher.addPath(mon)
            app_logger.info(f"Folder watcher active on: {mon}")

    def _ft_mon_dir_changed(self, path):
        """Called by QFileSystemWatcher when the watched directory changes.
        We defer the actual scan by 500ms so the OS finishes writing/copying
        all files in the batch before we try to open them."""
        QTimer.singleShot(500, lambda: self._ft_mon_scan(path))

    def _ft_mon_scan(self, path):
        """Scan the monitored folder for new audio files, move each to Processed/,
        and submit it for transcription immediately (no manual button needed)."""
        proc_dir = Path(path) / "Processed"
        try:
            proc_dir.mkdir(exist_ok=True)
        except Exception as e:
            app_logger.error(f"Monitor: cannot create Processed dir: {e}")
            return
        added = 0
        # Snapshot the directory first, then process the snapshot — avoids
        # the iteration-while-moving race that causes files to be skipped.
        try:
            candidates = [f for f in Path(path).iterdir()
                          if f.is_file() and f.suffix.lower() in self._FT_AUDIO_EXTS]
        except Exception as e:
            app_logger.error(f"Monitor: cannot list {path}: {e}")
            return
        for f in candidates:
            dst = proc_dir / f.name
            try:
                shutil.move(str(f), str(dst))
            except Exception as e:
                app_logger.error(f"Monitor: move failed for {f.name}: {e}")
                continue
            # Submit for transcription immediately
            self.transcriber.submit(str(dst), str(dst))  # src=path
            # Show in the status label so user can see what's happening
            added += 1
            app_logger.info(f"Monitor: auto-transcribing {f.name}")
        if added:
            app_logger.info(f"Monitor: submitted {added} file(s) from {path}")
            # Set _ft_total so the completion counter works correctly
            self._ft_total = getattr(self, "_ft_total", 0) + added
            self._ft_done  = getattr(self, "_ft_done",  0)
            if hasattr(self, 'ft_status_lbl'):
                self.ft_status_lbl.setText(f"Monitor: auto-transcribing {added} file(s)...")

    def eventFilter(self, obj, event):
        """Catch drag-and-drop onto ft_list."""
        from PyQt6.QtCore import QEvent
        from PyQt6.QtGui import QDragEnterEvent, QDropEvent
        if hasattr(self, 'ft_list') and obj is self.ft_list:
            if event.type() == QEvent.Type.DragEnter:
                if event.mimeData().hasUrls():
                    event.acceptProposedAction()
                    return True
            elif event.type() == QEvent.Type.Drop:
                for url in event.mimeData().urls():
                    p = url.toLocalFile()
                    if Path(p).suffix.lower() in self._FT_AUDIO_EXTS:
                        existing = [self.ft_list.item(i).text()
                                    for i in range(self.ft_list.count())]
                        if p not in existing:
                            self.ft_list.addItem(p)
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)

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
    # freeze_support() already called at module top — do not call again.
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
        app.setStyleSheet(_build_dark_style())
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
