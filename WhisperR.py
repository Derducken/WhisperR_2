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

            segments, _ = model.transcribe(
                audio_np,
                language=lang_code if lang_code != 'auto' else None,
                task='translate' if translate else 'transcribe',
                vad_filter=use_vad,
                initial_prompt=prompt or None,
            )
            seg_list = list(segments)
            # Emit each segment with its logprob so the main process can filter.
            # Filtering here would block nav/command triggers from being detected.
            if seg_list:
                _log(f"  segs={len(seg_list)}, "
                     f"logprobs={[round(s.avg_logprob,3) for s in seg_list]}")
            # Send as a list of (text, logprob) pairs so the receiver can filter
            seg_data = [(s.text.strip(), round(s.avg_logprob, 4)) for s in seg_list]
            result_q.put(('text', seg_data, src))
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
    QSystemTrayIcon, QMenu, QSlider, QListWidget, QListWidgetItem, QAbstractItemView, QSplitter,
    QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QObject, QEvent
from PyQt6.QtGui import (QPainter, QColor, QFont, QIcon, QAction, QKeyEvent, QPixmap, QPen,
                         QSyntaxHighlighter, QTextCharFormat, QKeySequence, QShortcut)

# --- 3. CONSTANTS ---
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]
LANG_MAP = {"Auto": None, "English": "en", "Greek": "el", "German": "de", "French": "fr", "Spanish": "es"}
# Known Whisper hallucination patterns — matched as SUBSTRINGS (case-insensitive).
# A transcription is dropped if its entire text (stripped) is one of these,
# OR if it STARTS WITH one of these hallucination prefixes.
HALLUCINATIONS = [
    "thank you.", "thanks for watching.", "god bless.", "god bless you.",
    "subtitles by", "amara.org", "translated by", "transcribed by",
    "please subscribe", "don't forget to subscribe", "like and subscribe",
    "thanks for watching, and i'll see you",
    "thank you for watching",
    "this video was",
]

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
            "editor_hotkey": "ctrl+shift+e",
            "rollback_hotkey": "ctrl+shift+z",
            "live_mode": "Auto-Pause",
            "dict_mode": "Auto-Pause", "auto_pause_sec": 2.0,
            "noise_floor": 50, "speech_vol": 211,
            "commands": {"Launch Notepad": "notepad.exe"},
            "terms": {"hexagon software": "Hexagon Software"},
            "hallucinations": [
                "thank you.", "thanks for watching.", "god bless.", "god bless you.",
                "subtitles by", "amara.org", "translated by", "transcribed by",
                "please subscribe", "don't forget to subscribe", "like and subscribe",
                "thanks for watching, and i'll see you",
                "thank you for watching",
                "this video was"
            ],
            "ind_show": True, "ind_type": "Both", "ind_pos": "Top-Left",
            "ind_size": 32, "ind_off": 5, "bar_edge": "Top", "bar_size": 5,
            "bar_thickness": 3, "ind_opacity": 220, "bar_opacity": 220,
            "ind_hide_idle": True,
            "log_level": "NONE", "use_vad": True,
            "ft_output_folder": str(Path.home() / "WhisperR_Output"),
            "ft_mon_folder": str(Path.home() / "WhisperR_Watch"),
            "ft_mon_enabled": False,
            "use_confidence": True, "min_confidence": 0.9,
            "editor_type_trigger":  "whisper type, whisper write",
            "editor_edit_trigger":  "whisper edit, whisper edit this",
            "editor_paste_trigger": "whisper paste, whisper done, whisper okay",
            "editor_hk_bold":      "Ctrl+B",  "editor_hk_italic":    "Ctrl+I",
            "editor_hk_strike":    "Ctrl+Shift+S", "editor_hk_highlight": "Ctrl+Shift+H",
            "editor_hk_code":      "Ctrl+`",
            "editor_hk_h1":        "Ctrl+1",  "editor_hk_h2":        "Ctrl+2",
            "editor_hk_h3":        "Ctrl+3",  "editor_hk_emdash":    "Ctrl+Shift+Minus",
            "editor_hk_bullet":    "Ctrl+Shift+B", "editor_hk_numlist": "Ctrl+Shift+N",
            "editor_hk_tasklist":  "Ctrl+Shift+T", "editor_hk_kbd":     "Ctrl+Shift+D",
            "editor_hk_link":    "Ctrl+K",
            "sendkeys_trigger":   "whisper send keys",
            "select_trigger":     "whisper select",
            "move_trigger":       "whisper move",
            "movebefore_trigger": "whisper before",
            "moveafter_trigger":      "whisper after",
            "replace_trigger":        "whisper replace",
            "insertbefore_trigger":   "whisper insert before",
            "insertafter_trigger":    "whisper insert after",
            "fuzzy_threshold":        0.75
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
        self._last_seg_data = []  # [(text, logprob)] from last transcription
        app_logger.info("Transcriber worker initialized")

    def _cfg_min_confidence(self):
        """Return the live min_confidence value from config settings."""
        return float(self.config.settings.get("min_confidence", 0.0))

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
                    _, seg_data, src = msg
                    # seg_data is list of (text, logprob) pairs from worker.
                    # Emit full text to on_text — confidence filtering happens
                    # there, AFTER wizard/command/trigger checks, so triggers
                    # are never silently dropped by the confidence gate.
                    # Hallucination check only — these are never valid speech.
                    text = ' '.join(st for st, _ in seg_data).strip()
                    if not text:
                        continue
                    _tl = text.lower().strip()
                    _hall_list = self.config.settings.get("hallucinations", HALLUCINATIONS)
                    _hall = any(_tl == h.lower() or _tl.startswith(h.lower())
                                for h in _hall_list)
                    if _hall:
                        app_logger.debug(f"  hallucination filtered: {text!r}")
                        continue
                    # Stash seg_data so on_text can apply per-segment confidence
                    # filtering on the paste path only (triggers see full text).
                    self._last_seg_data = seg_data
                    app_logger.info(f"Transcription: '{text[:50]}...'")
                    self.finished_text.emit(text, src)
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
    import pyautogui as _pag
    _pag.FAILSAFE = False

    # Normalise tag formats before tokenising:
    #   <CTRL>Z       → <CTRL+Z>
    #   <CTRL>+Z      → <CTRL+Z>
    #   <CTRL><Z>     → <CTRL+Z>   (two consecutive tags)
    # Strategy: collapse sequences of <TAG> tokens that are immediately
    # adjacent (no text between them) into a single <TAG+TAG> form.
    import re as _re
    # Step 1: <CTRL>+Z → <CTRL+Z>  (tag immediately followed by +key)
    text = _re.sub(r'<([^>]+)>\+([A-Za-z0-9]+)', r'<\1+\2>', text)
    # Step 2: <CTRL>Z → <CTRL+Z>   (tag immediately followed by bare key chars)
    text = _re.sub(r'<([^>]+)>([A-Za-z0-9]+)', r'<\1+\2>', text)
    # Step 3: <CTRL><Z> → <CTRL+Z> (two adjacent tags, no text between)
    text = _re.sub(r'<([^>]+)><([^>]+)>', r'<\1+\2>', text)

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



def _phrase_aliases(phrase: str) -> list[str]:
    """Split a comma-separated phrase field into individual detection aliases.

    "Launch Notepad, Lunch Not Bad, Lots Not Bat" -> three strings.
    Single-phrase fields (no comma) return a one-element list unchanged.
    """
    return [p.strip() for p in phrase.split(",") if p.strip()]


def _any_alias_matches(spoken: str, phrase_field: str, fuzz_thr: float = 0.75) -> bool:
    """True if any alias in phrase_field matches spoken text (exact or fuzzy).

    Whisper often inserts commas/periods into short commands, e.g. transcribing
    "whisper edit this" as "Whisper, edit this." — so we match against both the
    raw spoken text AND a punctuation-stripped version.
    """
    import re as _re_am
    from difflib import SequenceMatcher as _SM
    spoken_l        = spoken.lower()
    spoken_stripped = _re_am.sub(r"[^\w\s]", "", spoken_l).strip()
    for alias in _phrase_aliases(phrase_field):
        alias_l = alias.lower().strip()
        if alias_l in spoken_l or alias_l in spoken_stripped:
            return True
        if fuzz_thr > 0:
            aw = alias_l.split(); wn = len(aw)
            for candidate in (spoken_l, spoken_stripped):
                sw = candidate.split()
                for i in range(max(1, len(sw) - wn + 1)):
                    if _SM(None, alias_l, " ".join(sw[i:i+wn])).ratio() >= fuzz_thr:
                        return True
    return False


def _editor_trigger_matches(spoken: str, phrase_field: str) -> bool:
    """Exact-only match for editor open/close triggers.

    Editor triggers are short "whisper X" commands (2-3 words).  Using
    fuzzy matching between them causes fatal cross-fires:
      whisper type ↔ whisper edit  score 0.750  (exactly at threshold)
      whisper write ↔ whisper edit score 0.800
    So editor triggers use ONLY punctuation-stripped exact substring matching —
    no fuzzy scoring at all.
    """
    import re as _re_et
    spoken_l        = spoken.lower()
    spoken_stripped = _re_et.sub(r"[^\w\s]", "", spoken_l).strip()
    for alias in _phrase_aliases(phrase_field):
        alias_l = alias.lower().strip()
        if alias_l in spoken_l or alias_l in spoken_stripped:
            return True
    return False



def _capture_selection_from_hwnd(hwnd) -> str:
    """Send Ctrl+C to hwnd via SendInput and return whatever lands on the clipboard.

    Uses AttachThreadInput to give the target window keyboard focus just long
    enough to inject the keystrokes, then detaches.  Returns empty string on
    any failure or if the clipboard didn't change (nothing was selected).
    """
    import ctypes, ctypes.wintypes, time
    try:
        import pyperclip as _pcp
        _old = _pcp.paste()
    except Exception:
        _old = ""

    if not hwnd:
        return _old  # no source window — return whatever is already on clipboard

    try:
        u32 = ctypes.windll.user32

        VK_CONTROL       = 0x11
        VK_C             = 0x43
        KEYEVENTF_KEYUP  = 0x0002
        INPUT_KEYBOARD   = 1

        class _KI(ctypes.Structure):
            _fields_ = [
                ("wVk",         ctypes.wintypes.WORD),
                ("wScan",       ctypes.wintypes.WORD),
                ("dwFlags",     ctypes.wintypes.DWORD),
                ("time",        ctypes.wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
            ]
        class _INPUT(ctypes.Structure):
            class _U(ctypes.Union):
                _fields_ = [("ki", _KI)]
            _anonymous_ = ("_u",)
            _fields_    = [("type", ctypes.wintypes.DWORD), ("_u", _U)]

        def _make(vk, flags=0):
            i = _INPUT(); i.type = INPUT_KEYBOARD
            i.ki.wVk = vk; i.ki.dwFlags = flags
            return i

        fg      = u32.GetForegroundWindow()
        t_fg    = u32.GetWindowThreadProcessId(fg,   None)
        t_src   = u32.GetWindowThreadProcessId(hwnd, None)

        u32.AttachThreadInput(t_fg, t_src, True)
        u32.SetForegroundWindow(hwnd)
        u32.BringWindowToTop(hwnd)
        time.sleep(0.12)

        inputs = (_INPUT * 4)(
            _make(VK_CONTROL),
            _make(VK_C),
            _make(VK_C,       KEYEVENTF_KEYUP),
            _make(VK_CONTROL, KEYEVENTF_KEYUP),
        )
        u32.SendInput(4, inputs, ctypes.sizeof(_INPUT))
        time.sleep(0.25)   # wait for clipboard to update

        u32.AttachThreadInput(t_fg, t_src, False)

        try:
            new = _pcp.paste()
        except Exception:
            new = _old

        # If clipboard didn't change, nothing was selected — return empty
        # so the editor opens blank rather than with stale clipboard content.
        return new if new != _old else ""

    except Exception as _e:
        try:
            import logging; logging.getLogger("WhisperR").warning(f"_capture_selection_from_hwnd: {_e}")
        except Exception:
            pass
        return ""


def _smart_case_punct(original: str, new_text: str) -> str:
    """Adapt new_text capitalisation and terminal punctuation to match original.

    Rules:
      • If original starts with an uppercase letter → capitalise first letter of new_text.
      • If original is a "complete sentence" (ends with . ! ? … or an emoji-stop) AND
        new_text has no terminal punctuation → append the same terminal character.
      • "Complete sentence" = original ends with . ! ? or … (after stripping trailing spaces).
      • If new_text already ends with punctuation, leave it alone.
    """
    import re as _re_sc
    nt = new_text.strip()
    if not nt:
        return new_text

    # ── Capitalisation ───────────────────────────────────────────────────────
    orig_stripped = original.lstrip()
    if orig_stripped and orig_stripped[0].isupper():
        nt = nt[0].upper() + nt[1:]

    # ── Terminal punctuation ─────────────────────────────────────────────────
    _TERMINAL = ".!?…"
    orig_end = original.rstrip()
    orig_is_sentence = bool(orig_end) and orig_end[-1] in _TERMINAL
    new_has_punct    = bool(nt) and nt[-1] in _TERMINAL + ",;:'\")"
    if orig_is_sentence and not new_has_punct:
        nt = nt + orig_end[-1]   # replicate the exact terminal char (. vs ! vs ?)

    return nt

# ═══════════════════════════════════════════════════════════════════════════════
# WhisperEditor — built-in voice-driven text editor
#
# Launched by "whisper type" / "whisper write"  → blank slate
# Launched by "whisper edit" / "whisper edit this" → pre-fills from clipboard
#
# All replace/insert operations work directly on a QTextEdit, eliminating all
# pyautogui + focus-restoration fragility.
# ═══════════════════════════════════════════════════════════════════════════════

class _MdHighlighter(QSyntaxHighlighter):
    """Live Markdown syntax highlighter for WhisperEditor.

    Renders bold, italic, heading markers, and code spans with visible styling
    so the editor gives an Obsidian-style feel without a full render pipeline.
    """

    def __init__(self, document):
        super().__init__(document)
        self._rules = []

        def _rule(pattern, fmt):
            self._rules.append((pattern, fmt))

        def _fmt(color=None, bold=False, italic=False, size_pt=None):
            f = QTextCharFormat()
            if color:
                f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Weight.Bold)
            if italic:
                f.setFontItalic(True)
            if size_pt:
                f.setFontPointSize(size_pt)
            return f

        import re as _re_md
        # Headings
        _rule(_re_md.compile(r'^#{1,6}\s.*$', _re_md.MULTILINE),
              _fmt('#7ec8e3', bold=True))
        # Bold **...**
        _rule(_re_md.compile(r'\*\*[^*]+\*\*'), _fmt('#f9c97c', bold=True))
        # Italic *...*
        _rule(_re_md.compile(r'(?<!\*)\*[^*]+\*(?!\*)'), _fmt('#c3e88d', italic=True))
        # Inline code `...`
        _rule(_re_md.compile(r'`[^`]+`'), _fmt('#ff9580'))
        # Horizontal rule / bullets
        _rule(_re_md.compile(r'^[-*+]\s', _re_md.MULTILINE), _fmt('#888'))
        # Markdown links [text](url) — entire span in bold light-blue
        _rule(_re_md.compile(r'\[[^\]]*\]\([^)]*\)'), _fmt('#5bc8f5', bold=True))
        # Highlight ==text==
        _rule(_re_md.compile(r'==[^=]+=={0,2}'), _fmt('#f9e94e'))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


class _HotkeyFilteredTextEdit(QTextEdit):
    """QTextEdit subclass that eats keypress events matching registered hotkeys.

    pynput GlobalHotKeys does NOT suppress OS key events — it only fires a
    callback.  The raw key event still reaches Qt.  This subclass intercepts
    key events at the lowest Qt level before any character is inserted.
    """
    def keyPressEvent(self, e):
        app = QApplication.instance()
        main = next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == "WhisperRApp"), None)
        if main is not None:
            cfg = getattr(main, "config", None)
            if cfg:
                _hotkeys = [
                    cfg.settings.get("hotkey", ""),
                    cfg.settings.get("ptt_key", ""),
                    cfg.settings.get("visibility_hotkey", ""),
                    cfg.settings.get("editor_hotkey", ""),
                    cfg.settings.get("editor_edit_hotkey", ""),
                    cfg.settings.get("rollback_hotkey", ""),
                ]
                _MOD = Qt.KeyboardModifier
                _mod_map = {
                    "ctrl":  _MOD.ControlModifier,
                    "alt":   _MOD.AltModifier,
                    "shift": _MOD.ShiftModifier,
                    "win":   _MOD.MetaModifier,
                    "meta":  _MOD.MetaModifier,
                }
                _key_map = {c: getattr(Qt.Key, f"Key_{c.upper()}", None)
                            for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
                _key_map.update({"f"+str(i): getattr(Qt.Key, f"Key_F{i}", None)
                                 for i in range(1, 13)})
                mods = e.modifiers()
                key  = e.key()
                for hk_str in _hotkeys:
                    if not hk_str:
                        continue
                    parts = [p.strip().lower() for p in hk_str.split("+")]
                    exp_mods = _MOD.NoModifier
                    exp_key  = None
                    for part in parts:
                        if part in _mod_map:
                            exp_mods |= _mod_map[part]
                        elif part in _key_map and _key_map[part]:
                            exp_key = _key_map[part]
                    if exp_key and key == exp_key and mods == exp_mods:
                        # Check if this is the rollback hotkey → trigger undo
                        _rk = cfg.settings.get("rollback_hotkey", "")
                        if hk_str.lower().replace(" ", "") == _rk.lower().replace(" ", ""):
                            self.undo()
                        e.accept()
                        return  # eaten — never reaches QTextEdit's insert logic
        super().keyPressEvent(e)



class _CheatsheetWindow(QWidget):
    """Floating cheatsheet panel that attaches to the right of WhisperEditor.

    Displays three collapsible sections:
      1. Editor formatting shortcuts (buttons + hotkeys)
      2. App-level hotkeys from Settings
      3. User-defined Terms (trigger phrases only, no replacements)
    """

    def __init__(self, editor: "WhisperEditor"):
        super().__init__(editor,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Cheatsheet")
        self._editor = editor
        self._build_ui()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def closeEvent(self, event):
        """Sync toggle button when user closes via the window's X."""
        super().closeEvent(event)
        if self._editor and getattr(self._editor, "btn_cheatsheet", None):
            self._editor.btn_cheatsheet.setChecked(False)
        # Notify editor so it clears the reference
        self._editor._cheatsheet = None

    def _build_ui(self):
        self.setStyleSheet(
            "QWidget { background: #1a1a1a; color: #ddd; }"
            "QScrollArea { border: none; }"
            "QLabel#hdr { color:#aaa; font-size:8pt; padding:2px 4px; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self._inner_layout = QVBoxLayout(inner)
        self._inner_layout.setContentsMargins(8, 8, 8, 8)
        self._inner_layout.setSpacing(4)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._populate()

    def _populate(self):
        lay = self._inner_layout
        # clear existing
        while lay.count():
            item = lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        cfg = getattr(self._editor, "config", None)
        settings = cfg.settings if cfg and hasattr(cfg, "settings") else {}
        hk = lambda k, d="": settings.get(k, d)

        # ── Section helper ────────────────────────────────────────────────────
        def _section(title, rows):
            """Collapsible section. rows = list of (shortcode/key, description)."""
            grp = QWidget()
            grp_lay = QVBoxLayout(grp)
            grp_lay.setContentsMargins(0, 0, 0, 0)
            grp_lay.setSpacing(0)

            hdr_btn = QPushButton(f"▾  {title}")
            hdr_btn.setCheckable(True)
            hdr_btn.setChecked(True)
            hdr_btn.setStyleSheet(
                "QPushButton{background:#252525;border:none;border-bottom:1px solid #333;"
                "color:#ccc;font-weight:bold;font-size:9pt;text-align:left;"
                "padding:5px 8px;}"
                "QPushButton:hover{background:#2e2e2e;}"
                "QPushButton:checked{color:#fff;}")
            grp_lay.addWidget(hdr_btn)

            body = QWidget()
            body_lay = QVBoxLayout(body)
            body_lay.setContentsMargins(6, 4, 6, 4)
            body_lay.setSpacing(1)
            body.setStyleSheet("QWidget{background:#1a1a1a;}")

            for shortcode, desc in rows:
                row_w = QWidget()
                row_w.setStyleSheet("QWidget{background:transparent;}")
                row_lay = QHBoxLayout(row_w)
                row_lay.setContentsMargins(2, 1, 2, 1)
                row_lay.setSpacing(8)

                lbl_code = QLabel(shortcode)
                lbl_code.setStyleSheet(
                    "font-family:Consolas,monospace;font-size:9pt;"
                    "color:#5bc8f5;background:#111;border:1px solid #333;"
                    "border-radius:3px;padding:1px 5px;")
                lbl_code.setFixedWidth(130)
                lbl_code.setWordWrap(False)

                lbl_desc = QLabel(desc)
                lbl_desc.setStyleSheet("font-size:9pt;color:#bbb;")
                lbl_desc.setWordWrap(True)

                row_lay.addWidget(lbl_code)
                row_lay.addWidget(lbl_desc, 1)
                body_lay.addWidget(row_w)

            grp_lay.addWidget(body)

            def _toggle(checked):
                body.setVisible(checked)
                hdr_btn.setText(("▾  " if checked else "▸  ") + title)
            hdr_btn.toggled.connect(_toggle)

            lay.addWidget(grp)

        # ── Section 1: Editor shortcuts ───────────────────────────────────────
        ed_rows = [
            (hk("editor_hk_bold",      "Ctrl+B"),          "Bold  **text**"),
            (hk("editor_hk_italic",    "Ctrl+I"),          "Italic  *text*"),
            (hk("editor_hk_strike",    "Ctrl+Shift+S"),    "Strikethrough  ~~text~~"),
            (hk("editor_hk_highlight", "Ctrl+Shift+H"),    "Highlight  ==text=="),
            (hk("editor_hk_code",      "Ctrl+`"),          "Inline code  `text`"),
            (hk("editor_hk_kbd",       "Ctrl+Shift+D"),    "Keyboard key  <kbd>text</kbd>"),
            (hk("editor_hk_link",      "Ctrl+K"),          "Link  [text](url)"),
            ("Right-click 🔗",                              "Link with clipboard URL"),
            (hk("editor_hk_h1",        "Ctrl+1"),          "Heading 1  # text"),
            (hk("editor_hk_h2",        "Ctrl+2"),          "Heading 2  ## text"),
            (hk("editor_hk_h3",        "Ctrl+3"),          "Heading 3  ### text"),
            (hk("editor_hk_bullet",    "Ctrl+Shift+B"),    "Bullet list  - text"),
            (hk("editor_hk_numlist",   "Ctrl+Shift+N"),    "Numbered list  1. text"),
            (hk("editor_hk_tasklist",  "Ctrl+Shift+T"),    "Task list  - [ ] text"),
            ("Ctrl+Enter",                                  "Paste to App"),
        ]
        _section("✏️  Editor shortcuts", ed_rows)

        # ── Section 2: App hotkeys ────────────────────────────────────────────
        app_hk_rows = []
        pairs = [
            (hk("hotkey",           "<ctrl>+<alt>+z"), "Toggle dictation on/off"),
            (hk("ptt_key",          "ctrl+shift+space"), "Push-to-talk (hold)"),
            (hk("visibility_hotkey","ctrl+shift+alt+z"), "Show / hide WhisperR window"),
            (hk("rollback_hotkey",  "ctrl+shift+z"),   "Rollback last dictation"),
            (hk("editor_hotkey",    "ctrl+shift+e"),   "Toggle editor window"),
            (hk("editor_edit_hotkey",""),               "Copy & Edit (open editor with selection)"),
        ]
        for k, desc in pairs:
            if k:
                app_hk_rows.append((k, desc))
        if app_hk_rows:
            _section("⌨️  App hotkeys", app_hk_rows)

        # ── Section 3: Voice triggers ─────────────────────────────────────────
        voice_rows = [
            (hk("editor_edit_trigger",  "whisper edit"),  "Open editor with selected text"),
            (hk("editor_type_trigger",  "whisper type"),  "Open blank editor"),
            (hk("editor_paste_trigger", "whisper paste"), "Paste editor text to app"),
            (hk("select_trigger",       "whisper select"),"Select / copy text"),
            (hk("move_trigger",         "whisper move"),  "Move cursor"),
            (hk("replace_trigger",      "whisper replace"),"Replace text"),
            (hk("insertbefore_trigger", "whisper insert before"), "Insert before text"),
            (hk("insertafter_trigger",  "whisper insert after"),  "Insert after text"),
        ]
        _section("🎙  Voice triggers", voice_rows)

        # ── Section 4: Terms ──────────────────────────────────────────────────
        terms = settings.get("terms", {})
        if terms:
            def _fmt_repl(r):
                r = str(r)
                return (r[:47] + "…") if len(r) > 50 else r
            term_rows = [(phrase, f"→ {_fmt_repl(repl)}") for phrase, repl in terms.items()]
            _section("📝  Terms  (say to substitute)", term_rows)

        lay.addStretch()

    def refresh(self):
        """Rebuild content (call after settings change)."""
        self._populate()


class WhisperEditor(QWidget):
    """Voice-driven built-in text editor.

    Uses QWidget (not QDialog) so it has no implicit reject/close-on-deactivate
    behavior.  The host (WhisperRApp) calls append_text(text) to pipe dictation
    in, and may call execute_edit(op, target, replacement) for replace/insert.
    The editor signals paste_requested(text) when done.
    """

    paste_requested = pyqtSignal(str)   # user said "whisper paste" / clicked Paste to App
    finished = pyqtSignal()             # emitted on close (replaces QDialog.finished)

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, initial_text="", config=None, parent=None):
        super().__init__(parent,
                         Qt.WindowType.Window |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("WhisperR Editor")
        self.config = config or {}
        self._hotkeys_active = []   # keyboard listener handles registered here
        self._build_ui()
        self._apply_formatting_hotkeys()
        if initial_text:
            self.editor.setPlainText(initial_text)
        self._update_stats()
        self.resize(820, 620)
        self._centre_on_screen()
        # Prevent Qt from destroying this object when the window is closed —
        # the app may still hold a reference and the clipboard monitor needs it.
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self._cheatsheet: "_CheatsheetWindow | None" = None

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(6)

        # ── Stats bar ─────────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        stats_row.setSpacing(6)

        def _toggle_btn(emoji, tip, checked_bg="#5a3a00", checked_border="#cc8800",
                        checked_color="#ffcc44"):
            b = QPushButton(emoji)
            b.setCheckable(True)
            b.setFixedSize(28, 24)
            b.setToolTip(tip)
            b.setStyleSheet(
                f"QPushButton{{background:#2a2a2a;border:1px solid #444;"
                f"border-radius:3px;color:#aaa;}}"
                f"QPushButton:checked{{background:{checked_bg};border-color:{checked_border};"
                f"color:{checked_color};}}"
                "QPushButton:hover{border-color:#0078d7;}")
            return b

        # Three mutually-exclusive memory/clipboard toggles
        self.remember_toggle = _toggle_btn(
            "📌",
            "Remember content\n"
            "Preserves editor text when closed/hidden.\n"
            "On re-open via \"whisper edit\", new text appends below.\n"
            "Exclusive with the clipboard toggles.")
        self.clipboard_prefill_toggle = _toggle_btn(
            "📋",
            "Clipboard prefill\n"
            "Pre-populates editor with current clipboard on open.\n"
            "Exclusive with the other memory toggles.",
            checked_bg="#003a5a", checked_border="#0078d7", checked_color="#66bbff")
        self.clipboard_monitor_toggle = _toggle_btn(
            "👁",
            "Clipboard monitor\n"
            "Continuously watches clipboard; each new copy appends\n"
            "a new entry below existing text — ideal for research.\n"
            "Exclusive with the other memory toggles.",
            checked_bg="#003a1a", checked_border="#00aa44", checked_color="#66ee88")

        def _excl(sender, others):
            if sender.isChecked():
                for o in others:
                    o.setChecked(False)

        self.remember_toggle.toggled.connect(
            lambda: _excl(self.remember_toggle,
                [self.clipboard_prefill_toggle, self.clipboard_monitor_toggle]))
        self.clipboard_prefill_toggle.toggled.connect(
            lambda: _excl(self.clipboard_prefill_toggle,
                [self.remember_toggle, self.clipboard_monitor_toggle]))
        self.clipboard_monitor_toggle.toggled.connect(
            lambda chk: (
                self._start_clipboard_monitor() if chk else self._stop_clipboard_monitor(),
                _excl(self.clipboard_monitor_toggle,
                    [self.remember_toggle, self.clipboard_prefill_toggle])))

        for tog in (self.remember_toggle,
                    self.clipboard_prefill_toggle,
                    self.clipboard_monitor_toggle):
            stats_row.addWidget(tog)

        stats_row.addSpacing(8)
        self.lbl_words = QLabel("Words: 0")
        self.lbl_chars = QLabel("Chars: 0")
        self.lbl_remain = QLabel("")
        for lbl in (self.lbl_words, self.lbl_chars, self.lbl_remain):
            lbl.setStyleSheet("color: #aaa; font-size: 9pt;")
            stats_row.addWidget(lbl)
        stats_row.addStretch()
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(0)
        self.target_spin.setSpecialValueText("No target")
        self.target_spin.setToolTip("Target word count — words remaining shown on the left")
        self.target_spin.setFixedWidth(110)
        self.target_spin.valueChanged.connect(self._update_stats)
        stats_row.addWidget(QLabel("Target words:"))
        stats_row.addWidget(self.target_spin)
        root.addLayout(stats_row)

        # ── Formatting toolbar ────────────────────────────────────────────────
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(4)

        cfg = self.config if isinstance(self.config, dict) else getattr(self.config, 'settings', {})
        hk = lambda k, d: cfg.get(k, d)

        # Grouped button definitions: (label, tip, hotkey_key, default_hk, slot)
        # Groups separated by None entries (rendered as vertical dividers)
        self._fmt_btns = [
            ("**B**", "Bold",           "editor_hk_bold",      "Ctrl+B",          self._fmt_bold),
            ("*I*",   "Italic",         "editor_hk_italic",    "Ctrl+I",          self._fmt_italic),
            ("~~S~~", "Strikethrough",  "editor_hk_strike",    "Ctrl+Shift+S",    self._fmt_strike),
            ("==H==", "Highlight",      "editor_hk_highlight", "Ctrl+Shift+H",    self._fmt_highlight),
            None,  # ── group separator ──
            ("`C`",   "Inline code",    "editor_hk_code",      "Ctrl+`",          self._fmt_code),
            ("<kbd>", "Keyboard key",   "editor_hk_kbd",       "Ctrl+Shift+D",    self._fmt_kbd),
            ("🔗",    "Link",           "editor_hk_link",      "Ctrl+K",          self._fmt_link),
            None,  # ── group separator ──
            ("H1",    "Heading 1",      "editor_hk_h1",        "Ctrl+1",          lambda: self._fmt_heading(1)),
            ("H2",    "Heading 2",      "editor_hk_h2",        "Ctrl+2",          lambda: self._fmt_heading(2)),
            ("H3",    "Heading 3",      "editor_hk_h3",        "Ctrl+3",          lambda: self._fmt_heading(3)),
            None,  # ── group separator ──
            ("•",     "Bullet list",    "editor_hk_bullet",    "Ctrl+Shift+B",    self._fmt_bullet),
            ("1.",    "Numbered list",  "editor_hk_numlist",   "Ctrl+Shift+N",    self._fmt_numlist),
            ("☐",     "Task list",      "editor_hk_tasklist",  "Ctrl+Shift+T",    self._fmt_tasklist),
        ]

        _btn_ss = ("QPushButton{background:#2a2a2a;border:1px solid #444;border-radius:3px;"
                   "color:#ddd;font-size:10pt;}"
                   "QPushButton:hover{background:#0078d7;border-color:#0078d7;color:#fff;}")

        def _sep():
            ln = QFrame(); ln.setFrameShape(QFrame.Shape.VLine)
            ln.setStyleSheet("color:#555;"); ln.setFixedWidth(10)
            return ln

        for entry in self._fmt_btns:
            if entry is None:
                fmt_row.addWidget(_sep())
                fmt_row.addSpacing(2)
                continue
            label, tip, hk_key, hk_default, slot = entry
            hotkey = hk(hk_key, hk_default)
            btn = QPushButton(label)
            btn.setFixedSize(36, 28)
            btn.setToolTip(f"{tip}  [{hotkey}]")
            btn.clicked.connect(slot)
            btn.setStyleSheet(_btn_ss)
            if label == "🔗":
                btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                btn.customContextMenuRequested.connect(
                    lambda _: self._fmt_link(use_clipboard=True))
            fmt_row.addWidget(btn)

        fmt_row.addStretch()

        # Voice status indicator
        self.lbl_voice = QLabel("🎙 Voice active")
        self.lbl_voice.setStyleSheet("color:#28b450;font-size:9pt;")
        self.lbl_voice.setToolTip(
            "Green = dictation running and feeding into this editor.\n"
            "Grey = dictation is stopped. Start it from the main window or hotkey.")
        fmt_row.addWidget(self.lbl_voice)

        root.addLayout(fmt_row)

        # ── Text editor ───────────────────────────────────────────────────────
        self.editor = _HotkeyFilteredTextEdit()
        self.editor.setFont(QFont("Consolas", 11))
        self.editor.setAcceptDrops(True)
        self.editor.setPlaceholderText(
            "Start dictating, or drop a .txt / .md file here…")
        # Install event filter so hotkey key events are suppressed
        # before they land in the text area (see eventFilter below)
        self.editor.installEventFilter(self)
        self.editor.setMinimumHeight(360)
        self.editor.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        from PyQt6.QtGui import QTextOption as _QTO
        self.editor.setWordWrapMode(_QTO.WrapMode.WordWrap)
        self.editor.setStyleSheet(
            "QTextEdit{background:#111;color:#e0e0e0;border:1px solid #333;"
            "border-radius:4px;padding:6px;font-family:Consolas,monospace;}")
        self.editor.textChanged.connect(self._update_stats)
        # Drop support
        self.editor.dragEnterEvent = self._drag_enter
        self.editor.dropEvent = self._drop_event
        # Live MD highlighter
        self._highlighter = _MdHighlighter(self.editor.document())
        root.addWidget(self.editor)

        # ── Bottom button row ─────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        def _btn(label, tip, slot):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.setStyleSheet(
                "QPushButton{background:#2a2a2a;border:1px solid #444;padding:5px 10px;"
                "border-radius:4px;color:#ddd;}"
                "QPushButton:hover{background:#353535;border-color:#0078d7;}")
            return b

        btn_row.addWidget(_btn("📂 Import", "Load a .txt or .md file", self._import_file))
        btn_row.addWidget(_btn("💾 Export", "Save to .txt or .md file", self._export_file))
        btn_row.addWidget(_btn("📋 Copy", "Copy all text to clipboard", self._copy_all))
        btn_row.addStretch()

        self.btn_cheatsheet = _btn(
            "📖 Cheatsheet",
            "Show / hide the shortcut cheatsheet panel",
            self._toggle_cheatsheet)
        self.btn_cheatsheet.setCheckable(True)
        btn_row.addWidget(self.btn_cheatsheet)
        btn_row.addSpacing(20)

        self.btn_paste_app = _btn(
            "📤 Paste to App",
            "Paste text back to the previously active application\n"
            "Voice: \"whisper paste\" / \"whisper done\" / \"whisper okay\"\n"
            "Hotkey: Ctrl+Enter",
            self._paste_to_app)
        _paste_sc = QShortcut(QKeySequence("Ctrl+Return"), self)
        _paste_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        _paste_sc.activated.connect(self._paste_to_app)
        self._hotkeys_active.append(_paste_sc)
        self.btn_paste_app.setStyleSheet(
            "QPushButton{background:#0078d7;border:1px solid #005fa3;padding:5px 12px;"
            "border-radius:4px;color:#fff;font-weight:bold;}"
            "QPushButton:hover{background:#005fa3;}")
        btn_row.addWidget(self.btn_paste_app)

        btn_row.addWidget(_btn("✕ Close", "Close editor without pasting", self.close))
        root.addLayout(btn_row)

    # ── Positioning ───────────────────────────────────────────────────────────

    # ── Cheatsheet ────────────────────────────────────────────────────────────

    def _toggle_cheatsheet(self):
        """Show or hide the floating cheatsheet window."""
        if self._cheatsheet and self._cheatsheet.isVisible():
            self._cheatsheet.hide()
            self.btn_cheatsheet.setChecked(False)
        else:
            if not self._cheatsheet:
                self._cheatsheet = _CheatsheetWindow(self)
            self._reposition_cheatsheet()
            self._cheatsheet.show()
            self._cheatsheet.raise_()
            self.btn_cheatsheet.setChecked(True)

    def _reposition_cheatsheet(self):
        """Position cheatsheet window flush to the right of this editor."""
        if not self._cheatsheet:
            return
        geo = self.frameGeometry()
        cs_w = max(self.width() // 2, 360)
        cs_h = self.height()
        self._cheatsheet.resize(cs_w, cs_h)
        self._cheatsheet.move(geo.right() + 2, geo.top())

    def moveEvent(self, event):
        super().moveEvent(event)
        self._reposition_cheatsheet()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_cheatsheet()

    def _centre_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.move(screen.center() - self.rect().center())

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _update_stats(self):
        text = self.editor.toPlainText()
        words = len(text.split()) if text.strip() else 0
        chars = len(text)
        self.lbl_words.setText(f"Words: {words}")
        self.lbl_chars.setText(f"Chars: {chars}")
        target = self.target_spin.value()
        if target > 0:
            remaining = max(0, target - words)
            self.lbl_remain.setText(f"Remaining: {remaining}")
        else:
            self.lbl_remain.setText("")

    # ── Formatting helpers ────────────────────────────────────────────────────

    def _wrap_selection(self, before, after=None):
        """Wrap selected text with markdown markers, or insert at cursor."""
        after = after or before
        cur = self.editor.textCursor()
        sel = cur.selectedText()
        if sel:
            cur.insertText(before + sel + after)
        else:
            cur.insertText(before + after)
            cur.movePosition(cur.MoveOperation.Left, cur.MoveMode.MoveAnchor, len(after))
            self.editor.setTextCursor(cur)

    def _fmt_bold(self):      self._wrap_selection("**")
    def _fmt_italic(self):    self._wrap_selection("*")
    def _fmt_strike(self):    self._wrap_selection("~~")
    def _fmt_code(self):      self._wrap_selection("`")
    def _fmt_kbd(self):       self._wrap_selection("<kbd>", "</kbd>")
    def _fmt_emdash(self):
        cur = self.editor.textCursor(); cur.insertText("—"); self.editor.setTextCursor(cur)

    def _fmt_heading(self, level):
        cur = self.editor.textCursor()
        cur.movePosition(cur.MoveOperation.StartOfBlock)
        cur.movePosition(cur.MoveOperation.EndOfBlock, cur.MoveMode.KeepAnchor)
        line = cur.selectedText().lstrip("#").lstrip()
        cur.insertText("#" * level + " " + line)
        self.editor.setTextCursor(cur)

    def _fmt_highlight(self):  self._wrap_selection("==")

    def _fmt_link(self, use_clipboard=False):
        """Wrap selection as a Markdown link.
        use_clipboard=False → [sel](URL) placeholder
        use_clipboard=True  → [sel](clipboard contents)
        """
        cur = self.editor.textCursor()
        sel = cur.selectedText() or "link text"
        if use_clipboard:
            try:
                import pyperclip as _pclip
                url = _pclip.paste().strip() or "URL"
            except Exception:
                url = "URL"
        else:
            url = "URL"
        cur.insertText(f"[{sel}]({url})")
        # Position cursor inside the URL part if we used the placeholder
        if not use_clipboard:
            # Move back past the closing ) to select "URL" so user can type over it
            end = cur.position()
            cur.setPosition(end - len(url) - 1)
            cur.setPosition(end - 1, cur.MoveMode.KeepAnchor)
        self.editor.setTextCursor(cur)

    def _fmt_lines(self, prefix_fn):
        """Apply prefix_fn(line_index, line_text) → new_line to each selected line."""
        cur = self.editor.textCursor()
        # Expand selection to whole lines
        start = cur.selectionStart()
        end   = cur.selectionEnd()
        cur.setPosition(start)
        cur.movePosition(cur.MoveOperation.StartOfBlock)
        block_start = cur.position()
        cur.setPosition(end)
        cur.movePosition(cur.MoveOperation.EndOfBlock)
        block_end = cur.position()
        cur.setPosition(block_start)
        cur.setPosition(block_end, cur.MoveMode.KeepAnchor)
        selected = cur.selectedText()
        # QTextEdit uses \u2029 (paragraph separator) for line breaks in selectedText
        lines = selected.split("\u2029")
        new_lines = [prefix_fn(i, ln) for i, ln in enumerate(lines)]
        cur.insertText("\n".join(new_lines))
        self.editor.setTextCursor(cur)

    def _fmt_bullet(self):
        self._fmt_lines(lambda i, ln: "- " + ln)

    def _fmt_numlist(self):
        self._fmt_lines(lambda i, ln: f"{i+1}. " + ln)

    def _fmt_tasklist(self):
        self._fmt_lines(lambda i, ln: "- [ ] " + ln)

    # ── Clipboard monitor (delegated to WhisperRApp for persistence) ────────

    def _start_clipboard_monitor(self):
        """Delegate to the host app so the monitor survives hide/close."""
        app = QApplication.instance()
        host = next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == "WhisperRApp"), None)
        if host:
            host.start_clipboard_monitor()

    def _stop_clipboard_monitor(self):
        """Delegate stop to the host app."""
        app = QApplication.instance()
        host = next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == "WhisperRApp"), None)
        if host:
            host.stop_clipboard_monitor()

    def _apply_formatting_hotkeys(self):
        """Register QShortcut hotkeys active only while this window is open."""
        cfg = self.config if isinstance(self.config, dict) else getattr(self.config, 'settings', {})

        shortcuts = [
            (cfg.get("editor_hk_bold",      "Ctrl+B"),          self._fmt_bold),
            (cfg.get("editor_hk_italic",    "Ctrl+I"),          self._fmt_italic),
            (cfg.get("editor_hk_strike",    "Ctrl+Shift+S"),    self._fmt_strike),
            (cfg.get("editor_hk_highlight", "Ctrl+Shift+H"),    self._fmt_highlight),
            (cfg.get("editor_hk_code",      "Ctrl+`"),          self._fmt_code),
            (cfg.get("editor_hk_h1",        "Ctrl+1"),          lambda: self._fmt_heading(1)),
            (cfg.get("editor_hk_h2",        "Ctrl+2"),          lambda: self._fmt_heading(2)),
            (cfg.get("editor_hk_h3",        "Ctrl+3"),          lambda: self._fmt_heading(3)),
            (cfg.get("editor_hk_emdash",    "Ctrl+Shift+Minus"),self._fmt_emdash),
            (cfg.get("editor_hk_bullet",    "Ctrl+Shift+B"),    self._fmt_bullet),
            (cfg.get("editor_hk_numlist",   "Ctrl+Shift+N"),    self._fmt_numlist),
            (cfg.get("editor_hk_tasklist",  "Ctrl+Shift+T"),    self._fmt_tasklist),
            (cfg.get("editor_hk_kbd",       "Ctrl+Shift+D"),    self._fmt_kbd),
            (cfg.get("editor_hk_link",      "Ctrl+K"),          self._fmt_link),
            ("Ctrl+Shift+K",                                     lambda: self._fmt_link(use_clipboard=True)),
        ]
        for keys, slot in shortcuts:
            try:
                sc = QShortcut(QKeySequence(keys), self)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.activated.connect(slot)
                self._hotkeys_active.append(sc)
            except Exception:
                pass

    # ── Voice operations (called from WhisperRApp.on_text) ────────────────────

    def append_text(self, text: str):
        """Insert dictated text, replacing any current selection.

        If text is selected, the selection is replaced (like normal typing).
        If no selection, text is appended after a space if needed.
        """
        cur = self.editor.textCursor()
        if cur.hasSelection():
            # Replace selection — no leading space needed
            cur.insertText(text)
        else:
            existing = self.editor.toPlainText()
            if existing and not existing.endswith(" ") and not existing.endswith("\n"):
                text = " " + text
            cur.movePosition(cur.MoveOperation.End)
            cur.insertText(text)
        self.editor.setTextCursor(cur)
        self.editor.ensureCursorVisible()

    def set_voice_state(self, active: bool):
        """Called by WhisperRApp whenever dictation starts or stops."""
        if active:
            self.lbl_voice.setText("🎙 Voice active")
            self.lbl_voice.setStyleSheet("color:#28b450;font-size:9pt;")
        else:
            self.lbl_voice.setText("⏸ Voice disabled")
            self.lbl_voice.setStyleSheet("color:#666;font-size:9pt;")

    # Tag used to mark auto-inserted instance numbers so we can remove them cleanly
    _INST_TAG = "\x00WR_INST\x00"  # invisible sentinel (stripped on display, kept in plain text)

    def find_instances(self, target: str) -> int:
        """Return the count of case-insensitive occurrences of target in the editor."""
        import re as _re_fi
        return len(_re_fi.findall(_re_fi.escape(target), self.editor.toPlainText(), _re_fi.IGNORECASE))

    def annotate_instances(self, target: str) -> list[str]:
        """Number every occurrence of target inline: (1)Batman, (2)Batman, …

        Returns the list of annotation labels for the wizard prompt.
        Annotations are stored with an invisible sentinel so _remove_annotations
        can delete them without touching any existing parenthesised numbers.
        """
        import re as _re_ann
        buf = self.editor.toPlainText()
        pat = _re_ann.compile(_re_ann.escape(target), _re_ann.IGNORECASE)
        labels = []
        offset = 0
        new_buf = buf
        for i, m in enumerate(pat.finditer(buf), 1):
            tag = f"{self._INST_TAG}({i}){self._INST_TAG}"
            pos = m.start() + offset
            new_buf = new_buf[:pos] + tag + new_buf[pos:]
            offset += len(tag)
            labels.append(f"({i})")
        cur_pos = self.editor.textCursor().position()
        self.editor.setPlainText(new_buf)
        cur = self.editor.textCursor()
        cur.setPosition(min(cur_pos, len(new_buf)))
        self.editor.setTextCursor(cur)
        return labels

    def _remove_annotations(self):
        """Strip all auto-inserted instance annotations from the editor text."""
        import re as _re_rm
        buf = self.editor.toPlainText()
        # Remove sentinel-bracketed labels: \x00WR_INST\x00(N)\x00WR_INST\x00
        sentinel = _re_rm.escape(self._INST_TAG)
        cleaned = _re_rm.sub(sentinel + r"\(\d+\)" + sentinel, "", buf)
        if cleaned != buf:
            cur_pos = self.editor.textCursor().position()
            self.editor.setPlainText(cleaned)
            cur = self.editor.textCursor()
            cur.setPosition(max(0, min(cur_pos, len(cleaned))))
            self.editor.setTextCursor(cur)

    def execute_edit(self, op: str, target: str, new_text: str,
                     instance: int = -1) -> bool:
        """Perform replace/insertbefore/insertafter directly on editor text.

        instance: 1-based index of which occurrence to act on (-1 = last).
        Always removes any instance annotations before applying the edit.
        Returns True on success, False if target not found.
        """
        import re as _re_ed
        self._remove_annotations()
        buf = self.editor.toPlainText()
        pat = _re_ed.compile(_re_ed.escape(target), _re_ed.IGNORECASE)
        hits = list(pat.finditer(buf))
        if not hits:
            return False
        idx = (instance - 1) if (1 <= instance <= len(hits)) else len(hits) - 1
        m = hits[idx]
        if op == "replace":
            new_text = _smart_case_punct(m.group(), new_text)
            new_buf = buf[:m.start()] + new_text + buf[m.end():]
        elif op == "insertbefore":
            new_buf = buf[:m.start()] + new_text + " " + buf[m.start():]
        elif op == "insertafter":
            new_buf = buf[:m.end()] + " " + new_text + buf[m.end():]
        else:
            return False
        cur = self.editor.textCursor()
        pos = cur.position()
        self.editor.setPlainText(new_buf)
        cur2 = self.editor.textCursor()
        cur2.setPosition(min(pos, len(new_buf)))
        self.editor.setTextCursor(cur2)
        return True

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import file", "", "Text/Markdown (*.txt *.md);;All files (*)")
        if path:
            try:
                self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))
            except Exception as e:
                QMessageBox.warning(self, "Import failed", str(e))

    def _export_file(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export file", "", "Markdown (*.md);;Text (*.txt);;All files (*)")
        if path:
            try:
                Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")
            except Exception as e:
                QMessageBox.warning(self, "Export failed", str(e))

    def _copy_all(self):
        try:
            import pyperclip as _pc
            _pc.copy(self.editor.toPlainText())
        except Exception:
            QApplication.clipboard().setText(self.editor.toPlainText())

    def _paste_to_app(self):
        self.paste_requested.emit(self.editor.toPlainText())
        self.close()

    # ── Drag-and-drop ─────────────────────────────────────────────────────────

    def _drag_enter(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def _drop_event(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".txt", ".md")):
                try:
                    self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))
                except Exception:
                    pass
                break

    # ── Event filter (hotkey suppression on child QTextEdit) ─────────────────

    def eventFilter(self, obj, event):
        """Suppress keystrokes that match the app's global hotkeys so they
        don't land inside the QTextEdit while the editor window is focused."""
        if obj is self.editor and event.type() in (
                QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            app = QApplication.instance()
            main = next((w for w in app.topLevelWidgets()
                         if w.__class__.__name__ == "WhisperRApp"), None)
            if main is not None:
                cfg = getattr(main, "config", None)
                if cfg:
                    _hotkeys = [
                        cfg.settings.get("hotkey", ""),
                        cfg.settings.get("ptt_key", ""),
                        cfg.settings.get("visibility_hotkey", ""),
                        cfg.settings.get("editor_hotkey", ""),
                        cfg.settings.get("editor_edit_hotkey", ""),
                        cfg.settings.get("rollback_hotkey", ""),
                    ]
                    mods = event.modifiers()
                    key  = event.key()
                    _MOD = Qt.KeyboardModifier
                    _mod_map = {
                        "ctrl":  _MOD.ControlModifier,
                        "alt":   _MOD.AltModifier,
                        "shift": _MOD.ShiftModifier,
                        "win":   _MOD.MetaModifier,
                        "meta":  _MOD.MetaModifier,
                    }
                    _key_map = {c: getattr(Qt.Key, f"Key_{c.upper()}", None)
                                for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
                    _key_map.update({"f"+str(i): getattr(Qt.Key, f"Key_F{i}", None)
                                     for i in range(1, 13)})
                    for hk_str in _hotkeys:
                        if not hk_str:
                            continue
                        parts = [p.strip().lower() for p in hk_str.split("+")]
                        exp_mods = _MOD.NoModifier
                        exp_key  = None
                        for part in parts:
                            if part in _mod_map:
                                exp_mods |= _mod_map[part]
                            elif part in _key_map and _key_map[part]:
                                exp_key = _key_map[part]
                        if exp_key and mods == exp_mods and key == exp_key:
                            # If this is the rollback hotkey AND it's a KeyPress,
                            # trigger undo in the editor instead of ignoring it.
                            _rk = cfg.settings.get("rollback_hotkey", "")
                            _rk_parts = [p.strip().lower() for p in _rk.split("+")]
                            _rk_mods = _MOD.NoModifier
                            _rk_key  = None
                            for _p in _rk_parts:
                                if _p in _mod_map:
                                    _rk_mods |= _mod_map[_p]
                                elif _p in _key_map and _key_map[_p]:
                                    _rk_key = _key_map[_p]
                            if (event.type() == QEvent.Type.KeyPress and
                                    _rk_key and mods == _rk_mods and key == _rk_key):
                                self.editor.undo()
                            return True  # suppress — event eaten, not passed to QTextEdit
        return super().eventFilter(obj, event)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        # NOTE: clipboard monitor is NOT stopped here — it lives on the host
        # app and persists across hide/close. Only turning off the toggle stops it.
        for sc in self._hotkeys_active:
            sc.setEnabled(False)
        self._hotkeys_active.clear()
        self.finished.emit()   # notify app (replaces QDialog.finished)
        super().closeEvent(event)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close()
            return
        # Suppress keystrokes that match the app's global hotkeys so they
        # don't land in the text editor.  The app's keyboard listener fires
        # the action; we just need to eat the Qt key event here.
        app = QApplication.instance()
        main = next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == "WhisperRApp"), None)
        if main is not None:
            cfg = getattr(main, "config", None)
            if cfg:
                _hotkeys_to_suppress = [
                    cfg.settings.get("hotkey", ""),
                    cfg.settings.get("ptt_key", ""),
                    cfg.settings.get("visibility_hotkey", ""),
                    cfg.settings.get("editor_hotkey", ""),
                    cfg.settings.get("editor_edit_hotkey", ""),
                    cfg.settings.get("rollback_hotkey", ""),
                ]
                mods = e.modifiers()
                key  = e.key()
                # Build a set of (modifier_flags, key_code) pairs to compare
                _MOD = Qt.KeyboardModifier
                _mod_map = {
                    "ctrl":  _MOD.ControlModifier,
                    "alt":   _MOD.AltModifier,
                    "shift": _MOD.ShiftModifier,
                    "win":   _MOD.MetaModifier,
                    "meta":  _MOD.MetaModifier,
                }
                _key_map = {c: getattr(Qt.Key, f"Key_{c.upper()}", None)
                            for c in "abcdefghijklmnopqrstuvwxyz0123456789"}
                _key_map.update({"f"+str(i): getattr(Qt.Key, f"Key_F{i}", None)
                                 for i in range(1, 13)})
                for hk_str in _hotkeys_to_suppress:
                    if not hk_str:
                        continue
                    # Strip pynput angle-bracket format: <ctrl> → ctrl
                    parts = [p.strip().lower().strip("<>") for p in hk_str.split("+")]
                    expected_mods = _MOD.NoModifier
                    expected_key  = None
                    for part in parts:
                        if part in _mod_map:
                            expected_mods |= _mod_map[part]
                        elif part in _key_map and _key_map[part]:
                            expected_key = _key_map[part]
                    if expected_key and mods == expected_mods and key == expected_key:
                        e.accept()
                        return
        super().keyPressEvent(e)

# ── Voice-guided wizard overlay ───────────────────────────────────────────────
# A small always-on-top frameless window that prompts the user to speak a
# specific piece of information (e.g. "what to select", "replacement text").
# It shows live transcription feedback and closes automatically when the
# wizard step is satisfied.

class WizardOverlay(QDialog):
    """Non-blocking, always-on-top prompt window for multi-step voice commands.

    The host (WhisperRApp) creates this, shows it, and passes each new
    transcription result to feed() until the step is done.  The dialog never
    blocks the event loop — it is purely informational/cosmetic.
    """

    cancelled = pyqtSignal()   # user pressed Escape / Cancel button

    def __init__(self, title, prompt, parent=None, anchor=None):
        """anchor: if a QWidget, overlay is centred over it; otherwise bottom-right."""
        super().__init__(parent,
                         Qt.WindowType.Tool |
                         Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setModal(False)
        self.setFixedWidth(420)
        self._anchor = anchor

        self._build_ui(title, prompt)
        self._position()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self, title, prompt):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 10, 14, 10)
        root.setSpacing(6)

        # Title bar row
        title_row = QHBoxLayout()
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(
            "font-size: 11pt; font-weight: bold; color: #0078d7;")
        title_row.addWidget(lbl_title)
        title_row.addStretch()
        btn_cancel = QPushButton("✕")
        btn_cancel.setFixedSize(22, 22)
        btn_cancel.setToolTip("Cancel (Escape)")
        btn_cancel.setStyleSheet(
            "QPushButton { background: #333; border: none; color: #aaa; "
            "border-radius: 11px; font-size: 10pt; }"
            "QPushButton:hover { background: #c0392b; color: #fff; }")
        btn_cancel.clicked.connect(self._on_cancel)
        title_row.addWidget(btn_cancel)
        root.addLayout(title_row)

        # Prompt label
        self.lbl_prompt = QLabel(prompt)
        self.lbl_prompt.setWordWrap(True)
        self.lbl_prompt.setStyleSheet("color: #ddd; font-size: 10pt;")
        root.addWidget(self.lbl_prompt)

        # Live transcription feedback
        self.lbl_heard = QLabel("🎙 Listening…")
        self.lbl_heard.setWordWrap(True)
        self.lbl_heard.setStyleSheet(
            "color: #888; font-style: italic; font-size: 9pt; "
            "border-top: 1px solid #333; padding-top: 4px;")
        root.addWidget(self.lbl_heard)

        self.setStyleSheet(
            "WizardOverlay { background-color: #1a1a1a; "
            "border: 1px solid #0078d7; border-radius: 8px; }")

    def _position(self):
        """Centre over anchor widget if given, otherwise bottom-right of screen."""
        self.adjustSize()
        if self._anchor and self._anchor.isVisible():
            ag = self._anchor.geometry()
            gp = self._anchor.mapToGlobal(ag.topLeft()) if hasattr(self._anchor, "mapToGlobal") else ag.topLeft()
            # Centre of the anchor window on screen
            cx = self._anchor.frameGeometry().center().x()
            cy = self._anchor.frameGeometry().center().y()
            x = cx - self.width() // 2
            y = cy - self.height() // 2
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            x = screen.right()  - self.width()  - 24
            y = screen.bottom() - self.height() - 48
        self.move(x, y)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_prompt(self, prompt):
        """Update the prompt text (used when advancing to the next step)."""
        self.lbl_prompt.setText(prompt)
        self.lbl_heard.setText("🎙 Listening…")
        self.adjustSize()
        self._position()

    def feed(self, text):
        """Show the latest transcription result as live feedback."""
        short = text[:60] + ("…" if len(text) > 60 else "")
        self.lbl_heard.setText(f"🎙 Heard: \"{short}\"")
        self.adjustSize()
        self._position()

    def confirm(self, summary):
        """Briefly show a confirmation before the caller closes the window."""
        short = summary[:60] + ("…" if len(summary) > 60 else "")
        self.lbl_heard.setText(f"✓ {short}")
        self.adjustSize()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_cancel(self):
        self.cancelled.emit()
        self.close()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self._on_cancel()
        else:
            super().keyPressEvent(e)


def _score_trigger_match(spoken_text: str, trigger_phrase: str) -> float:
    """Multi-strategy similarity score between spoken_text and trigger_phrase.

    Returns a float in [0, 1].  Uses four complementary strategies:

    1. Word-level: compare first N spoken words against trigger words.
    2. Char-concat: join all chars, compare as single strings (catches
       phonetic approximations that break word boundaries).
    3. Sliding window: slide a window = len(trigger_chars) over spoken_chars,
       take best ratio (robust to extra words before/after trigger).
    4. Subsequence: check if the trigger chars appear as a subsequence in
       the spoken chars (catches scattered phonetic fragments).

    The final score is the maximum of all four.
    """
    from difflib import SequenceMatcher as _SM
    import re as _re_t

    def _clean(s):
        return _re_t.sub(r"[^a-z]", "", s.lower())

    s_words = [w for w in _re_t.sub(r"[^\w ]", "", spoken_text.lower()).split() if w]
    t_words = trigger_phrase.lower().split()
    n = len(t_words)

    best = 0.0

    # 1. Word-level window (existing approach)
    if s_words and len(s_words) >= n:
        for i in range(len(s_words) - n + 1):
            window = " ".join(s_words[i:i + n])
            r = _SM(None, trigger_phrase.lower(), window).ratio()
            if r > best:
                best = r

    # 2. Full char-concat comparison
    s_cat = _clean(spoken_text)
    t_cat = _clean(trigger_phrase)
    if s_cat and t_cat:
        r2 = _SM(None, t_cat, s_cat).ratio()
        if r2 > best:
            best = r2

    # 3. Sliding char window — slide len(t_cat) chars across s_cat
    tl = len(t_cat)
    if tl and len(s_cat) >= tl:
        for i in range(len(s_cat) - tl + 1):
            r3 = _SM(None, t_cat, s_cat[i:i + tl]).ratio()
            if r3 > best:
                best = r3
    elif tl and len(s_cat) < tl:
        # Spoken shorter than trigger — compare what we have
        r3 = _SM(None, t_cat[:len(s_cat)], s_cat).ratio() * (len(s_cat) / tl)
        if r3 > best:
            best = r3

    # 4. Subsequence bonus — if all trigger chars appear in order in spoken
    if t_cat:
        si = 0
        matched = 0
        for ch in s_cat:
            if si < len(t_cat) and ch == t_cat[si]:
                si += 1
                matched += 1
        subseq_ratio = matched / len(t_cat)
        # Only count as a bonus if it's a strong subsequence (>80% matched)
        if subseq_ratio > 0.8:
            r4 = subseq_ratio * 0.85  # discount to avoid false positives
            if r4 > best:
                best = r4

    return best

class WhisperRApp(QMainWindow):
    sig_toggle_vis    = pyqtSignal()
    sig_toggle_rec    = pyqtSignal()
    sig_toggle_editor = pyqtSignal()
    sig_editor_edit   = pyqtSignal()

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
            self.sig_toggle_editor.connect(self.toggle_editor_window)
            self.sig_editor_edit.connect(lambda: self._open_editor(from_clipboard=True))
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
            tm.addAction("Quit", self._quit_app)
            self.tray.setContextMenu(tm)
            app_logger.debug("  __init__: Tray context menu set")
            
            app_logger.debug("  __init__: Calling tray.show()...")
            self.tray.show()
            app_logger.debug(f"  __init__: tray.show() called, tray.isVisible={self.tray.isVisible()}")
            
            # Double-click or single-click tray icon → restore window
            self.tray.activated.connect(self._on_tray_activated)
            
            # Initialise _ptt_held before setup_logic so PTT polling can clear it
            self._ptt_held: set = set()
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
            # _ptt_held already initialised before setup_logic (above)
            self._last_paste_text: str = ""   # last pasted text (for rollback)
            self._session_buffer:  str = ""   # running concat of all pastes this session (for select/move)
            self._cursor_offset:   int = 0    # chars cursor is LEFT of end of session buffer (0=at end)
            self._session_paste_count: int = 0  # number of Ctrl+V pastes this session (for undo count)
            self._cursor_ops_pending: bool = False  # next transcription should be lowercased (select/move used)
            self._pending_edit = None  # (op, target) set by whisperreplace/insert triggers
            self._pre_clip_capture = None  # set by hotkey handler before Qt signal
            self._editor: WhisperEditor | None = None  # built-in text editor window
            self._editor_remember: bool = False      # persist editor content across open/close
            self._editor_saved_content: str = ""     # content preserved when remember is on
            # Load persisted editor content from disk (if "remember" was on last session)
            _ed_persist_path = Path(self.config.path).parent / "whisperr_editor.txt"
            _ed_state_path   = Path(self.config.path).parent / "whisperr_editor_state.json"
            self._editor_persist_path = _ed_persist_path
            self._editor_state_path   = _ed_state_path
            # Load full editor state (content + target + toggles) from JSON if present
            if _ed_state_path.exists():
                try:
                    import json as _json_es
                    _st = _json_es.loads(_ed_state_path.read_text(encoding="utf-8"))
                    if _st.get("remember", False):
                        self._editor_saved_content      = _st.get("content", "")
                        self._editor_remember           = True
                        self._editor_clipboard_prefill  = _st.get("clipboard_prefill", False)
                        self._editor_cb_monitor_was_on  = _st.get("cb_monitor", False)
                        self._editor_saved_target       = _st.get("target_words", 0)
                        app_logger.info(
                            f"Restored editor state: {len(self._editor_saved_content)}ch, "
                            f"target={self._editor_saved_target}")
                except Exception as _e:
                    app_logger.warning(f"Could not restore editor state: {_e}")
            elif _ed_persist_path.exists():  # legacy txt-only fallback
                try:
                    _saved = _ed_persist_path.read_text(encoding="utf-8")
                    if _saved:
                        self._editor_saved_content = _saved
                        self._editor_remember = True
                        app_logger.info(f"Restored editor content ({len(_saved)} chars) from legacy txt")
                except Exception as _e:
                    app_logger.warning(f"Could not restore editor content: {_e}")
            self._editor_clipboard_prefill: bool = False  # prefill with clipboard on open
            self._editor_saved_target: int = 0            # persisted target word count
            self._editor_return_hwnd = None           # window to restore focus to after paste
            self._cb_monitor_timer: QTimer | None = None  # app-level clipboard monitor
            self._cb_monitor_last: str = ""               # last seen clipboard content
            self._editor_cb_monitor_was_on: bool = False  # persists monitor toggle state
            # ── Wizard state ──────────────────────────────────────────
            # When a multi-step voice command is active, _wizard holds:
            #   op       : str   — "select"|"move"|"movebefore"|"moveafter"|
            #                      "replace"|"insertbefore"|"insertafter"
            #   step     : str   — current step name (op-specific)
            #   collected: dict  — data gathered so far
            # While _wizard is not None, on_text routes transcriptions to
            # _wizard_step() rather than the normal paste path.
            self._wizard: dict | None = None
            self._wizard_overlay: WizardOverlay | None = None
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

            ta = _HotkeyFilteredTextEdit()
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

        # Auto-scroll both panes to bottom on any content change
        from PyQt6.QtGui import QTextCursor as _QTC
        def _autoscroll(widget):
            widget.moveCursor(_QTC.MoveOperation.End)
            widget.ensureCursorVisible()
        self.results_area.textChanged.connect(
            lambda: _autoscroll(self.results_area))
        self.log_area.textChanged.connect(
            lambda: _autoscroll(self.log_area))

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

        self.btn_editor = QPushButton("📝 Open Editor")
        self.btn_editor.setFixedHeight(36)
        self.btn_editor.setToolTip(
            "Open the built-in voice text editor.\n"
            "You can also say \"whisper type\" or \"whisper write\".\n"
            "Use the Editor hotkey to toggle it from anywhere.")
        self.btn_editor.clicked.connect(self.toggle_editor_window)
        self.btn_editor.setStyleSheet(
            "QPushButton{background:#1a3a1a;border:1px solid #2d6a2d;"
            "color:#88cc88;padding:4px;border-radius:4px;}"
            "QPushButton:hover{background:#2d6a2d;color:#fff;border-color:#4caf50;}")
        l1.addWidget(self.btn_editor)
        
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
        ba_cmd = QPushButton("Append .txt")
        ba_cmd.clicked.connect(self._append_commands)
        ba_cmd.setToolTip("Append from file — skips duplicate phrases")
        be_cmd = QPushButton("Export .txt")
        be_cmd.clicked.connect(self._export_commands)
        btn_row.addWidget(ba)
        btn_row.addWidget(bd)
        btn_row.addWidget(bi_cmd)
        btn_row.addWidget(ba_cmd)
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
        tap = QPushButton("Append .txt")
        tap.clicked.connect(self._append_terms)
        tap.setToolTip("Append from file — skips duplicate phrases")
        te = QPushButton("Export .txt")
        te.clicked.connect(self._export_terms)
        terms_btn_row.addWidget(ta)
        terms_btn_row.addWidget(td)
        terms_btn_row.addWidget(ti)
        terms_btn_row.addWidget(tap)
        terms_btn_row.addWidget(te)
        l_terms.addLayout(terms_btn_row)
        self.tabs.addTab(t_terms, "Terms")

        self.tabs.addTab(t2, "Commands")

        # ===== HALLUCINATIONS TAB =====
        t_hall = QWidget()
        l_hall = QVBoxLayout(t_hall)
        lbl_hall = QLabel(
            "Hallucination Blocklist — phrases Whisper generates when it hears\n"
            "silence or background noise instead of real speech.\n"
            "Each entry is matched as a case-insensitive substring of the\n"
            "transcription. If the transcription starts with or equals any entry,\n"
            "it is silently discarded rather than pasted."
        )
        lbl_hall.setWordWrap(True)
        l_hall.addWidget(lbl_hall)

        self.hall_list = QListWidget()
        self.hall_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.hall_list.setAlternatingRowColors(True)
        for phrase in self.config.settings.get("hallucinations", []):
            self.hall_list.addItem(QListWidgetItem(phrase))
        l_hall.addWidget(self.hall_list)

        hall_btn_row = QHBoxLayout()
        hall_add = QPushButton("Add Phrase")
        hall_add.clicked.connect(self._hall_add)
        hall_edit = QPushButton("Edit Selected")
        hall_edit.clicked.connect(self._hall_edit)
        hall_del = QPushButton("Delete Selected")
        hall_del.clicked.connect(self._hall_delete)
        hall_imp = QPushButton("Import .txt")
        hall_imp.clicked.connect(self._hall_import)
        hall_app = QPushButton("Append .txt")
        hall_app.clicked.connect(self._hall_append)
        hall_app.setToolTip("Append from file — skips duplicate phrases")
        hall_exp = QPushButton("Export .txt")
        hall_exp.clicked.connect(self._hall_export)
        hall_btn_row.addWidget(hall_add)
        hall_btn_row.addWidget(hall_edit)
        hall_btn_row.addWidget(hall_del)
        hall_btn_row.addWidget(hall_imp)
        hall_btn_row.addWidget(hall_app)
        hall_btn_row.addWidget(hall_exp)
        l_hall.addLayout(hall_btn_row)

        self.tabs.addTab(t_hall, "Hallucinations")

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
        hotkey_group = QGroupBox("Hotkeys  (click to re-assign · × to clear)")
        hotkey_layout = QFormLayout()

        def _hk_row(btn, clear_cb):
            """Return a QWidget containing [hotkey-btn][×] for a form row."""
            from PyQt6.QtWidgets import QWidget, QHBoxLayout as _HBL
            w = QWidget(); row = _HBL(w); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(4)
            row.addWidget(btn)
            x = QPushButton("×")
            x.setFixedWidth(26)
            x.setToolTip("Clear this hotkey (disable it)")
            x.setStyleSheet("QPushButton{color:#e06c75;font-weight:bold;padding:0;}"
                            "QPushButton:hover{background:#3a2020;}")
            x.clicked.connect(clear_cb)
            row.addWidget(x)
            return w

        self.btn_hk1 = QPushButton(self.config.settings["hotkey"])
        self.btn_hk1.clicked.connect(lambda: self.cap_hk(self.btn_hk1, "hotkey"))
        self.btn_hk1.setToolTip("Toggle dictation on/off.")
        hotkey_layout.addRow("Toggle Dictation:", _hk_row(self.btn_hk1,
            lambda: self.btn_hk1.setText("")))

        self.btn_hk_vis = QPushButton(self.config.settings["visibility_hotkey"])
        self.btn_hk_vis.clicked.connect(lambda: self.cap_hk(self.btn_hk_vis, "visibility_hotkey"))
        self.btn_hk_vis.setToolTip("Show or hide the main window.")
        hotkey_layout.addRow("Show/Hide Window:", _hk_row(self.btn_hk_vis,
            lambda: self.btn_hk_vis.setText("")))

        self.btn_hk_editor = QPushButton(
            self.config.settings.get("editor_hotkey", "ctrl+shift+e"))
        self.btn_hk_editor.clicked.connect(
            lambda: self.cap_hk(self.btn_hk_editor, "editor_hotkey"))
        self.btn_hk_editor.setToolTip(
            "Toggle the built-in text editor window.\n"
            "Works even when the main window is hidden.")
        hotkey_layout.addRow("Toggle Editor Window:", _hk_row(self.btn_hk_editor,
            lambda: self.btn_hk_editor.setText("")))

        self.btn_hk_editor_edit = QPushButton(
            self.config.settings.get("editor_edit_hotkey", ""))
        self.btn_hk_editor_edit.clicked.connect(
            lambda: self.cap_hk(self.btn_hk_editor_edit, "editor_edit_hotkey"))
        self.btn_hk_editor_edit.setToolTip(
            "Copy selected text from the active app and open the\n"
            "editor pre-loaded with it. Same as saying \"whisper edit\".")
        hotkey_layout.addRow("Editor: Copy && Edit:", _hk_row(self.btn_hk_editor_edit,
            lambda: self.btn_hk_editor_edit.setText("")))

        self.btn_hk_rollback = QPushButton(self.config.settings["rollback_hotkey"])
        self.btn_hk_rollback.clicked.connect(lambda: self.cap_hk(self.btn_hk_rollback, "rollback_hotkey"))
        self.btn_hk_rollback.setToolTip(
            "Erases the last pasted segment and lowercases the first\n"
            "letter of the next transcription (for smooth sentence joining)."
        )
        hotkey_layout.addRow("Rollback Last:", _hk_row(self.btn_hk_rollback,
            lambda: self.btn_hk_rollback.setText("")))

        self.btn_hk2 = QPushButton(self.config.settings["ptt_key"])
        self.btn_hk2.clicked.connect(lambda: self.cap_hk(self.btn_hk2, "ptt_key"))
        self.btn_hk2.setToolTip(
            "Hold this key/combo to record (Push-To-Talk).\n"
            "Works regardless of Dictation Mode. Keys are suppressed\n"
            "while held so they don't reach other apps."
        )
        hotkey_layout.addRow("Push-to-Talk:", _hk_row(self.btn_hk2,
            lambda: self.btn_hk2.setText("")))

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
        self.cfg_sk_trigger = QLineEdit(self.config.settings.get("sendkeys_trigger", "whisper send keys"))
        self.cfg_sk_trigger.setToolTip(
            "When this word appears in a Command phrase, the action field\n"
            "is sent as a key sequence instead of launching a program.\n"
            "Example — Phrase: \"sendkeys undo\"  Action: \"<CTRL+Z>\""
        )
        sk_row.addWidget(self.cfg_sk_trigger)
        advanced_layout.addRow("Sendkeys Trigger Word:", sk_row)

        # ── WhisperEditor triggers ────────────────────────────────────────────
        _ed_help = (
            "Comma-separated aliases. Any of these trigger the editor.\n"
            "Example: \"whisper type, whisper write\""
        )
        self.cfg_ed_type_trigger = QLineEdit(
            self.config.settings.get("editor_type_trigger", "whisper type, whisper write"))
        self.cfg_ed_type_trigger.setToolTip(
            "Opens a blank editor.\n" + _ed_help)
        advanced_layout.addRow("Editor: New document trigger:", self.cfg_ed_type_trigger)

        self.cfg_ed_edit_trigger = QLineEdit(
            self.config.settings.get("editor_edit_trigger", "whisper edit, whisper edit this"))
        self.cfg_ed_edit_trigger.setToolTip(
            "Opens editor pre-filled with the current clipboard selection.\n" + _ed_help)
        advanced_layout.addRow("Editor: Edit clipboard trigger:", self.cfg_ed_edit_trigger)

        self.cfg_ed_paste_trigger = QLineEdit(
            self.config.settings.get("editor_paste_trigger", "whisper paste, whisper done, whisper okay"))
        self.cfg_ed_paste_trigger.setToolTip(
            "Pastes editor text back to the active app and closes the editor.\n" + _ed_help)
        advanced_layout.addRow("Editor: Paste-back trigger:", self.cfg_ed_paste_trigger)

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
            self.config.settings.get("select_trigger", "whisper select"))
        self.cfg_sel_trigger.setToolTip(
            "Select trigger — navigates to the target text and selects it.\n" + _nav_help)
        advanced_layout.addRow("WhisperSelect Trigger:", self.cfg_sel_trigger)

        self.cfg_move_trigger = QLineEdit(
            self.config.settings.get("move_trigger", "whisper move"))
        self.cfg_move_trigger.setToolTip(
            "Move-before trigger (synonym for WhisperBefore).\n" + _nav_help)
        advanced_layout.addRow("WhisperMove Trigger:", self.cfg_move_trigger)

        self.cfg_movebefore_trigger = QLineEdit(
            self.config.settings.get("movebefore_trigger", "whisper before"))
        self.cfg_movebefore_trigger.setToolTip(
            "Move-before trigger — places cursor immediately BEFORE the target.\n" + _nav_help)
        advanced_layout.addRow("WhisperBefore Trigger:", self.cfg_movebefore_trigger)

        self.cfg_moveafter_trigger = QLineEdit(
            self.config.settings.get("moveafter_trigger", "whisper after"))
        self.cfg_moveafter_trigger.setToolTip(
            "Move-after trigger — places cursor immediately AFTER the target.\n" + _nav_help)
        advanced_layout.addRow("WhisperAfter Trigger:", self.cfg_moveafter_trigger)

        self.cfg_replace_trigger = QLineEdit(
            self.config.settings.get("replace_trigger", "whisper replace"))
        self.cfg_replace_trigger.setToolTip(
            "Say this + a target phrase, then speak replacement text.\n"
            "Finds the target in the session buffer, replaces it in-memory,\n"
            "then undoes the last paste and repastes the corrected version.")
        advanced_layout.addRow("WhisperReplace Trigger:", self.cfg_replace_trigger)

        self.cfg_insertbefore_trigger = QLineEdit(
            self.config.settings.get("insertbefore_trigger", "whisper insert before"))
        self.cfg_insertbefore_trigger.setToolTip(
            "Say this + a target phrase, then speak text to insert BEFORE the target.")
        advanced_layout.addRow("WhisperInsertBefore Trigger:", self.cfg_insertbefore_trigger)

        self.cfg_insertafter_trigger = QLineEdit(
            self.config.settings.get("insertafter_trigger", "whisper insert after"))
        self.cfg_insertafter_trigger.setToolTip(
            "Say this + a target phrase, then speak text to insert AFTER the target.")
        advanced_layout.addRow("WhisperInsertAfter Trigger:", self.cfg_insertafter_trigger)

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

    def _append_terms(self):
        """Append terms from a .txt file, skipping any whose phrase already exists."""
        path, _ = QFileDialog.getOpenFileName(self, "Append Terms", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            existing = {
                self.terms_table.item(r, 0).text().lower()
                for r in range(self.terms_table.rowCount())
                if self.terms_table.item(r, 0)
            }
            added = 0
            for line in lines:
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    k = k.strip(); v = v.strip()
                    if k and k.lower() not in existing:
                        r = self.terms_table.rowCount()
                        self.terms_table.insertRow(r)
                        self.terms_table.setItem(r, 0, QTableWidgetItem(k))
                        self.terms_table.setItem(r, 1, QTableWidgetItem(v))
                        existing.add(k.lower())
                        added += 1
            app_logger.info(f"Terms: appended {added} new entry/entries from {path}")
            self.scratchpad.append(f"[Terms] Appended {added} new entry/entries.")
        except Exception as e:
            app_logger.error(f"Failed to append terms: {e}")
            QMessageBox.warning(self, "Append Failed", str(e))

    def _append_commands(self):
        """Append commands from a .txt file, skipping any whose phrase already exists."""
        path, _ = QFileDialog.getOpenFileName(self, "Append Commands", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            existing = {
                self.cmd_table.item(r, 0).text().lower()
                for r in range(self.cmd_table.rowCount())
                if self.cmd_table.item(r, 0)
            }
            added = 0
            for line in lines:
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    k = k.strip(); v = v.strip()
                    if k and k.lower() not in existing:
                        r = self.cmd_table.rowCount()
                        self.cmd_table.insertRow(r)
                        self.cmd_table.setItem(r, 0, QTableWidgetItem(k))
                        self.cmd_table.setItem(r, 1, QTableWidgetItem(v))
                        existing.add(k.lower())
                        added += 1
            app_logger.info(f"Commands: appended {added} new entry/entries from {path}")
            self.scratchpad.append(f"[Commands] Appended {added} new entry/entries.")
        except Exception as e:
            app_logger.error(f"Failed to append commands: {e}")
            QMessageBox.warning(self, "Append Failed", str(e))

    def _hall_append(self):
        """Append hallucination phrases from file, skipping duplicates."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Append Hallucinations", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            existing = {
                self.hall_list.item(i).text().lower()
                for i in range(self.hall_list.count())
            }
            added = 0
            for line in lines:
                phrase = line.strip()
                if phrase and phrase.lower() not in existing:
                    self.hall_list.addItem(QListWidgetItem(phrase))
                    existing.add(phrase.lower())
                    added += 1
            app_logger.info(f"Hallucinations: appended {added} phrase(s) from {path}")
            self.scratchpad.append(f"[Hallucinations] Appended {added} phrase(s).")
        except Exception as e:
            app_logger.error(f"Failed to append hallucinations: {e}")
            QMessageBox.warning(self, "Append Failed", str(e))

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

    # ── Hallucinations tab methods ───────────────────────────────────────────

    def _hall_add(self):
        """Prompt for a new hallucination phrase and add it to the list."""
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "Add Hallucination Phrase",
            "Enter phrase (case-insensitive substring to block):")
        if ok and text.strip():
            self.hall_list.addItem(QListWidgetItem(text.strip()))

    def _hall_edit(self):
        """Edit the currently selected hallucination phrase in-place."""
        from PyQt6.QtWidgets import QInputDialog
        item = self.hall_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Selection", "Select a phrase to edit.")
            return
        text, ok = QInputDialog.getText(
            self, "Edit Hallucination Phrase", "Phrase:", text=item.text())
        if ok and text.strip():
            item.setText(text.strip())

    def _hall_delete(self):
        """Delete all selected hallucination phrases."""
        for item in self.hall_list.selectedItems():
            self.hall_list.takeItem(self.hall_list.row(item))

    def _hall_import(self):
        """Import hallucination phrases from a .txt file (one phrase per line).
        Appends to the existing list rather than replacing it.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Hallucinations", "", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = Path(path).read_text(encoding="utf-8").splitlines()
            added = 0
            # Build set of existing phrases for deduplication
            existing = {
                self.hall_list.item(i).text().lower()
                for i in range(self.hall_list.count())
            }
            for line in lines:
                phrase = line.strip()
                if phrase and phrase.lower() not in existing:
                    self.hall_list.addItem(QListWidgetItem(phrase))
                    existing.add(phrase.lower())
                    added += 1
            app_logger.info(f"Hallucinations: imported {added} phrase(s) from {path}")
            self.scratchpad.append(f"[Hallucinations] Imported {added} phrase(s).")
        except Exception as e:
            app_logger.error(f"Failed to import hallucinations: {e}")
            QMessageBox.warning(self, "Import Failed", str(e))

    def _hall_export(self):
        """Export all hallucination phrases to a .txt file (one phrase per line)."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Hallucinations", "hallucinations.txt", "Text Files (*.txt)")
        if not path:
            return
        try:
            lines = [
                self.hall_list.item(i).text()
                for i in range(self.hall_list.count())
                if self.hall_list.item(i).text().strip()
            ]
            Path(path).write_text("\n".join(lines), encoding="utf-8")
            app_logger.info(f"Hallucinations exported to {path}")
        except Exception as e:
            app_logger.error(f"Failed to export hallucinations: {e}")
            QMessageBox.warning(self, "Export Failed", str(e))

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
            "editor_hotkey":      self.btn_hk_editor.text(),
            "editor_edit_hotkey": self.btn_hk_editor_edit.text(),
            "noise_floor": self.n_spin.value(),
            "speech_vol": self.s_spin.value(),
            "commands": cmds,
            "terms": {
                self.terms_table.item(r, 0).text(): self.terms_table.item(r, 1).text()
                for r in range(self.terms_table.rowCount())
                if self.terms_table.item(r, 0) and self.terms_table.item(r, 1)
                and self.terms_table.item(r, 0).text()
            },
            "hallucinations": [
                self.hall_list.item(i).text()
                for i in range(self.hall_list.count())
                if self.hall_list.item(i).text().strip()
            ],
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
            "editor_type_trigger":  self.cfg_ed_type_trigger.text().strip() or "whisper type, whisper write",
            "editor_edit_trigger":  self.cfg_ed_edit_trigger.text().strip() or "whisper edit, whisper edit this",
            "editor_paste_trigger": self.cfg_ed_paste_trigger.text().strip() or "whisper paste, whisper done, whisper okay",
            "sendkeys_trigger":   self.cfg_sk_trigger.text().strip() or "whisper send keys",
            "select_trigger":     self.cfg_sel_trigger.text().strip() or "whisper select",
            "move_trigger":       self.cfg_move_trigger.text().strip() or "whisper move",
            "movebefore_trigger": self.cfg_movebefore_trigger.text().strip() or "whisper before",
            "moveafter_trigger":      self.cfg_moveafter_trigger.text().strip() or "whisper after",
            "replace_trigger":        self.cfg_replace_trigger.text().strip() or "whisper replace",
            "insertbefore_trigger":   self.cfg_insertbefore_trigger.text().strip() or "whisper insert before",
            "insertafter_trigger":    self.cfg_insertafter_trigger.text().strip() or "whisper insert after",
            "fuzzy_threshold":        self.cfg_fuzzy_threshold.value()
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
            # _pre_rec_hwnd was already captured in on_toggle_hotkey (keyboard thread)
            # while the external app still had focus — don't overwrite it here.
            # Rollback handshake:
            # rollback_transcription() sets _rollback_armed (survives this block).
            # We consume it here and set _rollback_pending for this session only.
            # Any stale _rollback_pending from a previous no-speech session is cleared.
            self._rollback_pending = bool(getattr(self, '_rollback_armed', False))
            self._rollback_armed   = False
            if self._rollback_pending:
                app_logger.debug("toggle_rec: rollback armed — will lowercase first result")
            # Reset session buffer, cursor offset, and cursor-ops flag for the new session
            self._session_buffer = ""
            self._cursor_offset  = 0
            self._session_paste_count = 0
            self._cursor_ops_pending = False
            self._pending_edit = None
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



    # ── Voice wizard: multi-step guided voice commands ────────────────────────

    # Wizard steps per operation:
    #
    #  select / move / movebefore / moveafter
    #      step "target"  → ask "what?" → call navigate
    #
    #  replace
    #      step "target"  → ask "replace what?"
    #      step "replacement" → ask "replace with what?" → do in-buffer replace
    #
    #  insertbefore / insertafter
    #      step "target"  → ask "insert where?"
    #      step "text"    → ask "insert what?" → do in-buffer insert

    _WIZARD_PROMPTS = {
        # (op, step) → (overlay_title, overlay_body)
        ("select",      "target"):      ("Whisper Select",        "Say what you want to select."),
        ("move",        "target"):      ("Whisper Move",          "Say what you want to move to."),
        ("movebefore",  "target"):      ("Whisper Before",        "Say the word to move before."),
        ("moveafter",   "target"):      ("Whisper After",         "Say the word to move after."),
        ("replace",     "target"):      ("Whisper Replace",       "Say what you want to replace."),
        ("replace",     "replacement"): ("Whisper Replace",       "Now say the replacement text."),
        ("insertbefore","target"):      ("Whisper Insert Before", "Say the word to insert before."),
        ("insertbefore","text"):        ("Whisper Insert Before", "Now say the text to insert."),
        ("insertafter", "target"):      ("Whisper Insert After",  "Say the word to insert after."),
        ("insertafter", "text"):        ("Whisper Insert After",  "Now say the text to insert."),
    }

    # ── App-level clipboard monitor ──────────────────────────────────────────
    # Lives on WhisperRApp so it persists when the editor is hidden or closed.
    # The editor's toggle buttons start/stop it via delegation.

    def start_clipboard_monitor(self):
        """Start (or restart) the app-level clipboard polling timer."""
        if self._cb_monitor_timer and self._cb_monitor_timer.isActive():
            return  # already running
        try:
            import pyperclip as _pcp
            self._cb_monitor_last = _pcp.paste()
        except Exception:
            self._cb_monitor_last = ""
        self._cb_monitor_timer = QTimer(self)
        self._cb_monitor_timer.setInterval(500)
        self._cb_monitor_timer.timeout.connect(self._cb_monitor_poll)
        self._cb_monitor_timer.start()
        app_logger.info("Clipboard monitor started")

    def stop_clipboard_monitor(self):
        """Stop the clipboard polling timer."""
        if self._cb_monitor_timer:
            self._cb_monitor_timer.stop()
            self._cb_monitor_timer = None
        app_logger.info("Clipboard monitor stopped")

    def _cb_monitor_poll(self):
        """Poll clipboard; if changed, append new content to the editor.

        The editor window's VISIBILITY IS NOT CHANGED — if it's hidden it stays
        hidden, if visible it stays visible.  Text is always appended.
        """
        try:
            import pyperclip as _pcp
            current = _pcp.paste()
        except Exception:
            return
        if current == self._cb_monitor_last:
            return
        self._cb_monitor_last = current
        if not current.strip():
            return

        # Ensure editor exists (create silently if it was destroyed)
        if not self._editor:
            self._editor = WhisperEditor(config=self.config, parent=None)
            self._editor.paste_requested.connect(self._editor_paste_to_app)
            self._editor.finished.connect(self._on_editor_closed)
            # Restore the monitor toggle state so the user can still turn it off
            self._editor.clipboard_monitor_toggle.setChecked(True)
            # Don't show — visibility is unchanged from before

        cur = self._editor.editor.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        existing = self._editor.editor.toPlainText()
        sep = "\n\n" if existing.strip() else ""
        cur.insertText(sep + current)
        self._editor.editor.setTextCursor(cur)
        if self._editor.isVisible():
            self._editor.editor.ensureCursorVisible()

    def _open_editor(self, prefill: str = "", from_clipboard: bool = False):
        """Open the WhisperEditor window.

        prefill        : text to pre-populate (empty = blank slate)
        from_clipboard : if True, Ctrl+C the current selection first.
        Respects the remember-content toggle: if on, saved content is
        prepended/appended rather than replaced.
        """
        import time as _te_ed
        import pyperclip as _pc_ed

        try:
            import ctypes as _ct_ed
            self._editor_return_hwnd = _ct_ed.windll.user32.GetForegroundWindow()
        except Exception:
            self._editor_return_hwnd = None

        if from_clipboard:
            # ── Clipboard-capture sequence ────────────────────────────────────
            # Two paths:
            #   Hotkey path: on_editor_edit_hotkey already sent Ctrl+C on the
            #     pynput thread (before focus could shift) and stored the result
            #     in self._pre_clip_capture.  Use it directly, no Ctrl+C needed.
            #   Voice path: we must refocus the source window then send Ctrl+C,
            #     then poll until the clipboard changes (up to 1.5 s).
            new_text = ""
            _pre_captured = getattr(self, "_pre_clip_capture", None)
            # Consume the pre-captured value (None = not set, "" = nothing selected)
            self._pre_clip_capture = None

            if _pre_captured is not None:
                # ── Hotkey path: clipboard already captured ────────────────
                new_text = _pre_captured
                app_logger.info(
                    f"[EditThis] using pre-captured text ({len(new_text)}ch)")
            else:
                # ── Voice path: refocus + Ctrl+C + poll ───────────────────
                try:
                    import ctypes as _ct_fc, time as _t_fc
                    import pyperclip as _pcp_fc, pyautogui as _pag_fc

                    _u32 = _ct_fc.windll.user32
                    _k32 = _ct_fc.windll.kernel32

                    def _cb_read():
                        try:
                            return _pcp_fc.paste() or ""
                        except Exception:
                            pass
                        CF_UNICODETEXT = 13
                        try:
                            if not _u32.OpenClipboard(None):
                                return ""
                            try:
                                h = _u32.GetClipboardData(CF_UNICODETEXT)
                                if not h: return ""
                                p = _k32.GlobalLock(h)
                                if not p: return ""
                                try:
                                    return _ct_fc.wstring_at(p)
                                finally:
                                    _k32.GlobalUnlock(h)
                            finally:
                                _u32.CloseClipboard()
                        except Exception:
                            return ""

                    # Step 1: clear clipboard, then record the (now-empty) state
                    # This ensures identical re-selections are detected as a change.
                    try:
                        import pyperclip as _pcp_clr
                        _pcp_clr.copy("")
                        _t_fc.sleep(0.03)
                    except Exception:
                        pass
                    _clip_before = ""   # clipboard is now empty
                    _clip_original = _cb_read()  # should be "" but keep for safety
                    _fg_now      = _u32.GetForegroundWindow()
                    _src_hwnd    = getattr(self, "_pre_rec_hwnd", None) or 0
                    app_logger.info(
                        f"[EditThis] voice: fg_now=0x{_fg_now:08X} "
                        f"src=0x{_src_hwnd:08X} "
                        f"clip_before={len(_clip_before)}ch")
                    try:
                        _title_buf = _ct_fc.create_unicode_buffer(256)
                        _u32.GetWindowTextW(_src_hwnd, _title_buf, 256)
                        app_logger.info(f"[EditThis] target window: '{_title_buf.value}'")
                    except Exception:
                        pass

                    # Step 2: always refocus source window
                    # (even if _src_hwnd == _fg_now — the value may be stale)
                    if _src_hwnd:
                        try:
                            _t_cur = _k32.GetCurrentThreadId()
                            _t_fg  = _u32.GetWindowThreadProcessId(_fg_now, None)
                            _t_src = _u32.GetWindowThreadProcessId(_src_hwnd, None)
                            _u32.AttachThreadInput(_t_cur, _t_fg,  True)
                            _u32.AttachThreadInput(_t_cur, _t_src, True)
                            _u32.SetForegroundWindow(_src_hwnd)
                            _t_fc.sleep(0.15)
                            _fg_after = _u32.GetForegroundWindow()
                            app_logger.info(
                                f"[EditThis] refocus: target=0x{_src_hwnd:08X} "
                                f"actual=0x{_fg_after:08X} "
                                f"ok={_fg_after == _src_hwnd}")
                            _u32.AttachThreadInput(_t_cur, _t_fg,  False)
                            _u32.AttachThreadInput(_t_cur, _t_src, False)
                        except Exception as _ef:
                            app_logger.warning(f"[EditThis] refocus failed: {_ef}")

                    # Step 3: release any modifier keys still physically held
                    # (hotkey path: Ctrl+Shift+Alt+X may still be down when this runs,
                    # causing Ctrl+C to land as Ctrl+Shift+Alt+X+C in the target app)
                    try:
                        for _mod in ('ctrl', 'shift', 'alt', 'win'):
                            try: _pag_fc.keyUp(_mod)
                            except Exception: pass
                        _t_fc.sleep(0.05)   # let OS process the key-ups
                    except Exception:
                        pass

                    # Step 3b: send Ctrl+C
                    _pag_fc.hotkey('ctrl', 'c')
                    app_logger.info("[EditThis] Ctrl+C sent")

                    # Step 4: poll until clipboard changes (max 1.5 s)
                    for _i in range(30):
                        _t_fc.sleep(0.05)
                        _clip_after = _cb_read()
                        if _clip_after != _clip_before:
                            app_logger.info(
                                f"[EditThis] clipboard changed after {(_i+1)*50}ms "
                                f"({len(_clip_after)}ch)")
                            new_text = _clip_after
                            break
                    else:
                        _clip_after = _cb_read()
                        app_logger.info(
                            f"[EditThis] clipboard unchanged after 1.5s "
                            f"(still {len(_clip_after)}ch)")
                        new_text = ""

                except Exception as _e_fc:
                    app_logger.warning(f"[EditThis] capture failed: {_e_fc}", exc_info=True)
                    new_text = ""

            # If remember is on, append below existing content
            if self._editor_remember and self._editor_saved_content:
                prefill = self._editor_saved_content.rstrip() + "\n\n" + new_text
            else:
                prefill = new_text
        elif self._editor_remember and self._editor_saved_content and not prefill:
            prefill = self._editor_saved_content

        # Re-use existing visible editor
        if self._editor and self._editor.isVisible():
            if from_clipboard and prefill:
                self._editor.editor.setPlainText(prefill)
            self._editor.raise_()
            self._editor.activateWindow()
            return

        # Clipboard prefill: always load CURRENT clipboard on open (not saved).
        # The toggle state persists; the content is always fresh each open.
        if not from_clipboard and getattr(self, "_editor_clipboard_prefill", False):
            try:
                import pyperclip as _pc_pf
                prefill = _pc_pf.paste()  # always current, always replaces
            except Exception:
                pass

        self._editor = WhisperEditor(
            initial_text=prefill,
            config=self.config,
            parent=None)
        # Restore toggle states
        if self._editor_remember:
            self._editor.remember_toggle.setChecked(True)
        elif getattr(self, "_editor_clipboard_prefill", False):
            self._editor.clipboard_prefill_toggle.setChecked(True)
        # Restore monitor toggle if it was on (monitor itself already running on app)
        if getattr(self, "_editor_cb_monitor_was_on", False):
            self._editor.clipboard_monitor_toggle.setChecked(True)
        # Restore target word count
        if getattr(self, "_editor_saved_target", 0):
            self._editor.target_spin.setValue(self._editor_saved_target)
        self._editor.paste_requested.connect(self._editor_paste_to_app)
        # Save content when editor is closed/hidden
        self._editor.finished.connect(self._on_editor_closed)
        self._editor.show()
        self._editor.raise_()
        self._editor.activateWindow()
        self.scratchpad.append("[Editor] Opened — dictation now goes into the editor.")

    def _on_editor_closed(self):
        """Save content/state when editor window is closed."""
        if not self._editor:
            return
        # Snapshot all toggle states
        self._editor_cb_monitor_was_on = (
            getattr(self._editor, "clipboard_monitor_toggle", None) and
            self._editor.clipboard_monitor_toggle.isChecked())
        self._editor_clipboard_prefill = (
            getattr(self._editor, "clipboard_prefill_toggle", None) and
            self._editor.clipboard_prefill_toggle.isChecked())
        _remember_on = (getattr(self._editor, "remember_toggle", None) and
                        self._editor.remember_toggle.isChecked())
        # Persist remember state and content
        if _remember_on:
            self._editor_saved_content = self._editor.editor.toPlainText()
            self._editor_remember = True
            self._editor_saved_target = getattr(self._editor, "target_spin",
                                                None) and self._editor.target_spin.value() or 0
        else:
            self._editor_remember = False
            self._editor_saved_content = ""
            self._editor_saved_target = 0
        # Write full state JSON (includes all fields for future expansion)
        _state_path = getattr(self, "_editor_state_path", None)
        if _state_path:
            try:
                import json as _json_sv
                _state = {
                    "remember":        _remember_on,
                    "content":         self._editor_saved_content,
                    "target_words":    self._editor_saved_target,
                    "clipboard_prefill": self._editor_clipboard_prefill,
                    "cb_monitor":      self._editor_cb_monitor_was_on,
                }
                _state_path.write_text(
                    _json_sv.dumps(_state, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                if not _remember_on and _state_path.exists():
                    _state_path.unlink(missing_ok=True)  # clean up when remember off
            except Exception as _e_sv:
                app_logger.warning(f"Could not save editor state: {_e_sv}")
        # Keep the editor object alive if clipboard monitor is running —
        # the poll timer needs the reference to keep appending text.
        # Otherwise null it so _open_editor creates a fresh one next time.
        _monitor_running = (self._cb_monitor_timer and self._cb_monitor_timer.isActive())
        if not _monitor_running:
            self._editor = None

    def _editor_paste_to_app(self, text: str):
        """Called when user clicks 'Paste to App' or says the paste trigger.

        Restores focus to the previously active window and pastes the text.
        """
        import time as _te_ep
        import pyperclip as _pc_ep
        import pyautogui as _pag_ep
        import ctypes as _ct_ep

        hwnd = getattr(self, "_editor_return_hwnd", None)
        if hwnd:
            try:
                _u32 = _ct_ep.windll.user32
                _fg   = _u32.GetForegroundWindow()
                _tFG  = _u32.GetWindowThreadProcessId(_fg, None)
                _tTGT = _u32.GetWindowThreadProcessId(hwnd, None)
                _u32.AttachThreadInput(_tFG, _tTGT, True)
                _u32.SetForegroundWindow(hwnd)
                _u32.BringWindowToTop(hwnd)
                _u32.AttachThreadInput(_tFG, _tTGT, False)
                _te_ep.sleep(0.5)
            except Exception:
                pass

        _pc_ep.copy(text)
        _te_ep.sleep(self.config.settings.get("paste_delay", 0.5))
        _pag_ep.hotkey("ctrl", "v")
        self.scratchpad.append("[Editor] Text pasted to application.")

    def _wizard_start(self, op):
        """Launch a new wizard for `op`.  Called when a trigger is detected
        with no inline argument, or always for replace/insert ops."""
        self._wizard_cancel(silent=True)   # close any existing wizard first

        # Capture the currently active window BEFORE showing our overlay,
        # so we can restore focus to it when the wizard is done.
        try:
            import ctypes as _ct_wiz
            self._wizard_prev_hwnd = _ct_wiz.windll.user32.GetForegroundWindow()
        except Exception:
            self._wizard_prev_hwnd = None

        first_step = "target"
        title, prompt = self._WIZARD_PROMPTS[(op, first_step)]

        self._wizard = {"op": op, "step": first_step, "collected": {}}
        # Anchor overlay over editor if open, otherwise bottom-right
        _anchor = (self._editor
                   if (self._editor and self._editor.isVisible()) else None)
        self._wizard_overlay = WizardOverlay(title, prompt, parent=None, anchor=_anchor)
        self._wizard_overlay.cancelled.connect(self._wizard_cancel)
        self._wizard_overlay.show()
        self._wizard_overlay.raise_()
        # Only steal focus when editor is NOT open — when editor is open,
        # stealing and then returning focus triggers spurious close events.
        if not (self._editor and self._editor.isVisible()):
            self._wizard_overlay.activateWindow()
        self.scratchpad.append(f"[Wizard] {op} — step: {first_step}")
        app_logger.info(f"Wizard started: op={op!r}, prev_hwnd={getattr(self, '_wizard_prev_hwnd', None)}")

    def _wizard_step(self, text):
        """Route a transcription result into the active wizard.
        Returns True if the text was consumed by the wizard (suppress normal paste).
        """
        if self._wizard is None:
            return False

        op   = self._wizard["op"]
        step = self._wizard["step"]
        col  = self._wizard["collected"]

        # Show live feedback
        if self._wizard_overlay:
            self._wizard_overlay.feed(text)

        col[step] = text.strip()
        app_logger.info(f"Wizard step={step!r} heard: {text!r}")

        # ── After "target" step: check for multiple instances in editor ──
        if step == "target" and op not in ("select",) and \
                (self._editor and self._editor.isVisible()):
            import re as _re_ws
            _tgt_clean = text.strip().rstrip(".,!?;:")
            _tgt_clean = _re_ws.sub(
                r"^(to|before|after|the|for|with|a|an)\s+", "",
                _tgt_clean, flags=_re_ws.IGNORECASE).strip()
            _n_hits = self._editor.find_instances(_tgt_clean)
            if _n_hits > 1:
                # Number them in the editor and ask which one
                _labels = self._editor.annotate_instances(_tgt_clean)
                col["target"] = _tgt_clean   # store cleaned version
                self._wizard["step"] = "instance"
                _inst_prompt = (
                    f"Found {_n_hits} occurrences of \"{_tgt_clean}\" — "
                    "numbered in the editor.\n"
                    "Say which one: \"one\", \"two\", \"the third one\", or a number.")
                self._WIZARD_PROMPTS[(op, "instance")] = (f"Whisper {op.title()}", _inst_prompt)
                if self._wizard_overlay:
                    self._wizard_overlay.set_prompt(_inst_prompt)
                return True

        # ── Instance selection step ───────────────────────────────────────
        if step == "instance":
            # Parse spoken ordinal / number to 1-based index
            _ordinals = {
                "first":"1","one":"1","1":"1",
                "second":"2","two":"2","2":"2",
                "third":"3","three":"3","3":"3",
                "fourth":"4","four":"4","4":"4",
                "fifth":"5","five":"5","5":"5",
                "sixth":"6","six":"6","6":"6",
                "seventh":"7","seven":"7","7":"7",
                "eighth":"8","eight":"8","8":"8",
                "ninth":"9","nine":"9","9":"9",
                "tenth":"10","ten":"10","10":"10",
            }
            _spoken_l = text.lower().strip().rstrip(".,!?")
            _inst_num = None
            for _kw, _num in _ordinals.items():
                if _kw in _spoken_l:
                    _inst_num = int(_num); break
            import re as _re_num
            if _inst_num is None:
                _m_dig = _re_num.search(r"\d+", _spoken_l)
                if _m_dig:
                    _inst_num = int(_m_dig.group())
            if _inst_num is None:
                if self._wizard_overlay:
                    self._wizard_overlay.set_prompt(
                        "Didn't catch which one — please say a number or ordinal.")
                return True  # stay on instance step
            col["instance"] = _inst_num
            # Now advance to the next real step
            if op in ("select", "move", "movebefore", "moveafter"):
                self._wizard_finish(); return True
            elif op == "replace":
                self._wizard["step"] = "replacement"
                _, _p = self._WIZARD_PROMPTS[(op, "replacement")]
                if self._wizard_overlay: self._wizard_overlay.set_prompt(_p)
                return True
            elif op in ("insertbefore", "insertafter"):
                self._wizard["step"] = "text"
                _, _p = self._WIZARD_PROMPTS[(op, "text")]
                if self._wizard_overlay: self._wizard_overlay.set_prompt(_p)
                return True

        # ── Normal step progression ───────────────────────────────────────
        if op in ("select", "move", "movebefore", "moveafter"):
            self._wizard_finish()

        elif op == "replace":
            if step == "target":
                self._wizard["step"] = "replacement"
                title, prompt = self._WIZARD_PROMPTS[(op, "replacement")]
                if self._wizard_overlay:
                    self._wizard_overlay.set_prompt(prompt)
            else:  # step == "replacement"
                self._wizard_finish()

        elif op in ("insertbefore", "insertafter"):
            if step == "target":
                self._wizard["step"] = "text"
                title, prompt = self._WIZARD_PROMPTS[(op, "text")]
                if self._wizard_overlay:
                    self._wizard_overlay.set_prompt(prompt)
            else:  # step == "text"
                self._wizard_finish()

        return True   # consumed

    def _wizard_finish(self):
        """All steps collected — execute the operation and close the overlay."""
        if self._wizard is None:
            return

        op  = self._wizard["op"]
        col = self._wizard["collected"]

        import re as _re_wiz

        def _clean(s):
            """Strip leading filler words and trailing punctuation Whisper adds."""
            s = (s or "").strip().rstrip(".,!?;:")
            s = _re_wiz.sub(
                r"^(to|before|after|the|for|with|a|an)\s+", "",
                s, flags=_re_wiz.IGNORECASE)
            return s.strip()

        # Show confirmation in overlay, then close it and restore focus
        # before executing the action — the target app must be foreground.
        def _exec_action():
            try:
                # If editor is open, ops work on its text directly — no pyautogui needed
                _ed = self._editor if (self._editor and self._editor.isVisible()) else None

                if op in ("select", "move", "movebefore", "moveafter"):
                    target = _clean(col.get("target", ""))
                    if target:
                        if _ed:
                            # In editor: just move cursor to the word
                            _doc_text = _ed.editor.toPlainText()
                            import re as _re_edop
                            _m = _re_edop.search(_re_edop.escape(target), _doc_text, _re_edop.IGNORECASE)
                            if _m:
                                cur = _ed.editor.textCursor()
                                cur.setPosition(_m.start())
                                if op == "select":
                                    cur.setPosition(_m.end(), cur.MoveMode.KeepAnchor)
                                _ed.editor.setTextCursor(cur)
                                _ed.editor.ensureCursorVisible()
                            else:
                                self.scratchpad.append(f"[Editor] '{target}' not found.")
                        else:
                            self._whisper_navigate(op, target)
                    else:
                        self.scratchpad.append("[Wizard] Empty target — nothing to do.")

                elif op == "replace":
                    target      = _clean(col.get("target", ""))
                    replacement = col.get("replacement", "").strip()
                    if target and replacement:
                        if _ed:
                            _inst = col.get("instance", -1)
                            ok = _ed.execute_edit("replace", target, replacement, instance=_inst)
                            self.scratchpad.append(
                                f"[Editor] replace: '{target}' → '{replacement[:30]}'" if ok
                                else f"[Editor] '{target}' not found.")
                        else:
                            self._wizard_in_buffer_edit("replace", target, replacement)
                    else:
                        self.scratchpad.append("[Wizard] Replace: missing target or replacement.")

                elif op in ("insertbefore", "insertafter"):
                    target   = _clean(col.get("target", ""))
                    ins_text = col.get("text", "").strip()
                    if target and ins_text:
                        if _ed:
                            _inst = col.get("instance", -1)
                            ok = _ed.execute_edit(op, target, ins_text, instance=_inst)
                            self.scratchpad.append(
                                f"[Editor] {op}: '{target}' / '{ins_text[:30]}'" if ok
                                else f"[Editor] '{target}' not found.")
                        else:
                            self._wizard_in_buffer_edit(op, target, ins_text)
                    else:
                        self.scratchpad.append("[Wizard] Insert: missing target or text.")

            except Exception as e:
                app_logger.error(f"Wizard finish error: {e}", exc_info=True)
                self.scratchpad.append(f"[Wizard] Error: {e}")

        # Build a summary for the confirmation label
        if op in ("select", "move", "movebefore", "moveafter"):
            _summary = f"→ {_clean(col.get('target', ''))}"
        elif op == "replace":
            _summary = f"{_clean(col.get('target',''))} → {col.get('replacement','')[:30]}"
        else:
            _summary = f"{_clean(col.get('target',''))} / {col.get('text','')[:30]}"
        if self._wizard_overlay:
            self._wizard_overlay.confirm(_summary)

        # Restore focus to the previously active window, then delay before
        # sending keystrokes so it has time to fully activate.
        _hwnd = getattr(self, "_wizard_prev_hwnd", None)
        _delay_ms = max(int(self.config.settings.get("paste_delay", 0.5) * 1000), 500)

        def _restore_and_exec():
            # When the editor is open, ops run directly on the QTextEdit —
            # no focus change needed or wanted.
            _ed_open = self._editor and self._editor.isVisible()
            if _hwnd and not _ed_open:
                try:
                    import ctypes as _ct_r
                    _u32 = _ct_r.windll.user32
                    _fg   = _u32.GetForegroundWindow()
                    _tFG  = _u32.GetWindowThreadProcessId(_fg, None)
                    _tTGT = _u32.GetWindowThreadProcessId(_hwnd, None)
                    _u32.AttachThreadInput(_tFG, _tTGT, True)
                    _u32.SetForegroundWindow(_hwnd)
                    _u32.BringWindowToTop(_hwnd)
                    _u32.AttachThreadInput(_tFG, _tTGT, False)
                except Exception:
                    pass
            # For editor ops, execute immediately; for external, wait for focus
            from PyQt6.QtCore import QTimer as _QT2
            _delay = 0 if _ed_open else _delay_ms
            _QT2.singleShot(_delay, _exec_action)

        # Close overlay first, then restore + execute after brief pause
        if self._wizard_overlay:
            from PyQt6.QtCore import QTimer as _QT
            _ov = self._wizard_overlay
            _QT.singleShot(600, _ov.close)
            _QT.singleShot(650, _restore_and_exec)
        else:
            _restore_and_exec()

        self._wizard         = None
        self._wizard_overlay = None
        self._cursor_ops_pending = True

    def _wizard_cancel(self, silent=False):
        """Abort the current wizard without executing anything."""
        if self._wizard_overlay:
            self._wizard_overlay.close()
            self._wizard_overlay = None
        if self._wizard and not silent:
            self.scratchpad.append(f"[Wizard] Cancelled ({self._wizard['op']}).")
        # Remove any instance annotations left in the editor
        if self._editor and self._editor.isVisible():
            self._editor._remove_annotations()
        self._wizard = None

    def _wizard_in_buffer_edit(self, op, target, new_text):
        """Edit the session buffer in-place, then select+delete the original
        session text in the active app and repaste the corrected version.

        Uses select→delete→paste rather than Ctrl+Z, so it works regardless
        of the target app's undo history depth.
        """
        import re as _re_e
        import pyautogui as _pag_e
        import pyperclip as _pc_e
        import ctypes as _ct_e
        import time as _te

        buf   = self._session_buffer
        pat   = _re_e.compile(_re_e.escape(target), _re_e.IGNORECASE)
        hits  = list(pat.finditer(buf))
        if not hits:
            self.scratchpad.append(
                f"[Wizard] \"{target}\" not found in session buffer.")
            return

        m = hits[-1]
        if op == "replace":
            new_buf = buf[:m.start()] + new_text + buf[m.end():]
        elif op == "insertbefore":
            new_buf = buf[:m.start()] + new_text + " " + buf[m.start():]
        elif op == "insertafter":
            new_buf = buf[:m.end()] + " " + new_text + buf[m.end():]
        else:
            return

        paste_delay   = self.config.settings.get("paste_delay", 0.5)
        paste_count   = getattr(self, "_session_paste_count", 1)

        # Release any held modifier keys
        for _vk in (0x11, 0x10, 0x12):
            _ct_e.windll.user32.keybd_event(_vk, 0, 0x0002, 0)
        _te.sleep(0.06)

        # Undo every paste this session (one Ctrl+Z per paste = one clipboard
        # operation each), then repaste the corrected buffer in one shot.
        # This is reliable because each paste is a single undoable action —
        # unlike shift+selecting N chars which depends on cursor position.
        for _ in range(max(paste_count, 1)):
            _pag_e.hotkey("ctrl", "z")
            _te.sleep(0.06)

        # Brief pause to let the app settle after undo(s)
        _te.sleep(0.1)

        _pc_e.copy(new_buf)
        _te.sleep(paste_delay)
        _pag_e.hotkey("ctrl", "v")

        self._session_buffer = new_buf
        self._cursor_offset  = 0
        self._session_paste_count = 1  # the repaste above counts as one paste
        self.scratchpad.append(
            f"[Wizard] {op}: \"{target}\" → \"{new_text[:40]}\"")
        app_logger.info(f"Wizard in-buffer edit {op} on \"{target}\"")

    def _navigate_press_n(self, pag, key, n, interval=0.01):
        """Send `n` keypresses of `key`, using pyautogui with a small interval."""
        import time as _t
        for _ in range(n):
            pag.press(key)
            if n > 20:
                _t.sleep(interval)  # tiny pause for responsiveness on large moves

    def _whisper_navigate(self, operation, target_text):
        """Perform a cursor navigation / selection operation inside the active window.

        Works from self._session_buffer and self._cursor_offset.

        _cursor_offset tracks how many characters the cursor has been moved
        LEFTWARD from the end of the session buffer by previous navigate ops.
        0 = cursor is at end of all pasted text (initial state).
        N = cursor is N characters before the end.

        This allows chained navigate calls to work correctly even after the
        cursor has been moved by a previous whispermove/whisperselect.
        """
        import re as _re
        import pyautogui as _pag
        import ctypes as _ct
        import time as _time

        buf   = self._session_buffer
        clean = target_text.strip().rstrip(".,!?;:")
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

        m           = matches[-1]
        match_start = m.start()
        match_end   = m.end()
        buf_len     = len(buf)

        # ── Release held modifiers ──────────────────────────────────────────
        _VK_CONTROL = 0x11; _VK_SHIFT = 0x10; _VK_MENU = 0x12; _KEYUP = 0x0002
        for _vk in (_VK_CONTROL, _VK_SHIFT, _VK_MENU):
            _ct.windll.user32.keybd_event(_vk, 0, _KEYUP, 0)
        _time.sleep(0.06)

        # Current cursor absolute position in buffer (chars from buffer start)
        cur_abs = buf_len - self._cursor_offset   # where cursor is right now

        # Target positions
        before_abs = match_start   # just before the match
        after_abs  = match_end     # just after the match
        chars_in_match = match_end - match_start

        try:
            if operation in ("move", "movebefore"):
                delta = cur_abs - before_abs   # positive = move left
                if delta > 0:
                    self._navigate_press_n(_pag, "left", delta)
                elif delta < 0:
                    self._navigate_press_n(_pag, "right", -delta)
                self._cursor_offset = buf_len - before_abs
                self.scratchpad.append(
                    f"[Navigate] Cursor → before '{clean}' (moved {abs(delta)} {'◀' if delta>0 else '▶'})")

            elif operation == "moveafter":
                delta = cur_abs - after_abs
                if delta > 0:
                    self._navigate_press_n(_pag, "left", delta)
                elif delta < 0:
                    self._navigate_press_n(_pag, "right", -delta)
                self._cursor_offset = buf_len - after_abs
                self.scratchpad.append(
                    f"[Navigate] Cursor → after '{clean}' (moved {abs(delta)} {'◀' if delta>0 else '▶'})")

            elif operation == "select":
                # Move to just before match, then Shift+Right through it
                delta = cur_abs - before_abs
                if delta > 0:
                    self._navigate_press_n(_pag, "left", delta)
                elif delta < 0:
                    self._navigate_press_n(_pag, "right", -delta)
                # Now hold shift and press right for each char in the match
                _pag.keyDown("shift")
                try:
                    self._navigate_press_n(_pag, "right", chars_in_match)
                finally:
                    _pag.keyUp("shift")
                # After selection cursor is at match_end, offset unchanged from before
                self._cursor_offset = buf_len - after_abs
                self.scratchpad.append(
                    f"[Navigate] Selected '{clean}' "
                    f"(moved {abs(delta)} {'◀' if delta>0 else '▶'}, selected {chars_in_match} ▶)")

            # Arm lowercase for next transcription
            self._cursor_ops_pending = True
            app_logger.info(
                f"Navigate '{operation}' → '{clean}': cursor_offset now {self._cursor_offset}")

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

        # Sync editor voice-state indicator
        if self._editor and self._editor.isVisible():
            self._editor.set_voice_state(is_list)

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
            # ── Editor mode: if editor open, route dictation there ───────────
            # When WhisperEditor is visible, ALL live dictation goes into it.
            # Wizard/command triggers still fire first (so the user can say
            # "whisper replace" etc. to edit within the editor), but normal
            # text is appended to the editor instead of pasted externally.
            _cfg_ed = self.config.settings
            _ed_type_trigger = _cfg_ed.get("editor_type_trigger", "whisper type, whisper write")
            _ed_edit_trigger = _cfg_ed.get("editor_edit_trigger", "whisper edit, whisper edit this")
            _ed_paste_trigger = _cfg_ed.get("editor_paste_trigger", "whisper paste, whisper done, whisper okay")
            _ed_fuzz = float(_cfg_ed.get("fuzzy_threshold", 0.75))
            # "whisper paste" / "whisper done" — paste editor content to app
            if self._editor and self._editor.isVisible():
                if _editor_trigger_matches(text, _ed_paste_trigger):
                    self._editor._paste_to_app()
                    return
            # "whisper edit" / "whisper edit this" — open editor with clipboard
            # Checked BEFORE type trigger: "whisper edit" must not fall through
            # to "whisper type" via fuzzy matching.
            if _editor_trigger_matches(text, _ed_edit_trigger):
                self._open_editor(from_clipboard=True)
                return
            # "whisper type" / "whisper write" — open blank editor
            if _editor_trigger_matches(text, _ed_type_trigger):
                self._open_editor(prefill="")
                return

            # ── Terms pre-check: exact-phrase terms fire BEFORE wizard ──────
            # Terms with <KEY> tags (e.g. "Whisper undo" → "<CTRL+Z>") are
            # fully-specified user commands.  They must be matched and executed
            # here, BEFORE wizard scoring, so they are never mistaken for
            # move/select/replace triggers.  Only <KEY>-tag terms are checked
            # here; plain text substitutions still happen in the normal paste path.
            import re as _re_tp
            _terms_precheck_fired = False
            _cfg_terms = self.config.settings.get("terms", {})
            if _cfg_terms:
                _fuzz_tp = float(self.config.settings.get("fuzzy_threshold", 0.75))
                from difflib import SequenceMatcher as _SMtp
                for _tp_phrase, _tp_repl in _cfg_terms.items():
                    if not _tp_phrase.strip():
                        continue
                    if not ("<" in _tp_repl and ">" in _tp_repl):
                        continue  # only <KEY> terms are intercepted here
                    # Check all aliases (comma-separated)
                    _tp_matched = False
                    for _tp_alias in _phrase_aliases(_tp_phrase):
                        _tp_alias_l = _tp_alias.lower().strip()
                        _tp_text_l  = text.lower()
                        _tp_stripped = _re_tp.sub(r"[^\w\s]", "", _tp_text_l).strip()
                        # Exact match (with or without punctuation)
                        if _tp_alias_l in _tp_text_l or _tp_alias_l in _tp_stripped:
                            _tp_matched = True; break
                        # Fuzzy window match
                        if _fuzz_tp > 0:
                            _tp_words = _tp_stripped.split()
                            _tp_aw    = _tp_alias_l.split()
                            _tp_wn    = len(_tp_aw)
                            for _tp_i in range(max(1, len(_tp_words) - _tp_wn + 1)):
                                _tp_win = " ".join(_tp_words[_tp_i:_tp_i + _tp_wn])
                                if _SMtp(None, _tp_alias_l, _tp_win).ratio() >= _fuzz_tp:
                                    _tp_matched = True; break
                        if _tp_matched:
                            break
                    if _tp_matched:
                        app_logger.info(
                            f"[Terms] Pre-check matched '{_tp_phrase}' → sending key sequence")
                        self.scratchpad.append(f"[Term] {_tp_phrase} → {_tp_repl}")
                        try:
                            # Restore focus to last active window before sending keys
                            _hwnd_tp = getattr(self, "_pre_rec_hwnd", None) or getattr(self, "_last_paste_hwnd", None)
                            if _hwnd_tp:
                                try:
                                    import ctypes as _ct_tp
                                    _u32_tp = _ct_tp.windll.user32
                                    _fg_tp  = _u32_tp.GetForegroundWindow()
                                    _tFG_tp = _u32_tp.GetWindowThreadProcessId(_fg_tp, None)
                                    _tTG_tp = _u32_tp.GetWindowThreadProcessId(_hwnd_tp, None)
                                    _u32_tp.AttachThreadInput(_tFG_tp, _tTG_tp, True)
                                    _u32_tp.SetForegroundWindow(_hwnd_tp)
                                    _u32_tp.BringWindowToTop(_hwnd_tp)
                                    _u32_tp.AttachThreadInput(_tFG_tp, _tTG_tp, False)
                                    import time as _t_tp; _t_tp.sleep(0.1)
                                except Exception as _e_tp:
                                    app_logger.warning(f"[Terms] focus restore failed: {_e_tp}")
                            _send_keys_sequence(
                                _tp_repl,
                                paste_delay=self.config.settings.get("paste_delay", 0.5))
                            app_logger.info(f"[Terms] Key sequence sent: {_tp_repl}")
                        except Exception as _e_tpc:
                            app_logger.error(f"[Terms] send failed: {_e_tpc}", exc_info=True)
                        _terms_precheck_fired = True
                        break  # one term per transcription
            if _terms_precheck_fired:
                return

            # ── Wizard intercept: multi-step voice commands ──────────────────
            # If a wizard is active, route this transcription into it instead
            # of normal paste handling.  The wizard calls back into the app
            # (_whisper_navigate / _wizard_in_buffer_edit) when complete.
            if self._wizard is not None:
                # Single-word answers (e.g. "Batman") score low on logprob
                # because Whisper lacks surrounding context. If a wizard step
                # is active and the text is one word that isn't a hallucination,
                # accept it regardless of confidence — the user just said it.
                _wiz_words = text.split()
                _is_single = len(_wiz_words) == 1
                _hall_list_wiz = self.config.settings.get("hallucinations", HALLUCINATIONS)
                _is_hall = any(text.lower().strip() == h.lower() or
                               text.lower().strip().startswith(h.lower())
                               for h in _hall_list_wiz)
                if _is_single and _is_hall:
                    app_logger.debug(f"  wizard step: single-word hallucination ignored: {text!r}")
                else:
                    self._wizard_step(text)
                return

            # ── Command detection (runs BEFORE paste) ────────────────────────
            # ── WhisperNavigate: select / move cursor ──────────────────────
            # Check BEFORE command and paste handling — navigation is its own
            # operation; we do not paste the trigger phrase itself.
            _cfg      = self.config.settings
            _fuzz_thr = float(_cfg.get("fuzzy_threshold", 0.75))
            # Wizard trigger detection: best-match approach.
            # All triggers share the "whisper" prefix, so first-match-above-threshold
            # is unreliable — "whisper a place" (accent for "replace") would fire
            # "move" just because it appears first in the list and scores > 0.75.
            # Instead: require first spoken word ≈ "whisper", then find the
            # HIGHEST-scoring distinctive keyword match across all triggers.
            _trig_defs = [
                ("select",       _cfg.get("select_trigger",       "whisper select").lower()),
                ("move",         _cfg.get("move_trigger",         "whisper move").lower()),
                ("movebefore",   _cfg.get("movebefore_trigger",   "whisper before").lower()),
                ("moveafter",    _cfg.get("moveafter_trigger",    "whisper after").lower()),
                ("replace",      _cfg.get("replace_trigger",      "whisper replace").lower()),
                ("insertbefore", _cfg.get("insertbefore_trigger", "whisper insert before").lower()),
                ("insertafter",  _cfg.get("insertafter_trigger",  "whisper insert after").lower()),
            ]
            # Multi-strategy trigger detection.
            # Uses _score_trigger_match which combines word-level, char-concat,
            # sliding-window, and subsequence scoring — robust to the phonetic
            # approximations that arise with non-native accents.
            # Trigger threshold is lower than the general fuzzy threshold
            # (triggers are known short phrases; we should be aggressive).
            _trig_threshold = max(0.45, _fuzz_thr - 0.2)
            _trig_fired = False
            _best_score = 0.0
            _best_op    = None
            for _trig_op, _full_kw in _trig_defs:
                if not _full_kw:
                    continue
                # Score against the full trigger phrase (e.g. "whisper replace")
                _s = _score_trigger_match(text, _full_kw)
                if _s > _best_score:
                    _best_score = _s; _best_op = _trig_op
                # Also score against user-defined aliases (comma-separated)
                for _alias in _phrase_aliases(_full_kw):
                    _sa = _score_trigger_match(text, _alias)
                    if _sa > _best_score:
                        _best_score = _sa; _best_op = _trig_op
            if _best_op and _best_score >= _trig_threshold:
                # Guard 1: spoken text must be short (triggers are 2-4 words;
                # a full sentence like "when it connects to your camera …" cannot
                # be a trigger command regardless of substring score).
                _word_count = len(text.split())
                # Guard 2: first word must be phonetically close to "whisper"
                # (all triggers start with "whisper").  Use simple ratio check.
                from difflib import SequenceMatcher as _SM_g
                _first_word = text.split()[0].lower() if text.strip() else ""
                _first_ok = _SM_g(None, _first_word, "whisper").ratio() >= 0.55
                # Sanity: must score better than the runner-up by at least 0.04
                _runner_up = max(
                    (_score_trigger_match(text, kw) for op2, kw in _trig_defs
                     if op2 != _best_op and kw),
                    default=0.0)
                if _word_count <= 6 and _first_ok and _best_score - _runner_up >= 0.04:
                    app_logger.info(
                        f"Wizard trigger: op={_best_op!r} score={_best_score:.3f} "
                        f"runner_up={_runner_up:.3f}")
                    self.scratchpad.append(
                        f"[Wizard] Detected: {_best_op} (score {_best_score:.2f})")
                    self._wizard_start(_best_op)
                    _trig_fired = True
            if _trig_fired:
                return

            # If the transcribed text matches a voice command, execute it and
            # do NOT paste the text — the spoken phrase was a command, not prose.
            _cmd_fired = False
            app_logger.debug(f"  on_text: Checking {len(self.config.settings['commands'])} voice commands...")
            sk_trigger = self.config.settings.get("sendkeys_trigger", "whisper send keys").lower().strip()
            for phrase, cmd in self.config.settings["commands"].items():
                # phrase may be "Alias 1, Alias 2, Alias 3" — match any alias
                if _any_alias_matches(text, phrase, _fuzz_thr):
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
                # ── If editor is open, append text there instead of pasting ──
                if self._editor and self._editor.isVisible():
                    self._editor.append_text(text)
                    return

                # ── Confidence gate (paste path only) ────────────────────
                # Triggers / commands / wizard have already fired above, so
                # confidence filtering here only blocks actual pasting.
                if self.config.settings.get("use_confidence", False):
                    _mc  = float(self.config.settings.get("min_confidence", 0.0))
                    if _mc > 0.0:
                        _thr = -_mc
                        _seg_data = getattr(self.transcriber, "_last_seg_data", [])
                        if _seg_data:
                            _kept = [st for st, slp in _seg_data if slp >= _thr]
                            _dropped = [st for st, slp in _seg_data if slp < _thr]
                            for _ds in _dropped:
                                app_logger.debug(
                                    f"  paste confidence filtered: {_ds!r}")
                            text = " ".join(_kept).strip()
                            if not text:
                                app_logger.debug(
                                    "  all segments below confidence threshold — not pasting")
                                return

                # Apply term replacements (case-insensitive + fuzzy)
                import re as _re
                from difflib import SequenceMatcher as _SMt
                _key_tag_sequences = []  # term replacements that contain <KEY> tags
                for phrase, replacement in self.config.settings.get("terms", {}).items():
                    if not phrase.strip():
                        continue
                    # phrase col may be "alias1, alias2" — try each alias
                    _texact = False; _tfuzzy_match = None; _matched_alias = None
                    for _talias in _phrase_aliases(phrase):
                        _texact = bool(_re.search(_re.escape(_talias), text, _re.IGNORECASE))
                        if _texact:
                            _matched_alias = _talias; break
                        if _fuzz_thr > 0:
                            _tw = text.split(); _pw2 = _talias.split(); _wn2 = len(_pw2)
                            for _wi2 in range(max(1, len(_tw) - _wn2 + 1)):
                                _win2 = " ".join(_tw[_wi2:_wi2 + _wn2])
                                if _SMt(None, _talias.lower(), _win2.lower()).ratio() >= _fuzz_thr:
                                    _tfuzzy_match = _win2; _matched_alias = _talias; break
                        if _tfuzzy_match:
                            break
                    if _texact or _tfuzzy_match:
                        _pattern = (_re.escape(_matched_alias) if _texact
                                    else _re.escape(_tfuzzy_match))
                        if "<" in replacement and ">" in replacement:
                            # Contains special key tags — collect for sendkeys after paste
                            _key_tag_sequences.append(replacement)
                            text = _re.sub(_pattern, "", text, flags=_re.IGNORECASE).strip()
                        else:
                            text = _re.sub(_pattern, replacement,
                                           text, flags=_re.IGNORECASE)
                p_text = text + " " if auto_space else text
                # Strip trailing punctuation/spaces to find the last real word,
                # then record how many characters were actually output.
                self._last_paste_len = len(p_text)
                self._last_paste_text = p_text
                self._session_buffer += p_text  # accumulate for select/move
                self._cursor_offset   = 0       # fresh paste = cursor at new end
                self._session_paste_count += 1  # track for undo-based buffer replacement

                paste_delay = self.config.settings["paste_delay"]
                app_logger.debug(f"  on_text: auto_space={auto_space}, paste_delay={paste_delay}s, p_text length={len(p_text)}")
                try:
                    # Target window: use the hwnd captured at recording-start
                    # (GetForegroundWindow() here would return WhisperRApp itself)
                    import ctypes as _ct_bp
                    _u32_bp = _ct_bp.windll.user32
                    _k32_bp = _ct_bp.windll.kernel32
                    _paste_tgt = (getattr(self, "_pre_rec_hwnd", None) or
                                  getattr(self, "_last_paste_hwnd", None))
                    self._last_paste_hwnd = _paste_tgt

                    # Refocus target before paste
                    if _paste_tgt:
                        _fg_bp  = _u32_bp.GetForegroundWindow()
                        if _fg_bp != _paste_tgt:
                            try:
                                _tc_bp = _k32_bp.GetCurrentThreadId()
                                _tf_bp = _u32_bp.GetWindowThreadProcessId(_fg_bp,      None)
                                _ts_bp = _u32_bp.GetWindowThreadProcessId(_paste_tgt,  None)
                                _u32_bp.AttachThreadInput(_tc_bp, _tf_bp, True)
                                _u32_bp.AttachThreadInput(_tc_bp, _ts_bp, True)
                                _u32_bp.SetForegroundWindow(_paste_tgt)
                                _u32_bp.BringWindowToTop(_paste_tgt)
                                time.sleep(0.08)
                                _u32_bp.AttachThreadInput(_tc_bp, _tf_bp, False)
                                _u32_bp.AttachThreadInput(_tc_bp, _ts_bp, False)
                                app_logger.debug(
                                    f"  paste: refocused 0x{_paste_tgt:08X} "
                                    f"actual=0x{_u32_bp.GetForegroundWindow():08X}")
                            except Exception as _ef_bp:
                                app_logger.warning(f"  paste: refocus failed: {_ef_bp}")

                    pyperclip.copy(p_text)
                    time.sleep(paste_delay)
                    pyautogui.hotkey('ctrl', 'v')
                    app_logger.info(f"Text pasted: '{text[:30]}{'...' if len(text)>30 else ''}'")
                    # Fire any term sendkeys sequences now that the text is pasted
                    if _key_tag_sequences:
                        time.sleep(max(paste_delay, 0.1))
                        # Restore focus to the window that received the paste,
                        # otherwise keystrokes land in WhisperR itself.
                        try:
                            import ctypes as _ct_sk
                            _u32_sk = _ct_sk.windll.user32
                            _hwnd_sk = getattr(self, "_last_paste_hwnd", None)
                            if _hwnd_sk:
                                _fg_sk  = _u32_sk.GetForegroundWindow()
                                _tFG_sk = _u32_sk.GetWindowThreadProcessId(_fg_sk, None)
                                _tTG_sk = _u32_sk.GetWindowThreadProcessId(_hwnd_sk, None)
                                _u32_sk.AttachThreadInput(_tFG_sk, _tTG_sk, True)
                                _u32_sk.SetForegroundWindow(_hwnd_sk)
                                _u32_sk.BringWindowToTop(_hwnd_sk)
                                _u32_sk.AttachThreadInput(_tFG_sk, _tTG_sk, False)
                                time.sleep(0.1)
                        except Exception:
                            pass
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
            
            raw_editor_hk = self.config.settings.get("editor_hotkey", "ctrl+shift+e")
            editor_hotkey = self.normalize_hotkey(raw_editor_hk) if raw_editor_hk else None
            raw_editor_edit_hk = self.config.settings.get("editor_edit_hotkey", "")
            editor_edit_hotkey = self.normalize_hotkey(raw_editor_edit_hk) if raw_editor_edit_hk else None

            hotkey_map[toggle_hotkey]     = self.on_toggle_hotkey
            hotkey_map[visibility_hotkey] = self.on_visibility_hotkey
            if editor_hotkey:
                hotkey_map[editor_hotkey] = self.on_editor_hotkey
            if editor_edit_hotkey and editor_edit_hotkey != editor_hotkey:
                hotkey_map[editor_edit_hotkey] = self.on_editor_edit_hotkey
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
    
    def _quit_app(self):
        """Save persistent state then exit."""
        # Save editor content to disk if "remember" toggle is on
        try:
            _path = getattr(self, "_editor_persist_path", None)
            if _path:
                _content_to_save = ""
                # Prefer live editor content if open
                if self._editor:
                    _content_to_save = self._editor.editor.toPlainText()
                elif self._editor_remember and self._editor_saved_content:
                    _content_to_save = self._editor_saved_content
                if self._editor_remember and _content_to_save:
                    Path(_path).write_text(_content_to_save, encoding="utf-8")
                    app_logger.info(f"Editor content saved to {_path}")
                elif _path.exists() and not self._editor_remember:
                    Path(_path).unlink(missing_ok=True)
                # Also save full state JSON on exit
                _sp = getattr(self, "_editor_state_path", None)
                if _sp:
                    try:
                        import json as _json_quit
                        _target = 0
                        if self._editor:
                            _target = self._editor.target_spin.value()
                        elif hasattr(self, "_editor_saved_target"):
                            _target = self._editor_saved_target
                        _rem = self._editor_remember
                        if _rem:
                            _sp.write_text(_json_quit.dumps({
                                "remember": True,
                                "content":  _content_to_save,
                                "target_words": _target,
                                "clipboard_prefill": getattr(self, "_editor_clipboard_prefill", False),
                                "cb_monitor": getattr(self, "_editor_cb_monitor_was_on", False),
                            }, ensure_ascii=False, indent=2), encoding="utf-8")
                        elif _sp.exists():
                            _sp.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as _e:
            app_logger.warning(f"Could not save editor content: {_e}")
        QApplication.instance().quit()

    def on_toggle_hotkey(self):
        """Handler for toggle dictation hotkey - prevents subset conflicts.
        Captures foreground hwnd HERE (keyboard-thread) before Qt shifts focus.
        """
        app_logger.debug("Toggle dictation hotkey triggered (exact match)")
        try:
            import ctypes as _ct_hk
            _hwnd = _ct_hk.windll.user32.GetForegroundWindow()
            _own  = int(self.winId()) if hasattr(self, "winId") else 0
            if _hwnd and _hwnd != _own:
                self._pre_rec_hwnd = _hwnd
        except Exception:
            pass
        self.sig_toggle_rec.emit()
    
    def on_visibility_hotkey(self):
        """Handler for visibility hotkey - prevents subset conflicts"""
        app_logger.debug("Visibility hotkey triggered (exact match)")
        self.sig_toggle_vis.emit()

    def on_editor_hotkey(self):
        """Handler for editor toggle hotkey — runs on hotkey thread, emit to Qt."""
        app_logger.debug("Editor hotkey triggered")
        self.sig_toggle_editor.emit()

    def on_editor_edit_hotkey(self):
        """Open editor with current selection — identical to the voice path.

        Captures src_hwnd here (pynput thread, while source window still has
        focus), stores it in _pre_rec_hwnd, then emits sig_editor_edit which
        calls _open_editor(from_clipboard=True) on the Qt main thread.
        That method does the full refocus → Ctrl+C → poll sequence itself,
        exactly as it does for the voice command.  No separate Ctrl+C here.
        """
        app_logger.debug("Editor-edit hotkey triggered")
        try:
            import ctypes as _ct_eeh
            _hwnd = _ct_eeh.windll.user32.GetForegroundWindow()
            _own  = int(self.winId()) if hasattr(self, "winId") else 0
            _src  = _hwnd if (_hwnd and _hwnd != _own) else getattr(self, "_pre_rec_hwnd", 0)
            self._pre_rec_hwnd = _src
            app_logger.info(f"[EditThis] hotkey: src_hwnd=0x{_src:08X}")
        except Exception as _e:
            app_logger.warning(f"[EditThis] hotkey hwnd failed: {_e}")
        self.sig_editor_edit.emit()


    def toggle_editor_window(self):
        """Toggle the editor window: open if not open/visible, hide if visible.

        Respects the remember-content toggle inside the editor.
        When called by a voice trigger (open-new or open-edit), _open_editor
        is called instead of this method.
        """
        if self._editor and self._editor.isVisible():
            # Save content and full state if remember is on
            if getattr(self._editor, "remember_toggle", None) and \
                    self._editor.remember_toggle.isChecked():
                self._editor_saved_content = self._editor.editor.toPlainText()
                self._editor_remember = True
                self._editor_saved_target = self._editor.target_spin.value()
            else:
                self._editor_remember = False
                self._editor_saved_content = ""
                self._editor_saved_target = 0
            # Save monitor toggle state — monitor keeps running after hide
            self._editor_cb_monitor_was_on = (
                getattr(self._editor, "clipboard_monitor_toggle", None) and
                self._editor.clipboard_monitor_toggle.isChecked())
            # Persist to disk so state survives app restart too
            _sp = getattr(self, "_editor_state_path", None)
            if _sp and self._editor_remember:
                try:
                    import json as _json_tew
                    _sp.write_text(_json_tew.dumps({
                        "remember": True,
                        "content":  self._editor_saved_content,
                        "target_words": self._editor_saved_target,
                        "clipboard_prefill": getattr(self, "_editor_clipboard_prefill", False),
                        "cb_monitor": self._editor_cb_monitor_was_on,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            self._editor.hide()
            # Also hide cheatsheet when editor hides
            if getattr(self._editor, "_cheatsheet", None) and \
                    self._editor._cheatsheet.isVisible():
                self._editor._cheatsheet.hide()
        else:
            # Restore or open fresh
            prefill = self._editor_saved_content if self._editor_remember else ""
            self._open_editor(prefill=prefill)
            # Restore remember toggle state
            if self._editor and self._editor_remember:
                self._editor.remember_toggle.setChecked(True)
            # Re-show cheatsheet if it was open before
            if (self._editor and
                    getattr(self._editor, "_cheatsheet", None) and
                    getattr(self._editor, "btn_cheatsheet", None) and
                    self._editor.btn_cheatsheet.isChecked()):
                self._editor._cheatsheet.show()
                self._editor._reposition_cheatsheet()
    
    def normalize_hotkey(self, hotkey_str):
        """Convert our hotkey format to pynput format.

        Our format : "ctrl+shift+w"
        pynput format: "<ctrl>+<shift>+w"
        """
        parts = [p.strip().lower() for p in hotkey_str.lower().split('+')]
        normalized = []
        for part in parts:
            if part in ['ctrl', 'shift', 'alt', 'cmd', 'win']:
                normalized.append(f'<{part}>')
            elif part.startswith('f') and len(part) > 1 and part[1:].isdigit():
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
        self._quit_app()
    
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
