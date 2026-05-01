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


# ── Direct model downloader (bypasses huggingface_hub SSL issues) ──────────
#
# Per-model fallback file lists (used only when HF API is unreachable).
# large-v2/v3 do NOT have vocabulary.txt; smaller models DO.
_HF_FALLBACK_FILES = {
    "tiny":     (["config.json","model.bin","tokenizer.json","vocabulary.txt"],
                 ["tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
    "base":     (["config.json","model.bin","tokenizer.json","vocabulary.txt"],
                 ["tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
    "small":    (["config.json","model.bin","tokenizer.json","vocabulary.txt"],
                 ["tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
    "medium":   (["config.json","model.bin","tokenizer.json","vocabulary.txt"],
                 ["tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
    "large-v2": (["config.json","model.bin","tokenizer.json"],
                 ["vocabulary.txt","tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
    "large-v3": (["config.json","model.bin","tokenizer.json"],
                 ["vocabulary.txt","tokenizer_config.json","special_tokens_map.json","preprocessor_config.json"]),
}
_HF_CORE_REQUIRED = {"config.json", "model.bin", "tokenizer.json"}

# Harper-ls linter definitions: (internal_name, display_label, default, tooltip)
HARPER_LINTERS = [
    ("SpellCheck",                 "Spell Check",                  True,  "Detects misspelled words"),
    ("AnA",                        "A/An Usage",                   True,  "Detects wrong 'a' vs 'an' usage"),
    ("RepeatedWords",              "Repeated Words",               True,  "Detects repeated words (e.g. 'the the')"),
    ("Spaces",                     "Spacing Errors",               True,  "Detects double or missing spaces"),
    ("UnclosedQuotes",             "Unclosed Quotes",              True,  "Detects unclosed quote marks"),
    ("Matcher",                    "Common Errors",                True,  "Rule-based detection of common mistakes"),
    ("SentenceCapitalization",     "Capitalization",               True,  "Sentences must start with a capital letter"),
    ("LongSentences",              "Long Sentences",               True,  "Warns about overly long sentences"),
    ("MultipleSequentialPronouns", "Stacked Pronouns",             True,  "Detects stacked pronouns (e.g. 'I he went')"),
    ("Ellipsis",                   "Ellipsis Usage",               True,  "Proper ellipsis formatting"),
    ("Dashes",                     "Dash Spacing",                 True,  "Em-dash / en-dash spacing"),
    ("CompoundNouns",              "Compound Nouns",               True,  "Hyphenation of compound nouns"),
    ("PronounContractions",        "Pronoun Contractions",         True,  "Correct pronoun vs contraction usage"),
    ("WrongQuotes",                "Smart Quotes",                 False, "Converts straight quotes to typographic curly quotes"),
    ("SpelledNumbers",             "Spelled Numbers",              False, "Numbers should be spelled out"),
    ("CorrectNumberSuffix",        "Number Suffixes",              False, "Correct suffix usage (e.g. '1st' vs '1th')"),
    ("NumberSuffixCapitalization", "Ordinal Capitalization",       False, "Capitalization of ordinal suffixes"),
    ("LinkingVerbs",               "Linking Verbs",                False, "Checks linking verb usage"),
    ("AvoidCurses",                "Flag Profanity",               False, "Detects and flags curse words"),
    ("TerminatingConjunctions",    "Sentence-End Conjunctions",    False, "Conjunctions at the end of sentences"),
    ("OxfordComma",                "Oxford Comma",                 False, "Detects missing Oxford comma in lists"),
    ("BoringWords",                "Boring / Overused Words",      False, "Flags overused words (e.g. 'very', 'really')"),
]

def _harper_default_linters():
    """Return dict of linter → bool using HARPER_LINTERS defaults."""
    return {name: default for (name, _, default, _) in HARPER_LINTERS}


def _get_hf_file_list(model_name, ssl_ctx, log_fn):
    """Fetch the exact file list from the HF API. Returns list of filenames
    or None if unreachable."""
    import urllib.request as _ur, json as _j
    url = f"https://huggingface.co/api/models/Systran/faster-whisper-{model_name}"
    try:
        req = _ur.Request(url, headers={"User-Agent": "WhisperR/2.0"})
        with _ur.urlopen(req, timeout=8, context=ssl_ctx) as r:
            data = _j.loads(r.read())
            files = [s["rfilename"] for s in data.get("siblings", [])
                     if not s["rfilename"].startswith(".")]
            log_fn(f"  HF API: {len(files)} file(s) listed for {model_name}")
            return files   # everything listed is needed
    except Exception as e:
        log_fn(f"  HF API unreachable ({e}) — using built-in file list")
        return None


def _download_model_direct(model_name, hf_home, log_fn):
    """Download a faster-whisper model directly via urllib.

    - Queries HF API for the exact file list (no hardcoded guesses)
    - Falls back to a known-good per-model list if API is unreachable
    - Skips files that already exist and are non-zero (resume-friendly)
    - Retries each file once on transient failure
    - Uses atomic rename (file.part -> file) to avoid partial reads
    - Returns the local snapshot path on success, None on failure
    """
    import os, urllib.request as _ur, ssl

    repo_id  = f"Systran/faster-whisper-{model_name}"
    repo_dir = os.path.join(hf_home, f"models--Systran--faster-whisper-{model_name}")
    snap_dir = os.path.join(repo_dir, "snapshots", "main")
    os.makedirs(snap_dir, exist_ok=True)

    # Build SSL context
    ssl_ctx = ssl.create_default_context()
    _exe_dir = os.path.dirname(os.path.abspath(
        getattr(__import__("sys"), "executable", __file__)))
    _found_cert = False
    for _cert in [
        os.path.join(_exe_dir, "_internal", "certifi", "cacert.pem"),
        os.path.join(_exe_dir, "certifi", "cacert.pem"),
        os.path.join(_exe_dir, "_internal", "cacert.pem"),
    ]:
        if os.path.isfile(_cert):
            try:
                ssl_ctx.load_verify_locations(_cert)
                _found_cert = True
                log_fn(f"  SSL: cert bundle found")
                break
            except Exception:
                pass
    if not _found_cert:
        ssl_ctx = ssl._create_unverified_context()
        log_fn("  SSL: no cert bundle — using unverified context")

    # Test the SSL context with a cheap HEAD request.
    # If cert verification fails (e.g. VM clock skew), downgrade to
    # unverified rather than blocking all downloads.
    try:
        import urllib.request as _ur_test
        _tr = _ur_test.Request("https://huggingface.co",
                               headers={"User-Agent": "WhisperR/2.0"},
                               method="HEAD")
        _ur_test.urlopen(_tr, timeout=6, context=ssl_ctx)
    except Exception as _ssl_e:
        _ssl_es = str(_ssl_e)
        if "CERTIFICATE_VERIFY_FAILED" in _ssl_es or \
           "not yet valid" in _ssl_es or \
           "has expired" in _ssl_es:
            log_fn(f"  SSL cert check failed: {_ssl_e}")
            log_fn("  ⚠ This is usually caused by a wrong system clock.")
            log_fn("  Falling back to unverified SSL (safe for public model downloads).")
            ssl_ctx = ssl._create_unverified_context()
        # Other errors (e.g. network down) are handled per-file below

    # Get file list
    api_files = _get_hf_file_list(model_name, ssl_ctx, log_fn)
    if api_files is not None:
        # Everything the API lists is required; nothing is optional
        required_files = api_files
        optional_files = []
    else:
        required_files, optional_files = _HF_FALLBACK_FILES.get(
            model_name,
            (["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"], []))

    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    def _dl_file(fname, required):
        dest = os.path.join(snap_dir, fname)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            log_fn(f"  Skip (exists): {fname}")
            return True
        url  = f"{base_url}/{fname}"
        part = dest + ".part"
        for attempt in range(2):
            try:
                req = _ur.Request(url, headers={"User-Agent": "WhisperR/2.0"})
                with _ur.urlopen(req, timeout=300, context=ssl_ctx) as resp,                      open(part, "wb") as fout:
                    total    = int(resp.headers.get("Content-Length", 0))
                    done     = 0
                    last_pct = -10
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        fout.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done * 100 // total
                            if pct - last_pct >= 10:
                                last_pct = pct
                                log_fn(f"    {fname}: {pct}%"
                                       f"  ({done//(1<<20)}MB / {total//(1<<20)}MB)")
                os.replace(part, dest)
                sz = os.path.getsize(dest)
                log_fn(f"  ✓ {fname}"
                       f"  ({sz//(1<<20) if sz>=(1<<20) else sz//1024}"
                       f"{'MB' if sz>=(1<<20) else 'KB'})")
                return True
            except Exception as e:
                try: os.remove(part)
                except Exception: pass
                if attempt == 0:
                    log_fn(f"  ✗ {fname} attempt 1 failed: {e} — retrying...")
                else:
                    if required:
                        log_fn(f"  ✗ {fname} FAILED after 2 attempts: {e}")
                        return False
                    else:
                        log_fn(f"  ✗ {fname} skipped (optional): {e}")
                        return True   # optional failure is non-fatal
        return True

    all_files = [(f, True)  for f in required_files] +                 [(f, False) for f in optional_files]

    for fname, required in all_files:
        if not _dl_file(fname, required):
            log_fn(f"  Download aborted — {fname} is required but could not be downloaded")
            return None

    log_fn(f"  ✓ All files downloaded to {snap_dir}")
    return snap_dir


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

    # ── Pre-emptive cuDNN sub-library handling ──────────────────────────
    # PyInstaller's frozen import hook loads ALL collected DLLs for a package
    # the moment it is imported — before any Python code in that package runs.
    # cudnn_adv64_9.dll etc. are collected into _internal/ctranslate2/ but on
    # machines with no NVIDIA driver they fail to load, crashing the import.
    # Fix: try to pre-load them from the nvidia pip package (GPU path); if
    # that fails (no GPU), temporarily rename the bundle copies so the hook
    # skips them, then ctranslate2 falls back to CPU cleanly.
    _cudnn_sub_dlls = [
        "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_ops64_9.dll",
        "cudnn_engines_precompiled64_9.dll", "cudnn_engines_runtime_compiled64_9.dll",
        "cudnn_graph64_9.dll", "cudnn_heuristic64_9.dll",
    ]
    if os.name == "nt":
        import ctypes as _ct_pre
        _base_pre = os.path.dirname(sys.executable)
        _int_pre  = os.path.join(_base_pre, "_internal")
        _ct2_pre  = os.path.join(_int_pre, "ctranslate2")
        # Build search path: nvidia pip package bin dirs first
        _nv_bins = []
        for _pyroot in [os.path.dirname(os.path.dirname(sys.executable)),
                        r"C:\Python312", r"C:\Python311"]:
            _nv = os.path.join(_pyroot, "Lib", "site-packages", "nvidia")
            if os.path.isdir(_nv):
                for _pkg in os.listdir(_nv):
                    _bin = os.path.join(_nv, _pkg, "bin")
                    if os.path.isdir(_bin):
                        _nv_bins.append(_bin)
        _gpu_ok = False
        for _dll in _cudnn_sub_dlls:
            for _d in _nv_bins:
                _fp = os.path.join(_d, _dll)
                if os.path.isfile(_fp):
                    try:
                        _ct_pre.CDLL(_fp)
                        _gpu_ok = True
                    except Exception:
                        pass
                    break
        if not _gpu_ok:
            # No GPU driver found — rename bundle copies so PyInstaller hook skips them
            _renamed_pre = []
            for _dll in _cudnn_sub_dlls:
                _fp = os.path.join(_ct2_pre, _dll)
                if os.path.isfile(_fp):
                    try:
                        os.rename(_fp, _fp + ".disabled")
                        _renamed_pre.append(_dll)
                    except Exception:
                        pass
            if _renamed_pre:
                _log(f"  No GPU driver — disabled cuDNN sub-DLLs: {_renamed_pre}")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    _log(f"AI worker process started (pid={os.getpid()})")

    # Enable faulthandler only when WHISPERR_DEBUG env var is set.
    # Without the guard every run creates a file in TEMP — bad for SSDs.
    try:
        if os.environ.get("WHISPERR_DEBUG"):
            import faulthandler
            import tempfile
            _fh_path = os.path.join(tempfile.gettempdir(),
                                    f'whisperr_crash_{os.getpid()}.txt')
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
            # Add any extra DLL search paths here if needed
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
            # Only copy cuDNN sub-DLLs back if a GPU driver is present.
            # On CPU-only machines CUDA_VISIBLE_DEVICES=-1 was set by the
            # pre-emptive block; copying these DLLs back would re-enable
            # the broken import path that crashes ctranslate2.
            _gpu_present = os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1"
            _cuda_copied = []
            for _cdll in _cuda_copy_list:
                _csrc = os.path.join(_internal_d, _cdll)
                _cdst = os.path.join(_ct2_pkg_dir, _cdll)
                if _gpu_present and os.path.isfile(_csrc) and not os.path.isfile(_cdst):
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

        model_name, lang_code, compute_pref, audio_data, src, translate, use_vad, prompt, min_confidence, hotwords, vad_params = msg

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
                # ctranslate2 fails on machines with no NVIDIA GPU because
                # cudnn_adv64_9.dll loads but its sub-dependencies cannot be
                # satisfied without a working CUDA stack. Retry with CUDA
                # hidden — ctranslate2 skips all CUDA DLL loading and uses
                # its pure CPU backend instead.
                _log("Retrying import with CUDA disabled (CPU-only fallback)...")
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                    os.environ["CT2_CUDA_ALLOW_FP16"] = "0"
                    import sys as _sys_retry
                    for _mod in list(_sys_retry.modules.keys()):
                        if "ctranslate2" in _mod or "faster_whisper" in _mod:
                            del _sys_retry.modules[_mod]
                    import ctranslate2 as _ct2
                    _log(f"ctranslate2 {_ct2.__version__} imported OK (CPU-only)")
                    from faster_whisper import WhisperModel
                    _log("faster_whisper imported OK (CPU-only)")
                    compute_pref = "cpu"
                except Exception as e2:
                    _log(f"Import error (CPU retry): {type(e2).__name__}: {e2}")
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
                # Use HF_HOME env var (set by main process before worker start,
                # inheriting any custom cache path from Settings)
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
                        _snap_files  = os.listdir(_snap_path)
                        _has_weights = any(f.endswith(('.bin', '.ct2'))
                                           for f in _snap_files)
                        _has_config  = "config.json" in _snap_files
                        if _has_weights and _has_config:
                            # Model is cached — vocabulary.txt presence is
                            # model-dependent (large-v3 doesn't have it)
                            _model_path = _snap_path
                            _log(f"  Resolved: {_model_path} "
                                 f"({len(_snap_files)} files)")
                        else:
                            # No weights yet — partial or corrupt download
                            _log(f"  Snapshot incomplete ({_snap_files}) "
                                 f"— clearing for re-download")
                            try:
                                import shutil as _shu
                                _shu.rmtree(_repo_dir)
                                _log(f"  Cleared: {_repo_dir}")
                            except Exception as _rme:
                                _log(f"  Could not clear cache: {_rme}")
                            _model_path = model_name
                    else:
                        _log(f"  No snapshots — clearing empty repo dir")
                        try:
                            import shutil as _shu2
                            _shu2.rmtree(_repo_dir)
                            _log(f"  Cleared: {_repo_dir}")
                        except Exception as _rme2:
                            _log(f"  Could not clear cache: {_rme2}")
                        _model_path = model_name
                else:
                    _log(f"  Not cached: {_repo_dir} — download will be attempted")
                    # Don't force local_files_only — let WhisperModel download it
                    _model_path = model_name
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
                    _is_local_cuda = os.path.isdir(str(_model_path))
                    # If model not local, download it now before CUDA load
                    if not _is_local_cuda:
                        _log(f"  CUDA: model not local, downloading first")
                        # Use same TCP check as CPU path
                        _net_ok_cuda = False
                        try:
                            import socket as _sc2
                            _s2 = _sc2.socket(_sc2.AF_INET, _sc2.SOCK_STREAM)
                            _s2.settimeout(5); _s2.connect(("huggingface.co", 443)); _s2.close()
                            _net_ok_cuda = True
                        except Exception: pass
                        if _net_ok_cuda:
                            _dl_path_cu = _download_model_direct(model_name, _hf_home, _log)
                            if _dl_path_cu and os.path.isdir(_dl_path_cu):
                                _model_path  = _dl_path_cu
                                _is_local_cuda = True
                                _log(f"  CUDA download complete: {_dl_path_cu}")
                            else:
                                _log("  CUDA download failed — will skip CUDA")
                                cuda_device_count = 0  # force CPU fallback
                        else:
                            _log("  No network for CUDA download — falling back to CPU")
                            cuda_device_count = 0
                    _cuda_timeout = 30 if _is_local_cuda else 120
                    for ctype in ('float16', 'int8_float16'):
                        _log(f"  Trying CUDA {ctype} (timeout={_cuda_timeout}s)...")
                        _cuda_result = [None, None]  # [model, exception]
                        def _cuda_load_thread():
                            try:
                                _cuda_result[0] = WhisperModel(
                                    _model_path, device="cuda",
                                    compute_type=ctype,
                                    cpu_threads=4,
                                    num_workers=1,
                                    download_root=None,
                                    local_files_only=_is_local_cuda,
                                )
                            except Exception as _e_cl:
                                _cuda_result[1] = _e_cl
                        import threading as _thr_cuda
                        _ct = _thr_cuda.Thread(target=_cuda_load_thread, daemon=True)
                        _ct.start()
                        _ct.join(timeout=_cuda_timeout)
                        if _ct.is_alive():
                            _log(f"  CUDA {ctype} timed out after {_cuda_timeout}s")
                            # Report to main thread and give up on CUDA
                            if not _is_local_cuda:
                                _cache_dest_cu = os.path.join(
                                    _hf_home,
                                    f"models--Systran--faster-whisper-{model_name}")
                                _hf_url_cu = (f"https://huggingface.co/Systran/"
                                              f"faster-whisper-{model_name}/tree/main")
                                result_q.put(("model_not_found",
                                    model_name, _cache_dest_cu, _hf_url_cu))
                                result_q.put(('status', False))
                                loaded = True   # prevent CPU retry
                            break
                        elif _cuda_result[1] is not None:
                            _log(f"  CUDA {ctype} failed: "
                                 f"{type(_cuda_result[1]).__name__}: {_cuda_result[1]}")
                        else:
                            model = _cuda_result[0]
                            current_model_name = model_name
                            current_language = lang_code
                            _log(f"✓ {model_name} loaded on GPU ({ctype})")
                            loaded = True
                            break
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

                # _local_only: True when we resolved a real snapshot path
                _local_only = os.path.isdir(str(_model_path))
                # Also check if the repo dir exists at all (snapshot resolve
                # can fail while the model IS locally cached)
                _repo_exists = os.path.isdir(
                    os.path.join(_hf_home,
                                 f"models--Systran--faster-whisper-{model_name}"))
                _any_local = _local_only or _repo_exists
                _net_ok = True   # assume OK; overwritten below if download needed
                if not _any_local:
                    # Model not cached at all — must download. Check connectivity
                    # using a plain TCP connect (not HTTPS) so frozen-app SSL
                    # issues don't give a false "no network" result.
                    _net_ok = False
                    try:
                        import socket as _sock_nc
                        _s = _sock_nc.socket(_sock_nc.AF_INET, _sock_nc.SOCK_STREAM)
                        _s.settimeout(5)
                        _s.connect(("huggingface.co", 443))
                        _s.close()
                        _net_ok = True
                        _log("  Network TCP check OK (port 443)")
                    except Exception as _ne:
                        _log(f"  Network TCP check failed: {_ne}")
                    if _net_ok:
                        # Download files directly — huggingface_hub SSL hangs
                        # in frozen apps because cacert.pem isn't bundled.
                        _log(f"  Network OK — downloading {model_name} directly")
                        _dl_path = _download_model_direct(model_name, _hf_home, _log)
                        if _dl_path and os.path.isdir(_dl_path):
                            _model_path = _dl_path
                            _local_only = True
                            _log(f"  Download complete: {_dl_path}")
                        else:
                            _log("  Direct download failed — trying HF hub as fallback")
                            os.environ.pop("HF_HUB_OFFLINE", None)
                            os.environ.pop("TRANSFORMERS_OFFLINE", None)
                            os.environ.pop("HF_DATASETS_OFFLINE", None)
                    else:
                        _log(f"  No network — {model_name} not cached, cannot load")
                        _cache_dest = os.path.join(
                            _hf_home,
                            f"models--Systran--faster-whisper-{model_name}")
                        _hf_url = (f"https://huggingface.co/Systran/"
                                   f"faster-whisper-{model_name}/tree/main")
                        result_q.put(("model_not_found",
                            model_name, _cache_dest, _hf_url))
                        result_q.put(('status', False))
                        continue
                elif _repo_exists and not _local_only:
                    # Repo dir exists but snapshot resolve failed — try anyway
                    # with local_files_only so we don't attempt a download
                    _local_only = True
                    os.environ.pop("HF_HUB_OFFLINE", None)
                    os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                    _log(f"  Repo dir exists but snapshot resolve failed "
                         f"— will try loading with model name directly")
                    _model_path = model_name   # let WhisperModel find it
                for ctype in ('float32', 'int8'):
                    _log(f"  Trying CPU {ctype} (local_only={_local_only})...")
                    # Load in a thread with a hard timeout so a stalled
                    # download never blocks the worker indefinitely.
                    _load_result = [None, None]  # [model_or_None, exception_or_None]
                    def _load_thread():
                        try:
                            _load_result[0] = WhisperModel(
                                _model_path, device="cpu",
                                compute_type=ctype,
                                cpu_threads=4,
                                num_workers=1,
                                download_root=None,
                                local_files_only=_local_only,
                            )
                        except Exception as _e_lt:
                            _load_result[1] = _e_lt
                    import threading as _thr_load
                    _t = _thr_load.Thread(target=_load_thread, daemon=True)
                    _t.start()
                    # Timeout: 30 s for local load, 120 s if downloading
                    _timeout = 120 if _net_ok and not _local_only else 30
                    _t.join(timeout=_timeout)
                    if _t.is_alive():
                        _log(f"  CPU {ctype} timed out after {_timeout}s — aborting")
                        result_q.put(("error",
                            f"Loading '{model_name}' timed out after {_timeout}s.\n\n"
                            f"The download may be stalled. Options:\n"
                            f"• Check your internet connection\n"
                            f"• Copy the model folder to the local HuggingFace cache\n"
                            f"• Switch to a smaller cached model"
                        ))
                        result_q.put(('status', False))
                        loaded = True   # prevent further ctype attempts
                        break
                    elif _load_result[1] is not None:
                        _log(f"  CPU {ctype} failed: {type(_load_result[1]).__name__}: {_load_result[1]}")
                    else:
                        model = _load_result[0]
                        current_model_name = model_name
                        current_language = lang_code
                        _log(f"✓ {model_name} loaded on CPU ({ctype})")
                        loaded = True
                        break

            if not loaded:
                _log(f"All load attempts failed for {model_name}")
                # If no local cache either, emit richer error
                _is_cached = os.path.isdir(str(_model_path)) if "_model_path" in dir() else False
                if not _is_cached:
                    _hf_home_fb = os.path.join(
                        os.path.expanduser("~"), ".cache", "huggingface", "hub")
                    _cache_dest_fb = os.path.join(
                        _hf_home_fb,
                        f"models--Systran--faster-whisper-{model_name}")
                    _hf_url_fb = (f"https://huggingface.co/Systran/"
                                  f"faster-whisper-{model_name}/tree/main")
                    result_q.put(("model_not_found",
                        model_name, _cache_dest_fb, _hf_url_fb))
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
                vad_parameters=vad_params if (use_vad and vad_params) else None,
                initial_prompt=prompt or None,
                # faster-whisper expects hotwords as a single string, not a list
                hotwords=(" ".join(hotwords) if hotwords else None),
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
__version__ = "2.1.0"
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
    QDoubleSpinBox, QProgressBar, QFormLayout, QLineEdit, QGroupBox, QSpinBox, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea, QDialog, QMessageBox,
    QSystemTrayIcon, QMenu, QSlider, QListWidget, QListWidgetItem, QRadioButton, QAbstractItemView, QSplitter,
    QFrame, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect, QPoint, QObject, QEvent
from PyQt6.QtGui import (QPainter, QColor, QFont, QIcon, QAction, QKeyEvent, QPixmap, QPen,
                         QSyntaxHighlighter, QTextCharFormat, QKeySequence, QShortcut)

# --- 3. CONSTANTS ---
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

EDITOR_PRESETS = {
    "Interview":       "# Interview\n\n**Date:** \n**Interviewer:** \n**Subject:** \n\n---\n\n**Q:** \n\n**A:** \n\n",
    "Meeting Notes":   "# Meeting Notes\n\n**Date:** \n**Attendees:** \n**Agenda:** \n\n---\n\n## Discussion\n\n\n\n## Action Items\n\n- [ ] \n\n## Next Meeting\n\n",
    "Lecture / Talk":  "# Lecture: \n\n**Speaker:** \n**Date:** \n**Source:** \n\n---\n\n## Key Points\n\n\n\n## Notes\n\n\n\n## References\n\n",
    "Research Notes":  "# Research: \n\n**Topic:** \n**Date:** \n\n---\n\n## Summary\n\n\n\n## Sources\n\n\n\n## Questions\n\n",
    "Draft / Freewrite": "",
}
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
QMainWindow, QDialog, QScrollArea, QTabWidget, QTabBar, QStackedWidget { background-color: #121212; }
QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; font-size: 9pt; }
QWidget > QMenu, QMenu { background-color: #1e1e1e; color: #ddd; }
QFrame { background-color: #121212; }
QScrollArea > QWidget > QWidget { background-color: #121212; }
QTabBar::tab { background-color: #1e1e1e; color: #ccc; padding: 6px 14px; border: 1px solid #333; border-bottom: none; border-radius: 3px 3px 0 0; }
QTabBar::tab:selected { background-color: #0078d7; color: #fff; }
QTabBar::tab:hover { background-color: #2a2a2a; }
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
QMenu { background-color: #1e1e1e; color: #ddd; border: 1px solid #444; }
QMenu::item { padding: 4px 20px; }
QMenu::item:selected { background-color: #1a3a5c; }
QMenu::item:hover { background-color: #1a3a5c; }
QMenu::item:pressed { background-color: #0d2a4a; }
QMenu::separator { height: 1px; background: #333; margin: 2px 0; }
QMenu::item:disabled { color: #666; }
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

# Harper-specific debug logger — always writes regardless of app log_level.
_harper_log = logging.getLogger("harper_debug")
_harper_log.setLevel(logging.DEBUG)
_harper_path = os.path.join(BASE_DIR, "harper_debug.txt")
try:
    _hf = logging.FileHandler(_harper_path, mode='w', encoding='utf-8')
    _hf.setLevel(logging.DEBUG)
    _hf.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    _harper_log.addHandler(_hf)
except Exception:
    pass

def harper_log(msg):
    _harper_log.info(msg)

# --- 5. CONFIGURATION ---
class AppConfig:
    def __init__(self):
        self.path = os.path.join(BASE_DIR, "config.json")
        self.settings = {
            "model": "tiny", "lang_name": "English", "lang_code": "en",
            "hf_cache_path": "",   # empty = use default HF cache location
            "translate": False, "timestamps": False,
            "initial_prompt": "General professional writing. The speaker may use technical terminology across a variety of fields. Software names: Microsoft Word, Excel, PowerPoint, Google Docs, Sheets, Slides, VS Code, GitHub, ChatGPT, WhisperR. Technology terms: API, JSON, XML, HTML, CSS, JavaScript, Python, SQL, GPU, CPU, RAM, SSD, HDD, USB, Wi-Fi, Bluetooth, HTTPS. Business terms: KPI, ROI, B2B, B2C, SaaS, MVP, NDA, CRM. Measurements and abbreviations: GB, TB, MHz, GHz, ms, fps, dpi, OK, AI, ML, AR, VR, UI, UX. When the speaker says 'okay' in a sentence, transcribe it as 'OK'.",
            "hotwords": [],
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
            "editor_hotkey": "ctrl+shift+alt+a",
            "editor_edit_hotkey": "ctrl+shift+x",
            "rollback_hotkey": "ctrl+shift+z",
            "always_on_top": True,
            "aot_main": True, "aot_editor": True,
            "aot_notes": True, "aot_cheatsheet": True,
            "auto_backup_enabled": False,
            "auto_backup_interval": 10,
            "auto_backup_keep": 5,
            "cb_source_tag": True,
            "version_history_keep": 20,
            "version_history_infinite": False,
            # App-state snapshots
            "snapshots_enabled": False,
            "snapshots_mode": "count",  # "count" or "duration"
            "snapshots_keep_count": 60,   # snapshots to keep
            "snapshots_keep_hours": 24,   # hours of history to keep
            "live_mode": "Auto-Pause",
            "dict_mode": "Auto-Pause", "auto_pause_sec": 2.0,
            "noise_floor": 50, "speech_vol": 500,
            "commands": {"Launch Notepad": "notepad.exe"},
            "terms": {"whisper ar": "WhisperR", "youre": "you're", "dont": "don't", "cant": "can't", "wont": "won't", "its a": "it's a"},
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
            "vad_threshold": 0.5,
            "hotkey_cooldown_ms": 400,
            "manual_sentence_split": False,
            "mss_break_key": "shift",
            "harper": {"installed": False, "version": None},
            "vad_min_silence_ms": 2000,
            "vad_min_speech_ms": 250,
            "ft_output_folder": str(Path.home() / "WhisperR_Output"),
            "ft_mon_folder": str(Path.home() / "WhisperR_Watch"),
            "ft_mon_enabled": False,
            "use_confidence": True, "min_confidence": 0.5,
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
            "editor_hk_tagwrap": "Ctrl+Shift+W",
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
                # ── Config migrations ──────────────────────────────────
                # v2.1.0: editor_hk_kbd Ctrl+Shift+K → Ctrl+Shift+D
                if self.settings.get("editor_hk_kbd","") == "Ctrl+Shift+K":
                    self.settings["editor_hk_kbd"] = "Ctrl+Shift+D"
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
    finished_text       = pyqtSignal(str, str)
    status_changed      = pyqtSignal(bool)
    log_msg             = pyqtSignal(str)
    model_not_found_sig = pyqtSignal(str, str, str)  # model_name, dest_path, hf_url

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

    def preload_model(self, model_name: str = ""):
        """Ask the worker to pre-warm the model (runs before first recording).
        Pass model_name to override the saved config (e.g. on dropdown change).
        """
        cfg = self.config.settings
        name    = model_name or cfg['model']
        compute = 'cpu' if self._cuda_failed else cfg.get('compute_pref', 'auto')
        task = (
            name, cfg['lang_code'], compute,
            None, None,           # audio_data=None, src=None → preload sentinel
            False, False, '',     # translate, use_vad, prompt
            0.0,                  # min_confidence (unused on preload)
            [],                   # hotwords
            {},                   # vad_params
        )
        app_logger.info(f"TranscriberWorker.preload_model: queuing preload for model={name} compute={compute}")
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
            cfg.get('hotwords', []),
            {
                "threshold": float(cfg.get("vad_threshold", 0.5)),
                "min_silence_duration_ms": int(cfg.get("vad_min_silence_ms", 2000)),
                "min_speech_duration_ms":  int(cfg.get("vad_min_speech_ms", 250)),
            },
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

            # Propagate custom HF cache path to worker via environment
            _hf_custom = getattr(self.config, "settings", {}).get("hf_cache_path", "")
            if _hf_custom and os.path.isdir(_hf_custom):
                os.environ["HF_HOME"] = _hf_custom
                os.environ["HUGGINGFACE_HUB_CACHE"] = _hf_custom
                app_logger.info(f"Custom HF cache: {_hf_custom}")
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

                if msg[0] == "model_not_found":
                    _, _mn, _dest, _url = msg
                    app_logger.warning(f"Emitting model_not_found for {_mn}")
                    self.model_not_found_sig.emit(_mn, _dest, _url)
                    continue
                elif msg[0] == 'status':
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
            # PTT gate — only active when PTT has been pressed this session
            # (_ptt_was_pressed tracks whether PTT was ever engaged).
            # During normal toggle-mode dictation, ptt_pressed stays False
            # and _ptt_was_pressed stays False, so this block is skipped.
            if self.ptt_pressed or _ptt_was_pressed:
                # Detect release edge: was held, now released → dispatch immediately
                if _ptt_was_pressed and not self.ptt_pressed:
                    _ptt_was_pressed = False
                    if len(frames) > 5:
                        app_logger.debug(
                            f"PTT released — dispatching {len(frames)} frames")
                        self.speech_active.emit(False)
                        self.dispatch(frames, FIXED_RATE, FIXED_CHANNELS)
                        frames = []
                        last_speech = time.time()
                elif self.ptt_pressed:
                    _ptt_was_pressed = True
                if not self.ptt_pressed:
                    # PTT not held — idle-sleep until pressed again
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

# ── Harper LSP Integration ────────────────────────────────────────────────────
#
# harper-ls is a standalone binary Language Server that provides real-time
# spell and grammar checking via the Language Server Protocol (LSP) over stdio.
#
# Architecture:
#   • HarperLSPClient — manages one harper-ls subprocess per editor document.
#     Speaks JSON-RPC over stdin/stdout. Sends initialize → didOpen → didChange
#     on every (debounced) text change. Receives publishDiagnostics and converts
#     them to (line, col, end_line, end_col, message, suggestions) tuples.
#   • _MdHighlighter is extended to apply SpellCheckUnderline on those ranges.
#   • Right-click context menu offers quick fixes.
#   • Status indicator in editor toolbar reflects harper state.

import os, sys, json, threading, subprocess, platform

def _harper_binary_path():
    """Return the path to harper-ls executable, or None if not found."""
    exe = "harper-ls.exe" if sys.platform == "win32" else "harper-ls"
    # 1. Same folder as the app/exe
    app_dir = (os.path.dirname(sys.executable)
               if getattr(sys, "frozen", False)
               else os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(app_dir, exe)
    app_logger.debug(
        f"_harper_binary_path: checking {candidate!r} "
        f"exists={os.path.isfile(candidate)}")
    if os.path.isfile(candidate):
        return candidate
    # 2. System PATH
    import shutil
    found = shutil.which("harper-ls")
    app_logger.debug(f"_harper_binary_path: PATH lookup → {found!r}")
    return found  # may be None


def _harper_download(progress_cb=None, done_cb=None):
    """Download the correct harper-ls binary from GitHub in a background thread.

    progress_cb(msg: str) — called with status strings during download.
    done_cb(success: bool, msg: str) — called when finished.
    """
    def _worker():
        try:
            import urllib.request as _ur, zipfile, tarfile, tempfile, shutil

            # Determine the correct asset name
            machine = platform.machine().lower()
            plat    = sys.platform
            app_logger.info(f"Harper download: platform={plat}, machine={machine}")
            if plat == "win32":
                asset_name = "harper-ls-x86_64-pc-windows-msvc.zip"
            elif plat == "darwin":
                if "arm" in machine or "aarch64" in machine:
                    asset_name = "harper-ls-aarch64-apple-darwin.tar.gz"
                else:
                    asset_name = "harper-ls-x86_64-apple-darwin.tar.gz"
            else:  # Linux
                if "aarch64" in machine or "arm64" in machine:
                    asset_name = "harper-ls-aarch64-unknown-linux-gnu.tar.gz"
                else:
                    asset_name = "harper-ls-x86_64-unknown-linux-gnu.tar.gz"

            if progress_cb:
                progress_cb("Checking GitHub for the latest Harper release…")

            api_url = "https://api.github.com/repos/Automattic/harper/releases/latest"
            req = _ur.Request(api_url, headers={"User-Agent": "WhisperR"})
            with _ur.urlopen(req, timeout=15) as resp:
                release = json.loads(resp.read())

            version = release.get("tag_name", "unknown")
            assets  = release.get("assets", [])
            url     = next(
                (a["browser_download_url"] for a in assets
                 if a["name"] == asset_name), None)
            app_logger.info(f"Harper download: version={version}, asset={asset_name}, url_found={url is not None}")

            if not url:
                available = [a["name"] for a in assets]
                if done_cb:
                    done_cb(False,
                        f"Could not find the right download for your system.\n\n"
                        f"Expected: {asset_name}\n"
                        f"Available:\n" + "\n".join(f"  • {n}" for n in available[:8]) +
                        "\n\nYou can download manually from:\n"
                        "https://github.com/Automattic/harper/releases/latest")
                return

            if progress_cb:
                progress_cb(f"Downloading {asset_name} ({version})…")

            exe_name = "harper-ls.exe" if plat == "win32" else "harper-ls"
            app_dir  = (os.path.dirname(sys.executable)
                        if getattr(sys, "frozen", False)
                        else os.path.dirname(os.path.abspath(__file__)))
            dest     = os.path.join(app_dir, exe_name)
            app_logger.info(f"Harper download: dest={dest}")

            # Use mkdtemp() instead of TemporaryDirectory — on Windows,
            # TemporaryDirectory registers a weakref finaliser that Python
            # calls at interpreter shutdown. By that point AV software may
            # still hold the zip open, causing WinError 32. With mkdtemp we
            # delete manually with ignore_errors=True and are done.
            tmp = tempfile.mkdtemp(prefix="whisperr_harper_")
            src = None
            try:
                archive_path = os.path.join(tmp, asset_name)
                # Stream download with progress so UI stays responsive
                req2 = _ur.Request(url, headers={"User-Agent": "WhisperR"})
                with _ur.urlopen(req2, timeout=60) as _resp:
                    total = int(_resp.headers.get("Content-Length", 0))
                    downloaded = 0
                    chunk_size = 65536  # 64 KB
                    with open(archive_path, "wb") as _fout:
                        while True:
                            chunk = _resp.read(chunk_size)
                            if not chunk:
                                break
                            _fout.write(chunk)
                            downloaded += len(chunk)
                            if progress_cb and total:
                                pct = int(downloaded * 100 / total)
                                mb  = downloaded / 1048576
                                progress_cb(
                                    f"Downloading… {mb:.1f} MB ({pct}%)")
                app_logger.info(
                    f"Harper download: archive saved ({downloaded} bytes)")

                if progress_cb:
                    progress_cb("Extracting…")

                if asset_name.endswith(".zip"):
                    zf = zipfile.ZipFile(archive_path)
                    try:
                        members = zf.namelist()
                        target  = next(
                            (m for m in members
                             if m.endswith(exe_name) or m == exe_name), None)
                        if not target:
                            if done_cb:
                                done_cb(False,
                                    f"Could not find {exe_name} inside the archive.\n"
                                    f"Archive contents: {members}")
                            return
                        src = zf.extract(target, tmp)
                    finally:
                        zf.close()  # release handle before any rmtree attempt
                else:
                    tf = tarfile.open(archive_path)
                    try:
                        members = tf.getnames()
                        target  = next(
                            (m for m in members
                             if m.endswith(exe_name) or m == exe_name), None)
                        if not target:
                            if done_cb:
                                done_cb(False,
                                    f"Could not find {exe_name} inside the archive.\n"
                                    f"Archive contents: {members}")
                            return
                        member = tf.getmember(target)
                        tf.extract(member, tmp)
                        src = os.path.join(tmp, target)
                    finally:
                        tf.close()  # release handle before any rmtree attempt

                if src:
                    shutil.copy2(src, dest)
                    if plat != "win32":
                        os.chmod(dest, 0o755)
            finally:
                # Best-effort cleanup — AV scanners on Windows may keep the
                # zip locked briefly; ignore_errors means we never crash here.
                shutil.rmtree(tmp, ignore_errors=True)

            if done_cb:
                app_logger.info(f"Harper download: SUCCESS, binary at {dest}")
                done_cb(True, version)

        except OSError as e:
            app_logger.error(f"Harper download OSError: {e}")
            if done_cb:
                done_cb(False,
                    f"Could not write to the app folder.\n\n"
                    f"Error: {e}\n\n"
                    f"Try running WhisperR as Administrator, or download harper-ls.exe\n"
                    f"manually and place it in the same folder as WhisperR.exe.")
        except Exception as e:
            app_logger.error(f"Harper download Exception: {e}", exc_info=True)
            if done_cb:
                done_cb(False,
                    f"Download failed: {e}\n\n"
                    f"Check your internet connection, or download manually from:\n"
                    f"https://github.com/Automattic/harper/releases/latest")

    threading.Thread(target=_worker, daemon=True).start()


class _DiagBridge(QObject):
    """Signal bridge to deliver LSP diagnostics from reader thread to main thread."""
    ready = pyqtSignal(list)


class HarperLSPClient:
    """Manages one harper-ls subprocess and speaks LSP JSON-RPC over stdio.

    Usage:
        client = HarperLSPClient(on_diagnostics=callback)
        client.start()
        client.open_document(uri, text)
        client.change_document(uri, text)   # debounced internally
        client.stop()

    on_diagnostics(diags) is called on the main thread with a list of:
        {"range": {"start": {"line":N,"character":N},
                   "end":   {"line":N,"character":N}},
         "message": str,
         "suggestions": [str, ...]}
    """

    DEBOUNCE_MS = 300

    def __init__(self, on_diagnostics=None, binary_path=None, linters_callback=None):
        self._on_diagnostics = on_diagnostics
        self._binary   = binary_path or _harper_binary_path()
        self._proc     = None
        self._msg_id   = 0
        self._reader   = None
        self._lock     = threading.Lock()
        self._uri      = None
        self._version  = 0
        self._debounce  = None   # threading.Timer
        self._stopping  = False  # set True on explicit stop() to suppress restart
        self._init_event = threading.Event()  # set when harper-ls init succeeds
        self._last_text = ""   # cached doc text for keepalive / auto-restart
        self._linters_callback = linters_callback  # callable() -> dict{name: bool}
        # Qt signal bridge — emits from reader thread, fires callback on main thread
        try:
            self._bridge = _DiagBridge()
            self._bridge.ready.connect(
                self._on_diagnostics, type=Qt.ConnectionType.QueuedConnection)
            harper_log("BRIDGE: created")
        except Exception as _be:
            harper_log(f"BRIDGE: creation failed: {_be}")
            self._bridge = None
        # Response tracking for sync LSP requests (codeAction, etc.)
        self._pending_responses = {}  # id -> threading.Event
        self._response_data = {}       # id -> result dict

    def _get_linters_dict(self):
        """Return current linter config from callback, or defaults."""
        if self._linters_callback:
            try:
                cfg = self._linters_callback()
                if isinstance(cfg, dict):
                    return cfg
            except Exception:
                pass
        return _harper_default_linters()

    def refresh_linter_config(self):
        """Push updated linter config to harper-ls at runtime (no restart needed)."""
        if not self.running():
            return
        linters = self._get_linters_dict()
        settings = {
            "harper-ls": {
                "diagnosticSeverity": "hint",
                "linters": linters,
            }
        }
        harper_log(f"refresh_linter_config: pushing {linters}")
        self._send({
            "jsonrpc": "2.0",
            "method":  "workspace/didChangeConfiguration",
            "params":  {"settings": settings}
        })
        # Also trigger re-check by sending a new document change
        if self._uri and self._last_text:
            self._send_change(self._uri, self._last_text)

    def _auto_restart(self):
        """Called on main thread when harper-ls exits unexpectedly. Restart it."""
        if self._stopping:
            return
        app_logger.info("HarperLSPClient: auto-restarting harper-ls")
        # Kill old process if somehow still alive
        if self._proc and self._proc.poll() is None:
            try: self._proc.terminate()
            except Exception: pass
        self._proc = None
        self._diag_params_logged = False  # reset so new session logs first diag
        if self.start():
            uri  = self._uri
            text = getattr(self, "_last_text", "")
            if uri:
                self.open_document(uri, text)
                app_logger.info("HarperLSPClient: auto-restart successful")
        else:
            app_logger.warning("HarperLSPClient: auto-restart failed")

    def available(self):
        return bool(self._binary and os.path.isfile(self._binary))

    def running(self):
        return self._proc is not None and self._proc.poll() is None

    def start(self):
        if not self.available():
            harper_log(f"START: binary not found at {self._binary!r}")
            app_logger.warning(
                f"HarperLSPClient.start: binary not found at {self._binary!r}")
            return False
        if self.running():
            harper_log("START: already running")
            app_logger.debug("HarperLSPClient.start: already running")
            return True
        try:
            self._init_event.clear()
            harper_log(f"START: launching {self._binary}")
            app_logger.info(f"HarperLSPClient.start: launching {self._binary}")
            # CREATE_NO_WINDOW prevents a console flash on Windows
            _cflags = 0
            if sys.platform == "win32":
                _cflags = subprocess.CREATE_NO_WINDOW
            self._proc = subprocess.Popen(
                [self._binary, "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                creationflags=_cflags)
            harper_log(f"START: pid={self._proc.pid}")
            self._reader = threading.Thread(
                target=self._read_loop, daemon=True)
            self._reader.start()
            # Also drain stderr so it appears in the app log
            self._stderr_reader = threading.Thread(
                target=self._read_stderr_loop, daemon=True)
            self._stderr_reader.start()
            self._send_initialize()
            # Wait for harper-ls to respond to initialize (max 10s)
            if not self._init_event.wait(timeout=10.0):
                harper_log("START: initialization timed out")
                app_logger.warning(
                    "HarperLSPClient.start: initialization timed out")
            else:
                harper_log("START: initialization confirmed")
                app_logger.info(
                    "HarperLSPClient.start: initialization confirmed")
            return self.running()
        except Exception as _e:
            harper_log(f"START: failed: {_e}")
            app_logger.error(f"HarperLSPClient.start failed: {_e}", exc_info=True)
            self._proc = None
            return False

    def stop(self):
        self._stopping = True
        # Clean up the on-disk document file
        try:
            _uri = getattr(self, "_uri", None)
            if _uri and _uri.startswith("file:///"):
                _p = _uri[len("file:///"):].replace("/", os.sep)
                if os.path.exists(_p):
                    os.remove(_p)
        except Exception:
            pass
        if self._debounce:
            self._debounce.cancel()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def open_document(self, uri, text):
        self._uri     = uri
        self._version = 1
        harper_log(f"OPEN: uri={uri!r} text_len={len(text)}")
        app_logger.info(f"HarperLSPClient.open_document: uri={uri!r} text_len={len(text)}")
        # Write document to disk so harper-ls can resolve its path.
        # harper-ls checks if the file URI exists for file-local dictionaries.
        # This is a one-time write; subsequent changes go through didChange.
        self._write_doc_to_disk(uri, text)
        self._send({
            "jsonrpc": "2.0",
            "method":  "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri":        uri,
                    "languageId": "markdown",
                    "version":    self._version,
                    "text":       text,
                }
            }
        })

    def change_document(self, uri, text):
        """Debounced — waits DEBOUNCE_MS ms after last call before sending."""
        if self._debounce:
            self._debounce.cancel()
        self._debounce = threading.Timer(
            self.DEBOUNCE_MS / 1000.0,
            self._send_change, args=(uri, text))
        self._debounce.daemon = True
        self._debounce.start()

    def request_code_actions(self, uri, line, char_start, char_end, timeout=3.0):
        """Send textDocument/codeAction request and return list of suggestions.

        Runs synchronously — blocks until harper-ls responds or timeout.
        Returns list of {"title": str, "edit": dict} for each code action.
        """
        if not self.running():
            harper_log("CODE_ACTION: not running")
            return []
        req_id = self._next_id()
        evt = threading.Event()
        with self._lock:
            self._pending_responses[req_id] = evt
        harper_log(f"CODE_ACTION: sending request id={req_id} range={line}:{char_start}-{char_end}")
        self._send({
            "jsonrpc": "2.0",
            "id":      req_id,
            "method":  "textDocument/codeAction",
            "params": {
                "textDocument": {"uri": uri},
                "range": {
                    "start": {"line": line, "character": char_start},
                    "end":   {"line": line, "character": char_end},
                },
                "context": {"diagnostics": []},
            }
        })
        if evt.wait(timeout=timeout):
            with self._lock:
                result = self._response_data.pop(req_id, None)
                self._pending_responses.pop(req_id, None)
            if isinstance(result, list):
                harper_log(f"CODE_ACTION: got {len(result)} action(s)")
                return result
            else:
                harper_log(f"CODE_ACTION: unexpected result type: {type(result)}")
                return []
        else:
            harper_log(f"CODE_ACTION: timeout after {timeout}s")
            with self._lock:
                self._pending_responses.pop(req_id, None)
            return []

    # ── internal ─────────────────────────────────────────────────────────────

    def _next_id(self):
        with self._lock:
            self._msg_id += 1
            return self._msg_id

    def _send(self, obj):
        if not self.running():
            return
        body_str   = json.dumps(obj, ensure_ascii=False)
        body_bytes = body_str.encode("utf-8")
        # Content-Length MUST be byte count, not character count
        header     = f"Content-Length: {len(body_bytes)}\r\n\r\n"
        method = obj.get("method", f"response:{obj.get('id', '?')}")
        app_logger.debug(
            f"LSP → harper-ls: {method} ({len(body_bytes)}b)")
        with self._lock:
            try:
                self._proc.stdin.write(header.encode("utf-8") + body_bytes)
                self._proc.stdin.flush()
            except Exception as _se:
                app_logger.debug(f"HarperLSPClient._send error: {_se}")

    def _send_initialize(self):
        # rootUri MUST point to a directory that exists on disk.
        # harper-ls calls canonicalize() on it; if it fails (e.g. file:///
        # resolves to \ on Windows which doesn't exist), from_lsp_config
        # returns an Err and ALL linter config is silently discarded.
        _app_dir = (os.path.dirname(sys.executable)
                    if getattr(sys, "frozen", False)
                    else os.path.dirname(os.path.abspath(__file__)))
        _root_uri = "file:///" + _app_dir.replace("\\", "/").lstrip("/")
        harper_log(f"INIT: rootUri={_root_uri!r}")
        app_logger.info(f"HarperLSPClient: rootUri={_root_uri!r}")
        _linters = self._get_linters_dict()
        _settings = {
            "harper-ls": {
                "diagnosticSeverity": "hint",
                "linters": _linters,
            }
        }
        init_id = self._next_id()
        harper_log(f"INIT: sending initialize id={init_id}")
        self._send({
            "jsonrpc": "2.0",
            "id":      init_id,
            "method":  "initialize",
            "params": {
                "processId":        os.getpid(),
                "rootUri":          _root_uri,
                "workspaceFolders": [{"uri": _root_uri, "name": "WhisperR"}],
                "initializationOptions": _settings,
                "capabilities": {
                    "textDocument": {
                        "publishDiagnostics": {
                            "relatedInformation": True,
                            "versionSupport":     True,
                        },
                        "synchronization": {"dynamicRegistration": False},
                    },
                    "workspace": {
                        "workspaceFolders":      True,
                        "didChangeConfiguration": {"dynamicRegistration": False},
                        "configuration":          True,
                    }
                },
                "clientInfo": {"name": "WhisperR", "version": "2.1.0"}
            }
        })
        harper_log("INIT: sending initialized notification")
        # initialized notification — required by LSP spec
        self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
        harper_log("INIT: sending workspace/didChangeConfiguration")
        # Also push settings via workspace/didChangeConfiguration
        # (some harper-ls versions only read one or the other)
        self._send({
            "jsonrpc": "2.0",
            "method":  "workspace/didChangeConfiguration",
            "params":  {"settings": _settings}
        })

    def _send_keepalive(self):
        """Resend the last document text to prevent harper-ls idle timeout."""
        if not self.running():
            return
        uri  = getattr(self, "_uri",       None)
        text = getattr(self, "_last_text", None)
        if uri and text is not None:
            app_logger.debug("HarperLSPClient: keepalive ping")
            self._send_change(uri, text)

    def _write_doc_to_disk(self, uri, text):
        """Write the LSP document to disk so harper-ls can resolve it."""
        try:
            # Convert file:///E:/path/to/file.md → E:\path\to\file.md
            _path = uri[len("file:///"):].replace("/", os.sep)
            with open(_path, "w", encoding="utf-8") as _f:
                _f.write(text)
            app_logger.debug(f"HarperLSPClient: wrote doc to {_path!r}")
        except Exception as _we:
            app_logger.debug(f"HarperLSPClient: doc write skipped: {_we}")

    def _send_change(self, uri, text):
        self._last_text = text   # cache for auto-restart
        self._version += 1
        harper_log(f"CHANGE: version={self._version} text_len={len(text)}")
        app_logger.debug(
            f"LSP _send_change: version={self._version} "
            f"text_len={len(text)} preview={text[:40]!r}")
        self._write_doc_to_disk(uri, text)
        self._send({
            "jsonrpc": "2.0",
            "method":  "textDocument/didChange",
            "params": {
                "textDocument": {
                    "uri":     uri,
                    "version": self._version,
                },
                "contentChanges": [{"text": text}]
            }
        })

    def _read_loop(self):
        """Read LSP messages from harper-ls stdout in a dedicated thread."""
        buf = b""
        while self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    _exit_code = self._proc.poll()
                    harper_log(f"STDOUT CLOSED: harper-ls exit code={_exit_code}")
                    app_logger.info(
                        f"HarperLSPClient: stdout closed "
                        f"(harper-ls exited, code={_exit_code})")
                    if _exit_code not in (0, None):
                        harper_log(f"ERROR: harper-ls exit code {_exit_code}")
                        app_logger.error(
                            f"harper-ls exit code: {_exit_code} "
                            f"(non-zero = crash or error)")
                    # Schedule auto-restart on the main Qt thread
                    if self._on_diagnostics and not self._stopping:
                        from PyQt6.QtCore import QTimer
                        QTimer.singleShot(
                            5000, self._auto_restart)
                    break
                buf += chunk
                while True:
                    # Parse Content-Length header
                    hdr_end = buf.find(b"\r\n\r\n")
                    sep_len = 4
                    if hdr_end < 0:
                        # Try \n\n fallback
                        hdr_end = buf.find(b"\n\n")
                        sep_len = 2
                    if hdr_end < 0:
                        break
                    header  = buf[:hdr_end].decode("utf-8", errors="ignore")
                    length  = 0
                    for line in header.splitlines():
                        if line.lower().startswith("content-length:"):
                            try:
                                length = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                pass
                    body_start = hdr_end + sep_len
                    if len(buf) < body_start + length:
                        break   # need more data
                    body = buf[body_start:body_start + length]
                    buf  = buf[body_start + length:]
                    try:
                        msg = json.loads(body.decode("utf-8"))
                        harper_log(f"RECV: {len(body)}b parsed")
                        self._handle_message(msg)
                    except Exception as _he:
                        harper_log(f"PARSE ERROR: {_he}")
                        app_logger.warning(
                            f"HarperLSPClient._handle_message error: {_he}")
            except Exception as _rle:
                app_logger.warning(
                    f"HarperLSPClient read_loop exception: {_rle}")
                break

    def _read_stderr_loop(self):
        """Read harper-ls stderr and log it for diagnostics."""
        try:
            for line in iter(self._proc.stderr.readline, b""):
                txt = line.decode("utf-8", errors="replace").rstrip()
                if txt:
                    app_logger.debug(f"harper-ls stderr: {txt}")
        except Exception:
            pass

    def _handle_message(self, msg):
        method   = msg.get("method", "")
        _msg_id  = msg.get("id", "")
        _is_req  = "id" in msg and "method" in msg   # server→client request
        _is_resp = "id" in msg and "result" in msg    # our request's response
        if method:
            harper_log(f"MSG ← {method!r} id={_msg_id!r} req={_is_req} resp={_is_resp}")
        app_logger.debug(
            f"LSP ← harper-ls: {method!r} id={_msg_id!r} "
            f"req={_is_req} resp={_is_resp}")
        if method == "textDocument/publishDiagnostics":
            _params   = msg.get("params", {})
            diags_raw = _params.get("diagnostics", [])
            _doc_uri  = _params.get("uri", "?")
            harper_log(f"DIAG: {len(diags_raw)} diagnostic(s) for uri={_doc_uri!r}")
            # Log full raw message on first call
            if not getattr(self, "_diag_params_logged", False):
                import json as _jd
                harper_log(f"DIAG RAW: {_jd.dumps(msg, ensure_ascii=False)[:600]}")
                app_logger.info(
                    f"Harper LSP: RAW publishDiagnostics: "
                    f"{_jd.dumps(msg, ensure_ascii=False)[:800]}")
                self._diag_params_logged = True
            app_logger.debug(
                f"Harper LSP: received {len(diags_raw)} diagnostic(s) "
                f"for uri={_doc_uri!r}")
            if diags_raw:
                for _d in diags_raw[:3]:
                    harper_log(f"DIAG item: {_d}")
                    app_logger.debug(f"  diag: {_d}")
            diags = []
            for d in diags_raw:
                rng    = d.get("range", {})
                start  = rng.get("start", {})
                end    = rng.get("end", {})
                # Extract suggestions from codeActions if present
                suggs  = []
                for action in d.get("relatedInformation", []):
                    msg_txt = action.get("message", "")
                    if msg_txt:
                        suggs.append(msg_txt)
                # harper-ls puts suggestions in data.suggestions
                data = d.get("data") or {}
                if isinstance(data, dict):
                    suggs = data.get("suggestions", suggs)
                diags.append({
                    "range":       {"start": start, "end": end},
                    "message":     d.get("message", ""),
                    "suggestions": suggs,
                })
            if self._on_diagnostics:
                harper_log(f"DIAG: emitting {len(diags)} diag(s) via bridge")
                if self._bridge:
                    self._bridge.ready.emit(diags)
                else:
                    harper_log("DIAG: bridge is None, calling callback directly")
                    self._on_diagnostics(diags)

        elif method == "workspace/configuration":
            # harper-ls REQUIRES a response to workspace/configuration —
            # without it, it blocks indefinitely and never sends diagnostics.
            _req_id = msg.get("id")
            _items  = msg.get("params", {}).get("items", [])
            harper_log(f"WSCFG: request id={_req_id} items={len(_items)}")
            # harper-ls expects each result item to be {"harper-ls": {<settings>}}
            # It does result_item["harper-ls"] internally — unwrapped settings
            # cause "Settings must contain a 'harper-ls' key" error.
            _linters = self._get_linters_dict()
            _inner_cfg = {
                "diagnosticSeverity": "hint",
                "linters": _linters,
            }
            # harper-ls ALWAYS expects {"harper-ls": {settings}} wrapper
            # regardless of which section is requested. This is confirmed
            # by its stderr: "Settings must contain a 'harper-ls' key"
            # when we send the unwrapped inner dict.
            _wrapped = {"harper-ls": _inner_cfg}
            _results = [_wrapped for _ in _items] if _items else [_wrapped]
            import json as _json_wscfg
            harper_log(f"WSCFG: sending reply id={_req_id}")
            app_logger.debug(
                f"LSP workspace/configuration payload: "
                f"{_json_wscfg.dumps(_results[0] if _results else {}, ensure_ascii=False)[:200]}")
            self._send({"jsonrpc": "2.0", "id": _req_id, "result": _results})
            _sections_sent = [list(_r.keys()) if isinstance(_r, dict) else _r
                              for _r in _results]
            harper_log(f"WSCFG: reply sent, sections={_sections_sent}")
            app_logger.debug(
                f"LSP: replied to workspace/configuration id={_req_id} "
                f"sections={[i.get('section','?') for i in _items]} "
                f"result_keys={_sections_sent}")

        elif msg.get("id") is not None and "result" in msg:
            _res = msg.get("result")
            _rid = msg.get("id")
            if isinstance(_res, dict):
                _si = _res.get("serverInfo", {})
                if _si:
                    harper_log(f"INIT OK: server={_si.get('name','?')} v{_si.get('version','?')}")
                    app_logger.info(
                        f"Harper LSP server: {_si.get('name','?')} "
                        f"v{_si.get('version','?')} "
                        f"(result keys={list(_res.keys())})")
                    self._init_event.set()
                else:
                    harper_log(f"RESP: id={_rid} result keys={list(_res.keys())}")
                    app_logger.info(
                        f"Harper LSP response id={_rid}: "
                        f"result keys={list(_res.keys())}")
            else:
                harper_log(f"RESP: id={_rid} result type={type(_res).__name__} (list/scalar)")
            # Signal any pending request waiting for this response
            if _rid is not None:
                with self._lock:
                    self._response_data[_rid] = _res
                    if _rid in self._pending_responses:
                        self._pending_responses[_rid].set()
        elif msg.get("id") is not None and "error" in msg:
            harper_log(f"ERROR: id={msg['id']} error={msg['error']}")
            app_logger.error(
                f"Harper LSP error id={msg['id']}: {msg['error']}")
            _rid = msg.get("id")
            if _rid is not None:
                with self._lock:
                    self._response_data[_rid] = msg.get("error")
                    if _rid in self._pending_responses:
                        self._pending_responses[_rid].set()






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
        # Lint underlines — applied AFTER Markdown so they show on top
        block_start = self.currentBlock().position()
        self._apply_lint(block_start, text)

    def set_lint_errors(self, lint_list):
        """Update error spans and re-highlight. lint_list: [(start,end,msg,[sugg])]"""
        harper_log(f"set_lint_errors: {len(lint_list)} items, existing={len(getattr(self, '_lint_errors', []))}")
        if lint_list == getattr(self, "_lint_errors", None):
            harper_log("set_lint_errors: unchanged, skipping")
            return  # unchanged — skip rehighlight to avoid cursor jump
        self._lint_errors = lint_list
        harper_log("set_lint_errors: calling rehighlight()")
        # Save and restore cursor: rehighlight() can move it in some Qt builds.
        # QTextDocument.views() is not exposed in PyQt6 — use parent widget.
        _parent = self.parent()
        _saved = _parent.textCursor() if hasattr(_parent, "textCursor") else None
        self.rehighlight()
        harper_log("set_lint_errors: rehighlight() done")
        if _saved is not None:
            _parent.setTextCursor(_saved)

    def _apply_lint(self, block_start, text):
        """Apply wavy-underline format to lint spans overlapping this block."""
        if not hasattr(self, '_lint_errors') or not self._lint_errors:
            return
        block_end = block_start + len(text)
        err_fmt = QTextCharFormat()
        err_fmt.setUnderlineColor(QColor("#ff4444"))
        err_fmt.setUnderlineStyle(
            QTextCharFormat.UnderlineStyle.SpellCheckUnderline)
        harper_log(f"_apply_lint: block={block_start}-{block_end} text_len={len(text)} errors={len(self._lint_errors)}")
        for (start, end, _msg, _sugg) in self._lint_errors:
            harper_log(f"  error span: {start}-{end}")
            # Clamp to this block
            s = max(start - block_start, 0)
            e = min(end   - block_start, len(text))
            harper_log(f"  clamped: {s}-{e}")
            if s < e:
                self.setFormat(s, e - s, err_fmt)
                harper_log(f"  APPLIED format at {s} len={e-s}")



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
                    # Strip pynput angle-bracket format <ctrl> → ctrl
                    parts = [p.strip().lower().strip("<>") for p in hk_str.split("+")]
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




# ─────────────────────────────────────────────────────────────────────────────
# _NotesWindow  — sticky-note panel that attaches beside WhisperEditor
# ─────────────────────────────────────────────────────────────────────────────
class _NoteDragButton(QPushButton):
    """Compact drag-handle button in each note's top-bar.

    Does NOT use grabMouse() — that gives wrong coordinates on Windows inside
    a QScrollArea.  Instead, on press we install a QApplication-level event
    filter on _NotesWindow so it receives every subsequent mouse move/release
    regardless of which widget the cursor is over.
    """
    def __init__(self, text, note: "QWidget", parent=None):
        super().__init__(text, parent)
        self._note = note
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            nw = self._note._find_notes_win()
            if nw:
                nw._start_drag(self._note)
        event.accept()

    def mouseMoveEvent(self, event):
        event.accept()   # handled by the NotesWindow event filter

    def mouseReleaseEvent(self, event):
        event.accept()   # handled by the NotesWindow event filter



class _NoteWidget(QWidget):
    """A single sticky note with auto-expanding text area."""
    deleted = pyqtSignal(object)

    # (bg, border, text_color)
    NOTE_COLORS = [
        ("#fff9c4", "#e6d020", "#333"),  # yellow (default)
        ("#c8e6c9", "#4caf50", "#333"),  # green
        ("#b3e5fc", "#0288d1", "#333"),  # blue
        ("#f8bbd0", "#e91e63", "#333"),  # pink
        ("#ffe0b2", "#fb8c00", "#333"),  # orange
        ("#e1bee7", "#9c27b0", "#333"),  # purple
        ("#ffffff", "#cccccc", "#333"),  # white
        ("#e0e0e0", "#9e9e9e", "#333"),  # light grey
        ("#616161", "#424242", "#eee"),  # dark grey
        ("#121212", "#000000", "#eee"),  # black
    ]

    def __init__(self, text="", color_idx=0, parent=None):
        super().__init__(parent)
        self._color_idx = min(color_idx, len(self.NOTE_COLORS) - 1)
        self._collapsed = False
        self._build_ui(text)
        self._apply_color()
        # Expand width with parent
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def _build_ui(self, text):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── top bar ────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(3)
        self._swatches = []
        for i, (bg, border, _tc) in enumerate(self.NOTE_COLORS):
            sw = QPushButton()
            sw.setFixedSize(16, 16)
            sw.setStyleSheet(
                f"QPushButton{{background:{bg};border:1px solid {border};"
                f"border-radius:2px;padding:0;}}"
                f"QPushButton:hover{{border:2px solid #555;}}")
            sw.clicked.connect(lambda _, idx=i: self._set_color(idx))
            sw.setToolTip(f"Colour {i+1}")
            top.addWidget(sw)
            self._swatches.append(sw)
        top.addStretch()
        # ── drag-reorder button ──────────────────────────────────────────
        self._btn_drag = _NoteDragButton("⠿⠿", self)
        self._btn_drag.setFixedSize(52, 26)  # double width of ✕
        self._btn_drag.setToolTip("Hold and drag to reorder")
        self._btn_drag.setStyleSheet(
            "QPushButton{background:transparent;border:1px solid transparent;"
            "font-size:11px;color:#888;border-radius:3px;padding:0px;letter-spacing:3px;}"
            "QPushButton:hover{background:#2a2a2a;color:#aaa;border-color:#555;}")
        top.addWidget(self._btn_drag)
        # ── delete button ────────────────────────────────────────────────
        self._btn_del = QPushButton("✕")
        self._btn_del.setFixedSize(26, 26)
        self._btn_del.setToolTip("Delete this note")
        self._btn_del.setStyleSheet(
            "QPushButton{background:transparent;border:1px solid transparent;"
            "font-size:13px;font-weight:bold;color:#888;border-radius:3px;padding:0px;}"
            "QPushButton:hover{background:#e53935;color:#fff;border-color:#e53935;}")
        self._btn_del.clicked.connect(lambda: self.deleted.emit(self))
        top.addWidget(self._btn_del)
        root.addLayout(top)

        # ── text area ──────────────────────────────────────────────────────
        # Use _HotkeyFilteredTextEdit so global hotkeys (Ctrl+Alt+Z etc.)
        # are eaten here and never insert characters into the note.
        self.text_edit = _HotkeyFilteredTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setFrameShape(QFrame.Shape.NoFrame)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        fm = self.text_edit.fontMetrics()
        self._line_h = fm.lineSpacing()
        self._default_lines = 5
        self.text_edit.setMinimumHeight(self._line_h * self._default_lines + 14)
        self.text_edit.document().contentsChanged.connect(self._auto_resize)
        root.addWidget(self.text_edit)
        self._auto_resize()

    def _auto_resize(self):
        if self._collapsed:
            return
        doc_h = int(self.text_edit.document().size().height()) + 14
        min_h = self._line_h * self._default_lines + 14
        self.text_edit.setFixedHeight(max(doc_h, min_h))
        self.adjustSize()
        if self.parent():
            self.parent().adjustSize()

    def _set_color(self, idx):
        self._color_idx = idx
        self._apply_color()
        nw = self._find_notes_win()
        if nw and nw._color_filter:
            nw._apply_color_filter()

    def _apply_color(self):
        bg, border, tc = self.NOTE_COLORS[self._color_idx]
        self.setStyleSheet(
            f"_NoteWidget{{background:{bg};border:1px solid {border};"
            f"border-radius:6px;}}")
        self.text_edit.setStyleSheet(
            f"QTextEdit{{background:{bg};border:none;color:{tc};}}")

    def collapse(self):
        self._collapsed = True
        self.text_edit.setFixedHeight(self._line_h * 3 + 14)
        self.text_edit.setReadOnly(True)
        self.adjustSize()

    def uncollapse(self):
        self._collapsed = False
        self.text_edit.setReadOnly(False)
        self._auto_resize()

    def mousePressEvent(self, event):
        if self._collapsed:
            self.uncollapse()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._collapsed:
            self.uncollapse()
        super().mouseDoubleClickEvent(event)

    def get_text(self):       return self.text_edit.toPlainText()
    def get_color_idx(self):  return self._color_idx

    # ── Drag handle (the braille-dots bar at top of each note) ──────────

    def _find_notes_win(self):
        w = self.parent()
        while w:
            if isinstance(w, _NotesWindow):
                return w
            w = w.parent()
        return None


class _NotesWindow(QWidget):
    """Floating sticky-note panel that attaches beside WhisperEditor."""

    MAX_UNDO = 3

    def __init__(self, editor: "WhisperEditor"):
        super().__init__(editor, Qt.WindowType.Window)
        self.setWindowTitle("Notes")
        self._editor = editor
        self._notes: list[_NoteWidget] = []
        self._undo_stack: list[dict] = []
        self._dragging_note = None   # active drag target
        self._color_filter: set = set()  # empty = show all colors
        self._bulk_delete_snapshot: list = []  # for undoing Delete All
        self._build_ui()
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def closeEvent(self, event):
        btn = getattr(self._editor, "btn_notes", None)
        if btn:
            btn.setChecked(False)
        event.accept()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        self.setStyleSheet("QWidget{background:#1e1e1e;color:#e0e0e0;}")

        # ── header ─────────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        lbl = QLabel("📝 Notes")
        lbl.setStyleSheet("font-weight:bold;color:#ccc;font-size:10pt;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self._btn_undo_all = QPushButton("↩ Restore All")
        self._btn_undo_all.setToolTip("Restore all notes deleted by Delete All")
        self._btn_undo_all.setStyleSheet(
            "QPushButton{background:#3a2a00;border:1px solid #cc8800;"
            "color:#ffcc44;border-radius:4px;padding:3px 8px;font-weight:bold;}"
            "QPushButton:hover{background:#5a3a00;}")
        self._btn_undo_all.clicked.connect(self._undo_delete_all)
        self._btn_undo_all.setVisible(False)
        hdr.addWidget(self._btn_undo_all)
        self._btn_undo = QPushButton("↩ Undo")
        self._btn_undo.setToolTip("Restore last deleted note")
        self._btn_undo.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;padding:3px 8px;"
            "border-radius:4px;color:#ddd;font-size:9pt;}"
            "QPushButton:hover{background:#353535;border-color:#0078d7;}")
        self._btn_undo.clicked.connect(self._undo_delete)
        self._btn_undo.setVisible(False)
        hdr.addWidget(self._btn_undo)
        root.addLayout(hdr)

        # ── scroll area ─────────────────────────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setStyleSheet("QScrollArea{background:#1e1e1e;border:none;}")
        self._inner = QWidget()
        self._inner.setStyleSheet("QWidget{background:#1e1e1e;}")
        self._notes_layout = QVBoxLayout(self._inner)
        self._notes_layout.setContentsMargins(4, 4, 4, 4)
        self._notes_layout.setSpacing(20)
        self._notes_layout.addStretch()
        self._scroll.setWidget(self._inner)
        root.addWidget(self._scroll, 1)
        # Drop-indicator line (shown while dragging, hidden otherwise)
        self._drop_line = QFrame(self._inner)
        self._drop_line.setFrameShape(QFrame.Shape.HLine)
        self._drop_line.setFixedHeight(5)
        self._drop_line.setStyleSheet(
            "QFrame{background:#0078d7;border:none;border-radius:2px;}")
        self._drop_line.hide()

        # ── footer ──────────────────────────────────────────────────────────
        _ss = ("QPushButton{background:#2a2a2a;border:1px solid #444;"
               "border-radius:4px;color:#ddd;}"
               "QPushButton:hover{background:#353535;border-color:#0078d7;}")
        foot = QHBoxLayout()
        btn_add = QPushButton("＋ Add Note")
        btn_add.setToolTip(
            "Add a new note\n"
            "Shortcut: Ctrl+Enter while typing in any note")
        btn_add.setStyleSheet(_ss + "QPushButton{padding:4px 10px;}")
        btn_add.clicked.connect(lambda: self._add_note_after_focused())
        foot.addWidget(btn_add)
        # Delete All — sits right next to Add Note so users can't miss the pair
        self._btn_del_all = QPushButton("🗑 Delete All")
        self._btn_del_all.setToolTip(
            "Delete ALL notes\n"
            "Hold Shift to skip confirmation.\n"
            "Undoable with the Restore All button that appears.")
        self._btn_del_all.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;"
            "border-radius:4px;color:#cc4444;padding:4px 8px;}"
            "QPushButton:hover{background:#3a1a1a;border-color:#e53935;color:#ff6b6b;}")
        self._btn_del_all.clicked.connect(self._delete_all_notes)
        foot.addWidget(self._btn_del_all)
        # Ctrl+Enter shortcut — add note after the currently focused one
        _sc_add = QShortcut(QKeySequence("Ctrl+Return"), self)
        _sc_add.setContext(Qt.ShortcutContext.WindowShortcut)
        _sc_add.activated.connect(self._add_note_after_focused)
        # Right side: filter indicator + filter button
        self._filter_btn = QPushButton("🎨")
        self._filter_btn.setFixedSize(28, 28)
        self._filter_btn.setToolTip(
            "Filter notes by color\n"
            "Click color swatches to show/hide notes of that color.\n"
            "All colors shown = no filter active.")
        self._filter_btn.setStyleSheet(_ss)
        self._filter_btn.clicked.connect(self._show_color_filter_menu)
        self._filter_indicator = QPushButton("!")
        self._filter_indicator.setFixedSize(22, 22)
        self._filter_indicator.setToolTip("Some notes are hidden by the color filter")
        self._filter_indicator.setStyleSheet(
            "QPushButton{background:#5a3a00;border:1px solid #cc8800;"
            "color:#ffcc44;border-radius:3px;font-size:9px;font-weight:bold;padding:0;}"
            "QPushButton:hover{background:#7a5000;}")
        self._filter_indicator.hide()
        self._filter_indicator.clicked.connect(self._show_color_filter_menu)
        foot.addStretch()
        foot.addWidget(self._filter_indicator)
        foot.addWidget(self._filter_btn)
        btn_col = QPushButton("−")
        btn_col.setToolTip("Collapse all notes (show 3 lines each)")
        btn_col.setFixedSize(28, 28)
        btn_col.setStyleSheet(_ss + "QPushButton{font-size:16px;font-weight:bold;padding:0;}")
        btn_col.clicked.connect(self._collapse_all)
        btn_exp = QPushButton("+")
        btn_exp.setToolTip("Expand all notes")
        btn_exp.setFixedSize(28, 28)
        btn_exp.setStyleSheet(_ss + "QPushButton{font-size:16px;font-weight:bold;padding:0;}")
        btn_exp.clicked.connect(self._expand_all)
        foot.addWidget(btn_col)
        foot.addWidget(btn_exp)
        root.addLayout(foot)
        # ── import/export row ───────────────────────────────────────────
        _ss2 = ("QPushButton{background:#2a2a2a;border:1px solid #444;"
                "border-radius:4px;color:#aaa;font-size:9pt;padding:3px 8px;}"
                "QPushButton:hover{background:#353535;border-color:#0078d7;color:#ddd;}")
        io_row = QHBoxLayout()
        btn_imp = QPushButton("↑ Import Notes")
        btn_imp.setToolTip(
            "Import notes from a file.\n"
            "TXT/MD: each note separated by ---------- (10 dashes).\n"
            "WRP: only the notes are read; all other project data is ignored.\n"
            "Imported notes are appended to existing notes.")
        btn_imp.setStyleSheet(_ss2)
        btn_imp.clicked.connect(self._import_notes_file)
        btn_exp = QPushButton("↓ Export Notes")
        btn_exp.setToolTip(
            "Export notes to a file.\n"
            "TXT/MD: notes separated by ---------- (10 dashes).\n"
            "WRP (new): creates a notes-only project file.\n"
            "WRP (existing): appends notes to the file's existing notes.")
        btn_exp.setStyleSheet(_ss2)
        btn_exp.clicked.connect(self._export_notes_file)
        io_row.addWidget(btn_imp)
        io_row.addWidget(btn_exp)
        io_row.addStretch()
        root.addLayout(io_row)

        self._add_note()

    def _add_note_after_focused(self):
        """Add a new note immediately after the focused note (or at end)."""
        focused_idx = -1
        for i, n in enumerate(self._notes):
            if n.text_edit.hasFocus():
                focused_idx = i
                break
        note = self._add_note(after_idx=focused_idx)
        note.text_edit.setFocus()
        # Scroll to the new note — retry until it has been laid out
        QTimer.singleShot(0, lambda: self._scroll_to_note(note))

    def _scroll_to_note(self, note: "_NoteWidget", _attempts: int = 0):
        """Scroll the outer panel so the given note is fully visible.
        Retries every 10 ms (up to 20 times) until Qt has finished laying
        out the widget and it has a non-zero height.
        """
        if not hasattr(self, "_scroll") or not note:
            return
        if note.height() > 0:
            self._scroll.ensureWidgetVisible(note, 0, 20)
        elif _attempts < 20:
            QTimer.singleShot(10, lambda: self._scroll_to_note(note, _attempts + 1))

    def _scroll_cursor_visible(self, text_edit):
        """Scroll the outer panel so the cursor line inside text_edit is visible.

        text_edit has its own scrollbars disabled and auto-resizes, so the cursor
        rect is always relative to the top of the widget.  We map that rect into
        _inner coordinates, then ask the outer QScrollArea to show it.
        """
        if not hasattr(self, "_scroll"):
            return
        # Cursor rect in text_edit-local coords
        cr = text_edit.cursorRect()
        # Map to _inner coords (the QScrollArea's content widget)
        top_left = text_edit.mapTo(self._inner, cr.topLeft())
        bot_right = text_edit.mapTo(self._inner, cr.bottomRight())
        from PyQt6.QtCore import QRect
        cursor_rect_inner = QRect(top_left, bot_right)
        # ensureVisible takes (x, y, xmargin, ymargin) in content-widget coords
        self._scroll.ensureVisible(
            cursor_rect_inner.x(),
            cursor_rect_inner.bottom(),
            0, 30)   # 30 px margin below cursor so next line is readable

    def _add_note(self, text="", color_idx=0, after_idx=-1):
        note = _NoteWidget(text=text, color_idx=color_idx, parent=self._inner)
        note.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        note.deleted.connect(self._on_note_deleted)
        # Scroll to note when it receives focus (mouse click or Tab)
        _nw_ref = self
        _orig_focus = note.text_edit.focusInEvent
        def _on_focus(ev, _note=note):
            _orig_focus(ev)
            # Small delay so the layout has processed the focus change
            QTimer.singleShot(0, lambda: _nw_ref._scroll_to_note(_note))
        note.text_edit.focusInEvent = _on_focus
        # Keep the cursor line visible as the user types
        note.text_edit.cursorPositionChanged.connect(
            lambda _te=note.text_edit: _nw_ref._scroll_cursor_visible(_te))
        if after_idx >= 0 and after_idx < len(self._notes):
            # Insert after_idx in both list and layout
            self._notes.insert(after_idx + 1, note)
            # Layout index: each note occupies one slot before the trailing stretch
            self._notes_layout.insertWidget(after_idx + 1, note)
        else:
            count = self._notes_layout.count()
            self._notes_layout.insertWidget(count - 1, note)
            self._notes.append(note)
        if self._color_filter and note.get_color_idx() not in self._color_filter:
            note.hide()
            self._apply_color_filter()
        return note

    def _on_note_deleted(self, note):
        self._undo_stack.append({"text": note.get_text(), "color_idx": note.get_color_idx()})
        if len(self._undo_stack) > self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._notes.remove(note)
        note.setParent(None)
        note.deleteLater()
        self._btn_undo.setVisible(bool(self._undo_stack))

    def _undo_delete(self):
        if not self._undo_stack:
            return
        data = self._undo_stack.pop()
        self._add_note(data["text"], data["color_idx"])
        self._btn_undo.setVisible(bool(self._undo_stack))

    def _delete_all_notes(self):
        """Delete visible (filtered) notes, or all notes if no filter active.
        Stores a full snapshot for undo regardless.
        """
        shift_held = bool(QApplication.keyboardModifiers() &
                          Qt.KeyboardModifier.ShiftModifier)
        # Determine which notes will be deleted
        if self._color_filter:
            # Only delete notes whose color is in the active filter (visible)
            to_delete = [n for n in self._notes
                         if n.get_color_idx() in self._color_filter]
            label = f"Delete {len(to_delete)} Filtered Note(s)"
            question = (
                f"Delete {len(to_delete)} visible (filtered) note(s)?\n"
                f"The {len(self._notes) - len(to_delete)} hidden note(s) "
                f"will NOT be deleted.\n\n"
                "You can restore deleted notes with the \u21a9 Restore All button.")
        else:
            to_delete = list(self._notes)
            label = f"Delete All {len(to_delete)} Note(s)"
            question = (
                f"Delete all {len(to_delete)} note(s)?\n\n"
                "You can restore them with the \u21a9 Restore All button.")
        if not to_delete:
            return
        if not shift_held:
            reply = QMessageBox.question(
                self, label, question,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Snapshot ALL notes (including hidden) so Restore All is complete
        self._bulk_delete_snapshot = self.get_notes_data()
        self._undo_stack.clear()
        self._btn_undo.setVisible(False)
        for n in to_delete:
            self._notes.remove(n)
            n.setParent(None)
            n.deleteLater()
        # Ensure at least one blank note remains if everything was deleted
        if not self._notes:
            self._add_note()
        self._btn_undo_all.setVisible(True)
        self._apply_color_filter()

    def _undo_delete_all(self):
        """Restore notes snapshot saved by _delete_all_notes."""
        if not self._bulk_delete_snapshot:
            return
        for n in list(self._notes):
            n.setParent(None)
            n.deleteLater()
        self._notes.clear()
        for item in self._bulk_delete_snapshot:
            self._add_note(item.get("text", ""), item.get("color_idx", 0))
        self._bulk_delete_snapshot = []
        self._btn_undo_all.setVisible(False)
        if self._color_filter:
            self._apply_color_filter()

    def _collapse_all(self):
        for n in self._notes: n.collapse()

    def _expand_all(self):
        for n in self._notes: n.uncollapse()

    # ── Color filter ─────────────────────────────────────────────────────

    _COLOR_NAMES = ["Yellow","Green","Blue","Pink","Orange",
                    "Purple","White","Light Grey","Dark Grey","Black"]

    def _show_color_filter_menu(self):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(
            "QMenu{background:#1e1e1e;border:1px solid #444;padding:4px;}"
            "QMenu::item{padding:4px 12px;color:#ddd;}"
            "QMenu::item:selected{background:#2a2a2a;}")
        act_all = menu.addAction("Show All Colors")
        act_all.triggered.connect(lambda: self._set_color_filter(set()))
        menu.addSeparator()
        for i, (bg, border, _tc) in enumerate(_NoteWidget.NOTE_COLORS):
            name = (self._COLOR_NAMES[i]
                    if i < len(self._COLOR_NAMES) else f"Color {i}")
            tick = "[x]" if i in self._color_filter else "[ ]"
            px = QPixmap(14, 14)
            px.fill(QColor(bg))
            act = menu.addAction(QIcon(px), f"{tick} {name}")
            act.triggered.connect(lambda _, idx=i: self._toggle_color_filter(idx))
        menu.exec(self._filter_btn.mapToGlobal(
            self._filter_btn.rect().topLeft()))

    def _toggle_color_filter(self, color_idx: int):
        if color_idx in self._color_filter:
            self._color_filter.discard(color_idx)
        else:
            self._color_filter.add(color_idx)
        self._apply_color_filter()

    def _set_color_filter(self, new_filter: set):
        self._color_filter = new_filter
        self._apply_color_filter()

    def _apply_color_filter(self):
        hidden = 0
        for note in self._notes:
            if self._color_filter and note.get_color_idx() not in self._color_filter:
                note.hide()
                hidden += 1
            else:
                note.show()
        if self._color_filter:
            self._filter_btn.setStyleSheet(
                "QPushButton{background:#003a1a;border:2px solid #00cc55;"
                "color:#00ff77;border-radius:4px;font-size:14px;}")
        else:
            self._filter_btn.setStyleSheet(
                "QPushButton{background:#2a2a2a;border:1px solid #444;"
                "border-radius:4px;color:#ddd;font-size:14px;}"
                "QPushButton:hover{background:#353535;border-color:#0078d7;}")
        if hidden > 0:
            self._filter_indicator.setText(f"+{hidden}")
            self._filter_indicator.setToolTip(
                f"{hidden} note(s) hidden by color filter\n"
                f"Click the palette icon to change filter")
            self._filter_indicator.show()
        else:
            self._filter_indicator.hide()
        # Update Delete button label based on filter state
        visible_count = len(self._notes) - hidden
        if self._color_filter:
            self._btn_del_all.setText(f"🗑 Delete Filtered ({visible_count})")
            self._btn_del_all.setToolTip(
                f"Delete only the {visible_count} visible (filtered) note(s)\n"
                "Hidden notes are left untouched.\n"
                "Hold Shift to skip confirmation.\n"
                "Undoable with the Restore All button that appears.")
        else:
            self._btn_del_all.setText("🗑 Delete All")
            self._btn_del_all.setToolTip(
                "Delete ALL notes\n"
                "Hold Shift to skip confirmation.\n"
                "Undoable with the Restore All button that appears.")

    _NOTES_SEPARATOR = "----------"  # ten dashes — delimiter in TXT/MD exports

    def _import_notes_file(self):
        """Import notes from TXT, MD, or WRP — appending to existing notes."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Notes",
            "",
            "All supported (*.txt *.md *.wrp);;"
            "Text / Markdown (*.txt *.md);;"
            "WhisperR Project (*.wrp);;"
            "All files (*)")
        if not path:
            return
        try:
            ext = Path(path).suffix.lower()
            if ext == ".wrp":
                import json as _j
                data = _j.loads(Path(path).read_text(encoding="utf-8"))
                new_notes = data.get("notes", [])
                if not new_notes:
                    QMessageBox.information(
                        self, "Import Notes",
                        "No notes found in that project file.")
                    return
                for n in new_notes:
                    self._add_note(n.get("text", ""), n.get("color_idx", 0))
                self._scroll_to_note(self._notes[-1])
                QMessageBox.information(
                    self, "Import Notes",
                    f"Imported {len(new_notes)} note(s) from project file.")
            else:
                # TXT / MD — split on ten-dash separator
                raw = Path(path).read_text(encoding="utf-8")
                parts = [p.strip() for p in raw.split(self._NOTES_SEPARATOR)]
                parts = [p for p in parts if p]  # drop empty
                if not parts:
                    QMessageBox.information(
                        self, "Import Notes",
                        "No notes found in that file.\n\n"
                        "Notes should be separated by a line of ten dashes: ----------")
                    return
                for text in parts:
                    self._add_note(text, 0)
                self._scroll_to_note(self._notes[-1])
                QMessageBox.information(
                    self, "Import Notes",
                    f"Imported {len(parts)} note(s).")
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))

    def _export_notes_file(self):
        """Export notes to TXT, MD, or WRP."""
        notes = self.get_notes_data()
        if not notes:
            QMessageBox.information(
                self, "Export Notes", "No notes to export.")
            return
        path, selected = QFileDialog.getSaveFileName(
            self, "Export Notes",
            "",
            "Text file (*.txt);;"
            "Markdown (*.md);;"
            "WhisperR Project (*.wrp);;"
            "All files (*)")
        if not path:
            return
        try:
            ext = Path(path).suffix.lower()
            if ext == ".wrp":
                import json as _j
                if Path(path).exists():
                    # Existing WRP — merge notes, preserve everything else
                    existing = _j.loads(
                        Path(path).read_text(encoding="utf-8"))
                    old_notes = existing.get("notes", [])
                    existing["notes"] = old_notes + notes
                    Path(path).write_text(
                        _j.dumps(existing, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    added = len(notes)
                    total = len(existing["notes"])
                    QMessageBox.information(
                        self, "Export Notes",
                        f"Added {added} note(s) to existing file.\n"
                        f"File now contains {total} note(s) total.")
                else:
                    # New WRP — notes-only project skeleton
                    skeleton = {
                        "version":      1,
                        "text":         "",
                        "target_words": 0,
                        "notes":        notes,
                        "notes_filter": [],
                    }
                    Path(path).write_text(
                        _j.dumps(skeleton, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    QMessageBox.information(
                        self, "Export Notes",
                        f"Exported {len(notes)} note(s) to new project file.")
            else:
                # TXT / MD
                sep = f"\n{self._NOTES_SEPARATOR}\n"
                body = sep.join(n["text"] for n in notes)
                Path(path).write_text(body, encoding="utf-8")
                QMessageBox.information(
                    self, "Export Notes",
                    f"Exported {len(notes)} note(s).\n"
                    f"Notes are separated by: {self._NOTES_SEPARATOR}")
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))


    def get_notes_data(self):
        return [{"text": n.get_text(), "color_idx": n.get_color_idx()}
                for n in self._notes]

    def get_filter_state(self) -> list:
        return sorted(self._color_filter)

    def set_filter_state(self, state: list):
        self._color_filter = set(state) if state else set()
        self._apply_color_filter()


    def set_notes_data(self, data):
        # Clear undo stack — undo history is project-specific
        self._undo_stack.clear()
        self._btn_undo.setVisible(False)
        for n in list(self._notes):
            n.setParent(None)
            n.deleteLater()
        self._notes.clear()
        for item in data:
            self._add_note(item.get("text", ""), item.get("color_idx", 0))
        if not self._notes:
            self._add_note()
        if self._color_filter:
            self._apply_color_filter()

    # ── Drag infrastructure (app-level event filter approach) ─────────────

    def _start_drag(self, note: _NoteWidget):
        """Begin a drag session: set state, install event filter, update UI."""
        self._dragging_note = note
        QApplication.setOverrideCursor(Qt.CursorShape.ClosedHandCursor)
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        _eff = QGraphicsOpacityEffect(note)
        _eff.setOpacity(0.45)
        note.setGraphicsEffect(_eff)
        QApplication.instance().installEventFilter(self)

    def _end_drag(self, global_pos=None):
        """Finish or cancel the current drag session."""
        note = getattr(self, "_dragging_note", None)
        self._dragging_note = None
        QApplication.instance().removeEventFilter(self)
        QApplication.restoreOverrideCursor()
        self._drop_line.hide()
        if note:
            note.setGraphicsEffect(None)
            note._apply_color()
        if note and global_pos is not None:
            self._drag_drop(note, global_pos)

    def eventFilter(self, obj, event):
        from PyQt6.QtCore import QEvent
        note = getattr(self, "_dragging_note", None)
        if note is None:
            return False
        t = event.type()
        if t == QEvent.Type.MouseMove:
            gp = event.globalPosition().toPoint()
            self._drag_move(note, gp)
            return True
        if t == QEvent.Type.MouseButtonRelease:
            gp = event.globalPosition().toPoint()
            self._end_drag(gp)
            return True
        # Block press events so other drag buttons can't start a new drag
        if t == QEvent.Type.MouseButtonPress:
            return True
        return False

    def refresh(self):
        pass

    # ── Drag-to-reorder ────────────────────────────────────────────────
    def _drag_move(self, dragged: _NoteWidget, global_pos):
        """Show a drop-indicator line above/below target note."""
        local_y = self._inner.mapFromGlobal(global_pos).y()
        target_idx = self._drop_index(local_y, dragged)
        # Reset all note colours
        for n in self._notes:
            if n is not dragged:
                n._apply_color()
        # Position the drop-indicator line
        dl = self._drop_line
        margin = self._notes_layout.contentsMargins()
        inner_w = self._inner.width() - margin.left() - margin.right()
        if len(self._notes) > 1:
            if 0 <= target_idx < len(self._notes):
                ref_note = self._notes[target_idx]
                ref_y = ref_note.mapTo(self._inner, ref_note.rect().topLeft()).y()
                line_y = max(0, ref_y - 6)
            else:
                last = self._notes[-1]
                line_y = last.mapTo(self._inner, last.rect().bottomLeft()).y() + 2
            dl.setGeometry(margin.left(), line_y, inner_w, 5)
            dl.raise_()
            dl.show()
        else:
            dl.hide()

    def _drag_drop(self, dragged: _NoteWidget, global_pos=None):
        """Reorder notes list and rebuild layout after drop."""
        self._drop_line.hide()
        for n in self._notes:
            n._apply_color()
        # Find where the dragged note is now positioned on screen
        _gp = global_pos if global_pos else dragged.mapToGlobal(dragged.rect().center())
        local_y = self._inner.mapFromGlobal(_gp).y()
        new_idx = self._drop_index(local_y, dragged)
        old_idx = self._notes.index(dragged)
        # Clamp to valid range
        new_idx = min(new_idx, len(self._notes) - 1)
        if new_idx == old_idx:
            return
        # Reorder in _notes list
        self._notes.remove(dragged)
        self._notes.insert(new_idx, dragged)
        # Rebuild layout order
        # Remove all widgets from layout (keep trailing stretch)
        while self._notes_layout.count() > 1:
            item = self._notes_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        for note in self._notes:
            note.setParent(self._inner)
            self._notes_layout.insertWidget(
                self._notes_layout.count() - 1, note)
            note.show()

    def _drop_index(self, local_y: int, dragged: _NoteWidget) -> int:
        """Return the insertion index (in self._notes) where dragged should land.
        Computes against the visual midpoint of each non-dragged note.
        """
        others = [n for n in self._notes if n is not dragged]
        for i, n in enumerate(others):
            n_top = n.mapTo(self._inner, n.rect().topLeft()).y()
            n_mid = n_top + n.height() // 2
            if local_y < n_mid:
                # Insert before this note — find its actual index in self._notes
                return self._notes.index(n)
        # After all others — insert at end
        return len(self._notes)


class _CheatsheetWindow(QWidget):
    """Floating cheatsheet panel that attaches to the right of WhisperEditor.

    Displays three collapsible sections:
      1. Editor formatting shortcuts (buttons + hotkeys)
      2. App-level hotkeys from Settings
      3. User-defined Terms (trigger phrases only, no replacements)
    """

    def __init__(self, editor: "WhisperEditor"):
        super().__init__(editor,
                         Qt.WindowType.Window)
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
                         Qt.WindowType.Window)
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
        self._notes_win: "_NotesWindow | None" = None
        self._project_path = None  # Path to currently loaded .wrp project file

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
            "Clipboard monitor (left-click = append to text area)\n"
            "Watches clipboard continuously; each new copy is appended\n"
            "to the main text area — runs even when editor is closed.\n"
            "Right-click = Notes mode: each copy becomes a new note.\n"
            "Only one mode is active at a time.\n"
            "Exclusive with the other memory toggles.",
            checked_bg="#003a1a", checked_border="#00aa44", checked_color="#66ee88")
        self.clipboard_monitor_toggle.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        self.clipboard_monitor_toggle.customContextMenuRequested.connect(
            self._toggle_cb_notes_mode)
        # Track which mode is active: None / "text" / "notes"
        self._cb_mode: str = "text"   # updated by toggle handlers

        def _ensure_remember_on():
            """Turn on remember without firing any mutual-exclusion logic."""
            if not self.remember_toggle.isChecked():
                self.remember_toggle.blockSignals(True)
                self.remember_toggle.setChecked(True)
                self.remember_toggle.blockSignals(False)

        # remember_toggle: purely manual — never touched by other toggles
        # (no auto-off when another is enabled)

        # clipboard_prefill: mutually exclusive with monitor only
        def _on_prefill_toggled(checked):
            if checked:
                self.clipboard_monitor_toggle.blockSignals(True)
                self.clipboard_monitor_toggle.setChecked(False)
                self.clipboard_monitor_toggle.blockSignals(False)
                self.clipboard_monitor_toggle.setStyleSheet("QPushButton{}")
                self.clipboard_monitor_toggle._cb_notes_mode = False
                self._stop_clipboard_monitor()
        self.clipboard_prefill_toggle.toggled.connect(_on_prefill_toggled)

        # clipboard_monitor (left-click): text-append mode
        def _on_monitor_toggled(checked):
            if checked:
                # If currently in notes mode, switching to text mode
                _was_notes = getattr(
                    self.clipboard_monitor_toggle, "_cb_notes_mode", False)
                self.clipboard_monitor_toggle._cb_notes_mode = False
                # Turn off prefill
                self.clipboard_prefill_toggle.blockSignals(True)
                self.clipboard_prefill_toggle.setChecked(False)
                self.clipboard_prefill_toggle.blockSignals(False)
                # Apply green style for text mode
                self.clipboard_monitor_toggle.setStyleSheet(
                    "QPushButton{background:#003a1a;border:2px solid #00cc55;"
                    "color:#00ff77;border-radius:4px;padding:3px 8px;font-weight:bold;}")
                self._start_clipboard_monitor()
                _ensure_remember_on()
            else:
                # Only stop if not switching to notes mode
                if not getattr(
                        self.clipboard_monitor_toggle, "_cb_notes_mode", False):
                    self._stop_clipboard_monitor()
                    self.clipboard_monitor_toggle.setStyleSheet("QPushButton{}")
        self.clipboard_monitor_toggle.toggled.connect(_on_monitor_toggled)

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
            ("<>",    "Wrap in HTML/XML tag\nleft-click: <tag>text</tag>\nright-click: [tag]text[/tag]",
                      "editor_hk_tagwrap",   "Ctrl+Shift+W",    self._fmt_tagwrap),
            ("🔗",    "Link\nleft-click [Ctrl+K]: [text](placeholder-url)\nright-click [Ctrl+Shift+K]: [text](clipboard url)",           "editor_hk_link",      "Ctrl+K",          self._fmt_link),
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
            if label == "<>":
                btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
                btn.customContextMenuRequested.connect(
                    lambda _: self._fmt_tagwrap(square=True))
            fmt_row.addWidget(btn)

        fmt_row.addStretch()

        # Voice status indicator
        self.lbl_voice = QLabel("⏸ Voice disabled")
        self.lbl_voice.setStyleSheet("color:#666;font-size:9pt;")
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
        self.editor.textChanged.connect(self._autocorrect_terms)
        # Right-click context menu with lint corrections
        _editor_ref = self
        _orig_ctx = self.editor.contextMenuEvent
        def _ctx_menu(ev, _orig=_orig_ctx):
            from PyQt6.QtWidgets import QMenu as _QMctx
            from PyQt6.QtGui import QAction as _QActCtx
            cursor = self.editor.cursorForPosition(ev.pos())
            char_pos = cursor.position()
            lint_hit = _editor_ref._lint_at_cursor(char_pos)
            menu = self.editor.createStandardContextMenu()
            # Force proper hover highlighting on the context menu
            menu.setStyleSheet(
                "QMenu{background:#1e1e1e;color:#ddd;border:1px solid #444;}"
                "QMenu::item{padding:4px 20px;}"
                "QMenu::item:selected{background:#1a3a5c;}"
                "QMenu::item:hover{background:#1a3a5c;}"
                "QMenu::item:pressed{background:#0d2a4a;}"
                "QMenu::separator{height:1px;background:#333;margin:2px 0;}"
                "QMenu::item:disabled{color:#666;}")
            if lint_hit:
                msg, suggestions = lint_hit
                # Find the error span so we can highlight the word
                _err_start = _err_end = None
                for (st, en, _m, _sg) in _editor_ref._lint_errors:
                    if st <= char_pos < en:
                        _err_start, _err_end = st, en
                        break
                if _err_start is not None:
                    # Select the error word so user sees which word suggestions are for
                    _sel = self.editor.textCursor()
                    _sel.setPosition(_err_start)
                    _sel.setPosition(_err_end, _sel.MoveMode.KeepAnchor)
                    self.editor.setTextCursor(_sel)
                    menu.insertSeparator(menu.actions()[0])
                    _info = _QActCtx(f"⚠️ {msg}", menu)
                    _info.setEnabled(False)
                    menu.insertAction(menu.actions()[0], _info)
                    for sugg in suggestions:
                        def _make_fix(_checked=False, s=sugg, pos=char_pos):
                            # Find the error span and replace it
                            for (st, en, _m, _sg) in _editor_ref._lint_errors:
                                if st <= pos < en:
                                    cur = self.editor.textCursor()
                                    cur.setPosition(st)
                                    cur.setPosition(en,
                                        cur.MoveMode.KeepAnchor)
                                    cur.insertText(s)
                                    break
                        act = _QActCtx(f"✓ {sugg}", menu)
                        act.triggered.connect(_make_fix)
                        menu.insertAction(menu.actions()[2], act)
            # ── Quick-add from selection ──────────────────────────
            sel_text = self.editor.textCursor().selectedText().strip()
            if sel_text:
                menu.addSeparator()
                from PyQt6.QtWidgets import QApplication as _QAppc
                _app_w = next((w for w in _QAppc.topLevelWidgets()
                               if w.__class__.__name__ == "WhisperRApp"), None)

                def _make_hw_adder(t, aw):
                    def _do(_checked=False):
                        if not aw: return
                        hw = list(aw.config.settings.get("hotwords", []))
                        if t not in hw:
                            hw.append(t)
                            aw.config.settings["hotwords"] = hw
                            he = getattr(aw, "hotwords_edit", None)
                            if he: he.setPlainText("\n".join(hw))
                            aw.config.save()
                    return _do

                def _make_hall_adder(t, aw):
                    def _do(_checked=False):
                        if not aw: return
                        hall = list(aw.config.settings.get("hallucinations", []))
                        if t.lower() not in [h.lower() for h in hall]:
                            hall.append(t)
                            aw.config.settings["hallucinations"] = hall
                            hl = getattr(aw, "hall_list", None)
                            if hl:
                                from PyQt6.QtWidgets import QListWidgetItem as _LWI
                                hl.addItem(_LWI(t))
                            aw.config.save()
                    return _do

                def _make_term_adder_recog(t, aw):
                    def _do(_checked=False):
                        if not aw: return
                        from PyQt6.QtWidgets import QInputDialog, QLineEdit
                        repl, ok = QInputDialog.getText(
                            None, "Add Term Pair",
                            f'Replacement text for recognized phrase "{t}":',
                            QLineEdit.EchoMode.Normal, "")
                        if not ok or not repl.strip(): return
                        terms = dict(aw.config.settings.get("terms", {}))
                        terms[t.lower()] = repl.strip()
                        aw.config.settings["terms"] = terms
                        tt = getattr(aw, "terms_table", None)
                        if tt:
                            from PyQt6.QtWidgets import QTableWidgetItem
                            r = tt.rowCount(); tt.insertRow(r)
                            tt.setItem(r, 0, QTableWidgetItem(t.lower()))
                            tt.setItem(r, 1, QTableWidgetItem(repl.strip()))
                        aw.config.save()
                    return _do

                def _make_term_adder_repl(t, aw):
                    def _do(_checked=False):
                        if not aw: return
                        from PyQt6.QtWidgets import QInputDialog, QLineEdit
                        phrase, ok = QInputDialog.getText(
                            None, "Add Term Pair",
                            f'Recognized phrase that becomes "{t}":',
                            QLineEdit.EchoMode.Normal, "")
                        if not ok or not phrase.strip(): return
                        terms = dict(aw.config.settings.get("terms", {}))
                        terms[phrase.strip().lower()] = t
                        aw.config.settings["terms"] = terms
                        tt = getattr(aw, "terms_table", None)
                        if tt:
                            from PyQt6.QtWidgets import QTableWidgetItem
                            r = tt.rowCount(); tt.insertRow(r)
                            tt.setItem(r, 0, QTableWidgetItem(phrase.strip().lower()))
                            tt.setItem(r, 1, QTableWidgetItem(t))
                        aw.config.save()
                    return _do

                _lbl = sel_text[:25] + ("…" if len(sel_text) > 25 else "")
                _ah = _QActCtx(f'📖 Add "{_lbl}" to Vocabulary Boost', menu)
                _ah.triggered.connect(_make_hw_adder(sel_text, _app_w))
                menu.addAction(_ah)

                _ahal = _QActCtx(f'🚫 Add "{_lbl}" to Hallucinations', menu)
                _ahal.triggered.connect(_make_hall_adder(sel_text, _app_w))
                menu.addAction(_ahal)

                _ar = _QActCtx(f'🔁 Add "{_lbl}" as Recognized Phrase…', menu)
                _ar.triggered.connect(_make_term_adder_recog(sel_text, _app_w))
                menu.addAction(_ar)

                _arp = _QActCtx(f'🔁 Add "{_lbl}" as Replacement Text…', menu)
                _arp.triggered.connect(_make_term_adder_repl(sel_text, _app_w))
                menu.addAction(_arp)

            menu.exec(ev.globalPos())
        self.editor.contextMenuEvent = _ctx_menu
        # Debounced history snapshot
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(5000)
        self._history_timer.timeout.connect(self._push_history)
        self._history_timer.timeout.connect(self._snap_on_editor_change)
        self.editor.textChanged.connect(lambda: self._history_timer.start())
        # Harper LSP — start if binary available; textChanged drives debounced check
        self._lint_errors: list = []  # [(start,end,msg,[sugg])]
        self._harper_client = None
        self._harper_uri    = "file:///whisperr_editor_doc"
        # Start Harper automatically if the binary is present
        QTimer.singleShot(500, self._start_harper)
        # MSS state — next segment starts capitalised by default
        self._mss_next_capital = True
        # textChanged → debouncing is done inside HarperLSPClient
        self.editor.textChanged.connect(self._run_lint)
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

        # ── Menu bar ──────────────────────────────────────────────────────
        from PyQt6.QtWidgets import QMenuBar as _QMB
        _menubar = _QMB(self)
        _menubar.setStyleSheet(
            "QMenuBar{background:#1e1e1e;color:#ddd;border-bottom:1px solid #333;}"
            "QMenuBar::item{padding:4px 10px;background:transparent;}"
            "QMenuBar::item:selected{background:#2a2a2a;}"
            "QMenu{background:#1e1e1e;color:#ddd;border:1px solid #444;}"
            "QMenu::item{padding:4px 20px;}"
            "QMenu::item:selected{background:#1a3a5c;}"
            "QMenu::separator{height:1px;background:#333;margin:2px 0;}")

        # Helper: create a QAction with optional shortcut and add to menu
        from PyQt6.QtGui import QAction as _QAct, QKeySequence as _QKS
        def _ma(menu, label, slot, shortcut=None):
            act = _QAct(label, self)
            act.triggered.connect(slot)
            if shortcut:
                act.setShortcut(_QKS(shortcut))
            menu.addAction(act)
            return act

        # File menu
        _m_file = _menubar.addMenu("File")
        _ma(_m_file, "New Project",         self._new_project,      "Ctrl+N")
        _m_file.addSeparator()
        _ma(_m_file, "Load Project…",     self._load_project,     "Ctrl+O")
        _ma(_m_file, "Import Text File…",  self._import_file)
        _m_file.addSeparator()
        _ma(_m_file, "Save Project",         self._save_project,     "Ctrl+S")
        _ma(_m_file, "Export Text File…",  self._export_file)

        # Edit menu
        _m_edit = _menubar.addMenu("Edit")
        _ma(_m_edit, "Copy All",             self._copy_all,         "Ctrl+Shift+C")
        _ma(_m_edit, "Find && Replace",      self._show_find_replace, "Ctrl+H")
        _m_edit.addSeparator()
        from PyQt6.QtGui import QAction as _QActSC
        self._harper_menu_act = _QActSC("Spell && Grammar Checking: OFF", self)
        self._harper_menu_act.setCheckable(True)
        self._harper_menu_act.setChecked(False)
        self._harper_menu_act.triggered.connect(self._toggle_harper)
        _m_edit.addAction(self._harper_menu_act)

        # History menu (project version history — populated dynamically)
        self._m_history = _menubar.addMenu("History")
        self._m_history.aboutToShow.connect(self._populate_history_menu)

        # Snapshots menu (app-state snapshots — populated dynamically)
        self._m_snapshots = _menubar.addMenu("Snapshots")
        self._m_snapshots.aboutToShow.connect(self._populate_snapshots_menu)

        root.setMenuBar(_menubar)

        _btn_ss = ("QPushButton{background:#2a2a2a;border:1px solid #444;padding:5px 10px;"
                   "border-radius:4px;color:#ddd;}"
                   "QPushButton:hover{background:#353535;border-color:#0078d7;}")

        def _dual_btn(label_l, tip_l, slot_l, tip_r, slot_r):
            b = QPushButton(label_l)
            b.setToolTip(f"{tip_l}\nRight-click: {tip_r}")
            b.clicked.connect(slot_l)
            b.setStyleSheet(_btn_ss)
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda _: slot_r())
            return b

        def _btn(label, tip, slot):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            b.setStyleSheet(_btn_ss)
            return b

        # ── New Project ────────────────────────────────────────────
        btn_row.addWidget(_btn("✨ New", "New project — clears text and notes",
                               self._new_project))

        # ── Format Preset ─────────────────────────────────────────────────
        from PyQt6.QtWidgets import QComboBox as _QCBe
        self._preset_combo = _QCBe()
        self._preset_combo.addItem("📋 Preset…")
        for _pname in EDITOR_PRESETS:
            self._preset_combo.addItem(_pname)
        self._preset_combo.setToolTip(
            "Insert a template into the current editor.\n"
            "Replaces content only if editor is empty (or on New project).")
        self._preset_combo.setStyleSheet(
            "QComboBox{background:#2a2a2a;border:1px solid #444;"
            "padding:4px 8px;color:#ddd;border-radius:4px;}"
            "QComboBox:hover{border-color:#0078d7;}")
        self._preset_combo.currentTextChanged.connect(self._apply_preset)
        btn_row.addWidget(self._preset_combo)

        # ── Load / Import (dual) ────────────────────────────────────────────
        btn_row.addWidget(_dual_btn(
            "📂 Load",
            "Load a project file (.wrp) — restores text, notes and settings",
            self._load_project,
            "Import a .txt or .md file into the text area",
            self._import_file))

        # ── Save / Export (dual) ────────────────────────────────────────────
        btn_row.addWidget(_dual_btn(
            "💾 Save",
            "Save project to file (.wrp) — text, notes and settings",
            self._save_project,
            "Export text to a .txt or .md file",
            self._export_file))

        btn_row.addWidget(_btn("📋 Copy", "Copy all text to clipboard", self._copy_all))
        btn_row.addWidget(_btn("🔍 Find",
            "Find & Replace  [Ctrl+H]\n"
            "Close bar by pressing Ctrl+H again or clicking ✕.",
            self._show_find_replace))
        btn_row.addWidget(_btn("🕐 History",
            "Version history  [Ctrl+Alt+H]\n"
            "Snapshots taken automatically 5s after typing stops and on every save.",
            self._show_history))
        btn_row.addStretch()

        # ── Notes / Cheatsheet (dual) ────────────────────────────────────
        self.btn_notes = QPushButton("📝 Notes")
        self.btn_notes.setToolTip(
            "Show / hide the Notes panel\nRight-click: show / hide Cheatsheet")
        self.btn_notes.setCheckable(True)
        self.btn_notes.setStyleSheet(_btn_ss)
        self.btn_notes.clicked.connect(self._toggle_notes)
        self.btn_notes.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.btn_notes.customContextMenuRequested.connect(lambda _: self._toggle_cheatsheet())
        btn_row.addWidget(self.btn_notes)
        # Keep btn_cheatsheet alias so existing WhisperRApp code still works
        self.btn_cheatsheet = self.btn_notes
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

        # ── Harper spell-check indicator ────────────────────────────────
        self._harper_indicator = QPushButton("📝")
        self._harper_indicator.setFixedSize(32, 28)
        self._harper_indicator.setToolTip(
            "Spell & grammar checking: initialising…\n"
            "Go to Settings → Optional Tools to install Harper.")
        self._harper_indicator.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;"
            "border-radius:4px;color:#666;font-size:13px;padding:2px 6px;}"
            "QPushButton:hover{background:#353535;border-color:#888;}")
        self._harper_indicator.clicked.connect(self._toggle_harper)
        btn_row.addWidget(self._harper_indicator)

        root.addLayout(btn_row)


    # ── Positioning ───────────────────────────────────────────────────────────

    # ── Cheatsheet ────────────────────────────────────────────────────────────

    def _toggle_notes(self):
        """Show or hide the Notes panel, preserving notes across hide/show."""
        if self._notes_win and self._notes_win.isVisible():
            # Snapshot notes before hiding so they survive
            self._saved_notes_snapshot = self._notes_win.get_notes_data()
            self._notes_win.hide()
            self.btn_notes.setChecked(False)
        else:
            if not self._notes_win:
                self._notes_win = _NotesWindow(self)
            # Restore snapshot if available (keep snapshot in sync, don't clear)
            snap = getattr(self, "_saved_notes_snapshot", None)
            if snap:
                self._notes_win.set_notes_data(snap)
            self._reposition_panels()
            self._notes_win.show()
            self._notes_win.raise_()
            self.btn_notes.setChecked(True)
            # Apply always-on-top setting
            _host_n = next((w for w in QApplication.topLevelWidgets()
                            if w.__class__.__name__ == "WhisperRApp"), None)
            if _host_n and self.config.settings.get("aot_notes", False):
                from PyQt6.QtCore import Qt as _Qt_n
                f = self._notes_win.windowFlags() | _Qt_n.WindowType.WindowStaysOnTopHint
                self._notes_win.setWindowFlags(f)
                self._notes_win.show()
        self._reposition_panels()

    def _toggle_cheatsheet(self):
        """Show or hide the floating cheatsheet window."""
        if self._cheatsheet and self._cheatsheet.isVisible():
            self._cheatsheet.hide()
        else:
            if not self._cheatsheet:
                self._cheatsheet = _CheatsheetWindow(self)
            self._reposition_panels()
            self._cheatsheet.show()
            self._cheatsheet.raise_()
        self._reposition_panels()

    def _reposition_cheatsheet(self):
        """Legacy alias — delegates to _reposition_panels."""
        self._reposition_panels()

    def _reposition_panels(self):
        """Position Notes and/or Cheatsheet panels beside the editor.

        Each panel is half the editor width.
        Notes sits flush-right of editor; Cheatsheet right of Notes (or editor).
        """
        geo       = self.frameGeometry()
        notes_vis = bool(self._notes_win and self._notes_win.isVisible())
        cs_vis    = bool(self._cheatsheet and self._cheatsheet.isVisible())
        if not notes_vis and not cs_vis:
            return
        panel_w = max(self.width() // 2, 300)
        panel_h = self.height()
        x = geo.right() + 2
        if notes_vis:
            self._notes_win.resize(panel_w, panel_h)
            self._notes_win.move(x, geo.top())
            x += panel_w + 2
        if cs_vis:
            self._cheatsheet.resize(panel_w, panel_h)
            self._cheatsheet.move(x, geo.top())

    def moveEvent(self, event):
        super().moveEvent(event)
        self._reposition_panels()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_panels()

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

    def _autocorrect_terms(self):
        """Check if the last typed word matches a Term phrase; expand if so.
        Only fires for exact-match single-word terms (no spaces) to avoid
        interfering with normal typing of multi-word phrases mid-sentence.
        Multi-word terms are matched when the user types the final word followed
        by a space or punctuation.
        """
        cfg = self.config if isinstance(self.config, dict) else getattr(self.config, "settings", {})
        terms = cfg.get("terms", {}) if isinstance(cfg, dict) else {}
        if not terms:
            return
        cur = self.editor.textCursor()
        # Only trigger when the character just typed is a space, Enter, or punctuation
        pos = cur.position()
        text = self.editor.toPlainText()
        if pos == 0 or not text:
            return
        last_char = text[pos - 1] if pos <= len(text) else ""
        if last_char not in (" ", "\n", "\t", ".", ",", "!", "?", ";", ":"):
            return
        # Get word(s) just before the trigger character
        before = text[:pos - 1].rstrip()
        for phrase, replacement in terms.items():
            phrase_l = phrase.strip().lower()
            if not phrase_l:
                continue
            if before.lower().endswith(phrase_l):
                # Check it's at a word boundary (preceded by space/start or nothing)
                end_idx = len(before)
                start_idx = end_idx - len(phrase_l)
                if start_idx > 0 and before[start_idx - 1] not in (" ", "\n", "\t"):
                    continue  # not a word boundary
                # Replace: select the phrase and insert replacement
                # Block signal to avoid re-triggering
                self.editor.textChanged.disconnect(self._autocorrect_terms)
                try:
                    sel = self.editor.textCursor()
                    sel.setPosition(start_idx)
                    sel.setPosition(end_idx, sel.MoveMode.KeepAnchor)
                    sel.insertText(replacement)
                    self.editor.setTextCursor(sel)
                finally:
                    self.editor.textChanged.connect(self._autocorrect_terms)
                break

    def _fmt_tagwrap(self, square=False):
        """Prompt for a tag name, then wrap selection in <tag>…</tag> or [tag]…[/tag]."""
        from PyQt6.QtWidgets import QInputDialog
        style = "[]" if square else "<>"
        tag, ok = QInputDialog.getText(
            self, "Tag Wrap",
            f"Enter tag name ({style[0]}tag{style[1]}):")
        if not ok or not tag.strip():
            return
        tag = tag.strip()
        if square:
            self._wrap_selection(f"[{tag}]", f"[/{tag}]")
        else:
            self._wrap_selection(f"<{tag}>", f"</{tag}>")
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

    def _toggle_cb_notes_mode(self):
        """Right-click on clipboard monitor → toggle Notes mode.
        Right-click ON  → notes mode (blue), turn off text mode if active.
        Right-click OFF → stop monitor, leave remember unchanged.
        """
        host = self._get_host()
        if not host:
            return
        _btn = self.clipboard_monitor_toggle
        in_notes_mode = getattr(_btn, "_cb_notes_mode", False)
        if in_notes_mode:
            # Turn off notes mode — stop monitor, clear style, leave remember
            _btn._cb_notes_mode = False
            host.stop_clipboard_monitor()
            _btn.blockSignals(True)
            _btn.setChecked(False)
            _btn.blockSignals(False)
            _btn.setStyleSheet("QPushButton{}")
            return
        # Activate notes mode
        _btn._cb_notes_mode = True
        # Stop text-mode monitor if running (mode will change to notes)
        if host._cb_monitor_timer and host._cb_monitor_timer.isActive():
            host.stop_clipboard_monitor()
        # Turn off prefill
        self.clipboard_prefill_toggle.blockSignals(True)
        self.clipboard_prefill_toggle.setChecked(False)
        self.clipboard_prefill_toggle.blockSignals(False)
        # Set checked and apply blue style — block signals so _on_monitor_toggled
        # does not fire (we handle start manually here)
        _btn.blockSignals(True)
        _btn.setChecked(True)
        _btn.blockSignals(False)
        _btn.setStyleSheet(
            "QPushButton{background:#001a40;border:2px solid #0088ff;"
            "color:#44bbff;border-radius:4px;padding:3px 8px;font-weight:bold;}")
        host.start_clipboard_monitor(mode="notes")
        # Ensure remember is on (never turn it off)
        if hasattr(self, "remember_toggle") and not self.remember_toggle.isChecked():
            self.remember_toggle.blockSignals(True)
            self.remember_toggle.setChecked(True)
            self.remember_toggle.blockSignals(False)

    def _get_host(self):
        app = QApplication.instance()
        return next((w for w in app.topLevelWidgets()
                     if w.__class__.__name__ == "WhisperRApp"), None)

    def _start_clipboard_monitor(self):
        """Delegate to the host app so the monitor survives hide/close."""
        host = self._get_host()
        if host:
            host.start_clipboard_monitor(mode="text")

    def _stop_clipboard_monitor(self):
        """Delegate stop to the host app."""
        host = self._get_host()
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
            (cfg.get("editor_hk_tagwrap",   "Ctrl+Shift+W"),    self._fmt_tagwrap),
            (cfg.get("editor_hk_link",      "Ctrl+K"),          self._fmt_link),
            # Ctrl+Shift+K registered separately with ApplicationShortcut
            # so it works even when focus is inside the text widget
            ("Ctrl+H",                                           self._show_find_replace),
            ("Ctrl+Y",                                           self.editor.redo),
            ("Ctrl+Alt+H",                                       self._show_history),
        ]
        for keys, slot in shortcuts:
            try:
                sc = QShortcut(QKeySequence(keys), self)
                sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                sc.activated.connect(slot)
                self._hotkeys_active.append(sc)
            except Exception:
                pass
        # Ctrl+Shift+K: paste clipboard as URL — needs WindowShortcut so
        # it fires even when the QTextEdit has focus and captures keypresses
        try:
            _sc_url = QShortcut(QKeySequence("Ctrl+Shift+K"), self)
            _sc_url.setContext(Qt.ShortcutContext.WindowShortcut)
            _sc_url.activated.connect(lambda: self._fmt_link(use_clipboard=True))
            self._hotkeys_active.append(_sc_url)
        except Exception:
            pass

    # ── Voice operations (called from WhisperRApp.on_text) ────────────────────

    # ── Manual Sentence Splitting ─────────────────────────────────────────────
    #
    # When enabled: each dictated segment is "stitched" — trailing punctuation
    # is stripped and the next segment starts lowercase, all joined with a space.
    # Pressing the "sentence break" key (default: Left Shift, configurable) marks
    # the END of a sentence: a full stop is appended and the NEXT segment begins
    # with a capital letter.
    #
    # State: self._mss_next_capital (bool) — True = next text starts uppercase

    def _mss_apply(self, text: str) -> str:
        """Apply Manual Sentence Splitting rules to incoming text.

        Strips only sentence-ENDING punctuation (.!?…) from the tail.
        Mid-sentence marks (commas, parentheses, colons, semicolons) are kept.
        Capitalises the first letter if _mss_next_capital is True (start of
        a new sentence), otherwise forces it lowercase (mid-sentence continuation).
        After applying, _mss_next_capital is reset to False.
        """
        import re as _re_mss
        # Strip ONLY sentence-ending punctuation from the tail.
        # Commas, colons, semicolons, parentheses are intentionally kept.
        stripped = _re_mss.sub(r'[.!?\u2026]+$', '', text.rstrip()).rstrip()
        if not stripped:
            return ''

        if getattr(self, '_mss_next_capital', True):
            result = stripped[0].upper() + stripped[1:]
        else:
            result = stripped[0].lower() + stripped[1:]

        self._mss_next_capital = False
        return result


    def _mss_sentence_break(self):
        """Insert a period at the end of the current text and prepare
        for the next sentence to start with a capital letter.
        Called when the user presses the sentence-break key during dictation.
        """
        self._mss_next_capital = True
        # Move cursor to document end so period is always appended at the end
        cur = self.editor.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        doc_text = self.editor.toPlainText()
        last_non_space = doc_text.rstrip()
        if last_non_space and last_non_space[-1] not in '.!?…':
            # Strip any trailing spaces first, then add ". "
            while doc_text.endswith(" "):
                cur.deletePreviousChar()
                doc_text = doc_text[:-1]
            cur.insertText(". ")
        self.editor.setTextCursor(cur)
        self.editor.ensureCursorVisible()


    def append_text(self, text: str):
        """Insert dictated text at the current cursor position.

        - If text is selected, the selection is replaced.
        - If cursor is at end of non-empty, non-newline text, a space is prepended.
        - Otherwise text is inserted exactly at the cursor.
        - If Manual Sentence Splitting is enabled, text is transformed first.
        """
        cfg = getattr(self, "config", None)
        if cfg and cfg.settings.get("manual_sentence_split", False):
            text = self._mss_apply(text)
            if not text:
                return
        cur = self.editor.textCursor()
        # Wrap in an edit block so each append_text call is ONE undo step.
        # Without this, Qt merges consecutive insertions, making Ctrl+Z
        # undo multiple transcribed sentences at once.
        cur.beginEditBlock()
        if cur.hasSelection():
            cur.insertText(text)
        else:
            pos = cur.position()
            doc_text = self.editor.toPlainText()
            if pos > 0 and doc_text[pos - 1] not in (" ", "\n", "\t"):
                text = " " + text
            cur.insertText(text)
        cur.endEditBlock()
        self.editor.setTextCursor(cur)
        # Only scroll to cursor — don't force visual focus jump
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

    # ── Project load / save ───────────────────────────────────────────────────

    def _project_data(self) -> dict:
        """Snapshot the full editor state as a JSON-serialisable dict."""
        _ts = getattr(self, "target_spin", None)
        if _ts:
            _ts.interpretText()
        # Collect notes from live window if it exists, else from snapshot
        if self._notes_win:
            _notes = self._notes_win.get_notes_data()
        else:
            _notes = list(getattr(self, "_saved_notes_snapshot", []))
        return {
            "version": 1,
            "text":         self.editor.toPlainText(),
            "target_words": _ts.value() if _ts else 0,
            "notes":        _notes,
            "notes_filter": (self._notes_win.get_filter_state()
                             if self._notes_win else []),
        }

    def _apply_project_data(self, data: dict, project_path=None):
        """Restore editor state from a project dict."""
        self.editor.setPlainText(data.get("text", ""))
        _ts = getattr(self, "target_spin", None)
        if _ts:
            _ts.setValue(int(data.get("target_words", 0)))
        notes = data.get("notes", [])
        if notes:
            if not self._notes_win:
                self._notes_win = _NotesWindow(self)
            self._notes_win.set_notes_data(notes)
            # Show notes window immediately if not already visible
            if not self._notes_win.isVisible():
                self._notes_win.show()
                self._notes_win.raise_()
                _btn_n_p = getattr(self, "btn_notes", None)
                if _btn_n_p: _btn_n_p.setChecked(True)
                self._reposition_panels()
            # Cache so re-opening notes panel shows the data
            self._saved_notes_snapshot = list(notes)
        _filt = data.get('notes_filter', [])
        if _filt and self._notes_win:
            self._notes_win.set_filter_state(_filt)
        self._project_path = Path(project_path) if project_path else None
        # Reset and reload history from .wrp.history file
        self._version_history = []
        self._history_dirty_count = 0
        from datetime import datetime as _dt_lh
        self._history_last_flush = _dt_lh.now()
        self._load_history_from_project()
        self._update_title()
        # Turn on remember so closing doesn't lose the project
        rt = getattr(self, "remember_toggle", None)
        if rt:
            rt.setChecked(True)
        self._start_auto_backup()

    def _update_title(self):
        if self._project_path:
            self.setWindowTitle(f"WhisperR Editor — {self._project_path.stem}")
        else:
            self.setWindowTitle("WhisperR Editor")

    def _apply_preset(self, name: str):
        """Apply a format preset template to the editor."""
        if name == "📋 Preset…" or name not in EDITOR_PRESETS:
            return
        tpl = EDITOR_PRESETS[name]
        current = self.editor.toPlainText().strip()
        if current:
            reply = QMessageBox.question(
                self, "Apply Preset",
                f"Apply the \"{name}\" template?\n\n"
                "Current text will be replaced.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                # Reset combo without triggering again
                self._preset_combo.blockSignals(True)
                self._preset_combo.setCurrentIndex(0)
                self._preset_combo.blockSignals(False)
                return
        self.editor.setPlainText(tpl)
        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentIndex(0)
        self._preset_combo.blockSignals(False)
        # Move cursor to end of first line for immediate editing
        cur = self.editor.textCursor()
        cur.movePosition(cur.MoveOperation.End)
        self.editor.setTextCursor(cur)

    def _new_project(self):
        """Reset editor to blank slate — clear text, notes, project path."""
        reply = QMessageBox.question(
            self, "New Project",
            "Clear all text and notes and start a new project?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.editor.clear()
        _ts = getattr(self, "target_spin", None)
        if _ts: _ts.setValue(0)
        if self._notes_win:
            self._notes_win.set_notes_data([])
        self._saved_notes_snapshot = []
        self._project_path = None
        self._update_title()

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", "",
            "WhisperR Project (*.wrp);;All files (*)")
        if not path:
            return
        try:
            import json as _j
            data = _j.loads(Path(path).read_text(encoding="utf-8"))
            self._apply_project_data(data, path)
        except Exception as e:
            QMessageBox.warning(self, "Load failed", str(e))

    def _save_project(self):
        self._push_history()          # snapshot before save
        self._flush_history_to_project()  # flush dirty entries to .wrp.history
        init = str(self._project_path) if self._project_path else ""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", init,
            "WhisperR Project (*.wrp);;All files (*)")
        if not path:
            return
        if not path.endswith(".wrp"):
            path += ".wrp"
        try:
            import json as _j
            Path(path).write_text(
                _j.dumps(self._project_data(), ensure_ascii=False, indent=2),
                encoding="utf-8")
            self._project_path = Path(path)
            self._update_title()
            self._start_auto_backup()
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))


    # ── Version history ───────────────────────────────────────────────────────

    _MAX_HISTORY = 20   # overridden by config at runtime

    def _populate_history_menu(self):
        """Populate the History menu with version history entries."""
        from PyQt6.QtCore import Qt as _Qt3
        from datetime import datetime as _dt
        from collections import defaultdict as _dd
        m = self._m_history
        m.clear()
        m.addAction("Open History Picker… (Ctrl+Alt+H)", self._show_history)
        history = getattr(self, "_version_history", [])
        if not history:
            m.addSeparator()
            m.addAction("No snapshots yet").setEnabled(False)
            return
        m.addSeparator()
        by_day = _dd(list)
        for i, e in enumerate(history):
            try: dt = _dt.fromisoformat(e["ts"])
            except Exception: dt = _dt.now()
            by_day[dt.strftime("%Y-%m-%d")].append((i, dt, e))
        all_days = sorted(by_day.keys())
        span_days = ((_dt.fromisoformat(all_days[-1]) -
                      _dt.fromisoformat(all_days[0])).days + 1
                     if len(all_days) > 1 else 1)
        def _add_entry(parent, i, dt, e):
            w = e.get("words", len(e["text"].split()))
            label = f"{dt.strftime("%H:%M:%S")}  —  {w}w  —  {e["text"][:40].replace(chr(10)," ")}..."
            from PyQt6.QtGui import QAction as _QActH
            act = _QActH(label, parent)
            act.triggered.connect(lambda _c=False, _i=i: (
                self._push_history(),
                self.editor.setPlainText(history[_i]["text"])))
            parent.addAction(act)
        if span_days <= 1:
            entries = by_day[all_days[0]]
            for i, dt, e in reversed(entries[-20:] if len(entries) > 20
                                      else entries):
                _add_entry(m, i, dt, e)
        elif span_days <= 7:
            for day in reversed(all_days):
                sub = m.addMenu(_dt.fromisoformat(day).strftime("%A, %b %d"))
                for i, dt, e in reversed(by_day[day]):
                    _add_entry(sub, i, dt, e)
        else:
            from collections import defaultdict as _dd4
            by_month = _dd4(lambda: _dd(list))
            for day in all_days:
                mo = _dt.fromisoformat(day).strftime("%Y-%m")
                by_month[mo][day].extend(by_day[day])
            for mo in reversed(sorted(by_month.keys())):
                mo_dt = _dt.strptime(mo + "-01", "%Y-%m-%d")
                mo_sub = m.addMenu(mo_dt.strftime("%B %Y"))
                for day in reversed(sorted(by_month[mo].keys())):
                    day_sub = mo_sub.addMenu(
                        _dt.fromisoformat(day).strftime("%a %b %d"))
                    for i, dt, e in reversed(by_month[mo][day]):
                        _add_entry(day_sub, i, dt, e)

    def _populate_snapshots_menu(self):
        """Populate the Snapshots menu. Delegates to WhisperRApp."""
        m = self._m_snapshots
        m.clear()
        # Find parent WhisperRApp
        _app_w = next((w for w in QApplication.topLevelWidgets()
                       if w.__class__.__name__ == "WhisperRApp"), None)
        m.addAction("Open Snapshots Picker…",
            lambda: _app_w._show_app_snapshots() if _app_w else None)
        if not _app_w:
            return
        import json as _j
        from datetime import datetime as _dt
        from collections import defaultdict as _dd
        snaps = []
        sp = Path(BASE_DIR) / "whisperr_snapshots.jsonl"
        if sp.exists():
            for line in sp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try: snaps.append(_j.loads(line))
                    except Exception: pass
        ram = getattr(_app_w, "_app_snapshots_ram", [])
        on_disk = {s["ts"] for s in snaps}
        snaps += [s for s in ram if s["ts"] not in on_disk]
        snaps.sort(key=lambda s: s["ts"])
        if not snaps:
            m.addSeparator()
            m.addAction("No snapshots yet").setEnabled(False)
            return
        m.addSeparator()
        by_day = _dd(list)
        for i, s in enumerate(snaps):
            try: dt = _dt.fromisoformat(s["ts"])
            except Exception: dt = _dt.now()
            by_day[dt.strftime("%Y-%m-%d")].append((i, dt, s))
        all_days  = sorted(by_day.keys())
        span_days = ((_dt.fromisoformat(all_days[-1]) -
                      _dt.fromisoformat(all_days[0])).days + 1
                     if len(all_days) > 1 else 1)
        def _add_snap(parent, i, dt, s):
            txt = s.get("editor_text", "")
            label = (f"{dt.strftime("%H:%M:%S")}  —  {len(txt.split())}w  —  "
                     f"{txt[:35].replace(chr(10)," ")}...")
            from PyQt6.QtGui import QAction as _QActSn
            act = _QActSn(label, parent)
            act.triggered.connect(
                lambda _c=False, _s=s: _app_w._restore_app_snapshot(_s))
            parent.addAction(act)
        if span_days <= 1:
            entries = by_day[all_days[0]]
            for i, dt, s in reversed(entries[-20:] if len(entries) > 20
                                      else entries):
                _add_snap(m, i, dt, s)
        elif span_days <= 7:
            for day in reversed(all_days):
                sub = m.addMenu(_dt.fromisoformat(day).strftime("%A, %b %d"))
                for i, dt, s in reversed(by_day[day]):
                    _add_snap(sub, i, dt, s)
        else:
            from collections import defaultdict as _dd5
            by_month = _dd5(lambda: _dd(list))
            for day in all_days:
                mo = _dt.fromisoformat(day).strftime("%Y-%m")
                by_month[mo][day].extend(by_day[day])
            for mo in reversed(sorted(by_month.keys())):
                mo_dt = _dt.strptime(mo + "-01", "%Y-%m-%d")
                mo_sub = m.addMenu(mo_dt.strftime("%B %Y"))
                for day in reversed(sorted(by_month[mo].keys())):
                    day_sub = mo_sub.addMenu(
                        _dt.fromisoformat(day).strftime("%a %b %d"))
                    for i, dt, s in reversed(by_month[mo][day]):
                        _add_snap(day_sub, i, dt, s)

    def _start_harper(self):
        """Start the Harper LSP client for this editor document."""
        try:
            binary = _harper_binary_path()
            harper_log(f"_start_harper: binary={binary!r}")
            app_logger.info(f"_start_harper: binary={binary!r}")
            # Guard: if already running, nothing to do
            existing = getattr(self, "_harper_client", None)
            if existing is not None:
                try:
                    if existing.running():
                        harper_log("_start_harper: already running")
                        app_logger.debug("_start_harper: already running")
                        return
                except Exception as _re:
                    harper_log(f"_start_harper: existing client check failed: {_re}")
                    app_logger.warning(f"_start_harper: existing client check failed: {_re}")
            harper_log("_start_harper: creating HarperLSPClient")
            app_logger.info("_start_harper: creating HarperLSPClient")
            try:
                client = HarperLSPClient(
                    on_diagnostics=self._apply_lint_results,
                    linters_callback=lambda: self.config.settings.get("harper", {}).get("linters", {}),
                )
            except Exception as _ce:
                harper_log(f"_start_harper: client creation failed: {_ce}")
                app_logger.error(f"_start_harper: client creation failed: {_ce}", exc_info=True)
                return
            harper_log("_start_harper: calling client.start()")
            app_logger.info("_start_harper: HarperLSPClient created, calling start()")
            ok = client.start()
            harper_log(f"_start_harper: client.start() returned {ok}")
            app_logger.info(f"_start_harper: client.start() returned {ok}")
            if ok:
                self._harper_client = client
                # Use a file: URI within the workspace root so harper-ls
                # document filters match. The file does not need to exist
                # on disk — LSP carries the full content in open/change
                # notifications. This is standard LSP behaviour.
                # Simple in-memory document URI — no app-dir dependency.
                # rootUri is file:/// so document must also use file: scheme.
                # Document URI must be inside the workspace root
                _app_dir_e = (os.path.dirname(sys.executable)
                              if getattr(sys, "frozen", False)
                              else os.path.dirname(os.path.abspath(__file__)))
                _doc_base = ("file:///" +
                    _app_dir_e.replace("\\", "/").lstrip("/"))
                uri = _doc_base + "/whisperr-document.md"
                _doc_text = self.editor.toPlainText()
                self._harper_client.open_document(uri, _doc_text)
                self._harper_uri = uri
                app_logger.info(f"_start_harper: opened doc uri={uri!r}")
                # Keepalive timer — created on main thread, fires every 20s
                from PyQt6.QtCore import QTimer as _QTka
                self._harper_keepalive = _QTka(self)
                self._harper_keepalive.setInterval(5000)
                self._harper_keepalive.timeout.connect(
                    lambda: (
                        self._harper_client._send_keepalive()
                        if self._harper_client and self._harper_client.running()
                        else None))
                self._harper_keepalive.start()
                app_logger.info("_start_harper: keepalive timer started (5s)")
                # Trigger an initial lint after a short delay so harper-ls
                # gets content immediately after initialization
                from PyQt6.QtCore import QTimer as _QTinit
                _ed_ref = self  # capture for lambda
                _QTinit.singleShot(1000, _ed_ref._run_lint)
                self._update_harper_indicator(True)
                app_logger.info("_start_harper: Harper LSP active")
            else:
                self._harper_client = None
                self._update_harper_indicator(False)
                app_logger.warning("_start_harper: Harper LSP did not start")
        except Exception as _se:
            app_logger.error(f"_start_harper exception: {_se}", exc_info=True)
            self._harper_client = None
            self._update_harper_indicator(False)

    def _stop_harper(self):
        """Stop the Harper LSP client."""
        ka = getattr(self, "_harper_keepalive", None)
        if ka:
            ka.stop()
            self._harper_keepalive = None
        client = getattr(self, "_harper_client", None)
        if client:
            client.stop()
        self._harper_client = None
        self._lint_errors = []
        if hasattr(self, "_highlighter"):
            self._highlighter.set_lint_errors([])
        self._update_harper_indicator(False)

    def _toggle_harper(self):
        """Toggle spell/grammar checking on or off."""
        try:
            client = getattr(self, "_harper_client", None)
            is_running = False
            if client is not None:
                try:
                    is_running = client.running()
                except Exception:
                    pass
            if is_running:
                self._stop_harper()
            else:
                self._start_harper()
        except Exception as _te:
            app_logger.error(f"_toggle_harper exception: {_te}", exc_info=True)

    def _run_lint(self):
        """Send the current document to harper-ls (debounced via HarperLSPClient)."""
        client = getattr(self, "_harper_client", None)
        if not client or not client.running():
            harper_log("_run_lint: no running client")
            return
        uri  = getattr(self, "_harper_uri", None)
        if not uri:
            harper_log("_run_lint: no uri")
            return
        text = self.editor.toPlainText()
        harper_log(f"_run_lint: sending text_len={len(text)}")
        app_logger.debug(
            f"_run_lint: editor text_len={len(text)} preview={text[:40]!r}")
        client.change_document(uri, text)

    def _apply_lint_results(self, diags):
        """Receive LSP publishDiagnostics and convert to highlight spans.
        diags: list of {"range":{start,end}, "message":str, "suggestions":[str]}
        """
        harper_log(f"_apply_lint_results: {len(diags)} diag(s)")
        # Convert LSP line/char ranges to absolute char offsets
        text  = self.editor.toPlainText()
        lines = text.split("\n")
        # Build cumulative line-start offsets
        offsets = [0]
        for ln in lines:
            offsets.append(offsets[-1] + len(ln) + 1)
        errors = []
        for d in diags:
            rng   = d.get("range", {})
            sl    = rng.get("start", {}).get("line", 0)
            sc    = rng.get("start", {}).get("character", 0)
            el    = rng.get("end",   {}).get("line", 0)
            ec    = rng.get("end",   {}).get("character", 0)
            start = min(offsets[sl] + sc, len(text)) if sl < len(offsets) else 0
            end   = min(offsets[el] + ec, len(text)) if el < len(offsets) else 0
            harper_log(f"  diag: line={sl} ch={sc}-{ec} → offsets {start}-{end} text_len={len(text)}")
            if start < end:
                errors.append((
                    start, end,
                    d.get("message", ""),
                    d.get("suggestions", [])))
        self._lint_errors = errors
        harper_log(f"_apply_lint_results: {len(errors)} error(s) highlighted")
        if hasattr(self, "_highlighter"):
            self._highlighter.set_lint_errors(errors)
        else:
            harper_log("_apply_lint_results: NO HIGHLIGHTER FOUND")

    def _update_harper_indicator(self, active: bool):
        """Update the Harper status indicator in the toolbar and Edit menu."""
        btn = getattr(self, "_harper_indicator", None)
        # Sync Edit menu action
        act = getattr(self, "_harper_menu_act", None)
        if act:
            act.setChecked(active)
            act.setText(
                "Spell && Grammar Checking: ON"
                if active else
                "Spell && Grammar Checking: OFF")
        if btn is None:
            return
        if active:
            btn.setText("📝")
            btn.setToolTip("Spell & grammar checking: ON\nClick to disable")
            btn.setStyleSheet(
                "QPushButton{background:#1a3a1a;border:1px solid #4caf50;"
                "border-radius:4px;color:#4caf50;font-size:13px;padding:2px 6px;}"
                "QPushButton:hover{background:#2a5a2a;}")
        else:
            btn.setText("📝")
            btn.setToolTip(
                "Spell & grammar checking: OFF\n"
                "Click to enable  (requires harper-ls)\n"
                "Go to Settings → Optional Tools to install Harper.")
            btn.setStyleSheet(
                "QPushButton{background:#2a2a2a;border:1px solid #444;"
                "border-radius:4px;color:#666;font-size:13px;padding:2px 6px;}"
                "QPushButton:hover{background:#353535;border-color:#888;}")

    def _lint_at_cursor(self, pos):
        """Return (msg, suggestions) for the lint error at char position, or None."""
        for (start, end, msg, sugg) in getattr(self, "_lint_errors", []):
            if start <= pos < end:
                # If no suggestions cached, fetch code actions from harper-ls
                if not sugg:
                    client = getattr(self, "_harper_client", None)
                    uri = getattr(self, "_harper_uri", None)
                    if client and client.running() and uri:
                        text = self.editor.toPlainText()
                        lines = text[:start].split("\n")
                        line = len(lines) - 1
                        char_start = len(lines[-1])
                        char_end = char_start + (end - start)
                        harper_log(f"_lint_at_cursor: fetching code actions at L{line}:{char_start}-{char_end}")
                        actions = client.request_code_actions(uri, line, char_start, char_end)
                        # Extract suggestions from code action titles
                        # Harper "Replace with" actions use curly quotes: Replace with: "word"
                        # Skip "Add to dictionary" and "Ignore" actions
                        import re
                        for action in actions:
                            title = action.get("title", "")
                            if title.startswith("Replace with"):
                                # Match both curly quotes and straight quotes
                                m = re.search(r':\s*["\u201c]([^"\u201d]+)["\u201d]', title)
                                if m:
                                    sugg.append(m.group(1))
                                    harper_log(f"  suggestion: {m.group(1)}")
                        harper_log(f"_lint_at_cursor: extracted {len(sugg)} suggestion(s): {sugg}")
                return msg, sugg
        return None

    def _snap_on_editor_change(self):
        _app_w = next((w for w in QApplication.topLevelWidgets()
                       if w.__class__.__name__ == "WhisperRApp"), None)
        if not _app_w:
            return
        txt = self.editor.toPlainText()
        wc  = len(txt.split())
        # Calculate how many words were added/removed since last snapshot
        last_snap = (getattr(_app_w, "_app_snapshots_ram", [None])[-1]
                     if getattr(_app_w, "_app_snapshots_ram", []) else None)
        last_wc = len(last_snap["editor_text"].split()) if last_snap else 0
        delta = wc - last_wc
        if delta == 0:
            return   # no change since last snapshot
        snippet = " ".join(txt.split()[-7:])[:55]
        if delta > 0:
            reason = f"Editor: +{delta} word(s) (total {wc})  …  \"{snippet}\""
        else:
            reason = f"Editor: {delta} word(s) deleted (total {wc})"
        _app_w.take_app_snapshot(reason)

    # ── Version History ─────────────────────────────────────────────────────
    #
    # Each entry: {"ts": ISO-string, "text": str, "words": int, "chars": int}
    #
    # RAM buffer (_version_history) holds all in-session snapshots.
    # _history_dirty_count tracks how many new entries haven't been flushed
    # to the project file yet.  Flush occurs when:
    #   • count >= 10  (configurable later)
    #   • 10 minutes since last flush
    #   • user saves the project
    #   • editor is hidden / app quits

    def _cfg_history(self):
        """Return (keep_n, infinite) from live config."""
        cfg = (self.config if isinstance(self.config, dict)
               else getattr(self.config, "settings", {}))
        infinite = bool(cfg.get("version_history_infinite", False))
        keep     = int(cfg.get("version_history_keep", 20))
        return keep, infinite

    def _push_history(self):
        """Snapshot current text into version history (RAM buffer)."""
        from datetime import datetime as _dt
        text = self.editor.toPlainText()
        if not hasattr(self, "_version_history"):
            self._version_history: list[dict] = []
            self._history_dirty_count: int = 0
            self._history_last_flush = _dt.now()
        # Skip if identical to last entry
        if self._version_history and self._version_history[-1]["text"] == text:
            return
        entry = {
            "ts":    _dt.now().isoformat(timespec="seconds"),
            "text":  text,
            "words": len(text.split()),
            "chars": len(text),
        }
        self._version_history.append(entry)
        self._history_dirty_count += 1
        keep, infinite = self._cfg_history()
        if not infinite:
            while len(self._version_history) > max(keep, 1):
                self._version_history.pop(0)
        # Auto-flush to disk?
        now = _dt.now()
        minutes_since = (now - self._history_last_flush).total_seconds() / 60
        if (self._history_dirty_count >= 10 or minutes_since >= 10):
            self._flush_history_to_project()

    def _flush_history_to_project(self):
        """Append pending history entries to the project file if one is saved."""
        from datetime import datetime as _dt
        if not self._project_path or not getattr(self, "_version_history", None):
            return
        dirty = getattr(self, "_history_dirty_count", 0)
        if dirty == 0:
            return
        try:
            import json as _j
            hist_path = self._project_path.with_suffix(".wrp.history")
            existing: list[dict] = []
            if hist_path.exists():
                try:
                    existing = _j.loads(hist_path.read_text(encoding="utf-8"))
                except Exception:
                    existing = []
            # Append only the new (dirty) entries
            new_entries = self._version_history[-dirty:]
            merged = existing + new_entries
            # Trim by keep policy
            keep, infinite = self._cfg_history()
            if not infinite:
                while len(merged) > max(keep, 1):
                    merged.pop(0)
            hist_path.write_text(
                _j.dumps(merged, ensure_ascii=False, indent=2),
                encoding="utf-8")
            self._history_dirty_count = 0
            self._history_last_flush = _dt.now()
        except Exception as _e:
            import logging; logging.getLogger("whisperr").warning(
                f"History flush failed: {_e}")

    def _load_history_from_project(self):
        """Load persisted history entries from .wrp.history file into RAM."""
        from datetime import datetime as _dt
        if not hasattr(self, "_version_history"):
            self._version_history = []
            self._history_dirty_count = 0
            self._history_last_flush = _dt.now()
        if not self._project_path:
            return
        hist_path = self._project_path.with_suffix(".wrp.history")
        if not hist_path.exists():
            return
        try:
            import json as _j
            loaded = _j.loads(hist_path.read_text(encoding="utf-8"))
            # Merge: loaded entries go first; in-RAM entries appended
            ram_texts = {e["text"] for e in self._version_history}
            new_from_disk = [e for e in loaded if e["text"] not in ram_texts]
            self._version_history = new_from_disk + self._version_history
            keep, infinite = self._cfg_history()
            if not infinite:
                while len(self._version_history) > max(keep, 1):
                    self._version_history.pop(0)
        except Exception:
            pass

    def _show_history(self):
        """Show version history picker dialog with date/time/words/chars."""
        from PyQt6.QtWidgets import (QDialog, QTreeWidget, QTreeWidgetItem,
                                     QDialogButtonBox, QLabel, QHBoxLayout)
        from PyQt6.QtCore import Qt as _Qt
        from datetime import datetime as _dt

        history = getattr(self, "_version_history", [])
        if not history:
            QMessageBox.information(self, "Version History",
                "No history entries yet.\n\n"
                "Snapshots are taken automatically 5 seconds after you "
                "stop typing, and on every Save.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Version History")
        dlg.resize(640, 420)
        lay = QVBoxLayout(dlg)

        lbl = QLabel(f"{len(history)} snapshot(s) — select one to restore:")
        lbl.setStyleSheet("color:#aaa;font-size:11px;")
        lay.addWidget(lbl)

        tree = QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(["Date / Time", "Words", "Chars", "Preview"])
        tree.setColumnWidth(0, 160)
        tree.setColumnWidth(1, 60)
        tree.setColumnWidth(2, 60)
        tree.setColumnWidth(3, 280)
        tree.setStyleSheet(
            "QTreeWidget{background:#1e1e1e;color:#ddd;border:1px solid #444;}"
            "QTreeWidget::item:selected{background:#1a3a5c;}"
            "QHeaderView::section{background:#2a2a2a;color:#aaa;"
            "border:1px solid #333;padding:2px 4px;}")
        tree.setAlternatingRowColors(True)

        # Group entries by date for readability
        from collections import defaultdict as _dd
        by_day: dict = _dd(list)
        for i, entry in enumerate(history):
            try:
                dt = _dt.fromisoformat(entry["ts"])
            except Exception:
                dt = _dt.now()
            day_key = dt.strftime("%Y-%m-%d")
            by_day[day_key].append((i, dt, entry))

        # Determine grouping depth based on span
        all_days = sorted(by_day.keys())
        span_days = ((_dt.fromisoformat(all_days[-1]) -
                      _dt.fromisoformat(all_days[0])).days + 1
                     if len(all_days) > 1 else 1)

        def _make_leaf(i, dt, entry):
            words = entry.get("words", len(entry["text"].split()))
            chars = entry.get("chars", len(entry["text"]))
            preview = entry["text"][:100].replace("\n", " ")
            if len(entry["text"]) > 100:
                preview += "…"
            item = QTreeWidgetItem([
                dt.strftime("%H:%M:%S"),
                str(words),
                str(chars),
                preview,
            ])
            item.setData(0, _Qt.ItemDataRole.UserRole, i)
            return item

        if span_days <= 1:
            # Flat list, most recent first
            for i, dt, entry in reversed(list(by_day[all_days[0]])):
                tree.addTopLevelItem(_make_leaf(i, dt, entry))
        elif span_days <= 7:
            # Group by day
            for day in reversed(all_days):
                day_item = QTreeWidgetItem([
                    _dt.fromisoformat(day).strftime("%A, %b %d %Y"), "", "", ""])
                day_item.setExpanded(True)
                for i, dt, entry in reversed(by_day[day]):
                    day_item.addChild(_make_leaf(i, dt, entry))
                tree.addTopLevelItem(day_item)
        elif span_days <= 60:
            # Week → Day → entries
            from collections import defaultdict as _dd2
            by_week: dict = _dd2(lambda: _dd(list))
            for day in all_days:
                dt_d = _dt.fromisoformat(day)
                wk = dt_d.strftime("%Y-W%W")
                by_week[wk][day].extend(by_day[day])
            for wk in reversed(sorted(by_week.keys())):
                wk_dt = _dt.strptime(wk + "-1", "%Y-W%W-%w")
                wk_item = QTreeWidgetItem([
                    f"Week of {wk_dt.strftime('%b %d, %Y')}", "", "", ""])
                wk_item.setExpanded(True)
                for day in reversed(sorted(by_week[wk].keys())):
                    day_item = QTreeWidgetItem([
                        _dt.fromisoformat(day).strftime("%A, %b %d"), "", "", ""])
                    day_item.setExpanded(False)
                    for i, dt, entry in reversed(by_week[wk][day]):
                        day_item.addChild(_make_leaf(i, dt, entry))
                    wk_item.addChild(day_item)
                tree.addTopLevelItem(wk_item)
        else:
            # Month → Week → Day → entries
            from collections import defaultdict as _dd3
            by_month: dict = _dd3(lambda: _dd3(lambda: _dd3(list)))
            for day in all_days:
                dt_d = _dt.fromisoformat(day)
                mo  = dt_d.strftime("%Y-%m")
                wk  = dt_d.strftime("%Y-W%W")
                by_month[mo][wk][day].extend(by_day[day])
            for mo in reversed(sorted(by_month.keys())):
                mo_dt = _dt.strptime(mo + "-01", "%Y-%m-%d")
                mo_item = QTreeWidgetItem([mo_dt.strftime("%B %Y"), "", "", ""])
                mo_item.setExpanded(True)
                for wk in reversed(sorted(by_month[mo].keys())):
                    wk_dt = _dt.strptime(wk + "-1", "%Y-W%W-%w")
                    wk_item = QTreeWidgetItem([
                        f"Week of {wk_dt.strftime('%b %d')}", "", "", ""])
                    wk_item.setExpanded(False)
                    for day in reversed(sorted(by_month[mo][wk].keys())):
                        day_item = QTreeWidgetItem([
                            _dt.fromisoformat(day).strftime("%a %b %d"), "", "", ""])
                        day_item.setExpanded(False)
                        for i, dt, entry in reversed(by_month[mo][wk][day]):
                            day_item.addChild(_make_leaf(i, dt, entry))
                        wk_item.addChild(day_item)
                    mo_item.addChild(wk_item)
                tree.addTopLevelItem(mo_item)

        lay.addWidget(tree)

        # Stats bar
        info_row = QHBoxLayout()
        info_lbl = QLabel("")
        info_lbl.setStyleSheet("color:#888;font-size:10px;")
        info_row.addWidget(info_lbl)
        info_row.addStretch()
        lay.addLayout(info_row)

        def _on_selection():
            items = tree.selectedItems()
            if not items:
                return
            idx = items[0].data(0, _Qt.ItemDataRole.UserRole)
            if idx is None:
                info_lbl.setText("")
                return
            e = history[idx]
            w = e.get("words", len(e["text"].split()))
            c = e.get("chars", len(e["text"]))
            ts = e.get("ts", "")
            info_lbl.setText(
                f"Snapshot {idx+1}/{len(history)}  •  {w} words  •  {c} chars  •  {ts}")

        tree.itemSelectionChanged.connect(_on_selection)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            items = tree.selectedItems()
            if items:
                idx = items[0].data(0, _Qt.ItemDataRole.UserRole)
                if idx is not None:
                    self._push_history()   # save current state first
                    self.editor.setPlainText(history[idx]["text"])


    # ── Find & Replace ────────────────────────────────────────────────────────

    def _show_find_replace(self):
        """Open/close a non-modal Find / Replace bar inside the editor."""
        if getattr(self, "_fr_bar", None) and self._fr_bar.isVisible():
            self._fr_bar.close()
            self._fr_bar = None
            self.editor.setFocus()
            return
        from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QLabel
        bar = QWidget(self)
        bar.setStyleSheet(
            "QWidget{background:#252525;border-top:1px solid #444;}"
            "QLineEdit{background:#1e1e1e;border:1px solid #555;color:#ddd;"
            "padding:3px 6px;border-radius:3px;min-width:150px;}")
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)
        h.addWidget(QLabel("Find:"))
        _find  = QLineEdit(); _find.setPlaceholderText("search text…")
        h.addWidget(_find)
        h.addWidget(QLabel("Replace:"))
        _repl  = QLineEdit(); _repl.setPlaceholderText("replacement…")
        h.addWidget(_repl)
        _ss = ("QPushButton{background:#2a2a2a;border:1px solid #555;color:#ddd;"
               "padding:3px 8px;border-radius:3px;}"
               "QPushButton:hover{background:#353535;border-color:#0078d7;}")
        _btn_find = QPushButton("▶ Find"); _btn_find.setStyleSheet(_ss)
        _btn_repl = QPushButton("Replace"); _btn_repl.setStyleSheet(_ss)
        _btn_all  = QPushButton("Replace All"); _btn_all.setStyleSheet(_ss)
        _lbl_count = QLabel(""); _lbl_count.setStyleSheet("color:#aaa;font-size:9pt;")
        _btn_re   = QCheckBox(".*"); _btn_re.setToolTip("Use regular expressions")
        _btn_re.setStyleSheet("QCheckBox{color:#aaa;}")
        _btn_close = QPushButton("✕"); _btn_close.setFixedSize(22,22); _btn_close.setStyleSheet(_ss)
        for w in (_btn_find, _btn_repl, _btn_all, _lbl_count, _btn_re, _btn_close):
            h.addWidget(w)
        h.addStretch()
        self._fr_bar = bar
        self.layout().addWidget(bar)

        def _do_find():
            needle = _find.text()
            if not needle:
                return
            # Move cursor past current selection to advance, not re-find the same match
            cur = self.editor.textCursor()
            if cur.hasSelection():
                # Advance past current selection before searching
                pos = max(cur.selectionStart(), cur.selectionEnd())
                cur.setPosition(pos)
                self.editor.setTextCursor(cur)
            doc = self.editor.document()
            cur = self.editor.textCursor()
            if _btn_re.isChecked():
                found = doc.find(QRegularExpression(needle), cur)
            else:
                found = doc.find(needle, cur)
            if found.isNull():
                # Wrap around from start
                cur2 = self.editor.textCursor()
                cur2.movePosition(cur2.MoveOperation.Start)
                self.editor.setTextCursor(cur2)
                if _btn_re.isChecked():
                    found = doc.find(QRegularExpression(needle), self.editor.textCursor())
                else:
                    found = doc.find(needle, self.editor.textCursor())
            if not found.isNull():
                self.editor.setTextCursor(found)
                self.editor.ensureCursorVisible()
            else:
                _lbl_count.setText("not found")

        def _do_replace():
            needle = _find.text()
            if not needle:
                return
            cur = self.editor.textCursor()
            if cur.hasSelection():
                # Replace whatever is selected (result of a Find)
                cur.insertText(_repl.text())   # undo-able single op
                self.editor.setTextCursor(cur)
            # Always advance to next match after replacing
            _do_find()

        def _do_replace_all():
            import re as _re_fa
            needle = _find.text()
            repl   = _repl.text()
            if not needle:
                return
            # Use cursor-based replacement so Ctrl+Z undoes all at once
            doc = self.editor.document()
            cur = self.editor.textCursor()
            cur.beginEditBlock()
            cur.movePosition(cur.MoveOperation.Start)
            self.editor.setTextCursor(cur)
            count = 0
            while True:
                if _btn_re.isChecked():
                    found = doc.find(QRegularExpression(needle), self.editor.textCursor())
                else:
                    found = doc.find(needle, self.editor.textCursor())
                if found.isNull():
                    break
                if _btn_re.isChecked():
                    import re as _re_s
                    replaced = _re_s.sub(needle, repl, found.selectedText(), count=1)
                else:
                    replaced = repl
                found.insertText(replaced)
                self.editor.setTextCursor(found)
                count += 1
            cur.endEditBlock()
            _lbl_count.setText(f"{count} replaced")

        _btn_find.clicked.connect(_do_find)
        _find.returnPressed.connect(_do_find)
        _btn_repl.clicked.connect(_do_replace)
        _btn_all.clicked.connect(_do_replace_all)
        _btn_close.clicked.connect(lambda: (bar.close(), setattr(self, "_fr_bar", None)))
        _find.setFocus()

    # ── Auto-backup ───────────────────────────────────────────────────────────

    def _start_auto_backup(self):
        """Start (or restart) the auto-backup timer using current config."""
        cfg = self.config.settings if hasattr(self.config, "settings") else self.config
        if not cfg.get("auto_backup_enabled", False):
            self._stop_auto_backup()
            return
        interval_min = int(cfg.get("auto_backup_interval", 10))
        if not hasattr(self, "_backup_timer"):
            self._backup_timer = QTimer(self)
            self._backup_timer.timeout.connect(self._do_auto_backup)
        self._backup_timer.start(interval_min * 60 * 1000)

    def _stop_auto_backup(self):
        if getattr(self, "_backup_timer", None):
            self._backup_timer.stop()

    def _do_auto_backup(self):
        """Write a timestamped backup of the current project."""
        cfg = self.config.settings if hasattr(self.config, "settings") else self.config
        if not cfg.get("auto_backup_enabled", False):
            return
        text = self.editor.toPlainText()
        if not text.strip():
            return  # nothing to back up
        # Need a save path
        if not self._project_path:
            reply = QMessageBox.question(
                self, "Auto-backup",
                "Auto-backup is enabled but the project hasn't been saved yet.\n"
                "Save now so backups have a home?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes)
            if reply == QMessageBox.StandardButton.Yes:
                self._save_project()
            if not self._project_path:
                return  # user cancelled
        from datetime import datetime as _dt
        import json as _jb
        ts = _dt.now().strftime("%Y-%m-%d-%H-%M")
        stem = self._project_path.stem
        backup_path = self._project_path.parent / f"{stem}_{ts}.wrp.bak"
        try:
            # Build full project snapshot
            _ts_spin = getattr(self, "target_spin", None)
            if _ts_spin: _ts_spin.interpretText()
            data = {
                "version": 1,
                "text": text,
                "target_words": _ts_spin.value() if _ts_spin else 0,
                "notes": (self._notes_win.get_notes_data() if self._notes_win else []),
                "backup_timestamp": ts,
            }
            backup_path.write_text(
                _jb.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            # Prune old backups
            keep = int(cfg.get("auto_backup_keep", 5))
            import glob as _gl
            pattern = str(self._project_path.parent / f"{stem}_????-??-??-??-??.wrp.bak")
            backups = sorted(_gl.glob(pattern))
            for old in backups[:-keep]:
                try: Path(old).unlink()
                except Exception: pass
            app_logger.info(f"Auto-backup saved: {backup_path.name}")
        except Exception as e:
            app_logger.warning(f"Auto-backup failed: {e}")

    # ── Import / Export (text only) ───────────────────────────────────────────

    def _import_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import file", "", "Text/Markdown (*.txt *.md);;All files (*)")
        if path:
            try:
                self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))
            except Exception as e:
                QMessageBox.warning(self, "Import failed", str(e))

    def _export_file(self):
        """Export with format detection: md/txt always; docx/html/pdf if tools available."""
        import subprocess as _sp

        _has_pandoc = False
        _has_docx   = False
        try:
            import docx as _dx  # noqa
            _has_docx = True
        except ImportError:
            pass
        try:
            _r = _sp.run(["pandoc", "--version"], capture_output=True, timeout=3)
            _has_pandoc = _r.returncode == 0
        except Exception:
            pass

        filters = "Markdown (*.md);;Plain Text (*.txt);;HTML (*.html)"
        if _has_pandoc or _has_docx:
            filters += ";;Word Document (*.docx)"
        if _has_pandoc:
            filters += ";;PDF (*.pdf)"

        path, _ = QFileDialog.getSaveFileName(self, "Export", "", filters)
        if not path:
            return
        ext  = Path(path).suffix.lower()
        text = self.editor.toPlainText()
        try:
            if ext in (".md", ".txt", ""):
                Path(path).write_text(text, encoding="utf-8")

            elif ext == ".html":
                import html as _hl
                nl2  = "\n\n"
                nl1  = "\n"
                body = _hl.escape(text).replace(nl2, "</p><p>").replace(nl1, "<br>")
                html_out = (
                    "<!DOCTYPE html><html><head><meta charset='utf-8'>"
                    "<style>body{font-family:Georgia,serif;max-width:800px;"
                    "margin:2em auto;line-height:1.6;color:#222;}"
                    "p{margin:0.8em 0;}</style></head>"
                    "<body><p>" + body + "</p></body></html>"
                )
                Path(path).write_text(html_out, encoding="utf-8")

            elif ext == ".docx":
                if _has_pandoc:
                    _sp.run(["pandoc", "-f", "markdown", "-o", path],
                            input=text.encode("utf-8"), check=True, timeout=30)
                elif _has_docx:
                    import docx as _dx2
                    doc = _dx2.Document()
                    for para in text.split("\n\n"):
                        if para.strip():
                            doc.add_paragraph(para)
                    doc.save(path)
                else:
                    raise RuntimeError("Neither Pandoc nor python-docx is installed.")

            elif ext == ".pdf":
                if _has_pandoc:
                    _sp.run(["pandoc", "-f", "markdown", "-o", path],
                            input=text.encode("utf-8"), check=True, timeout=60)
                else:
                    raise RuntimeError(
                        "PDF export requires Pandoc.\n"
                        "Download from https://pandoc.org/installing.html")

            else:
                Path(path).write_text(text, encoding="utf-8")

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
                        # Strip pynput angle-bracket format <ctrl> → ctrl
                        parts = [p.strip().lower().strip("<>") for p in hk_str.split("+")]
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
                            _rk_parts = [p.strip().lower().strip("<>") for p in _rk.split("+")]
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
        # Stop Harper LSP client cleanly
        self._stop_harper()
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
            self.transcriber.model_not_found_sig.connect(self._on_model_not_found)
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
            # Periodic flush: write any pending RAM snapshots to disk every 10 min
            self._snapshot_flush_timer = QTimer(self)
            self._snapshot_flush_timer.timeout.connect(
                lambda: self._flush_app_snapshots())
            self._snapshot_flush_timer.start(600_000)
            # QFileSystemWatcher for real-time folder monitoring
            self._setup_ft_watcher()
            
            # No standalone pa_sys / meter_timer.
            # The live meter is driven purely by AudioRecorder.volume_out signal
            # (connected in toggle_rec). This avoids running a second PyAudio
            # instance alongside AudioRecorder's, which crashes in frozen mode.
            self.meter_stream = None  # kept for compat refs in toggle_rec cleanup
            
            app_logger.info("Application initialized successfully")
            self._apply_always_on_top()
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
            self._editor: WhisperEditor | None = None
            # Persisted editor state — defaults overwritten by JSON load below
            self._editor_remember: bool = True
            self._last_hk_time: float = 0.0
            self._editor_saved_content: str = ""
            self._editor_clipboard_prefill: bool = False
            self._editor_saved_target: int = 0
            self._editor_cb_monitor_was_on: bool = False
            self._editor_cheatsheet_open: bool = False
            self._editor_notes_open: bool = False
            self._editor_saved_notes: list = []
            self._editor_saved_filter: list = []   # notes color filter
            self._cb_monitor_mode: str = "text"
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
                        self._editor_saved_notes        = _st.get("notes", [])
                        self._editor_notes_open         = _st.get("notes_open", False)
                        self._editor_saved_filter       = _st.get("notes_filter", [])
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
            # Restart clipboard monitor after state is loaded
            if self._editor_cb_monitor_was_on:
                _mode_restart = self._cb_monitor_mode
                self.start_clipboard_monitor(mode=_mode_restart)
                app_logger.info(f"Clipboard monitor auto-restarted (mode={_mode_restart})")
            # Non-persisted runtime state
            self._editor_return_hwnd = None
            self._cb_monitor_timer: QTimer | None = None
            self._cb_monitor_last: str = ""
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
        self.tabs.currentChanged.connect(self._on_settings_tab_changed)
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
        
        # ===== TRANSCRIPTION STEERING TAB =====
        tp = QWidget()
        lp = QVBoxLayout(tp)
        lp.setSpacing(6)
        lp.setContentsMargins(8, 8, 8, 8)

        # ── Steering Prompt (~60% of space) ─────────────────────────
        _sp_lbl = QLabel("Steering Prompt")
        _sp_lbl.setStyleSheet("font-weight:bold;color:#ddd;")
        lp.addWidget(_sp_lbl)
        _sp_help = QLabel(
            "Describe what Whisper will hear — mention domain terms, proper nouns, "
            "abbreviations. Sent as context before every transcription.")
        _sp_help.setWordWrap(True)
        _sp_help.setStyleSheet("color:#888;font-size:11px;")
        lp.addWidget(_sp_help)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setText(self.config.settings["initial_prompt"])
        lp.addWidget(self.prompt_edit, 3)   # stretch=3 → ~60%
        hbp = QHBoxLayout()
        bi = QPushButton("Import .txt"); bi.clicked.connect(self.import_p)
        be = QPushButton("Export .txt"); be.clicked.connect(self.export_p)
        hbp.addWidget(bi); hbp.addWidget(be); hbp.addStretch()
        lp.addLayout(hbp)

        # ── Vocabulary Boost / Hotwords (~40% of space) ──────────────
        _hw_lbl = QLabel("Vocabulary Boost  (Hotwords)")
        _hw_lbl.setStyleSheet("font-weight:bold;color:#ddd;margin-top:4px;")
        lp.addWidget(_hw_lbl)
        _hw_help = QLabel(
            "Words/phrases Whisper should prioritise — one per line. "
            "Matched acoustically (not just as context). Keep list short.")
        _hw_help.setWordWrap(True)
        _hw_help.setStyleSheet("color:#888;font-size:11px;")
        lp.addWidget(_hw_help)
        self.hotwords_edit = QPlainTextEdit()
        self.hotwords_edit.setPlaceholderText("e.g.\nWhisperR\nHexagon Software")
        self.hotwords_edit.setPlainText(
            "\n".join(self.config.settings.get("hotwords", [])))
        lp.addWidget(self.hotwords_edit, 2)   # stretch=2 → ~40%
        hbhw = QHBoxLayout()
        bi_hw = QPushButton("Import .txt")
        be_hw = QPushButton("Export .txt")
        bi_hw.clicked.connect(self._import_hotwords)
        be_hw.clicked.connect(self._export_hotwords)
        hbhw.addWidget(bi_hw); hbhw.addWidget(be_hw); hbhw.addStretch()
        lp.addLayout(hbhw)

        self.tabs.addTab(tp, "Transcription Steering")
        # Auto-save: prompt changes debounced 5 s
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(5000)
        self._autosave_timer.timeout.connect(self._autosave_tabs)
        self.prompt_edit.textChanged.connect(self._autosave_timer.start)
        self.hotwords_edit.textChanged.connect(self._autosave_timer.start)

        # ===== COMMANDS TAB =====
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        
        l2.addWidget(QLabel("Voice Commands (phrase detection → action):"))
        
        self.cmd_table = QTableWidget(0, 2)
        self.cmd_table.setHorizontalHeaderLabels(["Phrase to Detect", "Command to Execute"])
        self.cmd_table.itemChanged.connect(
            lambda: self._autosave_timer.start() if hasattr(self, "_autosave_timer") else None)
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
        self.terms_table.itemChanged.connect(
            lambda: self._autosave_timer.start() if hasattr(self, "_autosave_timer") else None)
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
        self.hall_list.model().rowsInserted.connect(
            lambda *_: self._autosave_timer.start() if hasattr(self, "_autosave_timer") else None)
        self.hall_list.model().rowsRemoved.connect(
            lambda *_: self._autosave_timer.start() if hasattr(self, "_autosave_timer") else None)
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

        self.ft_progress = QProgressBar()
        self.ft_progress.setRange(0, 100)
        self.ft_progress.setValue(0)
        self.ft_progress.setVisible(False)
        self.ft_progress.setStyleSheet(
            "QProgressBar{border:1px solid #444;border-radius:4px;"
            "background:#1e1e1e;color:#ddd;text-align:center;}"
            "QProgressBar::chunk{background:#0078d7;border-radius:3px;}")
        l_ft.addWidget(self.ft_progress)

        self.ft_cancel_btn = QPushButton("Cancel")
        self.ft_cancel_btn.setVisible(False)
        self.ft_cancel_btn.setStyleSheet(
            "QPushButton{background:#5a1a1a;border:1px solid #aa3333;"
            "color:#ff8888;padding:4px 12px;border-radius:4px;}"
            "QPushButton:hover{background:#7a2222;}")
        self.ft_cancel_btn.clicked.connect(self._ft_cancel)
        l_ft.addWidget(self.ft_cancel_btn)
        self._ft_cancelled = False

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
            "Drop segments where Whisper is less than X% confident.\n"
            "0.50 = keep if Whisper is ≥50% sure (threshold logprob -0.693)\n"
            "0.90 = keep only clear speech    (threshold logprob -0.105)\n"
            "0.00 = disabled (keep everything)\n"
            "Typical real speech: logprob -0.1 (clear) to -0.6 (noisy).\n"
            "Recommended starting point: 0.30 – 0.50."
        )
        ai_layout.addRow("Min. Confidence (0-1):", self.cfg_conf_spin)

        # ── Model cache folder ──────────────────────────────────────────────
        import os as _os_hf
        _default_hf = _os_hf.path.join(
            _os_hf.path.expanduser("~"), ".cache", "huggingface", "hub")
        _saved_hf = self.config.settings.get("hf_cache_path", "") or _default_hf
        self.cfg_hf_path = QLineEdit(_saved_hf)
        self.cfg_hf_path.setToolTip(
            "Folder where Whisper model files are stored.\n"
            "WhisperR looks here for cached models before attempting a download.\n"
            "Default: %USERPROFILE%\\.cache\\huggingface\\hub")
        self.cfg_hf_path.setStyleSheet(
            "QLineEdit{background:#1a1a1a;color:#88ccff;font-family:monospace;font-size:9pt;}")
        _btn_hf_browse = QPushButton("Browse…")
        _btn_hf_browse.setToolTip("Choose a different folder containing model files.")
        _btn_hf_browse.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;padding:3px 10px;"
            "border-radius:4px;}QPushButton:hover{border-color:#0078d7;}")
        def _browse_hf_dir():
            chosen = QFileDialog.getExistingDirectory(
                self, "Select Model Cache Folder", self.cfg_hf_path.text())
            if chosen:
                self.cfg_hf_path.setText(chosen)
        _btn_hf_browse.clicked.connect(_browse_hf_dir)
        _btn_hf_open = QPushButton("📁 Open")
        _btn_hf_open.setToolTip("Open the model cache folder in Explorer.")
        _btn_hf_open.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;padding:3px 10px;"
            "border-radius:4px;}QPushButton:hover{border-color:#0078d7;}")
        def _open_hf_dir():
            import subprocess as _sp_hf2
            try: _sp_hf2.Popen(["explorer", self.cfg_hf_path.text()])
            except Exception: pass
        _btn_hf_open.clicked.connect(_open_hf_dir)
        _hf_row_w = QWidget()
        _hf_row = QHBoxLayout(_hf_row_w)
        _hf_row.setContentsMargins(0,0,0,0)
        _hf_row.addWidget(self.cfg_hf_path, 1)
        _hf_row.addWidget(_btn_hf_browse)
        _hf_row.addWidget(_btn_hf_open)
        ai_layout.addRow("Model cache:", _hf_row_w)

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
            "Seconds of volume-silence before recording stops and transcription begins.\n"
            "Uses the Noise Floor/Speech Vol settings to detect silence.\n\n"
            "Note: if VAD (Voice Activity Detection) is enabled in Advanced settings,\n"
            "the VAD Min Silence setting takes over this role and this value is ignored.\n\n"
            "1.0–2.0 s works well for most people."
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

        # Manual Sentence Splitting
        self.cfg_mss = QCheckBox("Manual Sentence Splitting")
        self.cfg_mss.setChecked(
            self.config.settings.get("manual_sentence_split", False))
        self.cfg_mss.setToolTip(
            "When enabled, Whisper's trailing punctuation is stripped from each\n"
            "segment and the next segment begins lowercase (continuing the\n"
            "sentence). Press the Sentence Break key to end a sentence:\n"
            "a full stop is appended and the next segment starts with a capital.\n\n"
            "Useful for long dictation where you want explicit control over\n"
            "sentence boundaries rather than relying on Whisper's auto-punctuation.")
        dict_layout.addRow(self.cfg_mss)
        from PyQt6.QtWidgets import QLineEdit as _QLE_mss
        self.cfg_mss_key = _QLE_mss()
        self.cfg_mss_key.setText(
            self.config.settings.get("mss_break_key", "shift"))
        self.cfg_mss_key.setToolTip(
            "Key to press to end the current sentence and start a new one.\n"
            "Default: shift  (Left Shift key)\n"
            "Enter the pynput key name, e.g.: shift, ctrl, alt, f1, space")
        self.cfg_mss_key.setMaximumWidth(120)
        dict_layout.addRow("Sentence Break key:", self.cfg_mss_key)

        dict_group.setLayout(dict_layout)
        main_layout.addWidget(dict_group)

        # --- Spell & Grammar Linters ---
        linter_group = QGroupBox("Spell & Grammar Linters")
        linter_grid = QGridLayout()
        linter_grid.setSpacing(4)
        linter_grid.setContentsMargins(6, 8, 6, 8)
        linter_col = 0
        linter_row = 0
        COLS = 3
        # Load saved linter config; merge with defaults for any missing keys
        _harper = self.config.settings.get("harper", {})
        if not isinstance(_harper, dict):
            _harper = {}
        _saved_linters = _harper.get("linters", {})
        if not isinstance(_saved_linters, dict):
            _saved_linters = {}
        _all_defaults = _harper_default_linters()
        self._linter_checkboxes = {}
        for (lname, label, default, tip) in HARPER_LINTERS:
            cb = QCheckBox(label)
            cb.setChecked(bool(_saved_linters.get(lname, default)))
            cb.setToolTip(tip)
            cb.toggled.connect(lambda checked, n=lname: self._on_linter_toggled(n, checked))
            self._linter_checkboxes[lname] = cb
            linter_grid.addWidget(cb, linter_row, linter_col)
            linter_col += 1
            if linter_col >= COLS:
                linter_col = 0
                linter_row += 1
        linter_group.setLayout(linter_grid)
        main_layout.addWidget(linter_group)

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

        from PyQt6.QtWidgets import QSpinBox as _SB2cd
        _cd_spin = _SB2cd()
        _cd_spin.setRange(0, 2000)
        _cd_spin.setSuffix(" ms")
        _cd_spin.setValue(
            int(self.config.settings.get("hotkey_cooldown_ms", 400)))
        _cd_spin.setToolTip(
            "After any hotkey fires, other hotkeys are ignored for this many\n"
            "milliseconds. This prevents Ctrl+Shift+Alt+Z from also triggering\n"
            "Ctrl+Alt+Z when you release Shift a fraction of a second later.\n"
            "Set to 0 to disable. Default: 400 ms.")
        self.cfg_hk_cooldown = _cd_spin
        hotkey_layout.addRow("Hotkey detection delay:", self.cfg_hk_cooldown)
        hotkey_group.setLayout(hotkey_layout)
        main_layout.addWidget(hotkey_group)

        # --- Always On Top ---
        aot_group = QGroupBox("Always On Top")
        aot_layout = QVBoxLayout()
        self.cfg_aot_master = QCheckBox("Keep app windows always on top of all windows")
        self.cfg_aot_master.setChecked(self.config.settings.get("always_on_top", False))
        self.cfg_aot_master.setToolTip(
            "When ON: all selected windows stay above every other window.\n"
            "When OFF: windows appear above the active app when opened,\n"
            "but do not float above unrelated windows.")
        aot_layout.addWidget(self.cfg_aot_master)
        # Sub-toggles
        _aot_indent_ss = "QCheckBox{margin-left:20px;}"
        self.cfg_aot_main = QCheckBox("Main window")
        self.cfg_aot_main.setChecked(self.config.settings.get("aot_main", False))
        self.cfg_aot_main.setStyleSheet(_aot_indent_ss)
        self.cfg_aot_editor = QCheckBox("Text Editor")
        self.cfg_aot_editor.setChecked(self.config.settings.get("aot_editor", False))
        self.cfg_aot_editor.setStyleSheet(_aot_indent_ss)
        self.cfg_aot_notes = QCheckBox("Notes panel")
        self.cfg_aot_notes.setChecked(self.config.settings.get("aot_notes", False))
        self.cfg_aot_notes.setStyleSheet(_aot_indent_ss)
        self.cfg_aot_cheatsheet = QCheckBox("Cheatsheet panel")
        self.cfg_aot_cheatsheet.setChecked(self.config.settings.get("aot_cheatsheet", False))
        self.cfg_aot_cheatsheet.setStyleSheet(_aot_indent_ss)
        for _aw in (self.cfg_aot_main, self.cfg_aot_editor,
                    self.cfg_aot_notes, self.cfg_aot_cheatsheet):
            aot_layout.addWidget(_aw)
        # Master toggle: set all subs ON/OFF
        def _aot_master_changed(checked):
            for _aw in (self.cfg_aot_main, self.cfg_aot_editor,
                        self.cfg_aot_notes, self.cfg_aot_cheatsheet):
                _aw.setChecked(checked)
        self.cfg_aot_master.toggled.connect(_aot_master_changed)
        aot_group.setLayout(aot_layout)
        main_layout.addWidget(aot_group)

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

        # --- Auto-Backup ---
        backup_group = QGroupBox("History & Auto-Backup (Text Editor)")
        backup_layout = QFormLayout()
        self.cfg_backup_enabled = QCheckBox("Enable auto-backup while editing")
        self.cfg_backup_enabled.setChecked(
            self.config.settings.get("auto_backup_enabled", False))
        self.cfg_backup_enabled.setToolTip(
            "Periodically saves a timestamped .wrp.bak snapshot of the active project.\n"
            "Backups are stored next to the project file.")
        backup_layout.addRow(self.cfg_backup_enabled)
        from PyQt6.QtWidgets import QSpinBox as _SB2
        self.cfg_backup_interval = _SB2()
        self.cfg_backup_interval.setRange(1, 120)
        self.cfg_backup_interval.setSuffix(" min")
        self.cfg_backup_interval.setValue(
            int(self.config.settings.get("auto_backup_interval", 10)))
        self.cfg_backup_interval.setToolTip("How often to take a backup snapshot.")
        backup_layout.addRow("Backup every:", self.cfg_backup_interval)
        self.cfg_backup_keep = _SB2()
        self.cfg_backup_keep.setRange(1, 100)
        self.cfg_backup_keep.setValue(
            int(self.config.settings.get("auto_backup_keep", 5)))
        self.cfg_backup_keep.setToolTip(
            "Maximum number of backup files to keep per project.\n"
            "Oldest backups are deleted when the limit is exceeded.")
        backup_layout.addRow("Keep up to:", self.cfg_backup_keep)
        # Version history — infinite toggle + depth spinbox
        self.cfg_vh_infinite = QCheckBox("Infinite version history")
        self.cfg_vh_infinite.setChecked(
            self.config.settings.get("version_history_infinite", False))
        def _vh_inf_immediate(checked):
            self.config.settings["version_history_infinite"] = checked
        self.cfg_vh_infinite.toggled.connect(_vh_inf_immediate)
        self.cfg_vh_infinite.setToolTip(
            "Keep ALL snapshots — the depth limit below is ignored.\n"
            "Snapshots are flushed to a .wrp.history file alongside\n"
            "the project file when you save or close the editor.")
        backup_layout.addRow(self.cfg_vh_infinite)
        self.cfg_version_history = _SB2()
        self.cfg_version_history.setRange(1, 10000)
        self.cfg_version_history.setValue(
            int(self.config.settings.get("version_history_keep", 20)))
        self.cfg_version_history.setToolTip(
            "Maximum snapshots to keep in session and in the .wrp.history file.\n"
            "Oldest entries are dropped when the limit is exceeded.\n"
            "Disabled (greyed out) when Infinite version history is enabled.\n"
            "Snapshots are taken 5 s after you stop typing and on every Save.\n"
            "Access them with Ctrl+Alt+H in the editor.")
        backup_layout.addRow("Version history depth:", self.cfg_version_history)
        # Grey out depth when infinite is on
        def _vh_inf_toggled(checked):
            self.cfg_version_history.setEnabled(not checked)
        self.cfg_vh_infinite.toggled.connect(_vh_inf_toggled)
        _vh_inf_toggled(self.cfg_vh_infinite.isChecked())
        _btn_browse_bak = QPushButton("📁 Browse Backup Folder")
        _btn_browse_bak.setToolTip(
            "Open the folder containing backups for the currently loaded project.")
        _btn_browse_bak.setStyleSheet(
            "QPushButton{background:#2a2a2a;border:1px solid #444;padding:4px 10px;"
            "border-radius:4px;color:#ddd;}"
            "QPushButton:hover{background:#353535;border-color:#0078d7;}")
        def _browse_bak():
            import subprocess as _sp_bak
            # Walk up to WhisperRApp to find active editor
            _app_w = next((w for w in QApplication.topLevelWidgets()
                           if w.__class__.__name__ == "WhisperRApp"), None)
            _ed = getattr(_app_w, "_editor", None) if _app_w else None
            _pp = getattr(_ed, "_project_path", None) if _ed else None
            folder = str(_pp.parent) if _pp else str(Path.home())
            try:
                _sp_bak.Popen(["explorer", folder])
            except Exception:
                QMessageBox.information(self, "Backup Folder", folder)
        _btn_browse_bak.clicked.connect(_browse_bak)
        backup_layout.addRow(_btn_browse_bak)
        backup_group.setLayout(backup_layout)
        main_layout.addWidget(backup_group)

        # --- App State Snapshots ---
        snap_group = QGroupBox("App State Snapshots")
        snap_layout = QFormLayout()
        self.cfg_snap_enabled = QCheckBox("Enable app-state snapshots")
        self.cfg_snap_enabled.setChecked(
            self.config.settings.get("snapshots_enabled", False))
        # Apply immediately on toggle — don't wait for Save Settings
        def _snap_immediate(checked):
            # Update in-memory only — full disk save happens via Save Settings.
            # Calling config.save() here would write a partial config because
            # other settings haven't been collected from the UI yet.
            self.config.settings["snapshots_enabled"] = checked
        self.cfg_snap_enabled.toggled.connect(_snap_immediate)
        self.cfg_snap_enabled.setToolTip(
            "Periodically snapshots EVERYTHING — settings, editor text, notes,\n"
            "word target, terms, commands — to a .snapshots file.\n"
            "Lets you recover work even if you never saved a project.")
        snap_layout.addRow(self.cfg_snap_enabled)
        # Mode: count vs duration (mutually exclusive)
        self.cfg_snap_mode_count = QRadioButton("Keep up to N snapshots")
        self.cfg_snap_mode_dur   = QRadioButton("Keep snapshots for a time period")
        _snap_mode = self.config.settings.get("snapshots_mode", "count")
        self.cfg_snap_mode_count.setChecked(_snap_mode == "count")
        self.cfg_snap_mode_dur.setChecked(_snap_mode == "duration")
        snap_layout.addRow(self.cfg_snap_mode_count)
        snap_layout.addRow(self.cfg_snap_mode_dur)
        # Count spinner
        self.cfg_snap_count = _SB2()
        self.cfg_snap_count.setRange(10, 50000)
        self.cfg_snap_count.setValue(
            int(self.config.settings.get("snapshots_keep_count", 60)))
        self.cfg_snap_count.setToolTip("Maximum number of snapshots to keep.")
        snap_layout.addRow("Max snapshots:", self.cfg_snap_count)
        # Duration spinner + unit combo
        _snap_dur_row = QHBoxLayout()
        self.cfg_snap_hours = _SB2()
        self.cfg_snap_hours.setRange(1, 8760)  # up to 1 year
        self.cfg_snap_hours.setValue(
            int(self.config.settings.get("snapshots_keep_hours", 24)))
        self.cfg_snap_hours.setToolTip("Keep snapshots taken within this many hours.")
        from PyQt6.QtWidgets import QComboBox as _QCB2
        self.cfg_snap_unit = _QCB2()
        self.cfg_snap_unit.addItems(["hours", "days", "weeks", "months"])
        self.cfg_snap_unit.setCurrentText(
            self.config.settings.get("snapshots_keep_unit", "hours"))
        _snap_dur_row.addWidget(self.cfg_snap_hours)
        _snap_dur_row.addWidget(self.cfg_snap_unit)
        _snap_dur_w = QWidget(); _snap_dur_w.setLayout(_snap_dur_row)
        snap_layout.addRow("Keep for:", _snap_dur_w)
        # Grayout logic
        def _snap_mode_changed():
            count_mode = self.cfg_snap_mode_count.isChecked()
            self.cfg_snap_count.setEnabled(count_mode)
            self.cfg_snap_hours.setEnabled(not count_mode)
            self.cfg_snap_unit.setEnabled(not count_mode)
        self.cfg_snap_mode_count.toggled.connect(_snap_mode_changed)
        self.cfg_snap_mode_dur.toggled.connect(_snap_mode_changed)
        _snap_mode_changed()
        # Enable/disable whole group
        def _snap_enabled_changed(checked):
            for w in [self.cfg_snap_mode_count, self.cfg_snap_mode_dur,
                      self.cfg_snap_count, self.cfg_snap_hours,
                      self.cfg_snap_unit]:
                w.setEnabled(checked if w not in
                    ([self.cfg_snap_count] if self.cfg_snap_mode_dur.isChecked()
                     else [self.cfg_snap_hours, self.cfg_snap_unit])
                    else False)
        self.cfg_snap_enabled.toggled.connect(lambda c: _snap_mode_changed() or _snap_enabled_changed(c))
        _snap_enabled_changed(self.cfg_snap_enabled.isChecked())
        snap_group.setLayout(snap_layout)
        main_layout.addWidget(snap_group)

        # --- Clipboard Monitor Options ---
        cbmon_group = QGroupBox("Clipboard Monitor Options")
        cbmon_layout = QFormLayout()
        self.cfg_cb_source_tag = QCheckBox(
            "Tag clipboard entries with their source window title")
        self.cfg_cb_source_tag.setChecked(
            self.config.settings.get("cb_source_tag", False))
        self.cfg_cb_source_tag.setToolTip(
            "Prepends [Window Title] to each clipboard entry\n"
            "so you can track where each clip came from.")
        cbmon_layout.addRow(self.cfg_cb_source_tag)
        cbmon_group.setLayout(cbmon_layout)
        main_layout.addWidget(cbmon_group)

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
        from PyQt6.QtWidgets import QDoubleSpinBox as _QDSB
        self.cfg_vad_threshold = _QDSB()
        self.cfg_vad_threshold.setRange(0.01, 0.99)
        self.cfg_vad_threshold.setSingleStep(0.05)
        self.cfg_vad_threshold.setDecimals(2)
        self.cfg_vad_threshold.setValue(
            float(self.config.settings.get("vad_threshold", 0.5)))
        self.cfg_vad_threshold.setToolTip(
            "How sensitive the microphone activity detector is.\n"
            "This is NOT the same as Confidence Filtering:\n"
            "  VAD Threshold controls when recording STARTS\n"
            "    (is there speech happening right now?).\n"
            "  Confidence Filtering controls whether the TRANSCRIBED TEXT\n"
            "    is accurate enough to keep after Whisper processes it.\n\n"
            "Higher = only starts recording on clearly audible speech.\n"
            "Lower = more sensitive, may trigger on quiet sounds or noise.\n"
            "Default: 0.50")
        advanced_layout.addRow("VAD Threshold:", self.cfg_vad_threshold)
        self.cfg_vad_min_silence = _SB2()
        self.cfg_vad_min_silence.setRange(100, 10000)
        self.cfg_vad_min_silence.setSuffix(" ms")
        self.cfg_vad_min_silence.setValue(
            int(self.config.settings.get("vad_min_silence_ms", 2000)))
        self.cfg_vad_min_silence.setToolTip(
            "How long a pause in speech (in ms) causes VAD to stop and send\n"
            "the audio to Whisper for transcription.\n\n"
            "This is similar to Auto-Pause Silence, but they operate differently:\n"
            "  Auto-Pause Silence uses the volume level to detect silence.\n"
            "  VAD Min Silence uses the Silero neural network to detect\n"
            "    whether speech is actually present (more accurate).\n\n"
            "If VAD is enabled, VAD Min Silence takes precedence.\n"
            "If VAD is disabled, Auto-Pause Silence is used instead.\n\n"
            "Lower = faster response but more sentence fragments.\n"
            "Higher = more complete sentences but more delay.\n"
            "Default: 2000 ms")
        advanced_layout.addRow("VAD Min Silence:", self.cfg_vad_min_silence)
        self.cfg_vad_min_speech = _SB2()
        self.cfg_vad_min_speech.setRange(50, 2000)
        self.cfg_vad_min_speech.setSuffix(" ms")
        self.cfg_vad_min_speech.setValue(
            int(self.config.settings.get("vad_min_speech_ms", 250)))
        self.cfg_vad_min_speech.setToolTip(
            "Minimum speech duration to count as a valid segment.\nDefault: 250 ms")
        advanced_layout.addRow("VAD Min Speech:", self.cfg_vad_min_speech)

        self.btn_setup = QPushButton("GPU Acceleration Setup Guide")
        self.btn_setup.setStyleSheet("background-color: #27ae60; color: white;")
        self.btn_setup.clicked.connect(self.setup_deps)
        advanced_layout.addRow(self.btn_setup)

        btn_open_log = QPushButton("Open Log File")
        btn_open_log.clicked.connect(self.open_log_file)
        advanced_layout.addRow(btn_open_log)

        advanced_group.setLayout(advanced_layout)
        main_layout.addWidget(advanced_group)

        # --- Optional Tools ---
        opt_group = QGroupBox("Optional Tools")
        opt_layout = QVBoxLayout()
        opt_lbl = QLabel(
            "These tools add optional features. WhisperR works fine without them.\n"
            "Hover over each item for installation instructions.")
        opt_lbl.setWordWrap(True)
        opt_lbl.setStyleSheet("color:#888;font-size:11px;")
        opt_layout.addWidget(opt_lbl)

        def _opt_row(tool_name, check_fn, frozen_msg, dev_msg, url):
            """Status row. frozen_msg = instructions for compiled app users.
            dev_msg = instructions for developers building from source."""
            row = QHBoxLayout()
            ok = False
            try: check_fn(); ok = True
            except Exception: pass
            if ok:
                lbl = QLabel(f"✅  {tool_name}  —  installed")
                lbl.setStyleSheet("color:#4caf50;font-size:11px;")
                row.addWidget(lbl)
            else:
                lbl = QLabel(f"❌  {tool_name}  —  not installed")
                lbl.setStyleSheet("color:#cc4444;font-size:11px;")
                row.addWidget(lbl)
                btn = QPushButton("How to install ↗")
                btn.setFixedHeight(22)
                btn.setStyleSheet(
                    "QPushButton{background:#2a2a2a;border:1px solid #555;"
                    "color:#aaa;font-size:10px;padding:1px 8px;border-radius:3px;}"
                    "QPushButton:hover{border-color:#0078d7;color:#fff;}")
                _frozen = getattr(sys, "frozen", False)
                _msg = frozen_msg if _frozen else dev_msg
                _u = url
                def _open(_c=False, _m=_msg, _u2=_u):
                    QMessageBox.information(
                        self, "How to install", _m)
                    import webbrowser; webbrowser.open(_u2)
                btn.clicked.connect(_open)
                row.addWidget(btn)
            row.addStretch()
            return row

        def _chk_harper():
            if not _harper_binary_path():
                raise RuntimeError("harper-ls binary not found")
        def _chk_pandoc():
            import subprocess as _sp2
            if _sp2.run(["pandoc","--version"], capture_output=True,
                        timeout=3).returncode != 0:
                raise RuntimeError
        def _chk_docx():
            import docx  # noqa

        # Pandoc — standalone installer, works both for frozen and source
        _pandoc_msg_frozen = (
            "Pandoc enables exporting to Word (.docx) and PDF.\n\n"
            "Steps:\n"
            "  1. Click OK to open the Pandoc download page\n"
            "  2. Download the Windows installer (.msi)\n"
            "  3. Run the installer — it adds pandoc.exe to your PATH\n"
            "  4. Restart WhisperR\n\n"
            "Size: ~100 MB. Free, no account needed.\n"
            "Clicking OK will open the download page.")

        _docx_msg_frozen = (
            "python-docx (basic Word export) is not available in this build.\n\n"
            "If you have Pandoc installed, Word export will use that instead\n"
            "and python-docx is not needed.\n\n"
            "Clicking OK will open the python-docx documentation page.")
        _docx_msg_dev = (
            "python-docx enables basic Word (.docx) export.\n\n"
            "To bundle it with the app, run this BEFORE building with PyInstaller:\n"
            "  pip install python-docx\n\n"
            "PyInstaller will then include it automatically.\n\n"
            "Note: Pandoc (if installed by the end user) will be used for DOCX export\n"
            "in preference to python-docx, so both can coexist.\n\n"
            "Clicking OK will open the python-docx page.")

        # Harper — auto-download button instead of generic _opt_row
        _harper_ok = False
        try: _chk_harper(); _harper_ok = True
        except Exception: pass
        _harper_row = QHBoxLayout()
        if _harper_ok:
            _hl = QLabel("✅  Harper  (spell & grammar)  —  installed")
            _hl.setStyleSheet("color:#4caf50;font-size:11px;")
            _harper_row.addWidget(_hl)
        else:
            _hl = QLabel("❌  Harper  (spell & grammar)  —  not installed")
            _hl.setStyleSheet("color:#cc4444;font-size:11px;")
            _harper_row.addWidget(_hl)
            _hbtn = QPushButton("Enable Spell && Grammar Checking (Harper)")
            _hbtn.setFixedHeight(24)
            _hbtn.setStyleSheet(
                "QPushButton{background:#0a3a6a;border:1px solid #0078d7;"
                "color:#7ec8ff;font-size:10px;padding:2px 10px;border-radius:3px;}"
                "QPushButton:hover{background:#0a4a8a;}")
            _hbtn.clicked.connect(lambda: self._show_harper_install_dialog())
            _harper_row.addWidget(_hbtn)
        _harper_row.addStretch()
        opt_layout.addLayout(_harper_row)
        opt_layout.addLayout(_opt_row(
            "Pandoc  (Word & PDF export)",
            _chk_pandoc,
            _pandoc_msg_frozen, _pandoc_msg_frozen,
            "https://pandoc.org/installing.html"))
        opt_layout.addLayout(_opt_row(
            "python-docx  (basic Word export, no Pandoc needed)",
            _chk_docx,
            _docx_msg_frozen, _docx_msg_dev,
            "https://python-docx.readthedocs.io"))
        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        # Save button
        btn_s = QPushButton("💾 SAVE ALL SETTINGS")
        btn_s.setObjectName("save_all_settings_btn")
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
    def _autosave_data(self):
        """Sync live UI tables into config and persist to disk immediately.
        Called after any mutation of terms, hallucinations, commands, or prompt
        so changes are never lost even if the user closes without clicking Save.
        """
        try:
            self.config.settings["terms"] = {
                self.terms_table.item(r, 0).text(): self.terms_table.item(r, 1).text()
                for r in range(self.terms_table.rowCount())
                if self.terms_table.item(r, 0) and self.terms_table.item(r, 1)
                and self.terms_table.item(r, 0).text()
            }
            self.config.settings["hallucinations"] = [
                self.hall_list.item(i).text()
                for i in range(self.hall_list.count())
                if self.hall_list.item(i).text().strip()
            ]
            cmds = {}
            for r in range(self.cmd_table.rowCount()):
                pi = self.cmd_table.item(r, 0)
                ci = self.cmd_table.item(r, 1)
                if pi and ci and pi.text().strip():
                    cmds[pi.text().strip()] = ci.text().strip()
            self.config.settings["commands"] = cmds
            self.config.settings["initial_prompt"] = self.prompt_edit.toPlainText()
            self.config.save()
            app_logger.debug("_autosave_data: saved terms/hall/cmds/prompt")
        except Exception as _e:
            app_logger.warning(f"_autosave_data failed: {_e}")

    def _on_linter_toggled(self, name: str, checked: bool):
        """Save linter toggle to config and push to harper-ls immediately."""
        try:
            _harper = self.config.settings.get("harper", {})
            if not isinstance(_harper, dict):
                _harper = {}
            _linters = _harper.get("linters", {})
            if not isinstance(_linters, dict):
                _linters = {}
            _linters[name] = checked
            _harper["linters"] = _linters
            self.config.settings["harper"] = _harper
            self.config.save()
            # Push to harper-ls immediately — no restart needed
            if hasattr(self, "_harper_client") and self._harper_client:
                self._harper_client.refresh_linter_config()
        except Exception as _e:
            app_logger.warning(f"_on_linter_toggled failed for {name}: {_e}")

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
            self._autosave_data()
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
            self._autosave_data()
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
            self._autosave_data()

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
        self._autosave_data()

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
            self._autosave_data()
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


    def _on_settings_tab_changed(self, idx):
        """Flush pending auto-save only when leaving a live-edit tab.
        Tab indices: 0=Main, 1=Transcription Steering, 2=Terms,
                     3=Commands, 4=Hallucinations (live-edit tabs).
        Switching TO any tab (including Settings) should not trigger a flush
        because that could call config.save() with a partial config.
        """
        LIVE_EDIT_TABS = {1, 2, 3, 4}   # tabs that have autosave
        # We only know the NEW index here — check if the timer was running
        # (i.e. a live-edit tab was being edited) and we are now leaving it.
        # Only flush if we are NOT switching INTO one of the live-edit tabs.
        if idx not in LIVE_EDIT_TABS:
            if hasattr(self, "_autosave_timer") and self._autosave_timer.isActive():
                self._autosave_timer.stop()
                self._autosave_tabs()

    def _autosave_tabs(self):
        """Auto-save AI Prompt, Terms, Commands, Hallucinations.
        Detects WHAT changed and builds a specific snapshot reason.
        """
        try:
            cfg = self.config.settings
            changes = []   # human-readable list of what changed

            # AI Prompt
            new_prompt = self.prompt_edit.toPlainText()
            if new_prompt != cfg.get("initial_prompt", ""):
                snippet = new_prompt[:60].replace("\n", " ")
                changes.append(f"AI Prompt edited: \"{snippet}\"")
            cfg["initial_prompt"] = new_prompt
            new_hw = [w.strip() for w in
                      self.hotwords_edit.toPlainText().splitlines() if w.strip()]
            if new_hw != cfg.get("hotwords", []):
                changes.append(f"Hotwords: {len(new_hw)} word(s)")
            cfg["hotwords"] = new_hw

            # Terms
            new_terms = {}
            for r in range(self.terms_table.rowCount()):
                k_item = self.terms_table.item(r, 0)
                v_item = self.terms_table.item(r, 1)
                if k_item and v_item:
                    k = k_item.text().strip().lower()
                    v = v_item.text().strip()
                    if k and v:
                        new_terms[k] = v
            old_terms = cfg.get("terms", {})
            added_t   = [k for k in new_terms if k not in old_terms]
            removed_t = [k for k in old_terms if k not in new_terms]
            changed_t = [k for k in new_terms
                         if k in old_terms and new_terms[k] != old_terms[k]]
            for k in added_t[:3]:
                changes.append(f"Term added: \"{k}\" → \"{new_terms[k]}\"")
            for k in removed_t[:3]:
                changes.append(f"Term removed: \"{k}\"")
            for k in changed_t[:3]:
                changes.append(f"Term changed: \"{k}\" → \"{new_terms[k]}\"")
            if len(added_t) + len(removed_t) + len(changed_t) > 3:
                changes.append(f"...and more term changes")
            cfg["terms"] = new_terms

            # Commands
            new_cmds = {}
            for r in range(self.cmd_table.rowCount()):
                k_item = self.cmd_table.item(r, 0)
                v_item = self.cmd_table.item(r, 1)
                if k_item and v_item:
                    k = k_item.text().strip()
                    v = v_item.text().strip()
                    if k and v:
                        new_cmds[k] = v
            old_cmds = cfg.get("commands", {})
            added_c   = [k for k in new_cmds if k not in old_cmds]
            removed_c = [k for k in old_cmds if k not in new_cmds]
            changed_c = [k for k in new_cmds
                         if k in old_cmds and new_cmds[k] != old_cmds[k]]
            for k in added_c[:2]:
                changes.append(f"Command added: \"{k}\"")
            for k in removed_c[:2]:
                changes.append(f"Command removed: \"{k}\"")
            for k in changed_c[:2]:
                changes.append(f"Command changed: \"{k}\"")
            cfg["commands"] = new_cmds

            # Hallucinations
            new_hall = []
            for r in range(self.hall_list.count()):
                item = self.hall_list.item(r)
                if item:
                    t = item.text().strip()
                    if t:
                        new_hall.append(t)
            old_hall = cfg.get("hallucinations", [])
            old_set  = set(old_hall)
            new_set  = set(new_hall)
            added_h   = list(new_set - old_set)
            removed_h = list(old_set - new_set)
            for h in added_h[:2]:
                changes.append(f"Hallucination added: \"{h[:40]}\"")
            for h in removed_h[:2]:
                changes.append(f"Hallucination removed: \"{h[:40]}\"")
            cfg["hallucinations"] = new_hall

            # Persist and snapshot
            self.config.save()
            if changes:
                reason = "; ".join(changes)
                app_logger.info(f"Auto-saved: {reason}")
                _app_w = next((w for w in QApplication.topLevelWidgets()
                               if w.__class__.__name__ == "WhisperRApp"), None)
                if _app_w:
                    _app_w.take_app_snapshot(reason)
            else:
                app_logger.debug("Auto-save: no changes detected")
        except Exception as _e:
            app_logger.warning(f"Auto-save of tabs failed: {_e}")

    def save_cfg(self):
        # Get reference to save button before any operations
        save_button = self.sender()
        if save_button is None:
            # Called programmatically (not from button click) — find btn
            save_button = self.findChild(QPushButton, "save_all_settings_btn")
        if save_button is None:
            return   # cannot proceed without button reference
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
        
        # Snapshot old values before update so we can diff for snapshot reason
        _cfg_before = {
            k: self.config.settings.get(k)
            for k in ("model", "lang_name", "lang_code", "noise_floor",
                      "speech_vol", "dict_mode", "live_mode", "auto_pause_sec",
                      "compute_pref", "min_confidence", "use_confidence",
                      "timestamps", "translate", "use_vad", "log_level",
                      "version_history_keep", "version_history_infinite",
                      "snapshots_enabled", "snapshots_mode",
                      "snapshots_keep_count", "snapshots_keep_hours",
                      "snapshots_keep_unit",
                      "auto_backup_enabled", "auto_backup_interval",
                      "auto_backup_keep", "cb_source_tag",
                      "min_to_tray", "auto_space",
                      "always_on_top", "aot_main", "aot_editor",
                      "aot_notes", "aot_cheatsheet",
                      "ind_show", "ind_type", "ind_pos", "ind_size",
                      "noise_floor", "speech_vol")
        }

        # Update all settings
        self.config.settings.update({
            "model": self.cfg_model.currentText(),
            "hf_cache_path": self.cfg_hf_path.text(),
            "lang_name": self.cfg_lang.currentText(),
            "lang_code": LANG_MAP[self.cfg_lang.currentText()],
            "audio_folder": self.cfg_folder.text(),
            "clear_exit": self.cfg_clear.isChecked(),
            "save_to_disk": not self.cfg_ram.isChecked(),
            "input_device_name": self.cfg_mic.currentText().strip(),
            "input_device_index": self.cfg_mic.currentData() if self.cfg_mic.currentData() != -1 else None,
            "dict_mode": self.cfg_dict_m.currentText(),
            "manual_sentence_split": self.cfg_mss.isChecked(),
            "mss_break_key": self.cfg_mss_key.text().strip() or "shift",
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
            "hotwords": [w.strip() for w in
                self.hotwords_edit.toPlainText().splitlines() if w.strip()],
            "always_on_top": self.cfg_aot_master.isChecked(),
            "aot_main": self.cfg_aot_main.isChecked(),
            "aot_editor": self.cfg_aot_editor.isChecked(),
            "aot_notes": self.cfg_aot_notes.isChecked(),
            "aot_cheatsheet": self.cfg_aot_cheatsheet.isChecked(),
            "auto_backup_enabled":  self.cfg_backup_enabled.isChecked(),
            "auto_backup_interval": self.cfg_backup_interval.value(),
            "auto_backup_keep":     self.cfg_backup_keep.value(),
            "cb_source_tag":        self.cfg_cb_source_tag.isChecked(),
            "version_history_keep": self.cfg_version_history.value(),
            "version_history_infinite": self.cfg_vh_infinite.isChecked(),
            "snapshots_enabled":     self.cfg_snap_enabled.isChecked(),
            "snapshots_mode":        "count" if self.cfg_snap_mode_count.isChecked() else "duration",
            "snapshots_keep_count":  self.cfg_snap_count.value(),
            "snapshots_keep_hours":  self.cfg_snap_hours.value(),
            "snapshots_keep_unit":   self.cfg_snap_unit.currentText(),
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
            "hotkey_cooldown_ms": self.cfg_hk_cooldown.value(),
            "vad_threshold": round(self.cfg_vad_threshold.value(), 2),
            "vad_min_silence_ms": self.cfg_vad_min_silence.value(),
            "vad_min_speech_ms": self.cfg_vad_min_speech.value(),
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
            self._apply_always_on_top()
            self.scratchpad.append("✓ Settings saved successfully")
            # Build snapshot reason from what actually changed
            _cfg_after = self.config.settings
            _changed = []
            _friendly = {
                "model": "Model",
                "lang_name": "Language",
                "dict_mode": "Dictation mode",
                "live_mode": "Live mode",
                "auto_pause_sec": "Auto-pause silence",
                "noise_floor": "Noise floor",
                "speech_vol": "Speech volume",
                "min_confidence": "Min confidence",
                "use_confidence": "Confidence filter",
                "timestamps": "Timestamps",
                "translate": "Translate",
                "use_vad": "VAD",
                "log_level": "Log level",
                "version_history_keep": "History depth",
                "version_history_infinite": "Infinite history",
                "snapshots_enabled": "Snapshots",
                "snapshots_mode": "Snapshot mode",
                "snapshots_keep_count": "Snapshot count",
                "snapshots_keep_hours": "Snapshot duration",
                "snapshots_keep_unit": "Snapshot unit",
                "auto_backup_enabled": "Auto-backup",
                "auto_backup_interval": "Backup interval",
                "cb_source_tag": "Source tagging",
                "min_to_tray": "Min to tray",
                "always_on_top": "Always on top",
                "ind_show": "Status indicator",
                "ind_type": "Indicator type",
                "ind_pos": "Indicator position",
            }
            for k, label in _friendly.items():
                old_v = _cfg_before.get(k)
                new_v = _cfg_after.get(k)
                if old_v != new_v:
                    # Format booleans nicely
                    if isinstance(new_v, bool):
                        _changed.append(f"{label}: {"ON" if new_v else "OFF"}")
                    else:
                        _changed.append(f"{label}: {old_v!r} → {new_v!r}")
            if _changed:
                _reason = "Settings changed: " + "; ".join(_changed[:6])
                if len(_changed) > 6:
                    _reason += f" (+{len(_changed)-6} more)"
            else:
                _reason = "Settings saved (no changes detected)"
            self.take_app_snapshot(_reason)
            
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
            # Disconnect speech_active so the dying thread cannot
            # re-light the recording indicator after we stop.
            # data_ready stays connected so any already-buffered
            # audio is still submitted and transcribed.
            try:
                self.recorder.speech_active.disconnect(self._on_speech_active)
            except Exception:
                pass
            self.btn_toggle.setText("Start Dictation")
            self._is_listening  = False
            self._speech_active = False
            self._update_app_state()
            app_logger.info("Dictation stopped")
            # Reset MSS flag so next dictation session starts capitalised
            if self.config.settings.get("manual_sentence_split", False):
                _ed2 = getattr(self, "_editor", None)
                if _ed2:
                    _ed2._mss_next_capital = True
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
            
            self.recorder.data_ready.connect(
                lambda d: self.transcriber.submit(d, "live"))
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
        # Ignore spurious signals from a recorder that fired just
        # before its signals were disconnected.
        if not getattr(self, "_is_listening", False):
            return
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
                    # Stop the recorder immediately on PTT release regardless
                    # of live_mode — PTT is always hold-to-talk.
                    if self.recorder and self.recorder.active:
                        self.sig_toggle_rec.emit()

            # ── MSS break-key polling ─────────────────────────────────
            # Poll via GetAsyncKeyState — only fires during active recording.
            # Avoids GlobalHotKeys <shift> which fires on every Ctrl+Shift+...
            if (self.config.settings.get("manual_sentence_split", False)
                    and self.recorder and self.recorder.active):
                _mss_vk = self._VK_MAP.get(
                    self.config.settings.get("mss_break_key", "shift")
                    .lower().strip(), 0)
                import ctypes as _ct_mss
                _mss_raw_key = self.config.settings.get("mss_break_key","shift").lower().strip()
                _mss_parts = [p.strip() for p in _mss_raw_key.replace("<","").replace(">","").split("+")]
                _mss_vks = [self._VK_MAP.get(p, 0) for p in _mss_parts]
                if _mss_vks and all(_mss_vks):
                    _mss_down = all(
                        _ct_mss.windll.user32.GetAsyncKeyState(v) & 0x8000
                        for v in _mss_vks)
                    if _mss_down and not getattr(self, "_mss_key_was_down", False):
                        app_logger.debug(f"MSS break key detected: {_mss_raw_key!r}")
                        self.on_mss_break()
                    self._mss_key_was_down = _mss_down
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

    def start_clipboard_monitor(self, mode="text"):
        """Start (or restart) the app-level clipboard polling timer.
        mode: "text" = append to editor textarea; "notes" = add as new notes.
        """
        self._cb_monitor_mode = mode
        if self._cb_monitor_timer and self._cb_monitor_timer.isActive():
            return  # already running — mode change takes effect immediately
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

        mode = getattr(self, "_cb_monitor_mode", "text")
        if mode == "notes":
            # Create notes window silently if needed
            _nw_poll = getattr(self._editor, "_notes_win", None)
            if not _nw_poll:
                self._editor._notes_win = _NotesWindow(self._editor)
                _nw_poll = self._editor._notes_win
            # Source tagging for notes mode
            _note_text = current
            if self.config.settings.get("cb_source_tag", False):
                try:
                    import ctypes as _ct_stn
                    _hwnd_stn = _ct_stn.windll.user32.GetForegroundWindow()
                    _buf_stn  = _ct_stn.create_unicode_buffer(256)
                    _ct_stn.windll.user32.GetWindowTextW(_hwnd_stn, _buf_stn, 256)
                    _src_stn = _buf_stn.value.strip()
                    if _src_stn:
                        _note_text = f"[{_src_stn}]\n{current}"
                except Exception:
                    pass
            _nw_poll._add_note(text=_note_text)
            # Keep snapshot and app-level flag in sync
            self._editor._saved_notes_snapshot = _nw_poll.get_notes_data()
            self._editor_notes_open = True
        else:
            cur = self._editor.editor.textCursor()
            cur.movePosition(cur.MoveOperation.End)
            existing = self._editor.editor.toPlainText()
            sep = "\n\n" if existing.strip() else ""
            # Source tagging: prepend window title if enabled
            _tagged = current
            if self.config.settings.get("cb_source_tag", False):
                try:
                    import ctypes as _ct_st
                    _hwnd_st = _ct_st.windll.user32.GetForegroundWindow()
                    _buf_st  = _ct_st.create_unicode_buffer(256)
                    _ct_st.windll.user32.GetWindowTextW(_hwnd_st, _buf_st, 256)
                    _src_title = _buf_st.value.strip()
                    if _src_title:
                        _tagged = f"[{_src_title}]\n{current}"
                except Exception:
                    pass
            cur.insertText(sep + _tagged)
            self._editor.editor.setTextCursor(cur)
            if self._editor.isVisible():
                self._editor.editor.ensureCursorVisible()
            # Keep saved content in sync so re-open shows accumulated text
            self._editor_saved_content = self._editor.editor.toPlainText()

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

        # Re-use existing editor (visible or kept alive by clipboard monitor)
        if self._editor:
            if self._editor.isVisible():
                if from_clipboard and prefill:
                    self._editor.editor.setPlainText(prefill)
                self._editor.raise_()
                self._editor.activateWindow()
                return
            # Editor exists but is hidden (monitor kept it) — just show it
            # Don't replace it; it already has all the accumulated content.
            _skip_create = True
        else:
            _skip_create = False

        # Clipboard prefill: always load CURRENT clipboard on open (not saved).
        # The toggle state persists; the content is always fresh each open.
        if not from_clipboard and getattr(self, "_editor_clipboard_prefill", False):
            try:
                import pyperclip as _pc_pf
                prefill = _pc_pf.paste()  # always current, always replaces
            except Exception:
                pass

        if not _skip_create:
            self._editor = WhisperEditor(
                initial_text=prefill,
                config=self.config,
                parent=None)
        # Restore toggle states — block signals to prevent _excl from
        # unchecking the others while we restore each one independently.
        for _tog in (self._editor.remember_toggle,
                     self._editor.clipboard_prefill_toggle,
                     self._editor.clipboard_monitor_toggle):
            _tog.blockSignals(True)
        self._editor.remember_toggle.setChecked(
            bool(getattr(self, "_editor_remember", True)))
        self._editor.clipboard_prefill_toggle.setChecked(
            bool(getattr(self, "_editor_clipboard_prefill", False)))
        self._editor.clipboard_monitor_toggle.setChecked(
            bool(getattr(self, "_editor_cb_monitor_was_on", False)))
        for _tog in (self._editor.remember_toggle,
                     self._editor.clipboard_prefill_toggle,
                     self._editor.clipboard_monitor_toggle):
            _tog.blockSignals(False)
        # If monitor was on, ensure it's still running
        if getattr(self, "_editor_cb_monitor_was_on", False):
            if not (self._cb_monitor_timer and self._cb_monitor_timer.isActive()):
                self.start_clipboard_monitor(
                    mode=getattr(self, "_cb_monitor_mode", "text"))
        # Restore target word count
        if getattr(self, "_editor_saved_target", 0):
            self._editor.target_spin.setValue(self._editor_saved_target)
        # Sync voice label with current dictation state
        self._editor.set_voice_state(
            getattr(self, "_is_listening", False))
        # Restore notes if they were open
        if getattr(self, "_editor_notes_open", False):
            _nw_op = getattr(self._editor, "_notes_win", None)
            if _nw_op is None:
                self._editor._notes_win = _NotesWindow(self._editor)
                _nw_op = self._editor._notes_win
            # Only restore from snapshot for a freshly-created editor.
            # If the editor was kept alive (monitor running), its _notes_win
            # already has the live up-to-date notes — overwriting would
            # discard any notes added by the clipboard poll while hidden.
            if not _skip_create:
                _saved_n = getattr(self, "_editor_saved_notes", [])
                if _saved_n:
                    _nw_op.set_notes_data(_saved_n)
                _saved_filt2 = getattr(self, "_editor_saved_filter", [])
                if _saved_filt2:
                    _nw_op.set_filter_state(_saved_filt2)
            _nw_op.show()
            _nw_op.raise_()
            _btn_n2 = getattr(self._editor, "btn_notes", None)
            if _btn_n2: _btn_n2.setChecked(True)
            self._editor._reposition_panels()
        # Restore cheatsheet visibility from last session
        if getattr(self, "_editor_cheatsheet_open", False):
            _cs3 = getattr(self._editor, "_cheatsheet", None)
            _btn3 = getattr(self._editor, "btn_cheatsheet", None)
            if _cs3 is None and _btn3 is not None:
                self._editor._toggle_cheatsheet()
            elif _cs3 is not None and not _cs3.isVisible():
                _cs3.show()
                self._editor._reposition_panels()
            if _btn3 is not None:
                _btn3.setChecked(True)
        if not _skip_create:
            self._editor.paste_requested.connect(self._editor_paste_to_app)
            # Save content when editor is closed/hidden
            self._editor.finished.connect(self._on_editor_closed)
        # Re-sync clipboard monitor toggle with running state
        if self._cb_monitor_timer and self._cb_monitor_timer.isActive():
            if hasattr(self._editor, "clipboard_monitor_toggle"):
                _mode = getattr(self, "_cb_monitor_mode", "text")
                self._editor.clipboard_monitor_toggle.setChecked(True)
                if _mode == "notes":
                    self._editor.clipboard_monitor_toggle._cb_notes_mode = True
                    self._editor.clipboard_monitor_toggle.setStyleSheet(
                        "QPushButton{background:#001a40;border:2px solid #0088ff;"
                        "color:#44bbff;border-radius:4px;padding:3px 8px;font-weight:bold;}")
                else:
                    self._editor.clipboard_monitor_toggle._cb_notes_mode = False
                    self._editor.clipboard_monitor_toggle.setStyleSheet(
                        "QPushButton{background:#003a1a;border:2px solid #00cc55;"
                        "color:#00ff77;border-radius:4px;padding:3px 8px;font-weight:bold;}")
        self._editor.show()
        self._editor.raise_()
        self._editor.activateWindow()
        self._apply_always_on_top()
        # Reposition panels after Qt processes the show event so geometry is fresh
        QTimer.singleShot(0, self._editor._reposition_panels)
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
        # Capture panel states — use isVisible() directly (btn_cheatsheet aliases btn_notes)
        _nw = getattr(self._editor, "_notes_win", None)
        self._editor_saved_notes = _nw.get_notes_data() if _nw else []
        # Flush pending history entries to disk
        if self._editor:
            try: self._editor._flush_history_to_project()
            except Exception: pass
        self._editor_notes_open = bool(_nw and _nw.isVisible())
        self._editor_saved_filter = (_nw.get_filter_state() if _nw else [])
        _cs_oc = getattr(self._editor, "_cheatsheet", None)
        self._editor_cheatsheet_open = bool(_cs_oc and _cs_oc.isVisible())
        # Persist text/target only when remember is on
        if _remember_on:
            _ts = getattr(self._editor, "target_spin", None)
            if _ts: _ts.interpretText()  # commit any uncommitted typed value
            self._editor_saved_target = _ts.value() if _ts else 0
            self._editor_saved_content = self._editor.editor.toPlainText()
            self._editor_remember = True
        else:
            self._editor_remember = False
            self._editor_saved_content = ""
            self._editor_saved_target = 0
        # Write / delete state JSON
        _state_path = getattr(self, "_editor_state_path", None)
        if _state_path:
            try:
                if _remember_on:
                    import json as _json_sv
                    _state = {
                        "remember":        True,
                        "content":         self._editor_saved_content,
                        "target_words":    self._editor_saved_target,
                        "clipboard_prefill": self._editor_clipboard_prefill,
                        "cb_monitor":      self._editor_cb_monitor_was_on,
                        "notes":           getattr(self, "_editor_saved_notes", []),
                        "notes_open":      getattr(self, "_editor_notes_open", False),
                        "notes_filter":    getattr(self, "_editor_saved_filter", []),
                    }
                    _state_path.write_text(
                        _json_sv.dumps(_state, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                else:
                    if _state_path.exists():
                        _state_path.unlink(missing_ok=True)
            except Exception as _e_sv:
                app_logger.warning(f"Could not save editor state: {_e_sv}")
        # Hide panels so they don't float orphaned after the editor closes
        if _nw and _nw.isVisible():
            _nw.hide()
        if _cs_oc and _cs_oc.isVisible():
            _cs_oc.hide()
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
        Updates config immediately and triggers preload/download right away.
        The user sees progress in the scratchpad without having to save first.
        """
        if not hasattr(self, 'transcriber'):
            return
        # Update config immediately so the worker gets the right model name
        self.config.settings['model'] = model_name
        try:
            self.config.save()
        except Exception:
            pass
        self.take_app_snapshot(f"Model changed to: {model_name}")
        app_logger.info(f"Model changed to: {model_name} — preloading immediately")
        self._model_loading = True
        self._update_app_state()
        self.scratchpad.append(
            f"[System] Model changed to {model_name}\n"
            f"[System] Downloading / loading in background — "
            f"progress shown below...")
        # Drain any pending preloads for previous model selections
        while not self.transcriber._pending.empty():
            try: self.transcriber._pending.get_nowait()
            except Exception: break
        # Queue the new model immediately, passing name directly
        self.transcriber.preload_model(model_name=model_name)

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
                # ── If editor is open AND focused, append text there ──────
                # If the user has clicked away to another app, fall through
                # to the normal paste path so text lands in that app.
                if self._editor and self._editor.isVisible():
                    try:
                        import ctypes as _ct_ed
                        _fg_ed = _ct_ed.windll.user32.GetForegroundWindow()
                        _own_ed = int(self._editor.winId())
                        _main_ed = int(self.winId())
                        _ed_has_focus = (_fg_ed in (_own_ed, _main_ed))
                        # Also check cheatsheet window
                        _cs_ed = getattr(self._editor, "_cheatsheet", None)
                        if _cs_ed and not _ed_has_focus:
                            try:
                                _ed_has_focus = (_fg_ed == int(_cs_ed.winId()))
                            except Exception:
                                pass
                    except Exception:
                        _ed_has_focus = True  # safe fallback
                    if _ed_has_focus:
                        self._editor.append_text(text)
                        return
                    # Editor open but another app is focused — fall through to paste

                # ── Confidence gate (paste path only) ────────────────────
                # Triggers / commands / wizard have already fired above, so
                # confidence filtering here only blocks actual pasting.
                if self.config.settings.get("use_confidence", False):
                    _mc  = float(self.config.settings.get("min_confidence", 0.0))
                    if _mc > 0.0:
                        import math as _math_cf
                        # mc is a probability (0-1); convert to logprob threshold.
                        # threshold = ln(mc), so mc=0.50 → -0.693, mc=0.90 → -0.105.
                        _thr = _math_cf.log(max(_mc, 1e-9))
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
            # MSS break key — only registered when MSS is enabled
            if self.config.settings.get("manual_sentence_split", False):
                _mss_raw = self.config.settings.get("mss_break_key", "shift")
                try:
                    _mss_norm = self.normalize_hotkey(_mss_raw)
                    if _mss_norm and _mss_norm not in hotkey_map:
                        hotkey_map[_mss_norm] = self.on_mss_break
                        app_logger.info(
                            f"MSS break key registered: {_mss_norm!r}")
                except Exception as _me:
                    app_logger.warning(f"MSS break key error: {_me}")
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
    
    # ── App-State Snapshot Engine ─────────────────────────────────────────────
    # Event-driven: take_app_snapshot(reason) is called wherever something changes.
    # Entries buffered in RAM, flushed to whisperr_snapshots.jsonl on disk when
    # 5+ new entries accumulate, every 10 minutes, or on quit.

    def take_app_snapshot(self, reason):
        if not self.config.settings.get("snapshots_enabled", False):
            app_logger.debug(f"take_app_snapshot skipped (disabled): {reason!r}")
            return
        if not hasattr(self, "_app_snapshots_ram"):
            self._app_snapshots_ram = []
            self._app_snapshots_dirty = 0
        snap = self._collect_app_snapshot(reason)
        last = self._app_snapshots_ram[-1] if self._app_snapshots_ram else None
        if last:
            if (last.get("editor_text") == snap["editor_text"]
                    and last.get("editor_notes") == snap["editor_notes"]
                    and last.get("terms") == snap["terms"]
                    and last.get("commands") == snap["commands"]
                    and last.get("model") == snap["model"]):
                return
        self._app_snapshots_ram.append(snap)
        self._app_snapshots_dirty += 1
        _nw = len(snap.get("editor_text", "").split())
        app_logger.info(
            f"App snapshot #{len(self._app_snapshots_ram)}: {reason!r} "
            f"({_nw}w, dirty={self._app_snapshots_dirty})")
        if self._app_snapshots_dirty >= 5:
            self._flush_app_snapshots()
            app_logger.info("App snapshots flushed to disk (threshold)")

    def _collect_app_snapshot(self, reason):
        from datetime import datetime as _dt
        cfg = self.config.settings
        editor_text, editor_notes, editor_target = "", [], 0
        if self._editor:
            editor_text = self._editor.editor.toPlainText()
            _nw = getattr(self._editor, "_notes_win", None)
            editor_notes = _nw.get_notes_data() if _nw else []
            _ts = getattr(self._editor, "target_spin", None)
            if _ts:
                editor_target = _ts.value()
        return {
            "ts":            _dt.now().isoformat(timespec="seconds"),
            "reason":        reason,
            "editor_text":   editor_text,
            "editor_notes":  editor_notes,
            "editor_target": editor_target,
            "model":         cfg.get("model", "tiny"),
            "lang_name":     cfg.get("lang_name", "English"),
            "terms":         dict(cfg.get("terms", {})),
            "commands":      dict(cfg.get("commands", {})),
            "hallucinations":list(cfg.get("hallucinations", [])),
            "initial_prompt":cfg.get("initial_prompt", ""),
        }

    def _flush_app_snapshots(self, final=False):
        import json as _j, os as _os
        dirty = getattr(self, "_app_snapshots_dirty", 0)
        ram   = getattr(self, "_app_snapshots_ram", [])
        if not ram or (dirty == 0 and not final):
            return
        snap_path = Path(BASE_DIR) / "whisperr_snapshots.jsonl"
        cfg = self.config.settings
        try:
            existing = []
            if snap_path.exists():
                for _ln in snap_path.read_text(encoding="utf-8").splitlines():
                    _ln = _ln.strip()
                    if _ln:
                        try: existing.append(_j.loads(_ln))
                        except Exception: pass
            merged = existing + (ram[-dirty:] if dirty else [])
            mode = cfg.get("snapshots_mode", "count")
            if mode == "count":
                keep_n = int(cfg.get("snapshots_keep_count", 60))
                while len(merged) > keep_n:
                    merged.pop(0)
            else:
                from datetime import datetime as _dt2, timedelta as _td
                unit = cfg.get("snapshots_keep_unit", "hours")
                val  = int(cfg.get("snapshots_keep_hours", 24))
                mul  = {"hours": 1, "days": 24, "weeks": 168, "months": 720}
                cut  = _dt2.now() - _td(hours=val * mul.get(unit, 1))
                merged = [s for s in merged
                          if _dt2.fromisoformat(s["ts"]) >= cut]
            out = _os.linesep.join(_j.dumps(s, ensure_ascii=False) for s in merged)
            snap_path.write_text(out + _os.linesep, encoding="utf-8")
            self._app_snapshots_dirty = 0
            self._app_snapshots_ram   = merged
        except Exception as _e:
            app_logger.warning("App snapshot flush failed: %s", _e)

    def _restore_app_snapshot(self, snap):
        ed = self._editor
        if ed:
            ed.editor.setPlainText(snap.get("editor_text", ""))
            _nw = getattr(ed, "_notes_win", None)
            if _nw and snap.get("editor_notes"):
                _nw.set_notes_data(snap["editor_notes"])
            _ts = getattr(ed, "target_spin", None)
            if _ts:
                _ts.setValue(int(snap.get("editor_target", 0)))
        cfg = self.config.settings
        for key in ("terms", "commands", "hallucinations",
                    "initial_prompt", "model", "lang_name"):
            if key in snap:
                cfg[key] = snap[key]
        try: self.config.save()
        except Exception: pass
        rsn = snap.get("reason", "")
        self.scratchpad.append("[System] App state restored from snapshot: " + rsn)

    def _show_app_snapshots(self):
        from PyQt6.QtWidgets import (QDialog, QTreeWidget, QTreeWidgetItem,
                                     QDialogButtonBox, QLabel)
        from PyQt6.QtCore import Qt as _Qt2
        from datetime import datetime as _dt
        from collections import defaultdict as _dd2
        import json as _j

        snaps = []
        snap_path = Path(BASE_DIR) / "whisperr_snapshots.jsonl"
        if snap_path.exists():
            for _ln in snap_path.read_text(encoding="utf-8").splitlines():
                _ln = _ln.strip()
                if _ln:
                    try: snaps.append(_j.loads(_ln))
                    except Exception: pass
        ram = getattr(self, "_app_snapshots_ram", [])
        on_disk = {s["ts"] for s in snaps}
        snaps += [s for s in ram if s["ts"] not in on_disk]
        snaps.sort(key=lambda s: s["ts"])

        if not snaps:
            QMessageBox.information(
                self, "App State Snapshots",
                "No snapshots yet.\n\n"
                "Enable app-state snapshots in Settings.\n"
                "Snapshots are taken when you save settings, modify terms/commands,\n"
                "change models, or type in the editor.")
            return

        parent_w = self._editor if self._editor else self
        dlg = QDialog(parent_w)
        dlg.setWindowTitle("App State Snapshots")
        dlg.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        dlg.resize(760, 520)
        lay = QVBoxLayout(dlg)
        lbl = QLabel(str(len(snaps)) + " snapshot(s)  —  select one to restore:")
        lbl.setStyleSheet("color:#aaa;font-size:11px;")
        lay.addWidget(lbl)

        tree = QTreeWidget()
        tree.setColumnCount(4)
        tree.setHeaderLabels(["Date / Time", "Words", "Trigger", "Preview"])
        tree.setColumnWidth(0, 155)
        tree.setColumnWidth(1, 55)
        tree.setColumnWidth(2, 205)
        tree.setColumnWidth(3, 235)
        tree.setStyleSheet(
            "QTreeWidget{background:#1e1e1e;color:#ddd;border:1px solid #444;}"
            "QTreeWidget::item:selected{background:#1a3a5c;}"
            "QHeaderView::section{background:#2a2a2a;color:#aaa;"
            "border:1px solid #333;padding:2px 4px;}")

        by_day = _dd2(list)
        for i, s in enumerate(snaps):
            try: dt = _dt.fromisoformat(s["ts"])
            except Exception: dt = _dt.now()
            by_day[dt.strftime("%Y-%m-%d")].append((i, dt, s))

        all_days  = sorted(by_day.keys())
        span_days = ((_dt.fromisoformat(all_days[-1]) -
                      _dt.fromisoformat(all_days[0])).days + 1
                     if len(all_days) > 1 else 1)

        def _leaf(i, dt, s):
            txt  = s.get("editor_text", "")
            prev = txt[:55].replace("\n", " ")
            if len(txt) > 55:
                prev += "..."
            item = QTreeWidgetItem([
                dt.strftime("%Y-%m-%d %H:%M:%S"),
                str(len(txt.split())),
                s.get("reason", ""),
                prev,
            ])
            item.setData(0, _Qt2.ItemDataRole.UserRole, i)
            return item

        if span_days <= 1:
            for i, dt, s in reversed(by_day[all_days[0]]):
                tree.addTopLevelItem(_leaf(i, dt, s))
        elif span_days <= 7:
            for day in reversed(all_days):
                di = QTreeWidgetItem(
                    [_dt.fromisoformat(day).strftime("%A, %b %d %Y"), "", "", ""])
                di.setExpanded(True)
                for i, dt, s in reversed(by_day[day]):
                    di.addChild(_leaf(i, dt, s))
                tree.addTopLevelItem(di)
        else:
            from collections import defaultdict as _dd3
            by_month = _dd3(lambda: _dd3(lambda: _dd3(list)))
            for day in all_days:
                dt_d = _dt.fromisoformat(day)
                mo = dt_d.strftime("%Y-%m")
                wk = dt_d.strftime("%Y-W%W")
                by_month[mo][wk][day].extend(by_day[day])
            for mo in reversed(sorted(by_month.keys())):
                mo_dt = _dt.strptime(mo + "-01", "%Y-%m-%d")
                mi = QTreeWidgetItem([mo_dt.strftime("%B %Y"), "", "", ""])
                mi.setExpanded(True)
                for wk in reversed(sorted(by_month[mo].keys())):
                    wk_dt = _dt.strptime(wk + "-1", "%Y-W%W-%w")
                    wi = QTreeWidgetItem(
                        ["Week of " + wk_dt.strftime("%b %d"), "", "", ""])
                    wi.setExpanded(False)
                    for day in reversed(sorted(by_month[mo][wk].keys())):
                        di = QTreeWidgetItem(
                            [_dt.fromisoformat(day).strftime("%a %b %d"), "", "", ""])
                        di.setExpanded(False)
                        for i, dt, s in reversed(by_month[mo][wk][day]):
                            di.addChild(_leaf(i, dt, s))
                        wi.addChild(di)
                    mi.addChild(wi)
                tree.addTopLevelItem(mi)

        lay.addWidget(tree)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() == QDialog.DialogCode.Accepted:
            items = tree.selectedItems()
            if items:
                idx = items[0].data(0, _Qt2.ItemDataRole.UserRole)
                if idx is not None:
                    rsn = snaps[idx].get("reason", "")
                    reply = QMessageBox.question(
                        parent_w, "Restore Snapshot",
                        "Restore snapshot:\n\"" + rsn + "\"\n\n"
                        "This replaces current editor text, notes, and settings.\n"
                        "This cannot be undone.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.Yes:
                        self._restore_app_snapshot(snaps[idx])

    def _show_harper_install_dialog(self):
        """Show the Harper installation dialog with auto-download option."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                                     QDialogButtonBox, QPushButton, QProgressBar)

        dlg = QDialog(self)
        dlg.setWindowTitle("Enable Spell & Grammar Checking")
        dlg.setMinimumWidth(520)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        title = QLabel("📝  Harper — Fast, Private Grammar & Spell Checking")
        title.setStyleSheet("font-weight:bold;font-size:13px;color:#ddd;")
        lay.addWidget(title)

        desc = QLabel(
            "Harper checks your spelling and grammar as you type — completely "
            "offline, with no data sent anywhere. It's fast, lightweight, and "
            "free.\n\n"
            "To use it, WhisperR needs the <b>harper-ls</b> executable placed in "
            "the same folder as WhisperR.exe.\n\n"
            "<b>Manual install:</b> Download <tt>harper-ls.exe</tt> from "
            "<tt>https://github.com/Automattic/harper/releases/latest</tt> "
            "and place it next to WhisperR.exe.")
        desc.setWordWrap(True)
        desc.setOpenExternalLinks(True)
        desc.setStyleSheet("color:#ccc;font-size:11px;")
        lay.addWidget(desc)

        status_lbl = QLabel("")
        status_lbl.setWordWrap(True)
        status_lbl.setStyleSheet("color:#aaa;font-size:10px;")
        lay.addWidget(status_lbl)

        progress = QProgressBar()
        progress.setRange(0, 0)   # indeterminate
        progress.setVisible(False)
        progress.setStyleSheet(
            "QProgressBar{border:1px solid #444;border-radius:3px;background:#1e1e1e;}"
            "QProgressBar::chunk{background:#0078d7;}")
        lay.addWidget(progress)

        auto_btn = QPushButton("⬇  Try Installing Automatically")
        auto_btn.setStyleSheet(
            "QPushButton{background:#0a3a6a;border:1px solid #0078d7;color:#7ec8ff;"
            "padding:6px 16px;border-radius:4px;font-weight:bold;}"
            "QPushButton:hover{background:#0a4a8a;}"
            "QPushButton:disabled{background:#1a1a1a;color:#555;border-color:#333;}")

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btns.accepted.connect(dlg.accept)

        lay.addWidget(auto_btn)
        lay.addWidget(btns)

        def _on_progress(msg):
            status_lbl.setText(msg)
            QApplication.processEvents()

        def _on_done(success, msg):
            progress.setVisible(False)
            auto_btn.setEnabled(True)
            if success:
                version = msg
                # Save to config
                self.config.settings["harper"] = {
                    "installed": True,
                    "version": version,
                }
                try: self.config.save()
                except Exception: pass
                status_lbl.setStyleSheet("color:#4caf50;font-size:10px;")
                status_lbl.setText(
                    f"✅  Harper {version} installed successfully!\n"
                    "Spell checking is now active in the text editor.\n"
                    "Reopen Settings to see the updated status.")
                # Update the harper status label in Optional Tools if visible
                for _hl_lbl in self.findChildren(QLabel):
                    if "Harper" in _hl_lbl.text() and "not installed" in _hl_lbl.text():
                        _hl_lbl.setText("✅  Harper  (spell & grammar)  —  installed")
                        _hl_lbl.setStyleSheet("color:#4caf50;font-size:11px;")
                        break
                # Start harper in any open editors
                for w in QApplication.topLevelWidgets():
                    ed = getattr(w, "_editor", None)
                    if ed:
                        ed._start_harper()
            else:
                status_lbl.setStyleSheet("color:#ff6b6b;font-size:10px;")
                status_lbl.setText(f"❌  {msg}")

        def _auto_install():
            auto_btn.setEnabled(False)
            progress.setVisible(True)
            status_lbl.setStyleSheet("color:#aaa;font-size:10px;")
            _harper_download(
                progress_cb=_on_progress,
                done_cb=_on_done)

        auto_btn.clicked.connect(_auto_install)
        dlg.exec()

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
                            _nw_q = None
                            if self._editor:
                                _nw_q = getattr(self._editor, "_notes_win", None)
                            _notes_q = (_nw_q.get_notes_data() if _nw_q
                                        else getattr(self, "_editor_saved_notes", []))
                            _sp.write_text(_json_quit.dumps({
                                "remember": True,
                                "content":  _content_to_save,
                                "target_words": _target,
                                "clipboard_prefill": getattr(self, "_editor_clipboard_prefill", False),
                                "cb_monitor": getattr(self, "_editor_cb_monitor_was_on", False),
                                "notes": _notes_q,
                                "notes_open": getattr(self, "_editor_notes_open", False),
                                "notes_filter": getattr(self, "_editor_saved_filter", []),
                            }, ensure_ascii=False, indent=2), encoding="utf-8")
                        elif _sp.exists():
                            _sp.unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception as _e:
            app_logger.warning(f"Could not save editor content: {_e}")
        # Flush any pending app-state snapshots to disk before exit
        try: self._flush_app_snapshots(final=True)
        except Exception: pass
        QApplication.instance().quit()

    def on_mss_break(self):
        """Called when the MSS sentence-break key is pressed.
        Only acts when dictation is currently active — avoids firing
        during normal typing or when the editor is idle.
        """
        # Only act while the recorder is actively recording
        rec = getattr(self, "recorder", None)
        if not rec or not rec.active:
            return
        from PyQt6.QtCore import QTimer as _QTmss
        def _do_break():
            _ed = getattr(self, "_editor", None)
            if _ed and _ed.isVisible():
                _ed._mss_sentence_break()
        _QTmss.singleShot(0, _do_break)

    def _hk_allowed(self) -> bool:
        """Return True if enough time has passed since the last hotkey fired."""
        import time as _t
        cooldown = self.config.settings.get("hotkey_cooldown_ms", 400) / 1000.0
        now = _t.monotonic()
        if now - self._last_hk_time < cooldown:
            app_logger.debug(
                f"Hotkey suppressed (cooldown {cooldown*1000:.0f}ms active)")
            return False
        self._last_hk_time = now
        return True

    def on_toggle_hotkey(self):
        """Handler for toggle dictation hotkey - prevents subset conflicts.
        Captures foreground hwnd HERE (keyboard-thread) before Qt shifts focus.
        Ignores the press when Shift is held (user is pressing the visibility
        superset combo Ctrl+Shift+Alt+Z and we must not steal the event).
        """
        # If the visibility hotkey is a superset (adds Shift to our combo),
        # skip when Shift is physically held — user pressed the visibility combo.
        # Use GetAsyncKeyState for reliable real-time key state on Windows.
        try:
            _vis_needs_shift = "shift" in getattr(
                self, "visibility_hotkey_normalized", "").lower()
            if _vis_needs_shift:
                import ctypes as _ctks
                VK_SHIFT = 0x10
                # High bit set = key is currently physically down
                _shift_down = bool(
                    _ctks.windll.user32.GetAsyncKeyState(VK_SHIFT) & 0x8000)
                if _shift_down:
                    app_logger.debug(
                        "Toggle hotkey skipped — Shift held (visibility combo)")
                    return
        except Exception:
            pass
        if not self._hk_allowed(): return
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
        """Handler for visibility hotkey.
        Does NOT use the shared cooldown — the toggle hotkey's cooldown must
        not suppress this. The Shift-state check in on_toggle_hotkey prevents
        the reverse problem (toggle firing during visibility combo).
        """
        app_logger.debug("Visibility hotkey triggered (exact match)")
        self.sig_toggle_vis.emit()

    def on_editor_hotkey(self):
        """Handler for editor toggle hotkey — runs on hotkey thread, emit to Qt."""
        if not self._hk_allowed(): return
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
        if not self._hk_allowed(): return
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
                self._editor.target_spin.interpretText()
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
                    _nw_tew = getattr(self._editor, "_notes_win", None)
                    _notes_tew = _nw_tew.get_notes_data() if _nw_tew else getattr(self, "_editor_saved_notes", [])
                    _sp.write_text(_json_tew.dumps({
                        "remember": True,
                        "content":  self._editor_saved_content,
                        "target_words": self._editor_saved_target,
                        "clipboard_prefill": getattr(self, "_editor_clipboard_prefill", False),
                        "cb_monitor": self._editor_cb_monitor_was_on,
                        "notes": _notes_tew,
                        "notes_open": bool(_nw_tew and _nw_tew.isVisible()),
                        "notes_filter": (_nw_tew.get_filter_state() if _nw_tew else []),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
            # Remember cheatsheet and notes open state before hiding
            _cs = getattr(self._editor, "_cheatsheet", None)
            # Use isVisible() directly — btn_cheatsheet is aliased to btn_notes
            self._editor_cheatsheet_open = bool(_cs and _cs.isVisible())
            _nw = getattr(self._editor, "_notes_win", None)
            self._editor_notes_open = bool(_nw and _nw.isVisible())
            # Snapshot notes content before hiding
            self._editor_saved_notes = _nw.get_notes_data() if _nw else []
            self._editor_saved_filter = (_nw.get_filter_state() if _nw else [])
            self._editor.hide()
            # Hide panels together with editor
            if _cs and _cs.isVisible():
                _cs.hide()
            if _nw and _nw.isVisible():
                _nw.hide()
        else:
            # Restore or open fresh
            prefill = self._editor_saved_content if self._editor_remember else ""
            self._open_editor(prefill=prefill)
            # Restore remember toggle state
            if self._editor and self._editor_remember:
                self._editor.remember_toggle.setChecked(True)
            # Sync voice label with current dictation state
            if self._editor:
                self._editor.set_voice_state(
                    getattr(self, "_is_listening", False))
            # Re-show notes if they were open
            _editor_notes_open = getattr(self, "_editor_notes_open", False)
            _monitor_live = bool(
                self._cb_monitor_timer and self._cb_monitor_timer.isActive())
            if self._editor and (_editor_notes_open or _monitor_live):
                _nw2 = getattr(self._editor, "_notes_win", None)
                _nw2_was_none = _nw2 is None
                if _nw2 is None:
                    self._editor._notes_win = _NotesWindow(self._editor)
                    _nw2 = self._editor._notes_win
                # Only restore from snapshot when notes window was freshly created.
                # If it already existed (editor kept alive by monitor), its notes
                # are live and must not be overwritten with the stale snapshot.
                if _nw2_was_none:
                    _saved_notes = getattr(self, "_editor_saved_notes", [])
                    if _saved_notes:
                        _nw2.set_notes_data(_saved_notes)
                if _editor_notes_open:
                    _nw2.show()
                    _nw2.raise_()
                    _btn_n = getattr(self._editor, "btn_notes", None)
                    if _btn_n: _btn_n.setChecked(True)
                    self._editor._reposition_panels()
            # Re-show cheatsheet if it was open
            if self._editor and getattr(self, "_editor_cheatsheet_open", False):
                _cs2 = getattr(self._editor, "_cheatsheet", None)
                _btn2 = getattr(self._editor, "btn_cheatsheet", None)
                if _cs2 is None and _btn2 is not None:
                    self._editor._toggle_cheatsheet()
                elif _cs2 is not None and not _cs2.isVisible():
                    _cs2.show()
                    self._editor._reposition_panels()
                if _btn2 is not None:
                    _btn2.setChecked(True)
    

    def _apply_always_on_top(self):
        """Apply WindowStaysOnTopHint to each window per settings."""
        cfg = self.config.settings

        def _set_aot(window, enabled):
            if not window:
                return
            flags = window.windowFlags()
            if enabled:
                flags |= Qt.WindowType.WindowStaysOnTopHint
            else:
                flags &= ~Qt.WindowType.WindowStaysOnTopHint
            was_visible = window.isVisible()
            window.setWindowFlags(flags)
            if was_visible:
                window.show()

        _ed = getattr(self, "_editor", None)
        _set_aot(self,  cfg.get("aot_main",        False))
        _set_aot(_ed,   cfg.get("aot_editor",      False))
        _set_aot(getattr(_ed, "_notes_win",  None), cfg.get("aot_notes",      False))
        _set_aot(getattr(_ed, "_cheatsheet", None), cfg.get("aot_cheatsheet", False))


    def _on_model_not_found(self, model_name: str, dest_path: str, hf_url: str):
        """Show a helpful dialog when a model can't be loaded or downloaded."""
        app_logger.warning(f"Model not found: {model_name}, dest={dest_path}")
        try:
            self._show_model_not_found_dialog(model_name, dest_path, hf_url)
        except Exception as _e:
            app_logger.error(f"model_not_found dialog failed: {_e}", exc_info=True)
            QMessageBox.critical(self, f"Model Not Found — {model_name}",
                f"Could not load model '{model_name}'.\n\n"
                f"Target folder:\n{dest_path}\n\n"
                f"Download from:\n{hf_url}")

    def _show_model_not_found_dialog(self, model_name: str, dest_path: str, hf_url: str):
        """Inner implementation — separated so exceptions surface cleanly."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel,
                                     QPushButton, QHBoxLayout, QTextEdit)
        import webbrowser as _wb

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Model Not Found — {model_name}")
        dlg.resize(580, 440)
        lay = QVBoxLayout(dlg)
        lay.setSpacing(10)

        lbl = QLabel(
            f"<b>WhisperR could not load or download: <code>{model_name}</code></b><br><br>"
            "This can happen when:<br>"
            "&nbsp;• The model has never been downloaded on this machine<br>"
            "&nbsp;• HuggingFace is unreachable (firewall, proxy, or no internet)<br>"
            "&nbsp;• The cached model folder is incomplete or corrupted<br><br>"
            "<b>How to fix it:</b><br>"
            "&nbsp;1. Click <i>Open Download Page</i> and download all model files.<br>"
            "&nbsp;2. Save them into the <b>Target folder</b> shown below (copy it first).<br>"
            "&nbsp;3. Restart WhisperR.<br><br>"
            "<b>If Windows Firewall is blocking the app:</b><br>"
            "&nbsp;• Open <i>Windows Defender Firewall → Allow an app through firewall</i><br>"
            "&nbsp;• Add WhisperR.exe to the allowed list for Private <i>and</i> Public networks.<br>"
            "&nbsp;• Then restart WhisperR — it will download the model automatically.")
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(lbl)

        path_lbl = QLabel("Target folder (save downloaded files here):")
        path_lbl.setStyleSheet("font-weight:bold;margin-top:4px;")
        lay.addWidget(path_lbl)
        path_box = QTextEdit()
        path_box.setPlainText(dest_path)
        path_box.setReadOnly(True)
        path_box.setFixedHeight(44)
        path_box.setStyleSheet(
            "background:#1a1a1a;color:#88ccff;border:1px solid #444;"
            "font-family:monospace;padding:4px;font-size:9pt;")
        lay.addWidget(path_box)

        btn_row = QHBoxLayout()
        _ss = ("QPushButton{background:#2a2a2a;border:1px solid #555;color:#ddd;"
               "padding:5px 14px;border-radius:4px;font-size:9pt;}"
               "QPushButton:hover{background:#353535;border-color:#0078d7;}")

        btn_clip = QPushButton("📋  Copy Path")
        btn_clip.setStyleSheet(_ss)
        def _copy_path():
            QApplication.clipboard().setText(dest_path)
            btn_clip.setText("✓  Copied!")
        btn_clip.clicked.connect(_copy_path)
        btn_row.addWidget(btn_clip)

        btn_browser = QPushButton("🌐  Open Download Page")
        btn_browser.setStyleSheet(_ss)
        btn_browser.clicked.connect(lambda: _wb.open(hf_url))
        btn_row.addWidget(btn_browser)

        btn_fw = QPushButton("🔒  Firewall Help")
        btn_fw.setStyleSheet(_ss)
        btn_fw.clicked.connect(lambda: _wb.open(
            "ms-settings:windowsdefender"))
        btn_row.addWidget(btn_fw)

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(_ss)
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)

        lay.addLayout(btn_row)
        dlg.exec()

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

    def _ft_cancel(self):
        self._ft_cancelled = True
        self.ft_status_lbl.setText("Cancelling...")

    def _ft_start_transcription(self):
        """Submit all queued files to the transcriber."""
        n = self.ft_list.count()
        app_logger.debug(f"_ft_start_transcription: ft_list.count()={n}")
        paths = [self.ft_list.item(i).text() for i in range(n)]
        if not paths:
            self.ft_status_lbl.setText("Queue is empty — use Add Files... to add audio files.")
            self.scratchpad.append("[File] Transcribe button pressed but queue is empty.")
            return
        self._ft_pending   = list(paths)
        self._ft_total     = len(paths)
        self._ft_cancelled = False
        self.ft_progress.setValue(0)
        self.ft_progress.setVisible(True)
        self.ft_cancel_btn.setVisible(True)
        self.ft_start_btn.setEnabled(False)
        self.ft_status_lbl.setText(
            f"Queuing {self._ft_total} file(s) for transcription...")
        self.scratchpad.append(
            f"[File] Starting transcription of {self._ft_total} file(s).")
        submitted = 0
        for i, p in enumerate(paths):
            if self._ft_cancelled:
                self.ft_status_lbl.setText(
                    f"Cancelled. {submitted}/{self._ft_total} file(s) queued.")
                break
            if not os.path.isfile(p):
                self.scratchpad.append(f"[File] Skipped (not found): {p}")
                app_logger.warning(
                    f"_ft_start_transcription: file not found: {p}")
                continue
            pct = int((i + 1) * 100 / self._ft_total)
            self.ft_progress.setValue(pct)
            self.ft_status_lbl.setText(
                f"[{i+1}/{self._ft_total}] Queuing: {Path(p).name}")
            QApplication.processEvents()
            self.transcriber.submit(p, p)
            app_logger.info(f"_ft_start_transcription: submitted {p}")
            submitted += 1
        if not self._ft_cancelled:
            self.ft_status_lbl.setText(
                f"All {submitted} file(s) queued. Transcribing in background...")
        self.ft_progress.setVisible(False)
        self.ft_cancel_btn.setVisible(False)
        self.ft_start_btn.setEnabled(True)
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

    def _import_hotwords(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Hotwords", "", "Text Files (*.txt);;All Files (*)")
        if path:
            try:
                text = Path(path).read_text(encoding="utf-8")
                self.hotwords_edit.setPlainText(text.strip())
            except Exception as e:
                QMessageBox.warning(self, "Import failed", str(e))

    def _export_hotwords(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Hotwords", "hotwords.txt",
            "Text Files (*.txt);;All Files (*)")
        if path:
            try:
                Path(path).write_text(
                    self.hotwords_edit.toPlainText(), encoding="utf-8")
            except Exception as e:
                QMessageBox.warning(self, "Export failed", str(e))

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
        
        # Flush hotwords and prompt from Settings UI to config on close
        try:
            hw_edit = getattr(self, "hotwords_edit", None)
            if hw_edit:
                self.config.settings["hotwords"] = [
                    w.strip() for w in hw_edit.toPlainText().splitlines()
                    if w.strip()]
                self.config.save()
        except Exception:
            pass
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


# Single-instance mutex handle — kept at module level so it is never GC'd
_WHISPERR_MUTEX_HANDLE = None

if __name__ == "__main__":
    # freeze_support() already called at module top — do not call again.
    app_logger.info("="*60)
    app_logger.info(f"{APP_NAME} v{__version__} - Starting")
    app_logger.info(f"Python: {sys.version}")
    app_logger.info(f"Platform: {sys.platform}\"")
    app_logger.info("="*60)
    
    # ── Single-instance guard ────────────────────────────────────
    # Handle stored at module level (_WHISPERR_MUTEX_HANDLE) so Python
    # never GC's it, keeping the Win32 mutex owned for the process lifetime.
    if sys.platform == "win32":
        try:
            import ctypes as _ctm
            _WHISPERR_MUTEX_HANDLE = _ctm.windll.kernel32.CreateMutexW(
                None, True, "WhisperR_SingleInstance_v210")
            if _ctm.windll.kernel32.GetLastError() == 183:
                # Another instance owns the mutex — show message and exit
                try:
                    _ctm.windll.kernel32.CloseHandle(_WHISPERR_MUTEX_HANDLE)
                except Exception: pass
                import tkinter as _tk, tkinter.messagebox as _mb
                _r = _tk.Tk(); _r.withdraw()
                _mb.showerror("WhisperR Already Running",
                    "WhisperR is already running.\nCheck the system tray.")
                _r.destroy()
                sys.exit(0)
        except SystemExit:
            raise
        except Exception as _me:
            app_logger.warning(f"Single-instance check failed: {_me}")
    try:
        app_logger.debug("→ Creating QApplication instance...")
        app = QApplication(sys.argv)
        app_logger.debug("✓ QApplication created")
        
        # Don't quit when the main window is hidden (e.g. minimized to tray)
        app.setQuitOnLastWindowClosed(False)
        app_logger.debug("✓ setQuitOnLastWindowClosed(False) set")
        
        app_logger.debug("→ Applying dark stylesheet...")
        # Force dark palette so Windows system theme doesn't
        # inject white backgrounds into native controls
        from PyQt6.QtGui import QPalette, QColor as _QColor
        _pal = QPalette()
        _dark  = _QColor("#121212")
        _mid   = _QColor("#1e1e1e")
        _light = _QColor("#e0e0e0")
        _acc   = _QColor("#0078d7")
        for _role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base,
                      QPalette.ColorRole.AlternateBase, QPalette.ColorRole.ToolTipBase):
            _pal.setColor(_role, _dark)
        for _role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                      QPalette.ColorRole.ButtonText, QPalette.ColorRole.ToolTipText,
                      QPalette.ColorRole.BrightText):
            _pal.setColor(_role, _light)
        _pal.setColor(QPalette.ColorRole.Button, _mid)
        _pal.setColor(QPalette.ColorRole.Highlight, _acc)
        _pal.setColor(QPalette.ColorRole.HighlightedText, _QColor("#ffffff"))
        app.setPalette(_pal)
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
