# WhisperR

**Free, local, offline-capable AI speech-to-text — plus a full-featured writing and research workbench. No subscription. No cloud. No one eavesdropping.**

![WhisperR v2.1.0](https://img.shields.io/badge/version-2.1.0-blue) ![Platform](https://img.shields.io/badge/platform-Windows-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/python-3.12-yellow)

---

## What is WhisperR?

WhisperR is a Windows desktop application that turns your voice into text — accurately, privately, and completely free — using OpenAI's Whisper AI models running entirely on your own machine. No internet required after the first model download. No API key. No monthly bill. No one on a server somewhere transcribing your conversations about your novel, your medical notes, or your late-night rants about office politics.

It also happens to ship with a surprisingly capable writing and research editor that professionals, students, and content creators have found genuinely useful for day-to-day work. That part is also free. All of it is free. We are not sorry about this.

## Vibe-coding alert!

WhisperR is **vibe-coded**, initially with Google Gemini and, later, using Claude.

A lot of work has gone in its design and the way all its features work and coexist to enable a voice-empowered writing/content-crafting workflow. It's the result of over thirty years of experience producing articles, tutorials, guides, and ebooks, for print and the web.

However, "I get it" if you don't like that - honestly, for remember, my very line of work is writing, and that's one of the fields hurt the most by modern LLMs. Still, if you're one of the fine folks who brand everything created with AI-assistance as "AI slop", and would prefer I'd actually copy-pasted snippets of code one-by-one from other GitHub projects and Stack Overflow, changed three lines and four variables, and called it a day, instead of having an LLM "translate" dozens of pages of detailed instructions of "how this should look and work" into python code, well, you'd better skip this :-)

---

## Why WhisperR instead of [insert paid service here]?

Let's be blunt. You've probably looked at — or already pay for — one of these:

- A cloud transcription service charging per minute of audio
- A "smart" note-taking app with a $12/month "Pro" tier that unlocks basic features
- An AI writing assistant that requires a subscription to do things your grandmother's typewriter could do for free
- Some combination of the above, each with its own login, privacy policy, and quarterly price increase

WhisperR does what all of them do, runs offline on hardware you already own, and costs nothing. The AI model that powers it (OpenAI's Whisper, in its faster-whisper incarnation) is the same underlying technology many of those services are built on — you're just cutting out the middleman, the server costs, and the investor expectations that make "free" tiers progressively worse every six months.

**The tradeoff:** WhisperR is a desktop app, not a cloud service. It doesn't sync to twelve devices automatically. It doesn't have a mobile app. If you want your notes on your phone, you'll have to copy them there yourself, like it's 2008, and life will go on.

---

## The Two Halves of WhisperR

WhisperR is two apps in one, and both halves are first-class citizens.

### Half One: The Dictation Engine

Point a microphone at your mouth, press a hotkey, talk, let go. Your words appear wherever your cursor is — in Word, Notepad, a browser textarea, your email client, anywhere. That's the core loop, and it works remarkably well.

- **Multiple Whisper models** from tiny (fast, runs on a potato) to large-v3 (extremely accurate, needs a decent GPU). Models are downloaded once and cached locally.
- **GPU acceleration** via CUDA for dramatically faster transcription on NVIDIA cards.
- **Auto-pause mode** that starts and stops recording based on silence detection — hands-free dictation that doesn't require button-pressing.
- **Push-to-talk mode** for when you're in a noisy environment or don't trust the silence detection.
- **Language support** for dozens of languages, with automatic detection if you're feeling spontaneous.
- **Translation** — transcribe non-English speech directly into English text.
- **Transcription Steering** — guide Whisper toward domain-specific vocabulary with a steering prompt (context-based) and a hotwords list (acoustically matched). Both are in the dedicated Transcription Steering settings tab.
- **Terms / Text Expansion** — teach WhisperR that when you say "hexagon software" you mean "Hexagon Software™", or that "contact info" should expand to your full email signature. Works for both voice input and typed text.
- **Voice Commands** — trigger custom actions, run scripts, or launch applications by saying predefined phrases.
- **Batch file transcription** — queue multiple audio files for transcription with a progress bar and cancel button; drop them in or use the Add Files browser
- **Folder monitoring** — drop audio files into a watched folder and have them transcribed automatically.
- **Confidence filtering** — discard low-confidence transcriptions that are more likely hallucinations than words you actually said.

### Half Two: The Writing and Research Editor

The editor is where WhisperR earns the "research workbench" description. It is emphatically not a pretty Markdown previewer with ambient sounds. It is a tool for people who produce large amounts of text and need to keep their thoughts organized while doing so.

**Core editing:**
- Distraction-free writing environment with word count and configurable word targets
- Full Markdown formatting via toolbar buttons or keyboard shortcuts (bold, italic, headings, lists, code, task lists, keyboard tags, and more)
- Find & Replace with regex support, full undo, and a match counter
- Tag-wrap for HTML/XML — select text, name a tag, get `<tag>text</tag>` instantly
- Link insertion with placeholder URL or automatic clipboard URL

**Research and organization:**
- **Notes panel** — a floating sticky-note panel that lives beside the editor. Ten color options, drag-to-reorder, collapsible, and filterable by color. Ctrl+Enter to add a note without reaching for the mouse. Use colors as priority levels (red = urgent, green = done, yellow = in progress), to group notes by topic, to mark source reliability, or however else your brain works — the filter lets you focus on just the colors you want to see.
- **Clipboard monitor (text mode)** — leave it running in the background and everything you copy from anywhere gets silently appended to your editor. Open a dozen browser tabs, copy quotes and statistics as you read, come back to a nicely accumulated buffer. The editor doesn't even need to be visible.
- **Clipboard monitor (notes mode)** — same idea, but each clipboard entry creates a new note instead of appending to the main text. Perfect for building a collection of source snippets with automatic source window tagging.
- **Source tagging** — when clipboard monitor is active, each captured item can be automatically prefixed with the title of the window it came from, so you always know whether that quote came from a PDF, a browser tab, or a Wikipedia article.
- **Version history** — automatic snapshots every few seconds of inactivity, plus on every save. Browse and restore any previous state without touching a separate backup system.
- **Auto-backup** — save timestamped `.wrp.bak` files of your project at configurable intervals. The oldest backups prune themselves.
- **Projects** — save and load `.wrp` files that preserve text, notes, word targets, and panel visibility. The editor remembers everything between sessions when "Remember Content" is enabled.
- **Text expansion** — the same Terms system that works for voice input also works as you type. Configure abbreviations that expand on space or punctuation.

---

## Who Is This For?

### Writers and authors
You have a first draft to get out of your head and onto the screen. WhisperR gets out of your way and transcribes as fast as you can speak. The editor handles long-form drafts with word targets, version history, and project saving. The clipboard monitor quietly accumulates research while you work. Text expansion turns your most-repeated phrases into two-keystroke affairs.

### Journalists and researchers
You interview someone, you get quotes. You read academic papers, you collect passages. WhisperR's clipboard monitor in notes mode is purpose-built for this: browse your sources, copy what matters, come back to a structured set of notes with the source window name stamped on each one. The editor handles the synthesis. The auto-backup handles the paranoia.

### Students
Recording lectures for later review is one thing. Having them transcribed automatically and appended to your notes is another. Leave WhisperR running in folder-monitor mode pointed at your recordings folder, go home, and find a text transcript waiting. Combine with the editor to organize, summarize, and build study materials. The version history is there for those moments when you reorganize everything and then immediately regret it.

### Content creators and podcasters
Scripting by voice is faster than typing for most people. The confidence filtering keeps your transcript clean of filler words you didn't intend to keep. Terms expansion handles your recurring catchphrases, sponsor read formats, and channel boilerplate. Folder monitoring handles batch transcription of recorded material.

### People with repetitive strain injuries or accessibility needs
Dictation as primary input is a genuine accessibility feature. WhisperR brings enterprise-grade transcription accuracy to users who need voice input but can't afford or don't want to depend on cloud services that can change pricing, restrict access, or simply go offline.

### Developers and technical writers
Voice-to-code works better than you'd expect for pseudocode, comments, and documentation. The `<kbd>` button, tag-wrap, and code formatting shortcuts mean the editor understands what you're building. Technical terms that Whisper might mishear can be corrected through the Terms system.

### Anyone who's tired of paying for things that should be free
This one's self-explanatory.

---

## Installation

1. Download the latest release from the [Releases page](../../releases)
2. Extract the zip to a folder of your choice
3. Run `WhisperR.exe`
4. On first launch, select a Whisper model from Settings and WhisperR will download it (once)
5. Configure your microphone and hotkeys
6. Talk

No installer. No admin rights required. No registry entries. Delete the folder to uninstall.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10 64-bit | Windows 11 |
| RAM | 4 GB | 8 GB+ |
| Storage | 500 MB (tiny model) | 5 GB (large-v3 model) |
| GPU | None (CPU works) | NVIDIA GPU with CUDA |
| GPU VRAM | N/A for CPU mode | Matches model size (see below) |
| Microphone | Any | A decent one |

**CPU vs GPU (CUDA) — what actually happens:** When using CPU mode, the model is loaded into your system RAM. When using CUDA (GPU), the model is loaded into your GPU's VRAM — not system RAM. This distinction matters:

- The GPU must have enough VRAM to hold the entire model. `large-v3` needs approximately 3 GB of VRAM. If your GPU has 4 GB of VRAM and other applications (games, other AI tools) are also using it, the load may fail.
- While a model is loaded in VRAM, that memory is not available to other GPU-intensive applications. Running `large-v3` alongside a demanding game on an 8 GB GPU will leave only ~5 GB of VRAM for the game — which may cause stuttering, crashes, or reduced graphics quality.
- For users with 8 GB or less of VRAM who also game or use other GPU-heavy applications: consider `small` or `medium` (both under 2 GB VRAM), or switch WhisperR to CPU mode while gaming.
- `tiny` and `base` run well on CPU for most use cases. `large-v3` on CPU is functional but slow — acceptable for batch transcription, frustrating for live dictation.

---

## A Note on Privacy

Every word you dictate is processed entirely on your machine. No audio leaves your computer. No text leaves your computer. WhisperR makes exactly one outbound connection: to download the model file from HuggingFace when you first select it. After that, it works without internet access.

The model download can be performed manually if you prefer: download the files from `https://huggingface.co/Systran/faster-whisper-{model-name}` and place them in `%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-{model-name}\snapshots\main\`.

---

## Credits

WhisperR is built on:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) by SYSTRAN
- [OpenAI Whisper](https://github.com/openai/whisper) (the underlying model)
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the UI
- [pynput](https://github.com/moses-palmer/pynput) for hotkeys
- [pyperclip](https://github.com/asweigart/pyperclip) for clipboard access

---

---

# Complete User Guide

*Everything you need to know about WhisperR, explained at a pace that assumes you are intelligent but have better things to do than read documentation. Which, if you're using a dictation app, is probably true.*

---

---

## Quick Start — From Zero to Talking in Five Minutes

*You don't need to read the full guide below to use WhisperR. This section gets you dictating in five minutes flat. Come back to the detailed guide when you actually need it — which may be never.*

---

### Step 1: Run the app

Double-click `WhisperR.exe`. A window appears. The app immediately begins downloading the `tiny` Whisper model in the background — you'll see progress messages in the scratchpad area at the bottom. Wait for it to finish. It's about 150 MB and takes a minute or two on a normal connection.

When you see `✓ tiny loaded` in the scratchpad, the app is ready.

> **Tip:** You only need an internet connection once, for the initial model download. After that, WhisperR works completely offline.

### Step 2: Select your microphone

Click **Settings** in the main window. Under **Audio Input Settings**, open the Microphone dropdown. Find your headset or microphone — look for your device name with **[Windows WASAPI]** at the end, which is the highest-quality option. Click **Save**.

### Step 3: Test it

Click the microphone button in the main window (or press **Ctrl+Alt+Z**). Speak normally. Click it again (or press **Ctrl+Alt+Z**) to stop. Your words should appear wherever your text cursor was — in Notepad, Word, a browser text field, anywhere.

If it worked: you're done. Enjoy your free transcription.

If the text appeared in the wrong place: make sure a text field in another application was active (had the cursor) before you pressed the hotkey. WhisperR inserts text wherever the keyboard focus is.

### Step 4: If accuracy is too low

The `tiny` model is fast but not the most accurate. If you're finding too many errors:

1. Open **Settings** → **AI Model Settings**
2. Select a larger model from the dropdown — try `small` or `medium` for a good balance, or `large-v3` if you have an NVIDIA GPU with at least 3 GB of VRAM
3. The new model downloads automatically in the background
4. Larger models are slower to load and transcribe, but significantly more accurate

### Step 5 (optional): Calibrate for your environment

If dictation is cutting off your words or triggering on background noise:

1. Open **Settings** → **Audio Input Settings**
2. Adjust **Noise Floor** (raise it if ambient noise triggers recording) and **Speech Volume** (lower it if your voice is being cut off)
3. The default mode is **Auto-Pause** — it records when you speak and stops when you go silent for 2 seconds, then transcribes. This works well for most people

That's it. The rest of this document explains every feature in detail for those who want to go further.

---

## Full Guide — Every Feature, Every Setting

*If you've read the Quick Start and want more, you're in the right place. If you haven't read the Quick Start, please do — it's shorter and gets you started faster than this wall of text.*

---

## Table of Contents

1. [First Launch and Initial Setup](#1-first-launch-and-initial-setup)
2. [The Main Window](#2-the-main-window)
3. [Settings — Complete Reference](#3-settings--complete-reference)
4. [Dictation — How It Actually Works](#4-dictation--how-it-actually-works)
5. [The Text Editor](#5-the-text-editor)
6. [The Notes Panel](#6-the-notes-panel)
7. [Clipboard Monitor](#7-clipboard-monitor)
8. [Terms, Commands, and Hallucinations](#8-terms-commands-and-hallucinations)
9. [File Transcription and Folder Monitor](#9-file-transcription-and-folder-monitor)
10. [Projects and Auto-backup](#10-projects-and-auto-backup)
11. [Workflow Examples by User Profile](#11-workflow-examples-by-user-profile)
12. [Troubleshooting](#12-troubleshooting)
13. [Hotkey Reference](#13-hotkey-reference)

---

## 1. First Launch and Initial Setup

When you run WhisperR for the first time, it will look functional but not do anything useful yet. This is because it needs a model.

### Choosing and Downloading a Model

On first launch WhisperR automatically begins downloading the `tiny` model in the background — you'll see progress in the scratchpad area. You can use the app immediately; transcription will become available once the download completes and the model loads.

To switch to a different model, open **Settings** (the Settings button in the main window) and go to **AI Model Settings**. You'll see a "Whisper Model" dropdown. The options are:

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny` | ~150 MB | Very fast | Acceptable | Testing, low-end hardware, quick notes |
| `base` | ~300 MB | Fast | Good | Everyday dictation on CPU |
| `small` | ~600 MB | Medium | Better | Good balance for CPU use |
| `medium` | ~1.5 GB | Slower | Very good | High accuracy without a GPU |
| `large-v3` | ~3 GB | Slow on CPU, fast on GPU | Excellent | Professional use with an NVIDIA GPU |

**Recommendation for most people:** Start with `tiny` to verify everything works. Switch to `large-v3` if you have an NVIDIA GPU and want the best accuracy.

Select your model from the dropdown. WhisperR immediately begins downloading it in the background — no Save button needed. Progress appears in the scratchpad area of the main window. You can use the app while it downloads, but transcription won't be available until the model fully loads (the scratchpad will confirm when it's ready).

### Microphone Selection

In Settings → **Audio Input Settings**, select your microphone from the dropdown. Windows exposes every physical microphone through multiple audio APIs, so the same physical device will appear several times in the list — typically as WASAPI, MME, DirectSound, and WDM-KS variants.

**Which one to pick:** Prefer **Windows WASAPI** when it appears for your device. WASAPI communicates directly with the audio hardware with lower latency and better quality than the older APIs. MME is a fallback for compatibility. DirectSound and WDM-KS are rarely the right choice for dictation.

If WASAPI isn't listed for your device, try the MME variant. If you're getting choppy audio or failed recordings, try a different API variant for the same physical device.

The device name shown is often truncated (Windows limits it to 31 characters). If you're unsure which entry corresponds to your headset or USB microphone, check Windows Sound settings to see the full device name.

### Hotkeys

The default hotkeys are:
- **Ctrl+Alt+Z** — Toggle dictation on/off
- **Ctrl+Shift+Space** — Push-to-talk (hold while speaking)
- **Ctrl+Shift+Alt+Z** — Show/hide the WhisperR window
- **Ctrl+Shift+Alt+A** — Toggle the text editor

You can change all of these in Settings → **Hotkeys**. If a hotkey conflicts with another application, change it. The app will tell you if a hotkey fails to register.

---

## 2. The Main Window

The main window has three areas:

**Top bar:** Shows status (idle/recording/transcribing), the current microphone, and quick-access buttons for Settings and the Editor.

**Scratchpad:** A read-only log of recent activity. Model loading progress, transcription results, warnings, and system messages appear here. It is not meant for writing in — it's a status display. Think of it as the app talking to you rather than you talking to it.

**Bottom bar:** The primary controls — dictation mode selector, language, and the record button (which you'll rarely click because hotkeys are faster).

### Dictation Modes

WhisperR has two mode selectors in Settings → **Dictation Settings**: **Dictation Mode** and **Live Mode**. They control the same three recording behaviours but apply to different triggers: Dictation Mode governs recordings started from the main window button, Live Mode governs recordings started from the global hotkey. You can set them differently if you want different behaviour depending on how you start recording.

The three available modes are:

- **Simple** — Records from when you press the hotkey/button until you press it again, then transcribes. No automatic silence detection. The most predictable mode — what you record is what gets transcribed, with no surprises.
- **Auto-Pause** — Detects silence during recording. After the configured silence duration (default 2 seconds), automatically stops recording and sends the audio for transcription. Best for natural dictation where you speak in bursts with pauses between thoughts. Requires tuning the Noise Floor and Speech Volume settings for your microphone.
- **Continuous** — Transcribes in a rolling loop while active, sending audio in segments without waiting for silence. Best for unbroken streams of speech.

**Push-to-Talk** is not a mode — it's a separate hotkey (Ctrl+Shift+Space by default) that records only while held, regardless of which mode is selected.

> **VAD vs Auto-Pause:** When VAD is enabled in Settings → Advanced, it replaces Auto-Pause's volume-based silence detection with a neural network that actually recognises speech. VAD is generally more accurate in noisy environments. The three VAD tuning parameters (Threshold, Min Silence, Min Speech) are described in the VAD Fine-Tuning section below.

**Configuring Noise Floor and Speech Volume for Auto-Pause:** These two settings in Audio Input Settings determine when Auto-Pause considers you to be speaking vs silent. The Noise Floor is the threshold below which audio is treated as background noise. The Speech Volume is the threshold above which audio is treated as active speech. The gap between them prevents rapid toggling. If Auto-Pause is cutting off your words, lower the Speech Volume. If it's triggering on ambient noise, raise the Noise Floor. Start the app speaking normally at your typical dictation volume and adjust from there.

### VAD Fine-Tuning (Settings → Advanced)

When VAD (Voice Activity Detection) is enabled, three additional parameters in Settings → Advanced control its behaviour:

**VAD Threshold** — Silero VAD sensitivity from 0.01 to 0.99. Higher values mean only confident speech activates recording; lower values are more sensitive and may trigger on background noise. Default: 0.50.

**VAD Min Silence** — Milliseconds of silence required before a speech segment is sent for transcription. Lower values give faster response but more sentence fragments; higher values produce fuller sentences with more latency. Default: 2000 ms.

**VAD Min Speech** — Minimum speech duration for a segment to be considered valid. Sounds shorter than this (brief noise bursts, throat-clearing) are ignored. Default: 250 ms.

### Where Text Goes

By default, transcribed text is inserted wherever your keyboard cursor is — in any application, any text field, anywhere. WhisperR does this by simulating keyboard input after transcription completes.

To change the destination to the built-in editor, open the editor window first. While the editor is open and focused, dictation goes into it instead.

---

## 3. Settings — Complete Reference

The Settings window is a single tabbed window. The main content tab is **Settings**, plus separate tabs for **AI Prompt**, **Terms**, **Commands**, **Hallucinations**, and **File Transcription**. The Settings tab is scrollable and contains the groups below.

### AI Model Settings

**Whisper Model** — The transcription model. Larger = more accurate, slower, more RAM. See the table in Section 1.

**Language** — The language you'll be speaking. Set this correctly — Whisper is significantly more accurate when it knows what language to expect. "Auto" lets Whisper detect the language each recording, which adds slight latency.

**Include timestamps** — Adds `[HH:MM:SS]` markers to transcribed output. Useful for transcribing interviews or lectures where timing matters.

**Translate to English** — When enabled, speech in other languages is transcribed directly to English rather than the original language. Useful if you're processing multilingual source material.

**Min. Confidence** — A threshold between 0 and 1. Transcription segments below this confidence score are discarded. Start at 0 (disabled) and increase it only if you're getting frequent hallucinations — nonsense output during silence or background noise. Typical useful range is 0.3–0.5. Setting it too high will silently discard legitimate transcriptions.

**Model cache folder** — Where Whisper model files are stored. Defaults to `%USERPROFILE%\.cache\huggingface\hub`. You can point this at a different drive if your C: drive is short on space. Use the Browse button to select a folder containing already-downloaded models.

### Audio Input Settings

**Microphone** — Your recording device. If you have multiple audio inputs, select the one you actually speak into.

**Noise Floor** — The volume threshold below which audio is considered background noise rather than speech. Default 50. Increase if WhisperR starts recording every time your cat walks past the microphone. Decrease if it's cutting off the beginnings of your words.

**Speech Volume** — The volume threshold above which audio is considered active speech. Default 500. The gap between Noise Floor and Speech Volume is the "hysteresis zone" that prevents rapid toggling.

**Auto-pause duration** — How many seconds of silence triggers an auto-pause and sends the accumulated audio for transcription. Default 2 seconds. Decrease for faster response; increase if you pause mid-thought and don't want it to transcribe incomplete sentences.

### Hotkeys

All hotkeys are configurable here. Click the field and press your desired key combination. The X button clears a hotkey if you don't want it assigned.

Key hotkeys:
- **Toggle Dictation** — Start/stop recording
- **Push-to-Talk** — Hold to record, release to transcribe
- **Show/Hide Window** — Bring WhisperR to front or minimize it
- **Toggle Editor** — Show or hide the text editor window
- **Copy & Edit** — Copy the currently selected text in any application and open it in the editor for editing. Useful for post-processing dictated text.
- **Rollback** — Undo the last transcription (removes the last inserted text from wherever it landed)

### Visual Indicators

The status indicator is a small floating overlay that appears in a corner of your screen showing the current state: idle (grey), loading (amber), recording (red), transcribing (blue).

**Show status indicator overlay** — Toggle it on or off.
**Indicator type** — Dot (small), Bar (thin horizontal), or Both.
**Position** — Which corner of the screen.
**Size** — How prominent you want it.

Disable it entirely if you find it distracting. The tray icon color also reflects the state.

### Logging Level

Found in Settings → **Advanced**. Controls how much WhisperR writes to its log file (`app_log.txt` in the WhisperR folder).

- **NONE** — No log file written. Best for everyday use and for SSD longevity. The app still shows status in the scratchpad.
- **INFO** — Logs normal operation: model loads, transcriptions, hotkey registrations. Useful if something seems wrong but the app isn't crashing.
- **WARNING / ERROR** — Logs only problems. A good middle ground if you want a record of errors without a large log file.
- **DEBUG** — Logs everything, including every audio volume reading, every poll cycle, and internal state changes. The log file grows quickly. Only use this when actively diagnosing a problem and share it when reporting bugs.

**Recommendation:** Leave it at NONE for normal use. Switch to DEBUG only when something isn't working and you need to see what the app is doing internally.

### Always On Top

Controls which WhisperR windows float above all other windows. The master toggle sets all four (main window, editor, notes panel, cheatsheet panel) at once; the sub-toggles let you configure them individually.

Useful if you're dictating while using other applications and want the WhisperR status visible at all times.

### Auto-Backup (Text Editor)

**Enable auto-backup** — Periodically saves timestamped `.wrp.bak` snapshots of the active project.
**Backup every N minutes** — How often to snapshot.
**Keep up to N backups** — Maximum backup files per project; oldest are deleted when exceeded.
**Browse Backup Folder** — Opens Explorer to the folder containing the current project's backups.

Backups are stored alongside the project file (same folder) with the naming convention `ProjectName_YYYY-MM-DD-HH-MM.wrp.bak`. They can be loaded directly via the editor's Load button if you need to recover an earlier version.

### Optional Tools

Found at the bottom of Settings → **Settings** tab. Shows which optional tools are installed and provides installation guidance.

**How optional tools work in the compiled app:** WhisperR is a self-contained `.exe`. Python packages like `harper-py` or `python-docx` cannot be installed into it after the fact — they must be bundled at build time. The Optional Tools panel shows their status and explains what to do:

- **Pandoc** — a standalone installer (`.msi`) that adds `pandoc.exe` to your system PATH. Works with the compiled app because WhisperR finds it via PATH, not as a Python package. Download from [pandoc.org/installing.html](https://pandoc.org/installing.html), install it, restart WhisperR.
- **Harper** — must be bundled at build time. End users of the compiled app cannot install it separately. Check the WhisperR GitHub releases page for builds that include Harper.
- **python-docx** — must be bundled at build time. If Pandoc is installed, DOCX export uses Pandoc instead and python-docx is not needed.

### Clipboard Monitor Options

**Tag clipboard entries with their source window title** — When the clipboard monitor captures a clip, prepend `[Window Title]` to the text. Lets you see at a glance whether a passage came from a browser tab, a PDF viewer, or another source.

### History & Auto-Backup (Text Editor)

**Version history depth** — How many in-session editor snapshots to keep (0 = disabled). Snapshots are taken automatically as you type and on every save. Access them with Ctrl+Alt+H in the editor. This setting lives alongside auto-backup because both are about preserving your work — version history for the current session, auto-backup for named projects on disk.

The remaining auto-backup settings (interval, keep count, browse folder) are described in the Auto-backup section above — they now share this group in Settings.

### Advanced

Contains voice trigger phrases (how you invoke the editor by voice), the Sendkeys trigger word for keyboard automation commands, and log level (see Logging Level section above).

### File Storage

**RAM-only mode** — When enabled (the default), audio recordings are processed entirely in memory and never written to disk. The recording exists only long enough to be transcribed, then is discarded. This is better for SSD longevity (no repeated small writes) and privacy (no audio files accumulating on disk). Disable it only if you need WhisperR to save recordings as audio files — for example, to keep a copy of the original audio alongside the transcription.

**Clear recordings on exit** — If RAM-only mode is disabled and WhisperR is saving audio files, this option deletes them when the app closes.

**Minimize to system tray** — When enabled, pressing the **Minimize** button (not X) sends the WhisperR window to the system tray instead of the taskbar. The app keeps running silently — hotkeys, clipboard monitor, and folder watch all continue working. Click the tray icon to restore the window. The X button always closes and exits the app regardless of this setting.

---

## 4. Dictation — How It Actually Works

### The Recording Cycle

1. You trigger recording (hotkey, PTT, or button)
2. WhisperR records audio from your microphone
3. In Auto-Pause mode, when silence is detected for the configured duration, recording pauses and the audio is sent for transcription
4. The Whisper model processes the audio (this takes 0.5–3 seconds depending on model size and hardware)
5. The transcribed text is inserted at the cursor position in the active application

### Improving Accuracy

**Speak clearly but naturally.** Whisper is trained on natural speech and handles normal pace well. Over-articulating can actually hurt accuracy.

**Use the right model for your hardware.** Transcription errors are often a model size problem rather than a speech problem. If you're getting frequent errors, try a larger model.

**Configure Terms.** Proper nouns, technical terms, product names, and unusual words that Whisper consistently mishears can be added to the Terms system with their correct spellings.

**Adjust Noise Floor.** If ambient noise is triggering false recordings, increase the Noise Floor setting.

**Don't rely on spoken punctuation.** Whisper is inconsistent about interpreting words like "comma" or "period" as punctuation symbols vs transcribing them literally as the words *comma* and *period*. Results vary by model, language, and context. The larger models handle it better than `tiny` or `base`. In practice, most users find it easier to add punctuation manually in the editor rather than dictating it.

### The Rollback Hotkey (Ctrl+Shift+Z)

Rollback is not an undo — it doesn't remove the last transcription. It's a **sentence continuation tool**.

When Whisper transcribes a segment it often adds a period or ellipsis at the end ("I was going to..."). If you want to keep speaking and continue the same sentence, that trailing punctuation is a problem — the next transcription will start with a capital letter and feel like a new sentence.

Pressing Rollback strips only the trailing punctuation and whitespace from the last transcription, leaving a clean word boundary. The next dictation then pastes in lowercase, continuing the sentence seamlessly.

Example: you say "I was going to" → Whisper transcribes "I was going to." → you press Rollback → the period is deleted → you say "finish that thought" → you get "I was going to finish that thought."

---

## 5. The Text Editor

Open the editor with **Ctrl+Shift+Alt+A** or the Editor button in the main window.

### The Button Row

Running along the bottom of the editor is a row of buttons:

**✨ New** — Start a new project. Clears text and notes. Prompts for confirmation (it's not a monster).

**📂 Load / Import** — Left-click to load a `.wrp` project file. Right-click to import a plain `.txt` or `.md` file into the current text area.

**💾 Save / Export** — Left-click to save as a `.wrp` project (preserves text, notes, word target, and all editor state). Right-click to export in your chosen format — available options depend on what's installed on your system:

- **Markdown (.md) and Plain Text (.txt)** — always available
- **HTML (.html)** — always available; produces a self-contained file with light CSS styling
- **Word Document (.docx)** — available if [Pandoc](https://pandoc.org) or `python-docx` is installed; Pandoc produces better results
- **PDF (.pdf)** — available if Pandoc is installed

Pandoc is detected automatically at export time — no configuration needed.


**📋 Copy** — Copies all text in the editor to the clipboard.

**📋 Preset dropdown** — Select a document template to instantly populate the editor with a structured starting point. Available: Interview, Meeting Notes, Lecture/Talk, Research Notes, Draft/Freewrite. Prompts for confirmation if the editor has content.

**🔍 Find** — Opens the Find & Replace bar (or close it if already open). Also Ctrl+H.

**🕐 History** — Opens the version history browser. Also Ctrl+Alt+H.

**📌 Remember** — When enabled (amber), the editor remembers its content between sessions. Close it, reopen it, your text is still there. This also enables remember for clipboard monitor content — if you close the editor while the monitor is running, everything accumulates silently and reappears when you open the editor again.

**👁 Clipboard Monitor** — Left-click to start appending clipboard clips to the main textarea. Right-click to switch to Notes mode (each clip creates a new note). The button turns green for text-append mode, blue for notes mode. Running the clipboard monitor automatically enables Remember.

**📋 Clipboard Prefill** — When enabled, opening the editor pre-fills it with whatever is currently in the clipboard. Useful for "open the editor around what I just copied."

**📝 Notes** — Toggle the notes panel open/closed.

**📖 Cheatsheet** — Toggle the formatting shortcuts cheatsheet.

**↩ Undo** — Restores the last deleted note (notes panel undo, not text undo).

**⬆ Paste to App** — Takes the editor's current text and pastes it into whichever application was active before you opened the editor. The primary way to get edited text back into the application you were working in.

### Formatting Toolbar

The formatting buttons use Markdown syntax. The editor is a plain-text Markdown editor, not a WYSIWYG editor — `**bold**` will appear as `**bold**`, not as **bold**. This is intentional. The formatted output is meant to be pasted into tools that render Markdown (web CMSes, GitHub, Notion, Obsidian, etc.) or converted as needed.

| Button | Syntax | Shortcut |
|--------|--------|----------|
| **B** | `**text**` | Ctrl+B |
| *I* | `*text*` | Ctrl+I |
| ~~S~~ | `~~text~~` | Ctrl+Shift+S |
| ==H== | `==text==` | Ctrl+Shift+H |
| `C` | `` `text` `` | Ctrl+` |
| `<kbd>` | `<kbd>text</kbd>` | Ctrl+Shift+D |
| `<>` | `<tag>text</tag>` (prompts for tag) | Ctrl+Shift+W |
| 🔗 | `[text](url)` | Ctrl+K |
| 🔗 right-click | `[text](clipboard-url)` | Ctrl+Shift+K |
| H1/H2/H3 | `# text` / `## text` / `### text` | Ctrl+1/2/3 |
| • | `- text` | Ctrl+Shift+B |
| 1. | `1. text` | Ctrl+Shift+N |
| ☐ | `- [ ] text` | Ctrl+Shift+T |

### Spell and Grammar Checking

If `harper-py` is installed (`pip install harper-py`), WhisperR checks spelling and grammar as you type. Errors appear as red wavy underlines. Right-click any underlined word to see the error description and up to five suggested corrections — clicking one applies it immediately. The check runs in a background thread 2 seconds after you stop typing.

Harper is fully offline with no network calls. It catches common English errors: subject-verb agreement, punctuation misuse, commonly confused words. Install it once and it activates automatically.

### Find & Replace (Ctrl+H)

The Find & Replace bar opens at the bottom of the editor. It supports literal text and regex (toggle the `.*` checkbox). Find advances through matches in order and wraps around. Replace replaces the current match and advances to the next. Replace All uses a single undo operation — one Ctrl+Z to revert everything.

### Version History (Ctrl+Alt+H)

WhisperR takes a snapshot of the editor text 5 seconds after you stop typing and on every manual save. The history picker shows each snapshot with a timestamp and the first 80 characters of text. Selecting a snapshot restores it (and saves your current state first, so you can undo the restore if needed).

### Terms Autocorrect in the Editor

The Terms system doesn't only work for dictation — it also works as you type. If your Terms include `"hexsw": "Hexagon Software"`, typing `hexsw ` (followed by a space or punctuation) will automatically replace it with `Hexagon Software`. This is text expansion for the keyboard, not just the microphone.

### Word Target

The target word count field in the corner sets a goal. The remaining count updates as you type. Useful for keeping yourself honest about article length, avoiding the eternal "is this long enough?" anxiety, and for feeling smug when you exceed the target.

---

## 6. The Notes Panel

The notes panel is a floating window that attaches to the right of the editor. Open it with the 📝 button.

### Basic Operations

**Adding a note:** Click "＋ Add Note" at the bottom, or press **Ctrl+Enter** while typing in any note to add one directly below the current note.

**Editing a note:** Click on a note to uncollapse it (if collapsed) and type. Notes auto-resize as you add content.

**Deleting a note:** The ✕ button in the top-right of each note. The ↩ Undo button in the editor restores the last deleted note (up to 3 levels).

**Reordering notes:** Each note has a ⠿⠿ drag handle in the top-right (before the ✕). Press and hold it, drag up or down. A blue indicator line shows where the note will land. Release to drop. The cursor changes to a closed hand while dragging.

**Changing color:** The row of colored squares in the top-left of each note changes its background color. Ten options from yellow to black.

**Collapsing notes:** The − button in the footer collapses all notes to show 3 lines each. The + button expands all. Individual notes can be uncollapsed by clicking on them.

**🗑 Delete All:** The trash button in the footer deletes all notes at once. A confirmation dialog appears first (hold Shift to skip it). Immediately after deleting, an amber **↩ Restore All** button appears in the panel header — click it to bring all notes back exactly as they were. The Restore All option disappears if you add new notes or close and reopen the panel, so use it promptly.

**Color filter (🎨 button):** Click the palette button in the footer to open the color filter menu. Each color has a toggle — enable the colors you want to see, and notes of all other colors are hidden. The filter button turns green when a filter is active. An amber `+N` badge appears next to it showing how many notes are currently hidden, as a reminder that they exist. Click "Show All Colors" to clear the filter.

Color filter state is saved with project files and restored on load, so your filtered view survives closing and reopening. It is also preserved in the session state when Remember Content is enabled.

**Color as a system:** Colors are most useful when they mean something consistent. Some approaches:
- Priority: red = urgent, orange = soon, yellow = someday, green = done
- Source type: blue = web sources, pink = books, purple = interviews, white = your own ideas
- Status: yellow = raw capture, blue = needs verification, green = confirmed, grey = archived
- Project area: one color per chapter, section, or topic thread

The filter makes any of these systems actually useful — instead of scrolling past everything, you can view only the color(s) relevant to what you're doing right now.

### Undo, History, and What Gets Remembered

**Per-note undo (↩ Undo button):** Deleting a single note with its ✕ button pushes it onto a small undo stack (last 3 deletions). The ↩ Undo button in the panel header restores the most recently deleted note. This stack is cleared when you load a new project.

**Delete All undo (↩ Restore All button):** The 🗑 Delete All button snapshots all notes before deleting them. The amber ↩ Restore All button appears immediately after and restores the entire set. This is a one-shot undo — it holds one snapshot, not a chain.

**Notes and the Remember toggle:** Notes are stored on the editor object itself and persist as long as the editor is alive (which includes while the clipboard monitor is running). They are shown automatically on reopen only if the notes panel was open when you last closed the editor. This means notes from a previous session can accumulate silently — open the Notes panel to check. If you want a clean slate, use 🗑 Delete All.

**Notes are NOT part of the editor's version history.** The version history (Ctrl+Alt+H) tracks text content only, not notes. If you delete notes, version history cannot restore them — only the ↩ Restore All button can, and only immediately after the deletion. For notes you want to keep long-term, save them as a project (💾 Save).

**Notes and projects:** Notes are fully saved and restored with .wrp project files. The color filter state is also saved with projects and restored on load.


### Notes and Projects

Notes are saved with the project when you use Save, and restored when you Load. If Remember Content is enabled, notes are also persisted between sessions automatically.

---

## 7. Clipboard Monitor

The clipboard monitor watches your clipboard continuously and does something useful with each new item you copy. It runs in the background and keeps working even when the editor window is closed.

### Text Mode (Left-click the 👁 button)

Every time you copy something to the clipboard, the text is appended to the editor's main textarea. Each clip is separated by a blank line. This is ideal for:

- Accumulating research from multiple sources as you browse
- Collecting quotes from PDFs, web pages, and other documents
- Building a buffer of content to edit and synthesize later

The editor doesn't need to be visible. Copy things all day from anywhere, open the editor when you're ready, and everything is there waiting.

### Notes Mode (Right-click the 👁 button)

Same idea, but each clipboard item becomes a separate note rather than being appended to the main text. Each note is independent, color-codeable, and reorderable. This is better for:

- Building a collection of source material you want to keep organized
- Research where individual items have distinct meaning rather than flowing together
- Situations where you want to see each clip as a discrete unit rather than a stream

### Source Tagging

Enable "Tag clipboard entries with their source window title" in Settings → Clipboard Monitor Options. Each clip will be prefixed with `[Window Title]` — for example, `[Article - Google Chrome]` or `[report.pdf - Adobe Acrobat]`. Lets you track provenance without any extra effort.

### Remember and the Monitor

Enabling the clipboard monitor also enables the Remember Content toggle. This ensures that content accumulated while the editor was closed is still there when you reopen it. Turn off Remember manually only if you specifically want a clean slate.

---

## 8. Terms, Commands, and Hallucinations

These three settings tabs handle the more advanced behavioral customization of WhisperR.

### Terms (Text Expansion and Autocorrection)

Terms are phrase-replacement rules. Each term has a **trigger phrase** and a **replacement text**.

When you dictate, WhisperR checks the transcribed text for trigger phrases and replaces them with the replacement text before inserting anything. The same replacement also happens when you type the trigger phrase in the editor and follow it with a space or punctuation.

**Use cases:**

*Correcting persistent mishearings:*
- Trigger: `whisper ar` → Replacement: `WhisperR`
- Trigger: `hexagon software` → Replacement: `Hexagon Software™`

*Text expansion for frequently typed content:*
- Trigger: `mymail` → Replacement: `firstname.lastname@email.com`
- Trigger: `addr` → Replacement: `123 Example Street, City, State 12345`
- Trigger: `sig` → Replacement: a full email signature

*Formatting shortcuts:*
- Trigger: `btw` → Replacement: `by the way`
- Trigger: `afaik` → Replacement: `as far as I know`

Terms matching is case-insensitive for the trigger, but the replacement is applied exactly as written.

### Commands

Commands let you trigger actions by saying specific phrases during dictation. Each command has a **trigger phrase** and a **system command** to execute.

Examples:
- `launch notepad` → `notepad.exe`
- `open calculator` → `calc.exe`
- `open my brief` → `C:\Documents\brief.docx`

Say the phrase during dictation and WhisperR runs the command instead of inserting the text. The phrase itself does not appear in the transcription output.

### Hallucinations

The hallucinations list contains phrases that Whisper commonly generates when it hears background noise, silence, or unintelligible audio — phrases it confidently "hallucinates" rather than transcribes. Common examples include `Thank you`, `Thanks for watching`, and `You`.

Any transcription that exactly matches or starts with a phrase in the hallucinations list is silently discarded. This prevents the editor from periodically inserting `Thank you for watching!` into your legal brief because your air conditioning rattled at the wrong pitch.

You can add model-specific hallucinations you encounter. If you find that a particular phrase keeps appearing in your transcriptions during silence, add it here.

### Transcription Steering

Found in Settings → **Transcription Steering** tab. Two complementary tools for guiding Whisper toward domain-specific vocabulary.

**Steering Prompt** — Freeform context sent to Whisper before every transcription. Describe what you'll be talking about: content type, domain terms, proper nouns, abbreviations. This biases the model's language predictions.

Example for medical dictation:
> *Medical transcription. Patient notes. Drug names: metformin, lisinopril.*

Example for a software interview:
> *Software engineering. Python, Kubernetes, REST API, CI/CD pipeline.*

**Vocabulary Boost (Hotwords)** — One word or phrase per line that Whisper should prioritise acoustically — matched against what it actually hears, not just used as context. Best for specific terms that the prompt alone doesn't fix. Keep the list short; too many entries can hurt overall accuracy.

**When to use which:** The prompt is better for setting general context. Hotwords are better for specific terms that keep being transcribed incorrectly despite being in the prompt.

Both the prompt and hotwords have Import/Export buttons for `.txt` files, and both are saved to the app config and restored on next launch.

---

## 9. File Transcription and Folder Monitor

### Manual File Transcription

The **Transcribe File** option (available from the main window) lets you select an audio file and have it transcribed to text. Supported formats: **WAV** and **MP3**.

The transcription appears in the scratchpad and can be saved from there.

### Folder Monitor

Set a folder in Settings → Output Folder, and WhisperR will watch it continuously for new audio files. Any file dropped into the folder is automatically transcribed and the result saved as a `.txt` file alongside the original.

This is the hands-off batch transcription feature. Leave it running, drop recordings into the folder, come back later to find text files waiting. Useful for:
- Transcribing recorded meetings or lectures after the fact
- Processing a backlog of voice memos
- Automated workflows where audio files arrive from other sources

---

## 10. Projects and Auto-backup

### What Is a Project?

A `.wrp` project file is a JSON file that stores everything about your current editor session: the full text content, all notes (with their colors and order), the word target, and panel visibility state. It's portable — copy it to another machine with WhisperR and open it there.

### Session Persistence vs. Projects

There are two ways your work is preserved:

**Remember Content (session persistence):** The 📌 toggle in the editor. When enabled, closing and reopening the editor restores everything automatically, without a named project file. This is for ongoing work that you don't need to formally name or archive — your working buffer that's always there.

**Projects:** Named `.wrp` files for work you want to organize, archive, or return to later. Use the 📂 Load and 💾 Save buttons.

The clipboard monitor automatically enables Remember Content. This means: enable the monitor, close the editor, accumulate content, reopen the editor, everything is there. This is by design and not a bug.

### Auto-backup

When enabled (Settings → Auto-Backup), WhisperR saves `ProjectName_YYYY-MM-DD-HH-MM.wrp.bak` next to your project file every N minutes. The most recent N backups are kept; older ones are deleted automatically.

If you haven't saved your project yet when the first backup is due, WhisperR will prompt you to save. There's no backup without somewhere to back up to.

Backups can be loaded directly via the Load button — they're full project files, just with a different extension. Browse to them using the Browse Backup Folder button in Settings.

---

## 11. Workflow Examples by User Profile

### The Novelist

*Setup:* `large-v3` model (GPU), Auto-Pause mode, 2-second pause threshold, Remember Content enabled.

*Workflow:* Open the editor, set a word target for the day. Dictate prose sections by voice, using the keyboard only for editing and reformatting. Use Terms to handle character names and place names that Whisper mishears. Use the clipboard monitor to capture research notes from browsers and PDFs into the notes panel while drafting. Use version history to step back to that paragraph you deleted twenty minutes ago and immediately regretted deleting.

*Key features:* Dictation to editor, Terms for proper nouns, clipboard-to-notes, version history, word targets.

### The Student

*Setup:* `small` or `medium` model (CPU), Folder Monitor pointing to a recordings folder.

*Workflow:* Record lectures with a phone or recorder. Drop the recordings into the monitor folder. Come home to text transcripts. Open the editor, load a transcript, use it as a base for organized notes — cut, paste, summarize, annotate in the notes panel. Save each lecture as a named project.

For study sessions: dictate summary notes by voice while reviewing material (faster than typing), use Terms for course-specific terminology, use clipboard monitor to accumulate textbook passages for reference.

*Key features:* Folder monitor, project saving, notes panel for annotation, Terms for terminology.

### The Journalist

*Setup:* `large-v3` (GPU), notes-mode clipboard monitor, source tagging enabled.

*Workflow:* During research, enable clipboard-to-notes mode. Browse sources, copy relevant quotes and facts. Each clip becomes a note tagged with the source window title. After research, disable the monitor and review notes — drag to reorder by importance, color-code by topic or source reliability. Begin drafting in the main textarea, using notes as a reference panel. Use the color filter to focus on one source type at a time — for example, show only blue (web) notes while verifying online claims, then switch to pink (books) while adding citations.

For interview transcription: drop the audio file into folder monitor, get a text transcript, load it into the editor, annotate key quotes as notes.

*Key features:* Clipboard monitor (notes mode), source tagging, drag-reorder notes, folder monitor.

### The Teacher / Trainer

*Setup:* `medium` model, Terms configured for course vocabulary, Commands set up to launch relevant tools.

*Workflow:* Dictate lesson notes, quiz questions, and course materials by voice. Use heading formatting (H1/H2/H3) to structure content for easy conversion to course documents. Export to plain text, paste into your LMS or document editor.

Configure Commands to launch presentation software, open specific folders, or execute scripts with voice triggers.

*Key features:* Dictation to external applications, formatting toolbar, Terms for consistency, Commands for workflow automation.

### The Developer / Technical Writer

*Setup:* `small` or `medium` model, Terms for code-related corrections, Push-to-Talk mode.

*Workflow:* Use Push-to-Talk for focused dictation when needed (voice comments, docstrings, design notes). Use the editor for longer documentation with the `<kbd>` and code formatting buttons. Use tag-wrap for XML/HTML documentation. Use clipboard monitor to accumulate code snippets and error messages from multiple terminal windows into a debugging notes panel.

*Key features:* Push-to-Talk, code formatting, tag-wrap, clipboard-to-notes for debugging sessions.

### The Researcher / Academic

*Setup:* `large-v3` (GPU), clipboard monitor with source tagging, Auto-backup enabled, projects for each paper or chapter.

*Workflow:* One project per paper. Use clipboard-to-notes mode while reviewing literature — each clipped passage becomes a note tagged with its source. Use version history for major structural revisions. Use auto-backup for peace of mind. Export finished drafts as plain text for import into LaTeX or Word.

*Key features:* Project files, clipboard-to-notes, source tagging, version history, auto-backup.

---

## 12. Troubleshooting

### Model won't load / keeps re-downloading

Check that the model cache folder (Settings → AI Model Settings → Model cache folder) is a valid, writable path. If you see "Snapshot incomplete" in the logs, the previous download was interrupted — WhisperR will automatically clear it and re-download.

### Transcription doesn't appear anywhere

Make sure a dictation mode is selected and that the recording indicator (overlay dot or tray icon) shows red when you're recording. If you have the editor open, text goes there; otherwise it goes to the active application's cursor position. If no application has focus, the text goes nowhere.

### Hotkeys don't work

Another application may be capturing the same key combination. Change the conflicting hotkey in Settings. Some combinations (involving Windows key, for example) are system-reserved and can't be reliably captured.

### App says model is loaded but transcription quality is poor

Try a larger model. `tiny` and `base` sacrifice accuracy for speed. Also check your microphone selection in Settings — using a built-in laptop microphone at the same input level as a headset microphone will produce different results.

### "Worker crashed" messages in the scratchpad

The worker process handles transcription in a separate process to isolate crashes from the main app. If it crashes repeatedly, try forcing CPU mode (compute preference = CPU only) to rule out a GPU driver issue. Check the app log (in the same folder as WhisperR.exe, named `whisperr_YYYY-MM-DD.log`) for details.

### VM / clock-related SSL errors

If you're running in a virtual machine and see `certificate is not yet valid` errors, the VM's system clock is wrong. Sync it: open an admin PowerShell and run `w32tm /resync /force`, or right-click the clock → Adjust date/time → Sync now.

---

## 13. Hotkey Reference

| Action | Default | Configurable |
|--------|---------|:---:|
| Toggle dictation | Ctrl+Alt+Z | ✓ |
| Push-to-talk | Ctrl+Shift+Space | ✓ |
| Show/hide main window | Ctrl+Shift+Alt+Z | ✓ |
| Toggle editor | Ctrl+Shift+Alt+A | ✓ |
| Copy & Edit (open selection in editor) | Ctrl+Shift+X | ✓ |
| Rollback last transcription | Ctrl+Shift+Z | ✓ |
| Find & Replace (in editor) | Ctrl+H | ✓ |
| Version history (in editor) | Ctrl+Alt+H | ✓ |
| Bold | Ctrl+B | ✓ |
| Italic | Ctrl+I | ✓ |
| Strikethrough | Ctrl+Shift+S | ✓ |
| Highlight | Ctrl+Shift+H | ✓ |
| Inline code | Ctrl+` | ✓ |
| Keyboard tag | Ctrl+Shift+D | ✓ |
| Tag wrap (<>) | Ctrl+Shift+W | ✓ |
| Link (placeholder URL) | Ctrl+K | ✓ |
| Link (clipboard URL) | Ctrl+Shift+K | — |
| Heading 1/2/3 | Ctrl+1 / Ctrl+2 / Ctrl+3 | ✓ |
| Bullet list | Ctrl+Shift+B | ✓ |
| Numbered list | Ctrl+Shift+N | ✓ |
| Task list | Ctrl+Shift+T | ✓ |
| Em dash | Ctrl+Shift+– | ✓ |
| Add note (in notes panel) | Ctrl+Enter | — |
| Save project | Ctrl+S | — |

---

*WhisperR v2.1.0 — Free, local, yours.*
