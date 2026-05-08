# WhisperR Upgrade Plan

> Last updated: 2026-05-08
> Version: 2.1.0

## Overview

- **Goal:** Modularize and enhance WhisperR while maintaining existing functionality
- **Current:** Single monolithic file (`WhisperR.py`, ~13,720 lines) + stub module structure
- **Philosophy:** Incremental improvements, always working app between changes

---

## PROGRESS SUMMARY

### Actually Completed (in WhisperR.py):
- ✅ Enhanced crash handler (OS, RAM, PyQt version info)
- ✅ Auto-save indicator ("💾 Auto-saved at HH:MM:SS" in status bar)
- ✅ Theme system (Dark/True Black/Light selector in Settings)
- ✅ Comprehensive tooltips on all major UI elements
- ✅ Enhanced keyboard cheat sheet (searchable, categorized)
- ✅ Table insertion dialog
- ✅ Notes panel with colors, reorder, filter
- ✅ Version history with word-level diff (uses `difflib` from stdlib)
- ✅ Copy-to-clipboard from history dialog
- ✅ Keyboard shortcuts: Ctrl+, (Settings), Ctrl+Shift+N (Notes), Ctrl+Shift+G (Cheatsheet), Ctrl+F (Find)
- ✅ Editor formatting toolbar (Bold, Italic, Strikethrough, Highlight, Code, KBD, Tag, Link, H# dropdown, lists, Insert Table)
- ✅ 10+ editor stylesheets updated to use theme variables (button_background, button_hover, input_border, primary, surface, text, etc.)
- ✅ Notes window stylesheets updated to use theme variables

### Not Done / Stubs Only:
- ❌ Modular refactoring — `core/`, `ui/`, `utils/` modules are placeholders, never imported
- ❌ Write/Preview Markdown toggle — dead code removed entirely (never worked)
- ❌ `core/diff_engine.py` — exists but never imported; app uses stdlib `difflib`
- ❌ `core/config.py` — standalone stub, not integrated

### Code Statistics:
- **Current:** ~13,720 lines in `WhisperR.py`
- **Module stubs:** 18 files across `core/`, `ui/`, `utils/`, `themes/`, `styles/`

---

## Editor Toolbar Layout (current)

| Button | Width | Shortcut |
|--------|-------|----------|
| Bold (B) | 42px | Ctrl+B |
| Italic (I) | 38px | Ctrl+I |
| Strikethrough (S) | 44px | Ctrl+Shift+S |
| Highlight (=) | 46px | Ctrl+Shift+H |
| Code (`) | 44px | Ctrl+` |
| KBD (<kbd>) | 48px | Ctrl+Shift+D |
| Tag (<>) | 38px | Ctrl+Shift+W |
| Link (🔗) | 38px | Ctrl+K |
| H# (dropdown) | 54px | Ctrl+1/2/3/4/5 |
| Bullet list | 36px | Ctrl+Shift+B |
| Numbered list | 36px | Ctrl+Shift+N |
| Task list | 36px | Ctrl+Shift+T |

---

## Theme-Aware Editor Styling

The editor toolbar buttons and menus now read colors from the active theme dict instead of hardcoded dark-only values.  Theme keys used:

- `button_background`, `button_hover` — button fill
- `input_border` — button borders
- `text`, `text_secondary`, `text_disabled` — text colors
- `primary` — hover accent, active borders
- `surface`, `border` — backgrounds, container borders
- `selection_background`, `selection_text` — selection highlights

Three themes available: `Dark`, `True Black (OLED)`, `Light`.

---

## Current State

The app is a single monolithic ~13,720-line file that works well. All improvements have been made directly in `WhisperR.py`. The module directories exist but are not used — any future modular refactoring would start from scratch.

---

## PHASE 1: Foundation (Structural Changes) — NOT STARTED

> Goal: Break the monolith into manageable pieces
> Status: Stub modules exist but NOT integrated

### Created (never integrated):
- `core/config.py` — AppConfig class (copy, not imported)
- `core/transcription.py` — Transcription utilities stub
- `core/audio.py` — Audio recording stub
- `core/terms.py` — Terms processor stub
- `core/diff_engine.py` — Diff engine stub
- `ui/components.py` — Reusable components stub
- `ui/main_window.py` — Placeholder
- `ui/editor.py` — Placeholder
- `ui/notes.py` — Placeholder
- `ui/settings.py` — Placeholder
- `ui/cheatsheet.py` — Placeholder
- `ui/indicators.py` — Placeholder
- `utils/logging_utils.py` — Logging stub
- `utils/file_utils.py` — File utils stub
- `themes/dark.py` — Dark theme colors
- `themes/dark_true_black.py` — True black for OLED
- `themes/light.py` — Light theme
- `styles/styles.py` — QSS generator

**Integration needed:** Import each module into main app, test, remove duplicated code from WhisperR.py.

---

## PHASE 2: User-Facing Improvements — MOSTLY DONE

All features in this phase are complete and integrated directly into `WhisperR.py`.

### Task 2.1: Comprehensive Tooltips ✅
### Task 2.2: Auto-Save Indicator ✅
### Task 2.3: Enhanced Keyboard Cheat Sheet ✅
### Task 2.4: Theme System ✅

---

## PHASE 3: Editor Enhancements — PARTIALLY DONE

### Task 3.1: Write/Preview Toggle ❌ REMOVED
Dead code was removed. No Markdown preview feature exists.

### Task 3.2: Word-Level Diff ✅
Uses stdlib `difflib` (not `diff-match-patch`).

### Task 3.3: Table Support ✅

---

## Dependencies

No additional dependencies beyond PyQt6 and whisper-standalone-windows.
`markdown` and `diff-match-patch` are NOT required.
The themes are pure Python dicts in `themes/*.py` — no extra packages.

---

## Testing Strategy

1. After each change, verify the app launches without import errors
2. Test the specific feature changed
3. If anything breaks, fix before proceeding
4. Never leave the app in a broken state

---

*End of upgrade plan*
