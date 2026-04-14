# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**track-lrc-align** is an AI-powered lyrics-to-music alignment tool. It takes a lyrics text file + audio file and produces a timestamped LRC file with per-word (or per-line) timing. It supports Chinese, Japanese, Korean, English, and Russian.

## Commands

**Dependencies (Python 3.9/3.10):**
```bash
pip install -r requirements.txt
```

**CLI usage:**
```bash
python main.py <lyrics_file> <audio_file> \
  -f lrc|srt \         # output format (default: lrc; srt accepted but not yet implemented)
  -l 0|1 \             # 0=word-level (default), 1=line-level
  -v 0|1 \             # 1=separate vocals via Demucs (default), 0=skip vocal separation
  -m mdx_extra \       # demucs model (mdx|mdx_extra|mdx_q|mdx_extra_q)
  -i -1                # demucs model index (-1=ensemble)
```

## Architecture

### Processing Pipeline

```
lyrics text + audio file
  → t2l/phonetic.py: language detection (regex CJK → fastText fallback)
                     → phonetize to ASCII phonemes per language
  → t2l/t2l.py: audio loading (torchaudio → librosa fallback)
               → vocal separation via Demucs (mdx_extra model) [skipped if vocalize=False]
               → resample to 22050 Hz
  → t2l/mtl/wrapper.py: mel-spectrogram → CNN-RNN acoustic model
                        → phoneme posteriorgram prediction
                        → DTW alignment (utils.alignment / alignment_bdr)
  → t2l/t2l.py: gen_lrc() converts frame indices to [MM:SS.mmm] timestamps
  → stdout / file output
```

### Key Constants
- Sample rate: **22050 Hz**
- Frame resolution: `256 / 22050 * 3` seconds per frame (~0.0348s)
- Mel features: 128, FFT: 512, used in `train_audio_transforms` (wrapper.py)

### Model Checkpoints (`./checkpoints/`)
- `checkpoint_Baseline` — single-task acoustic model, 41 phoneme classes
- `checkpoint_MTL` — multi-task learning, classes (41, 47); default for CLI
- `checkpoint_BDR` — boundary detection model (1.8 MB), used with `*_BDR` methods

Model loading path is relative to CWD: `./checkpoints/checkpoint_<type>`. The process **must be run from the repo root**.

### The `process()` signature (`t2l/t2l.py`)
```python
process(txt_lines, audio_file, mtl_model='MTL', demucs_model='mdx_extra',
        demucs_idx=-1, line_only=False, out_file=None, verbose=True,
        vocalize=True, format='lrc')
```
- `vocalize=False` skips Demucs vocal separation and runs alignment directly on the raw audio.
- `format` is accepted for forward-compatibility but only `lrc` output is currently implemented.

### The `method` parameter in `align()`
Accepts a string `"Baseline"`, `"MTL"`, `"Baseline_BDR"`, or `"MTL_BDR"`, or a pre-loaded tuple `(ac_model, bdr_model, model_type, bdr_flag, device)` from `load_mtl_model()`. Pre-loading via `t2l/init_model.py` can be used to avoid repeated model initialization.

### Language Detection (`t2l/phonetic.py`)
Regex takes priority over fastText for CJK scripts:
1. Japanese katakana/hiragana range → `ja`
2. CJK unified ideographs → `zh`
3. Hangul → `ko`
4. Cyrillic → `ru`
5. fastText `lid.176.ftz` for remaining cases (fallback)

### Error Types
- `TxtValueError` — invalid/empty lyrics after parsing
- `AudioValueError` — audio load failure (both torchaudio and librosa failed)

## Important Implementation Details

- `gen_lrc()` in `t2l/t2l.py` has boundary guards: if `word_align` is shorter than the total words in `lines`, it truncates rather than crashing (critical fix from AI-129).
- Audio loading tries `torchaudio.load` first; falls back to `librosa.load` if it returns empty tensor (handles formats torchaudio can't decode).
- Demucs ensemble models (`mdx_extra`) contain 4 sub-models; `demucs_idx=-1` uses the ensemble, `0-3` selects a single sub-model.
- `pykakasi` (Japanese romanization) is GPL-licensed — take note if changing licensing.
