# track-lrc-align

AI-powered lyrics-to-music alignment tool. Given a plain lyrics text file and an audio file, it produces a timestamped LRC file with per-word (or per-line) timing. Supports **Chinese**, **Japanese**, **Korean**, **English**, and **Russian**.

## Features

- **Multi-language support**: CJK (zh/ja/ko), English, and Russian with automatic language detection
- **Word-level & line-level alignment**: Choose between precise per-word timestamps or simpler per-line timestamps
- **Vocal separation**: Optional Demucs-based vocal/instrumental separation for more accurate alignment
- **Multiple output formats**: LRC (default), SRT (experimental)
- **Pre-trained models**: Three checkpoint variants (Baseline, MTL, BDR) for different accuracy/speed trade-offs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/liu-xiaoran/auto-lrc.git
cd auto-lrc

# Install dependencies (Python 3.9/3.10 recommended)
pip install -r requirements.txt
```

> **Note**: This project uses Git LFS for model checkpoint files. Make sure [Git LFS](https://git-lfs.github.com/) is installed before cloning.

### Basic Usage

```bash
python main.py <lyrics_file> <audio_file>
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --format` | Output format: `lrc` or `srt` | `lrc` |
| `-l, --line_only` | `1` = line-level, `0` = word-level | `0` |
| `-v, --vocalize` | `1` = separate vocals via Demucs, `0` = skip | `1` |
| `-m, --model` | Demucs model: `mdx`, `mdx_extra`, `mdx_q`, `mdx_extra_q` | `mdx_extra` |
| `-i, --idx` | Demucs model index (`-1` = ensemble, `0-3` = single sub-model) | `-1` |

### Examples

```bash
# Word-level alignment with vocal separation (default)
python main.py demofile/original_txt.txt demofile/original_track.mp3

# Line-level alignment, skip vocal separation
python main.py lyrics.txt song.mp3 -l 1 -v 0

# Output as SRT format
python main.py lyrics.txt song.mp3 -f srt
```

## Architecture

```
lyrics + audio
  → phonetic.py: language detection (regex CJK → fastText fallback) + phonemization
  → t2l.py: audio loading (torchaudio → librosa fallback)
          → vocal separation via Demucs
          → resample to 22050 Hz
  → wrapper.py: mel-spectrogram → CNN-RNN → phoneme posteriorgram → DTW alignment
  → gen_lrc(): frame indices → [MM:SS.mmm] timestamps
  → LRC output
```

### Model Checkpoints

| Model | Description | Classes |
|-------|-------------|---------|
| `checkpoint_Baseline` | Single-task acoustic model | 41 |
| `checkpoint_MTL` | Multi-task learning model (default) | (41, 47) |
| `checkpoint_BDR` | Boundary detection model | — |

Models are stored in `./checkpoints/` and managed via Git LFS.

## Programmatic API

```python
from t2l.t2l import process

lrc = process(
    txt_lines,          # list of lyrics lines
    "song.mp3",         # audio file path
    mtl_model="MTL",    # model type: "Baseline" | "MTL" | "Baseline_BDR" | "MTL_BDR"
    demucs_model="mdx_extra",
    demucs_idx=-1,
    line_only=False,
    out_file=None,
    verbose=True,
    vocalize=True,
    format="lrc"
)
print(lrc)
```

## Output Format

```lrc
[00:00.120]<00:00.120>Hel<00:00.240>lo<00:00.360> <00:00.480>World
[00:01.000]<00:01.000>这<00:01.120>是<00:01.240>中<00:01.360>文
```

## License

See [LICENSE](LICENSE) for details. Note that `pykakasi` (Japanese romanization) is GPL-licensed.
