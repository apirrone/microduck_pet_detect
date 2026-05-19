# microduck_pet_detect

A tiny audio classifier that detects when the microduck robot is being pet on
the head (where the onboard mic sits). Trains in Python, ships as a Rust binary
that does sub-millisecond inference per window on a Raspberry Pi Zero 2W.

- **Features**: 40-band log-mel spectrogram over a 1 s window @ 16 kHz mono.
- **Model**: ~20 KB tiny CNN (Conv → BN → ReLU → MaxPool → Conv → BN → ReLU → GAP → Linear).
- **Output**: `PettingEvent::Start` / `PettingEvent::End` with hysteresis. Also
  consumable as a stdout-line stream from the `pet_detect` binary.

## Install on the Pi

```bash
curl -sSL https://raw.githubusercontent.com/apirrone/microduck_pet_detect/main/install.sh | bash
```

Installs `pet_detect` + `pet_features` to `/usr/local/bin` and the ONNX model
to `/opt/microduck/pet_detect.onnx`. Re-run anytime to upgrade.

## Quick test

```bash
arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -t raw \
  | pet_detect --model /opt/microduck/pet_detect.onnx
```

You'll see one line per inference (~every 250 ms):
`<ts_ms>  <p_petting>  <state>`. Scratch the head and `p_petting` shoots up.

Flags: `--threshold 0.95` (enter), `--exit-threshold 0.85` (leave; hysteresis).

## Record training data

On the Pi:

```bash
arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -d 30 /tmp/normal_NN.wav
arecord -D plughw:aic3104,0 -f S16_LE -r 16000 -c 1 -d 30 /tmp/petting_NN.wav
```

Then `scp` into the repo:

```
data/normal/    # walking, falling, motors moving, idle, ambient — anything that isn't petting
data/petting/   # head scratches at varying speeds / pressures / fingers
```

Aim for at least a few minutes per class with varied conditions.

## Train

```bash
cargo build --release --bin pet_features   # only needed if features.rs / lib.rs changed
uv run training/train.py
```

`train.py` invokes the Rust `pet_features` binary to extract features — same
code path as the runtime, so there's no train/infer drift. Writes
`models/pet_detect.onnx`.

## Publish a new release

```bash
git add models/pet_detect.onnx
git commit -m "retrained model"
git tag v0.1.X
git push --tags
```

CI cross-builds for aarch64 and publishes a release tarball containing the
binaries + `libonnxruntime.so*` + the ONNX model. On the Pi, re-run the install
script to pull it.

## Use as a library

The `PettingDetector` struct in `src/lib.rs` is the integration surface:

```rust
use microduck_pet_detect::{PettingDetector, PettingDetectorConfig, PettingEvent};

let mut det = PettingDetector::new(&model_path, PettingDetectorConfig::default())?;
let (events, last_p) = det.push_samples(&samples_f32);
for ev in events {
    match ev {
        PettingEvent::Start => { /* be happy */ }
        PettingEvent::End   => { /* calm down */ }
    }
}
```

Used by `microduck_runtime` via a git dep — see its `pet_worker.rs` for the
arecord-subprocess pattern.
