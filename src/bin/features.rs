// Dump log-mel features of a WAV to stdout as f32 little-endian, shape [N_MELS, WINDOW_FRAMES].
// Slides a 1-second window with 50% overlap; one feature block per stdout write.
//
// Used by the Python training script to extract features identical to the
// runtime Rust implementation — guarantees train/infer parity.

use std::io::Write;

use anyhow::Result;
use clap::Parser;

use microduck_pet_detect::{
    load_wav_mono_16k, MelExtractor, HOP, N_MELS, WINDOW_FRAMES, WINDOW_SAMPLES,
};

#[derive(Parser)]
struct Args {
    /// Path to a WAV file (any sample rate / channel count — auto-resampled to 16k mono).
    wav: String,
    /// Hop between windows in samples (default: WINDOW_SAMPLES / 2, i.e. 50% overlap).
    #[arg(long)]
    stride: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let samples = load_wav_mono_16k(&args.wav)?;
    let stride = args.stride.unwrap_or(WINDOW_SAMPLES / 2);
    let extractor = MelExtractor::new();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut start = 0;
    let mut n_blocks = 0;
    while start + WINDOW_SAMPLES <= samples.len() {
        let mel = extractor.log_mel(&samples[start..start + WINDOW_SAMPLES]);
        // Write as raw f32 LE.
        let bytes: &[u8] = bytemuck_cast(&mel);
        out.write_all(bytes)?;
        n_blocks += 1;
        start += stride;
    }
    eprintln!(
        "wrote {} blocks of [{},{}] f32 ({} bytes each, hop={} samples)",
        n_blocks,
        N_MELS,
        WINDOW_FRAMES,
        N_MELS * WINDOW_FRAMES * 4,
        HOP
    );
    Ok(())
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain old data, length matches.
    unsafe {
        std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * std::mem::size_of::<f32>())
    }
}
