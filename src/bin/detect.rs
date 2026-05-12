// Live petting detector wrapper.
// Reads 16-bit signed little-endian mono PCM from stdin at 16 kHz, runs the
// PettingDetector, and prints one line per inference: "<ts_ms>\t<p>\t<state>".

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;

use microduck_pet_detect::{i16_to_f32, PettingDetector, PettingDetectorConfig, WINDOW_SAMPLES};

#[derive(Parser)]
struct Args {
    /// Path to the trained ONNX model.
    #[arg(long)]
    model: PathBuf,
    /// Hop between inference windows in samples (default ≈ 250 ms).
    #[arg(long, default_value_t = WINDOW_SAMPLES / 4)]
    stride: usize,
    /// Probability above which petting is declared started.
    #[arg(long, default_value_t = 0.98)]
    threshold: f32,
    /// Probability below which petting is declared ended (hysteresis).
    #[arg(long, default_value_t = 0.5)]
    exit_threshold: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut detector = PettingDetector::new(
        &args.model,
        PettingDetectorConfig {
            stride: args.stride,
            enter_threshold: args.threshold,
            exit_threshold: args.exit_threshold,
        },
    )?;

    let start_t = Instant::now();
    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut buf = [0u8; 4096];
    let mut i16_buf: Vec<i16> = Vec::with_capacity(2048);

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if n % 2 != 0 {
            return Err(anyhow!("odd byte count, expected i16 stream"));
        }
        i16_buf.clear();
        for chunk in buf[..n].chunks_exact(2) {
            i16_buf.push(i16::from_le_bytes([chunk[0], chunk[1]]));
        }
        let f32_samples = i16_to_f32(&i16_buf);
        let (events, last_p) = detector.push_samples(&f32_samples);
        if let Some(p) = last_p {
            let ts = start_t.elapsed().as_millis();
            let state = if detector.is_petting() { "petting" } else { "normal" };
            println!("{}\t{:.3}\t{}", ts, p, state);
        }
        for ev in events {
            eprintln!("event: {:?}", ev);
        }
    }
    Ok(())
}
