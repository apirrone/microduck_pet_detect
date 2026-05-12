// Live petting detector.
// Reads 16-bit signed little-endian mono PCM from stdin at 16 kHz, computes
// log-mel features over sliding 1-second windows, and prints one line per
// inference: "<timestamp_ms>\t<p_petting>\t<label>".
//
// Pipe from arecord, e.g.:
//   arecord -D hw:aic3104,0 -f S16_LE -r 16000 -c 1 -t raw | pet_detect --model models/pet.onnx

use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use clap::Parser;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use microduck_pet_detect::{MelExtractor, HOP, N_MELS, WINDOW_FRAMES, WINDOW_SAMPLES};

#[derive(Parser)]
struct Args {
    /// Path to the trained ONNX model.
    #[arg(long)]
    model: PathBuf,
    /// Hop between inference windows in samples (default = 25% of window, ~250 ms).
    #[arg(long, default_value_t = WINDOW_SAMPLES / 4)]
    stride: usize,
    /// Decision threshold on petting probability.
    #[arg(long, default_value_t = 0.6)]
    threshold: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .commit_from_file(&args.model)?;

    let extractor = MelExtractor::new();
    let mut ring: Vec<f32> = Vec::with_capacity(WINDOW_SAMPLES * 2);
    let start_t = Instant::now();

    let stdin = std::io::stdin();
    let mut reader = stdin.lock();
    let mut buf = [0u8; 4096];
    let mut next_infer_at = WINDOW_SAMPLES;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if n % 2 != 0 {
            return Err(anyhow!("odd byte count, expected i16 stream"));
        }
        for chunk in buf[..n].chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]);
            ring.push(s as f32 / 32768.0);
        }

        while ring.len() >= next_infer_at {
            let start = next_infer_at - WINDOW_SAMPLES;
            let mel = extractor.log_mel(&ring[start..start + WINDOW_SAMPLES]);
            // Shape: [1, 1, N_MELS, WINDOW_FRAMES] for the tiny CNN.
            let input = Tensor::from_array(([1usize, 1, N_MELS, WINDOW_FRAMES], mel))?;
            let outputs = session.run(ort::inputs![input])?;
            let (_shape, probs) = outputs[0].try_extract_tensor::<f32>()?;
            // Model outputs softmax over [normal, petting] — index 1 is petting.
            let p_pet = probs[1];
            let label = if p_pet >= args.threshold { "petting" } else { "normal" };
            let ts = start_t.elapsed().as_millis();
            println!("{}\t{:.3}\t{}", ts, p_pet, label);
            next_infer_at += args.stride;
        }

        // Keep ring at most 2× window to bound memory.
        if ring.len() > WINDOW_SAMPLES * 4 {
            let drop = ring.len() - WINDOW_SAMPLES * 2;
            ring.drain(..drop);
            next_infer_at -= drop;
        }
    }
    // Silence the unused-import warning for HOP.
    let _ = HOP;
    Ok(())
}
