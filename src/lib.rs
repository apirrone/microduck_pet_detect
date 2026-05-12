use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex32, Fft, FftPlanner};

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const HOP: usize = 160; // 10 ms
pub const WIN: usize = 400; // 25 ms
pub const N_MELS: usize = 40;
pub const WINDOW_FRAMES: usize = 100; // 1.0 s of audio
pub const WINDOW_SAMPLES: usize = (WINDOW_FRAMES - 1) * HOP + WIN; // 16240 samples
pub const FMIN: f32 = 0.0;
pub const FMAX: f32 = 8000.0;
pub const LOG_EPS: f32 = 1e-6;

pub struct MelExtractor {
    fft: Arc<dyn Fft<f32>>,
    hann: Vec<f32>,
    mel_filters: Vec<Vec<(usize, f32)>>, // sparse: for each mel band, (fft_bin, weight)
}

impl MelExtractor {
    pub fn new() -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let hann: Vec<f32> = (0..WIN)
            .map(|n| 0.5 - 0.5 * (2.0 * PI * n as f32 / (WIN as f32 - 1.0)).cos())
            .collect();
        let mel_filters = build_mel_filterbank(SAMPLE_RATE, N_FFT, N_MELS, FMIN, FMAX);
        Self { fft, hann, mel_filters }
    }

    /// Process a 1-second window of mono f32 samples in [-1,1].
    /// Returns log-mel matrix flattened in row-major order [N_MELS * WINDOW_FRAMES].
    pub fn log_mel(&self, samples: &[f32]) -> Vec<f32> {
        assert!(
            samples.len() >= WINDOW_SAMPLES,
            "need at least {} samples, got {}",
            WINDOW_SAMPLES,
            samples.len()
        );
        let mut out = vec![0.0f32; N_MELS * WINDOW_FRAMES];
        let mut fft_buf = vec![Complex32::new(0.0, 0.0); N_FFT];

        for frame in 0..WINDOW_FRAMES {
            let start = frame * HOP;
            // Windowed frame, zero-padded to N_FFT.
            for i in 0..N_FFT {
                fft_buf[i] = if i < WIN {
                    Complex32::new(samples[start + i] * self.hann[i], 0.0)
                } else {
                    Complex32::new(0.0, 0.0)
                };
            }
            self.fft.process(&mut fft_buf);

            // Power spectrum, only first N_FFT/2 + 1 bins matter.
            let n_bins = N_FFT / 2 + 1;
            let mut power = vec![0.0f32; n_bins];
            for (i, c) in fft_buf.iter().take(n_bins).enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            // Apply mel filterbank, take log.
            for (m, filt) in self.mel_filters.iter().enumerate() {
                let mut s = 0.0;
                for &(bin, w) in filt {
                    s += w * power[bin];
                }
                out[m * WINDOW_FRAMES + frame] = (s + LOG_EPS).ln();
            }
        }
        out
    }
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}

fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10f32.powf(m / 2595.0) - 1.0)
}

fn build_mel_filterbank(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<(usize, f32)>> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    // n_mels+2 evenly-spaced points on the mel scale.
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    // FFT-bin centers in Hz.
    let bin_hz: Vec<f32> = (0..n_bins)
        .map(|k| k as f32 * sr as f32 / n_fft as f32)
        .collect();

    let mut filters = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let (lo, ctr, hi) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        let mut filt = Vec::new();
        for (k, &fk) in bin_hz.iter().enumerate() {
            let w = if fk <= lo || fk >= hi {
                0.0
            } else if fk <= ctr {
                (fk - lo) / (ctr - lo)
            } else {
                (hi - fk) / (hi - ctr)
            };
            if w > 0.0 {
                filt.push((k, w));
            }
        }
        filters.push(filt);
    }
    filters
}

/// Load a mono WAV (or downmix stereo by averaging) and resample naively to 16 kHz
/// using linear interpolation. Returns samples in [-1, 1].
pub fn load_wav_mono_16k(path: &str) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample as i32;
            let scale = (1i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<_>, _>>()?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()?,
    };
    // Downmix to mono.
    let mono: Vec<f32> = if channels == 1 {
        raw
    } else {
        raw.chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    // Resample to SAMPLE_RATE.
    if spec.sample_rate as usize == SAMPLE_RATE {
        return Ok(mono);
    }
    Ok(resample_linear(&mono, spec.sample_rate as usize, SAMPLE_RATE))
}

fn resample_linear(input: &[f32], sr_in: usize, sr_out: usize) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let ratio = sr_in as f64 / sr_out as f64;
    let out_len = (input.len() as f64 / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let i0 = src.floor() as usize;
        let frac = (src - i0 as f64) as f32;
        let s0 = input[i0];
        let s1 = if i0 + 1 < input.len() { input[i0 + 1] } else { s0 };
        out.push(s0 * (1.0 - frac) + s1 * frac);
    }
    out
}
