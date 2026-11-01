#![allow(dead_code)]
#[path = "../src/cleanup.rs"]
mod cleanup;
#[path = "../src/state.rs"]
mod state;

use serde::{Deserialize, Serialize};
use std::{
    io::{self, BufRead},
    path::PathBuf,
    time::Instant,
};

#[derive(Deserialize)]
struct Input {
    id: serde_json::Value,
    text: String,
    #[serde(default)]
    context: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Output {
    id: serde_json::Value,
    text: String,
    error: Option<String>,
    model_id: String,
    device: String,
    inference_ms: u128,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let (mut path, mut model_id, mut device, mut mode) = (None, None, None, None);
    while let Some(flag) = args.next() {
        let value = args.next().ok_or("Missing argument value")?;
        match flag.as_str() {
            "--model" => path = Some(PathBuf::from(value)),
            "--model-id" => model_id = Some(value),
            "--device" => device = Some(value),
            "--mode" => mode = Some(value),
            _ => return Err(format!("Unknown argument: {flag}").into()),
        }
    }
    let path = path.ok_or("Missing --model")?;
    let model_id = model_id.ok_or("Missing --model-id")?;
    let device = device.ok_or("Missing --device")?;
    let mode = match mode.as_deref() {
        Some("format") => state::CleanupMode::Format,
        Some("intent") => state::CleanupMode::Intent,
        _ => return Err("--mode must be format or intent".into()),
    };
    let runtime = match cleanup::Runtime::load(&path, &model_id, &device) {
        Ok(runtime) => runtime,
        Err(_) if device == "auto" => cleanup::Runtime::load(&path, &model_id, "cpu")?,
        Err(error) => return Err(error.into()),
    };
    for line in io::stdin().lock().lines() {
        let line = line?;
        let input: Input = serde_json::from_str(&line)?;
        let started = Instant::now();
        let (text, error) = match runtime.generate(&input.text, &input.context, mode, &[]) {
            Ok(text) => (text, None),
            Err(error) => (input.text, Some(error)),
        };
        let output = Output {
            id: input.id,
            text,
            error,
            model_id: model_id.clone(),
            device: runtime.device.clone(),
            inference_ms: started.elapsed().as_millis(),
        };
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(())
}
