use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EngineResponse {
    ok: bool,
    error: Option<String>,
    instrument: Option<String>,
    confidence: Option<f32>,
    model: Option<String>,
    midi_path: Option<String>,
    note_count: Option<usize>,
    duration_seconds: Option<f32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptionResult {
    instrument: String,
    confidence: f32,
    model: String,
    note_count: usize,
    duration_seconds: f32,
    midi_file_name: String,
    midi_bytes: Vec<u8>,
}

fn sanitize_file_name(file_name: &str) -> String {
    file_name
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri should live inside the project root")
        .to_path_buf()
}

fn python_executable(project_root: &PathBuf) -> PathBuf {
    let venv_python = project_root.join(".venv").join("bin").join("python");
    if venv_python.exists() {
        venv_python
    } else {
        PathBuf::from("python3")
    }
}

#[tauri::command]
fn transcribe_audio(file_name: String, audio_bytes: Vec<u8>) -> Result<TranscriptionResult, String> {
    if audio_bytes.is_empty() {
        return Err("The selected audio file is empty.".into());
    }

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_millis();
    let work_dir = std::env::temp_dir().join("soundclay").join(timestamp.to_string());
    fs::create_dir_all(&work_dir).map_err(|error| error.to_string())?;

    let safe_file_name = sanitize_file_name(&file_name);
    let input_path = work_dir.join(&safe_file_name);
    fs::write(&input_path, audio_bytes).map_err(|error| error.to_string())?;

    let root = project_root();
    let engine_path = root.join("engine").join("soundclay_engine.py");
    let output_dir = work_dir.join("out");
    let output = Command::new(python_executable(&root))
        .arg(&engine_path)
        .arg("--input")
        .arg(&input_path)
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TMPDIR", "/private/tmp")
        .env("PYTHONPYCACHEPREFIX", "/private/tmp/soundclay-pycache")
        .output()
        .map_err(|error| format!("Failed to launch Python engine: {error}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let json_line = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| format!("Engine did not return JSON. stderr: {}", stderr.trim()))?;
    let response: EngineResponse = serde_json::from_str(json_line.trim()).map_err(|error| {
        format!("Engine returned invalid JSON: {error}. stderr: {}", stderr.trim())
    })?;

    if !output.status.success() || !response.ok {
        return Err(response
            .error
            .unwrap_or_else(|| format!("Engine failed: {}", stderr.trim())));
    }

    let midi_path = response
        .midi_path
        .ok_or_else(|| "Engine did not return a MIDI path.".to_string())?;
    let midi_bytes = fs::read(&midi_path).map_err(|error| error.to_string())?;
    let midi_file_name = PathBuf::from(&safe_file_name)
        .with_extension("mid")
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("soundclay.mid")
        .to_string();

    Ok(TranscriptionResult {
        instrument: response.instrument.unwrap_or_else(|| "Unknown".into()),
        confidence: response.confidence.unwrap_or(0.0),
        model: response.model.unwrap_or_else(|| "unknown".into()),
        note_count: response.note_count.unwrap_or(0),
        duration_seconds: response.duration_seconds.unwrap_or(0.0),
        midi_file_name,
        midi_bytes,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![transcribe_audio])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
