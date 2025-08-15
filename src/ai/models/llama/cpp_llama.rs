use std::{net::{TcpStream, SocketAddr, IpAddr, Ipv4Addr}, process::{Child, Command, Stdio}, thread, time::Duration};

use anyhow::{Context, Result};
use hf_hub::api::Progress;
use serde_json::json;

use crate::ai::models::llama::TextGenerator;

pub struct CppLlama {
    port: u16,
    model_id: String,
    terminators: Vec<String>,
    server: Option<Child>,
}

impl CppLlama {
    pub fn new<P: Progress + Clone>(model_id: &str, _revision: &str, mut _progress: P) -> Result<CppLlama> {
        // Check llama-server availability by attempting to execute it with --help
        let server_cmd = "llama-server";
        let available = Command::new(server_cmd)
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok();
        if !available {
            return Err(anyhow::anyhow!("llama-server executable not found on PATH. See llama.cpp docs: https://github.com/ggml-org/llama.cpp/tree/master"));
        }

        let port: u16 = 8905;

        // Start llama-server in background for the provided model id
        // Example: llama-server -hf ggml-org/gemma-3-1b-it-GGUF --port 8905
        let mut child = Command::new(server_cmd)
            .args(["-hf", model_id, "--port", &port.to_string(), "-ngl", "32"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| "failed to spawn llama-server process")?;

        // Wait for the server to become ready by probing the TCP port
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let mut ready = false;
        for _ in 0..1000 { // ~10s
            if TcpStream::connect(addr).is_ok() {
                ready = true;
                break;
            }
            // If process exited, bubble up an error
            if let Ok(Some(status)) = child.try_wait() {
                // Capture any stderr for diagnostics
                let mut diag = String::new();
                if let Some(mut stderr) = child.stderr.take() { use std::io::Read; let _ = stderr.read_to_string(&mut diag); }
                let msg = if diag.is_empty() { format!("llama-server exited early with status: {}", status) } else { format!("llama-server exited early with status: {}\n{}", status, diag) };
                return Err(anyhow::anyhow!(msg));
            }
            thread::sleep(Duration::from_millis(100));
        }
        if !ready {
            return Err(anyhow::anyhow!("timed out waiting for llama-server to be ready on port 8905"));
        }

        println!("llama-server ready on port {}", port);

        Ok(CppLlama {
            port,
            model_id: model_id.to_string(),
            terminators: Vec::new(),
            server: Some(child),
        })
    }
}

impl Drop for CppLlama {
    fn drop(&mut self) {
        if let Some(child) = &mut self.server {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl TextGenerator for CppLlama {
    fn add_terminator(&mut self, terminator: &str) {
        self.terminators.push(terminator.to_string());
    }

    fn generate(
        &mut self,
        prompt: String,
        _sample_len: Option<usize>,
        _channel: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<String> {

        println!("Generating with CppLlama");
        // print as curl command
        println!("curl -X POST http://localhost:{}/v1/completions -H \"Content-Type: application/json\" -d '{{ \"prompt\": \"{}\" }}'", self.port, prompt);

        let url = format!("http://localhost:{}/v1/completions", self.port);
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .context("failed to build HTTP client")?;
        // Grammar to force 1-3 comma-separated alphabetic words (3-20 chars)
        let grammar = r#"root ::= ws? category ("," ws? category){0,2} ws?
ws ::= " "*
category ::= alpha{3,20}
alpha ::= [A-Za-z]"#;

        let payload = json!({
            "prompt": prompt,
            "temperature": 0.0,
            "stop": ["\n", "<", "</output>"],
            "max_tokens": 16,
            "grammar": grammar,
            "stream": false,
            "model": self.model_id
        });

        let resp = client
            .post(url)
            .json(&payload)
            .send()
            .context("failed to POST to llama-server /v1/chat/completions")?
            .error_for_status()
            .context("llama-server returned error status")?;

        let mut text = String::new();
        let v: serde_json::Value = resp.json().context("failed to parse JSON response")?;

        println!("Response: {:?}", v);
        
        // Parse OpenAI-compatible completions shape first, then fallback to chat
        if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
            if let Some(content) = choice.get("text").and_then(|c| c.as_str()) {
                text = content.to_string();
            } else if let Some(content) = choice.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                text = content.to_string();
            }
        }

        // Apply simple terminator truncation if configured
        if !self.terminators.is_empty() && !text.is_empty() {
            if let Some(idx) = self
                .terminators
                .iter()
                .filter_map(|t| text.find(t).map(|i| (i)))
                .min()
            {
                text.truncate(idx);
            }
        }

        Ok(text)
    }
}