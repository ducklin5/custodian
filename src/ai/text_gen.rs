use anyhow::{Context, Error, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama as LlamaModel};
use hf_hub::{Repo, RepoType, api::sync::Api};
use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;


pub struct TextGenerator {
    model: LlamaModel,
    config: LlamaConfig,
    tokenizer: Tokenizer,
    device: Device,
    cache: Cache,
    eos_token: u32,
    sample_len: Option<usize>,
    channel: Option<std::sync::mpsc::Sender<String>>,
    terminators: Option<Vec<String>>,
}

impl TextGenerator {
    pub fn new(model_id: &str, revision: &str) -> Result<Self> {
        let device = Device::cuda_if_available(0).context("Failed to select CUDA device")?;

        println!("Using device: {:?}", device);

        let api = Api::new().context("Failed to create Hugging Face API client")?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let tokenizer_filename = repo
            .get("tokenizer.json")
            .context("Failed to get tokenizer.json")?;
        let config_filename = repo
            .get("config.json")
            .context("Failed to get config.json")?;

        // Read and parse the config
        let config_str = std::fs::read_to_string(&config_filename)?;
        let config_json: Value = serde_json::from_str(&config_str)?;

        println!("Model architecture: {:?}", config_json.get("model_type"));

        // Extract config values manually from JSON with better defaults for SmolLM2
        let vocab_size = config_json["vocab_size"].as_u64().unwrap_or(32000) as usize;
        let hidden_size = config_json["hidden_size"].as_u64().unwrap_or(512) as usize;
        let intermediate_size = config_json["intermediate_size"].as_u64().unwrap_or(1408) as usize;
        let num_hidden_layers = config_json["num_hidden_layers"].as_u64().unwrap_or(24) as usize;
        let num_attention_heads =
            config_json["num_attention_heads"].as_u64().unwrap_or(16) as usize;

        // For SmolLM2, num_key_value_heads should typically equal num_attention_heads
        let num_key_value_heads =
            if let Some(kv_heads) = config_json["num_key_value_heads"].as_u64() {
                kv_heads as usize
            } else {
                num_attention_heads
            };

        let max_position_embeddings = config_json["max_position_embeddings"]
            .as_u64()
            .unwrap_or(2048) as usize;
        let rms_norm_eps = config_json["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32;
        let rope_theta = config_json["rope_theta"].as_f64().unwrap_or(10000.0) as f32;

        // Check if the model ties embeddings (shares input and output embeddings)
        let tie_word_embeddings = config_json["tie_word_embeddings"]
            .as_bool()
            .unwrap_or(false);

        // Create Llama config - we'll try both tied and untied embeddings
        let config = LlamaConfig {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps: rms_norm_eps.into(),
            rope_theta,
            use_flash_attn: false,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            tie_word_embeddings, // Start with the config value
        };

        println!(
            "Loaded config: vocab_size={}, hidden_size={}, num_layers={}, num_attention_heads={}, num_key_value_heads={}, tie_word_embeddings={}",
            config.vocab_size,
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.tie_word_embeddings
        );

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;

        // Load safetensors files - try to find all model files
        let mut safetensors_files = Vec::new();

        // First try the standard patterns
        if let Ok(file) = repo.get("model.safetensors") {
            safetensors_files.push(file);
        } else {
            // Try sharded files
            let mut shard_index = 1;
            loop {
                let filename = if shard_index == 1 {
                    format!("model-{:05}-of-{:05}.safetensors", 1, 1)
                } else {
                    break;
                };

                if let Ok(file) = repo.get(&filename) {
                    safetensors_files.push(file);
                    shard_index += 1;
                } else {
                    break;
                }
            }

            // If no sharded files, try the simple naming
            if safetensors_files.is_empty() {
                safetensors_files.push(
                    repo.get("pytorch_model.bin")
                        .context("Failed to get model files")?,
                );
            }
        }

        if safetensors_files.is_empty() {
            return Err(Error::msg("No model files found"));
        }

        println!("Loading {} model file(s)", safetensors_files.len());

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensors_files, DType::F32, &device)
                .context("Failed to load model weights")?
        };

        // Debug: Print available tensor names to understand the model structure
        println!("Attempting to load model...");

        // Try loading with different configurations to handle various model variants
        let model = match LlamaModel::load(vb.clone(), &config) {
            Ok(model) => model,
            Err(e) => {
                println!("First load attempt failed: {}", e);
                println!("Trying with tie_word_embeddings=true...");

                // Try with tied embeddings
                let mut config_tied = config.clone();
                config_tied.tie_word_embeddings = true;

                match LlamaModel::load(vb.clone(), &config_tied) {
                    Ok(model) => {
                        println!("Successfully loaded with tied embeddings!");
                        model
                    }
                    Err(e2) => {
                        println!("Second load attempt also failed: {}", e2);
                        return Err(anyhow::anyhow!(
                            "Failed to load model with both tied and untied embeddings. Original error: {}, Tied error: {}. \
                            This model variant might not be compatible with the candle Llama loader.",
                            e,
                            e2
                        ));
                    }
                }
            }
        };

        // Initialize cache
        let cache = Cache::new(true, DType::F32, &config, &device)?;

        let eos_token = {
            // Try SmolLM2's EOS token first
            let eos_encoding = tokenizer.encode("<|end|>", false).map_err(Error::msg)?;
            if eos_encoding.get_ids().is_empty() {
                // Try alternative EOS tokens
                let alt_eos = tokenizer.encode("</s>", false).map_err(Error::msg)?;
                if alt_eos.get_ids().is_empty() {
                    println!("Warning: Could not find EOS token, using token ID 2");
                    2u32 // Common EOS token ID
                } else {
                    println!("Using </s> as EOS token, ID: {}", alt_eos.get_ids()[0]);
                    alt_eos.get_ids()[0]
                }
            } else {
                println!(
                    "Using <|end|> as EOS token, ID: {}",
                    eos_encoding.get_ids()[0]
                );
                eos_encoding.get_ids()[0]
            }
        };

        Ok(Self {
            model,
            config,
            tokenizer,
            device,
            cache,
            eos_token,
            sample_len: None,
            channel: None,
            terminators: None,
        })
    }

    fn config(&self) -> &LlamaConfig {
        &self.config
    }

    pub fn with_sample_len(mut self, sample_len: usize) -> Self {
        self.sample_len = Some(sample_len);
        self
    }

    pub fn with_channel(mut self, channel: std::sync::mpsc::Sender<String>) -> Self {
        self.channel = Some(channel);
        self
    }

    pub fn with_terminators(mut self, terminators: Vec<&str>) -> Self {
        let _term: Vec<String> = terminators.into_iter().map(|s| s.to_string()).collect();
        self.terminators = Some(_term);
        self
    }

    pub fn set_terminators(&mut self, terminators: Vec<&str>) {
        let terminators: Vec<String> = terminators.into_iter().map(|s| s.to_string()).collect();
        self.terminators = Some(terminators);
    }

    pub fn generate(
        &mut self,
        prompt: String,
        sample_len: Option<usize>,
        channel: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<String> {
        let sample_len = sample_len.or(self.sample_len).unwrap_or(200);

        // Reset cache for each new generation (like in the official example)
        self.cache = Cache::new(true, DType::F32, &self.config, &self.device)?;

        // Use the prompt as-is since it already contains the SmolLM2 chat format
        let formatted_prompt = prompt;

        // Debug: Let's see what tokens we're getting
        let debug_tokens = self
            .tokenizer
            .encode(&*formatted_prompt, true)
            .map_err(Error::msg)?;

        let mut tokens = self
            .tokenizer
            .encode(formatted_prompt.as_str(), true)
            .map_err(Error::msg)?
            .get_ids()
            .to_vec();

        let mut response = String::new();

        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&tokens[start_pos..], &self.device)?.unsqueeze(0)?;

            // Run the model forward pass
            let logits = self.model.forward(&input_ids, start_pos, &mut self.cache)?;

            // Handle different logits tensor shapes
            let final_logits = if logits.dims().len() == 3 {
                // Shape is [batch_size, seq_len, vocab_size]
                logits.i((0, logits.dim(1)? - 1))?
            } else if logits.dims().len() == 2 {
                // Shape is [seq_len, vocab_size] or [batch_size, vocab_size]
                if logits.dim(0)? == 1 {
                    // [1, vocab_size] -> take the single row
                    logits.i(0)?
                } else {
                    // [seq_len, vocab_size] -> take the last row
                    logits.i(logits.dim(0)? - 1)?
                }
            } else {
                return Err(anyhow::anyhow!(
                    "Unexpected logits shape: {:?}. Expected 2D or 3D tensor.",
                    logits.shape()
                ));
            };

            // Ensure we awaithave a valid 1D tensor for argmax
            if final_logits.dims().len() != 1 {
                return Err(anyhow::anyhow!(
                    "Final logits should be 1D but got shape: {:?}",
                    final_logits.shape()
                ));
            }

            if final_logits.elem_count() == 0 {
                return Err(anyhow::anyhow!(
                    "Final logits tensor is empty. This suggests an issue with the model forward pass."
                ));
            }

            // Simple greedy sampling (argmax)
            let next_token = final_logits.argmax(0)?;
            let next_token_id = next_token.to_scalar::<u32>()?;


            // Don't stop on the first EOS token, as it might be part of the prompt
            if next_token_id == self.eos_token && index > 0 {
                break;
            }

            tokens.push(next_token_id);

            // Decode and print the new token
            let token_vec = vec![next_token_id];
            if let Ok(text) = self.tokenizer.decode(&token_vec, false) {
                // break if token contains any of the terminators
                if let Some(terminators) = &self.terminators {
                    if terminators.iter().any(|t| text.contains(t)) {
                        break;
                    }
                }

                let text = text.replace("▁", " ");
                response.push_str(&text);

                if let Some(channel) = &channel {
                    channel.send(text.clone())?;
                }

                if let Some(channel) = &self.channel {
                    channel.send(text)?;
                }
            } else {
                println!("Failed to decode token {}", next_token_id);
            }
        }
        Ok(response)
    }

    pub fn prompt(&mut self, prompt: String) -> Result<String> {
        self.generate(prompt, None, None)
    }
}

pub fn init_text_gen(
    tx: Option<std::sync::mpsc::Sender<String>>,
    terminators: Option<Vec<&str>>,
) -> Result<Arc<Mutex<TextGenerator>>> {
    let model_id = "tiiuae/Falcon3-1B-Base";
    println!("Loading model '{}'...", model_id);
    
    let mut all_eos = vec!["<|end|>", "</s>", "\n\n"];
    if let Some(terminators) = terminators {
        for t in &terminators {
            all_eos.push(t);
        }
    }

    let mut generator = TextGenerator::new(model_id, "main")?
        .with_sample_len(50)
        .with_terminators(all_eos);

    if let Some(tx) = tx {
        generator = generator.with_channel(tx);
    }
    Ok(Arc::new(Mutex::new(generator)))
}
