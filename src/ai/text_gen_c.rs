use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    module::{Module, Param},
    tensor::{Int, Tensor, TensorData},
};
use tokenizers::Tokenizer as GTokenizer;

use crate::ai::{models::transformer::TransformerRecord, operators::sampling::Sampler};

use anyhow::{Context, Error, Result};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

use wgpu::{Instance, InstanceDescriptor, Backends, RequestAdapterOptions};
use md5;
use pollster::block_on;

// Define the backend. We use Wgpu with optimized tensor operations.
// WGPU backend provides good cross-platform GPU performance.
type LBackend = Wgpu;

use crate::ai::models::llama::LlamaConfig;

use super::models::transformer::{KeyValueCache, Transformer};
use super::models::custom_rope::{CustomRotaryEncoding, CustomRotaryEncodingConfig};

// Helper function to compute tensor hash for debugging - concatenate all elements as strings and MD5 hash
fn tensor_hash<B: burn::tensor::backend::Backend, const D: usize>(tensor: &Tensor<B, D>) -> u64 {
    let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();
    
    // Round each element to 3 decimal places and concatenate to string
    let tensor_string: String = data.iter()
        .map(|&val| format!("{:.3}", val))
        .collect::<Vec<String>>()
        .join("");
    
    // MD5 hash the concatenated string
    let hash_bytes = md5::compute(tensor_string.as_bytes());
    
    // Convert first 8 bytes to u64 for display (little-endian to match Python)
    u64::from_le_bytes([
        hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
        hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
    ])
}

struct HFLlama {
    config: LlamaConfig,
    tokenizer: GTokenizer,
    model: Transformer<LBackend>,
    rope: CustomRotaryEncoding<LBackend>,
    cache: Vec<KeyValueCache<LBackend>>,
    device: WgpuDevice,
}

impl HFLlama {
    pub fn generate(
        &mut self,
        prompt: String,
        sample_len: Option<usize>,
        channel: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<String> {
        let sample_len = sample_len.unwrap_or(50); // Default to 50 tokens if not specified
        
        println!("\n=== RUST DEBUG INFO ===");
        println!("Prompt: '{}'", prompt);
        
        // Reset cache for each new generation
        for cache in self.cache.iter_mut() {
            cache.reset();
        }
        
        // Tokenize the prompt
        let tokens = self.tokenizer.encode(prompt.clone(), true).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        
        println!("Input tokens: {:?}", tokens);
        
        // Print config for comparison
        println!("Model config:");
        println!("  vocab_size: {}", self.config.vocab_size);
        println!("  hidden_size: {}", self.config.hidden_size);
        println!("  num_hidden_layers: {}", self.config.num_hidden_layers);
        println!("  num_attention_heads: {}", self.config.num_attention_heads);
        println!("  num_key_value_heads: {}", self.config.num_key_value_heads);
        println!("  intermediate_size: {}", self.config.intermediate_size);
        println!("  max_seq_len: {}", self.config.max_seq_len);
        println!("  rope_theta: {}", self.config.rope_theta);
        println!("  tie_word_embeddings: {}", self.config.tie_word_embeddings);
        
        let max_new_tokens = std::cmp::min(sample_len, self.config.max_seq_len - tokens.len());
        let mut response = String::new();
        
        // Use smart sampling based on temperature (defaulting to greedy for consistency)
        let mut sampler = Sampler::Argmax;
        
        println!("\n=== FORWARD PASS DEBUG ===");
        println!("Input shape: [1, {}]", tokens.len());
        
        // Initial forward pass (prefill)
        let mut pos = 0;
        let token_data = TensorData::new(tokens.clone(), [1, tokens.len()]);
        let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
        
        println!("Input tensor hash: {:016x}", {
            let float_tensor = token_tensor.clone().float();
            tensor_hash(&float_tensor)
        });
        
        // Run initial forward pass (debug enabled for first token)
        let mut out = self.model.forward_with_debug(token_tensor, pos, &mut self.cache, &self.rope, true);
        pos += tokens.len();

        println!("Model output tensor hash: {:016x}", tensor_hash(&out));
        
        println!("Output shape: {:?}", out.dims());
        
        // Extract logits for last position
        let seq_len = tokens.len();
        let logits = out.select(1, [seq_len - 1].into()).flatten(1, 2);
        println!("Logits shape: {:?}", logits.dims());
        
        // Get logits values for debugging
        let logits_data: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
        let min_logit = logits_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("Logits min/max: {:.6} / {:.6}", min_logit, max_logit);
        
        // Get top 10 tokens for comparison
        let mut indexed_logits: Vec<(usize, f32)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        println!("\nTop 10 tokens:");
        for i in 0..10.min(indexed_logits.len()) {
            let (token_id, value) = indexed_logits[i];
            let token_text = self.tokenizer.decode(&[token_id as u32], false).unwrap();
            println!("  {}. Token {} ('{}'): {:.6}", i+1, token_id, token_text, value);
        }
        
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        let greedy_token_text = self.tokenizer.decode(&[next_token], false).unwrap();
        println!("\nGreedy next token: {} ('{}')", next_token, greedy_token_text);
        
        // Decode first generated token
        let token_text = self.tokenizer.decode(&[next_token], false).unwrap();
        println!("First generated token: {} (ID: {})", token_text, next_token);
        response.push_str(&token_text);
        
        // Send to channel if provided
        if let Some(channel) = &channel {
            if let Err(_) = channel.send(token_text.clone()) {
                // Channel receiver has disconnected, but continue generation
            }
        }
        
        tokens.push(next_token as i32);
        
        println!("\n=== STEP-BY-STEP GENERATION ===");
        
        // Generate remaining tokens  
        let single_token_shape = [1, 1];
        
        // Continue normal generation for remaining tokens
        for _i in 1..max_new_tokens {
            let token_data = TensorData::new(vec![next_token as i32], single_token_shape);
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
            
            out = self.model.forward_with_debug(token_tensor, pos, &mut self.cache, &self.rope, false);
            pos += 1;
            
            let logits = out.flatten(1, 2);
            next_token = sampler.sample(logits).into_scalar() as u32;
            
            if next_token == self.config.eos_token_id {
                break;
            }
            
            let token_text = self.tokenizer.decode(&[next_token], false).unwrap();
            response.push_str(&token_text);
            
            if let Some(channel) = &channel {
                if let Err(_) = channel.send(token_text.clone()) {
                    // Channel receiver has disconnected, but continue generation
                }
            }
            
            tokens.push(next_token as i32);
        }
        
        Ok(response)
    }

    fn new(model_id: &str, revision: &str, device: WgpuDevice) -> Result<HFLlama> {
        let api =
            hf_hub::api::sync::Api::new().context("Failed to create Hugging Face API client")?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));

        // Download necessary files from the Hub
        let config_filename = repo
            .get("config.json")
            .context("Failed to get config.json")?;
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .context("Failed to get tokenizer.json")?;
        let model_filename = repo
            .get("model.safetensors")
            .context("Failed to get model weights")?;

        // Load the config file
        println!("Loading config...");
        let config_filename = config_filename
            .to_str()
            .context("Config filename is not valid UTF-8")?;
        let config = LlamaConfig::from_file(config_filename)
            .context("Failed to load LlamaConfig from file")?;
        println!(
            "Loaded config: num_hidden_layers={:?}, num_attention_heads={:?}, num_key_value_heads={:?}",
            config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads
        );

        // Load the tokenizer
        println!("Loading tokenizer...");
        let tokenizer = GTokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
        println!(
            "Loaded tokenizer with vocab size: {}",
            tokenizer.get_vocab_size(false)
        );

        // Load the model weights
        println!("Loading model weights...");
        let load_args = LoadArgs::new(model_filename)
            // Map lm_head.* -> output.*
            .with_key_remap("lm_head\\.(.+)", "output.$1")
            // Remove model. prefix
            .with_key_remap("model\\.(.+)", "$1")
            // Map embed_tokens.* -> tok_embeddings.*
            .with_key_remap("embed_tokens\\.(.+)", "tok_embeddings.$1")
            // Map layers.[i].input_layernorm.* -> layers.[i].attention_norm.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.input_layernorm\\.(.+)",
                "$1.attention_norm.$2",
            )
            // Map layers.[i].post_attention_layernorm.* -> layers.[i].ffn_norm.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.post_attention_layernorm\\.(.+)",
                "$1.ffn_norm.$2",
            )
            // Map layers.[i].mlp.down_proj.* -> layers.[i].feed_forward.mlp.down_proj.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.down_proj\\.(.+)",
                "$1.feed_forward.mlp.down_proj.$2",
            )
            // Map layers.[i].mlp.gate_proj.* -> layers.[i].feed_forward.mlp.gate_proj.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.gate_proj\\.(.+)",
                "$1.feed_forward.mlp.gate_proj.$2",
            )
            // Map layers.[i].mlp.up_proj.* -> layers.[i].feed_forward.mlp.up_proj.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.up_proj\\.(.+)",
                "$1.feed_forward.mlp.up_proj.$2",
            )
            // Map layers.[i].self_attn.k_proj.* -> layers.[i].attention.wk.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.self_attn\\.k_proj\\.(.+)",
                "$1.attention.wk.$2",
            )
            // Map layers.[i].self_attn.o_proj.* -> layers.[i].attention.wo.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.self_attn\\.o_proj\\.(.+)",
                "$1.attention.wo.$2",
            )
            // Map layers.[i].self_attn.q_proj.* -> layers.[i].attention.wq.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.self_attn\\.q_proj\\.(.+)",
                "$1.attention.wq.$2",
            )
            // Map layers.[i].self_attn.v_proj.* -> layers.[i].attention.wv.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.self_attn\\.v_proj\\.(.+)",
                "$1.attention.wv.$2",
            )
            // Map norm.weight -> norm.gamma for all layers
            .with_key_remap("(.*)norm\\.weight", "${1}norm.gamma");
        let load_args = if !config.tie_word_embeddings {
            load_args.with_key_remap("lm_head\\.weight", "output.weight")
        } else {
            load_args
        };

        let record: TransformerRecord<_> = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
            .load(load_args, &device)
            .context("Should decode state successfully")?;

        // Create TransformerConfig with correct field names
        let transformer_config = super::models::transformer::TransformerConfig {
            vocab_size: config.vocab_size,
            n_layers: config.num_hidden_layers,
            d_model: config.hidden_size,
            hidden_size: config.intermediate_size,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            max_seq_len: config.max_seq_len,
            norm_eps: 1e-5,
            tie_word_embeddings: config.tie_word_embeddings,
        };

        let model = transformer_config
            .init(&device)
            .load_record(record);

        let cache = (0..config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    config.num_key_value_heads,
                    config.max_seq_len,
                    config.hidden_size / config.num_attention_heads,
                    &device,
                )
            })
            .collect::<Vec<_>>();

        let rope_config = CustomRotaryEncodingConfig::new(
            config.max_seq_len,
            config.hidden_size / config.num_attention_heads,
        )
        .with_theta(config.rope_theta as f32);

        let rope = rope_config.init(&device);

        Ok(HFLlama {
            config,
            model,
            cache,
            rope,
            tokenizer,
            device,
        })
    }
}

mod test {
    use std::io::Write;
    use std::time::Instant;

    use crate::ai::operators::sampling::{Sampler, TopP};

    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};
    

    #[test] 
    fn test_prefill_performance() -> Result<()> {
        let max_tokens = 1; // Just test prefill performance
        
        // Start total timing
        let total_start = Instant::now();

        let device = WgpuDevice::default();
        println!("WgpuDevice selected: {:?}", device);
        let instance = Instance::new(InstanceDescriptor { 
            backends: Backends::all(),
            ..Default::default() 
        });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions { 
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default() 
        })).unwrap();
        let info = adapter.get_info();
        print!("Selected device name: {}\n", info.name);
        let temperature = 0.0; // Use greedy for fastest sampling
        let top_p = 0.9;
        let seed = 10;
        let model_id = "HuggingFaceTB/SmolLM-135M";
        let revision = "main";
        
        // Time model loading
        let load_start = Instant::now();
        println!("Loading model '{}' from revision '{}'", model_id, revision);
        let llama =
            HFLlama::new(model_id, revision, device).context("Failed to load Llama model")?;
        println!("⏱️  Model loading took: {:.2}s", load_start.elapsed().as_secs_f32());

        let HFLlama {
            model,
            rope,
            mut cache,
            device,
            tokenizer,
            config,
        } = llama;

        // Test different prompt lengths to measure prefill scaling
        let test_prompts = vec![
            "Hello",
            "The quick brown fox",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog and runs through the forest",
        ];

        for prompt in test_prompts {
            let tokenize_start = Instant::now();
            let tokens = tokenizer.encode(prompt.to_string(), true).unwrap();
            let tokens = tokens
                .get_ids()
                .iter()
                .map(|&x| x as i32)
                .collect::<Vec<_>>();
            println!("⏱️  Tokenization took: {:.2}ms", tokenize_start.elapsed().as_millis());

            println!("\n🧪 Testing prefill with {} tokens: {:?}", tokens.len(), tokens);

            // Reset cache for each test
            for c in cache.iter_mut() {
                c.reset();
            }

            // Use optimized sampling for inference
            let mut sampler = Sampler::Argmax;

            // Time ONLY the prefill (initial forward pass)
            let prefill_start = Instant::now();
            let mut pos = 0;
            
            let token_data = TensorData::new(tokens.clone(), [1, tokens.len()]);
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &device);
            
            // Run initial forward pass
            let out = model.forward(token_tensor, pos, &mut cache, &rope);
            let prefill_time = prefill_start.elapsed().as_millis();
            
            println!("⏱️  Prefill ({} tokens) took: {:.2}ms ({:.2}ms per token)", 
                tokens.len(), prefill_time, prefill_time as f32 / tokens.len() as f32);
        }

        Ok(())
    }

    #[test]
    fn test_generate_method() -> Result<()> {
        let device = WgpuDevice::default();
        let model_id = "HuggingFaceTB/SmolLM-135M";
        let revision = "main";
        
        println!("Loading model for generate test...");
        let mut llama = HFLlama::new(model_id, revision, device)
            .context("Failed to load Llama model")?;
        
        let prompt = "The following is a popular long poem: ".to_string();
        let response = llama.generate(prompt.clone(), Some(100), None)?;
        
        println!("Prompt: {}", prompt);
        println!("Generated: {}", response);
        
        // Verify response is not empty
        assert!(!response.is_empty(), "Generated response should not be empty");
        
        Ok(())
    }
}