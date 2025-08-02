use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{Int, Tensor, TensorData},
};
use tokenizers::Tokenizer as GTokenizer;

use crate::ai::operators::sampling::Sampler;

use anyhow::{Context, Error, Result};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

use wgpu::{Instance, InstanceDescriptor, Backends, RequestAdapterOptions};
use pollster::block_on;

// Define the backend. We use Wgpu with optimized tensor operations.
// WGPU backend provides good cross-platform GPU performance.
type LBackend = Wgpu;

use crate::ai::models::llama::LlamaConfig;

use super::models::transformer::{KeyValueCache, Transformer};


struct HFLlama {
    config: LlamaConfig,
    tokenizer: GTokenizer,
    model: Transformer<LBackend>,
    rope: RotaryEncoding<LBackend>,
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
        
        // Run initial forward pass
        let mut out = self.model.forward(token_tensor, pos, &mut self.cache, &self.rope);
        pos += tokens.len();
        
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
        
        for i in 1..max_new_tokens.min(5) { // Limit to first 5 tokens for debugging
            // Create tensor for single token
            let token_data = TensorData::new(vec![next_token as i32], single_token_shape);
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
            
            // Forward pass for single token
            out = self.model.forward(token_tensor, pos, &mut self.cache, &self.rope);
            pos += 1;
            
            // Extract logits
            let logits = out.flatten(1, 2);
            
            // Debug logits for this step
            let logits_data: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
            let mut indexed_logits: Vec<(usize, f32)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            next_token = sampler.sample(logits).into_scalar() as u32;
            
            // Check for EOS token
            if next_token == self.config.eos_token_id {
                println!("Step {}: Hit EOS token", i);
                break;
            }
            
            // Decode and add to response
            let token_text = self.tokenizer.decode(&[next_token], false).unwrap();
            println!("Step {}: Token {} ('{}') - Top 3: {}/{}/{}", 
                i, next_token, token_text,
                indexed_logits[0].0, indexed_logits[1].0, indexed_logits[2].0);
            response.push_str(&token_text);
            
            // Send to channel if provided
            if let Some(channel) = &channel {
                if let Err(_) = channel.send(token_text.clone()) {
                    // Channel receiver has disconnected, but continue generation
                }
            }
            
            tokens.push(next_token as i32);
        }
        
        // Continue normal generation for remaining tokens
        for _i in 5..max_new_tokens {
            let token_data = TensorData::new(vec![next_token as i32], single_token_shape);
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
            
            out = self.model.forward(token_tensor, pos, &mut self.cache, &self.rope);
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
            // Map layers.[i].mlp.down_proj.* -> layers.[i].feed_forward.w2.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.down_proj\\.(.+)",
                "$1.feed_forward.w2.$2",
            )
            // Map layers.[i].mlp.gate_proj.* -> layers.[i].feed_forward.swiglu.linear_inner.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.gate_proj\\.(.+)",
                "$1.feed_forward.swiglu.linear_inner.$2",
            )
            // Map layers.[i].mlp.up_proj.* -> layers.[i].feed_forward.swiglu.linear_outer.*
            .with_key_remap(
                "(layers\\.[0-9]+)\\.mlp\\.up_proj\\.(.+)",
                "$1.feed_forward.swiglu.linear_outer.$2",
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

        let record = SafetensorsFileRecorder::<FullPrecisionSettings>::default()
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

        let rope = RotaryEncodingConfig::new(
            config.max_seq_len,
            config.hidden_size / config.num_attention_heads,
        )
        .with_theta(config.rope_theta as f32);

        let rope = rope.init(&device);

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
    fn test_load_llama_from_hf() -> Result<()> {
        let max_tokens: usize = 10; // Reduced for faster testing

        // Start total timing
        let total_start = Instant::now();

        // Use default GPU device for stability
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
        let temperature = 0.7;
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

        // Time tokenization
        let tokenize_start = Instant::now();
        let prompt = "The quick brown".to_string();
        let tokens = tokenizer.encode(prompt.clone(), true).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        println!("⏱️  Tokenization took: {:.2}ms", tokenize_start.elapsed().as_millis());

        let remaining = std::cmp::min(max_tokens, config.max_seq_len) - tokens.len();
        println!("Remaining: {}", remaining);
        println!("Initial tokens: {:?}", tokens);

        println!("\n🚀 Starting inference...\n");
        print!("{}", prompt);

        // Use optimized sampling for inference
        let mut sampler = if temperature < 0.01 {
            // For very low temperature, use greedy sampling (faster)
            Sampler::Argmax
        } else {
            Sampler::TopP(TopP::new(top_p, seed))
        };

        // Start inference timing
        let inference_start = Instant::now();
        
        // Time initial forward pass (prefill) - optimized tensor creation
        let prefill_start = Instant::now();
        let mut pos = 0;
        
        // Create tensor more efficiently without cloning
        let token_data = TensorData::new(tokens.clone(), [1, tokens.len()]);
        let mut token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &device);
        
        // Run initial forward pass
        let mut out = model.forward(token_tensor, pos, &mut cache, &rope);
        pos += tokens.len();
        
        // Extract logits for last position more efficiently
        let seq_len = tokens.len();
        let logits = out.select(1, [seq_len - 1].into()).flatten(1, 2);
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        
        println!("\n⏱️  Prefill (initial forward pass) took: {:.2}ms", prefill_start.elapsed().as_millis());
        
        // Time first token generation
        let first_token_start = Instant::now();
        let token_text = tokenizer.decode(&[next_token], false).unwrap();
        print!("{}", token_text);
        std::io::stdout().flush().unwrap();
        tokens.push(next_token as i32);
        println!("\n⏱️  First token generation took: {:.2}ms", first_token_start.elapsed().as_millis());
        
        // Time subsequent token generation
        let mut total_decode_time = 0.0f32;
        let mut max_decode_time = 0.0f32;
        let mut min_decode_time = f32::MAX;
        
        // Pre-allocate single token tensor to reuse
        let single_token_shape = [1, 1];
        
        for i in 0..(remaining - 1) {
            let decode_start = Instant::now();
            
            // Create tensor more efficiently for single token
            let token_data = TensorData::new(vec![next_token as i32], single_token_shape);
            token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &device);
            
            // Forward pass for single token
            out = model.forward(token_tensor, pos, &mut cache, &rope);
            pos += 1;
            
            // Extract logits (already flattened for single token)
            let logits = out.flatten(1, 2);
            next_token = sampler.sample(logits).into_scalar() as u32;
            
            // Early exit on EOS
            if next_token == config.eos_token_id { 
                println!("\n⏱️  Hit EOS token at position {}", i + 1);
                break; 
            }
            
            // Decode and print token
            let token_text = tokenizer.decode(&[next_token], false).unwrap();
            print!("{}", token_text);
            std::io::stdout().flush().unwrap();
            tokens.push(next_token as i32);
            
            // Track timing
            let decode_time = decode_start.elapsed().as_millis() as f32;
            total_decode_time += decode_time;
            max_decode_time = max_decode_time.max(decode_time);
            min_decode_time = min_decode_time.min(decode_time);
            
            // Print timing for every 5th token
            if (i + 1) % 5 == 0 {
                println!("\n⏱️  Token {}: {:.2}ms", i + 1, decode_time);
            }
        }
        
        let total_inference_time = inference_start.elapsed().as_secs_f32();
        let total_time = total_start.elapsed().as_secs_f32();
        
        println!("\n\n📊 PERFORMANCE SUMMARY:");
        println!("⏱️  Total time: {:.2}s", total_time);
        println!("⏱️  Inference time: {:.2}s", total_inference_time);
        println!("⏱️  Tokens generated: {}", remaining.min(tokens.len() - 3));
        println!("⏱️  Tokens/second: {:.2}", (remaining.min(tokens.len() - 3) as f32) / total_inference_time);
        if remaining > 1 {
            println!("⏱️  Avg decode time per token: {:.2}ms", total_decode_time / (remaining - 1) as f32);
            println!("⏱️  Min decode time: {:.2}ms", min_decode_time);
            println!("⏱️  Max decode time: {:.2}ms", max_decode_time);
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
        
        let prompt = "The quick brown".to_string();
        let response = llama.generate(prompt.clone(), Some(100), None)?;
        
        println!("Prompt: {}", prompt);
        println!("Generated: {}", response);
        
        // Verify response is not empty
        assert!(!response.is_empty(), "Generated response should not be empty");
        
        Ok(())
    }
}

//pub struct TextGenerator {
//    model: Llama<LBackend>,
//    config: LlamaConfig,
//    device: WgpuDevice,
//    eos_token: u32,
//    sample_len: Option<usize>,
//    channel: Option<std::sync::mpsc::Sender<String>>,
//    terminators: Option<Vec<String>>,
//}

//impl TextGenerator {
//    pub fn new(model_id: &str, revision: &str) -> Result<Self> {
//        // Configure the WGPU backend to use Vulkan.
//        // Burn will automatically select the best available device.
//        let device = WgpuDevice::default();
//        println!("Using device: {:?}", device);
//
//
//
//        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
//
//        println!("Loading model weights from Hugging Face...");
//        // `llama-burn` provides a convenient function to load models directly
//        // from Hugging Face repos. It handles the weight conversion.
//        let model: Llama<Backend> = load_llama_model_from_hf(repo, &device)?;
//
//        let eos_token = {
//            let eos_encoding = tokenizer.encode("<|end|>", false).map_err(Error::msg)?;
//            if eos_encoding.get_ids().is_empty() {
//                let alt_eos = tokenizer.encode("</s>", false).map_err(Error::msg)?;
//                if alt_eos.get_ids().is_empty() {
//                    println!("Warning: Could not find EOS token, using token ID 2");
//                    2u32
//                } else {
//                    alt_eos.get_ids()[0]
//                }
//            } else {
//                eos_encoding.get_ids()[0]
//            }
//        };
//
//        Ok(Self {
//            model,
//            config,
//            tokenizer,
//            device,
//            eos_token,
//            sample_len: None,
//            channel: None,
//            terminators: None,
//        })
//    }
//
//    pub fn with_sample_len(mut self, sample_len: usize) -> Self {
//        self.sample_len = Some(sample_len);
//        self
//    }
//
//    pub fn with_channel(mut self, channel: std::sync::mpsc::Sender<String>) -> Self {
//        self.channel = Some(channel);
//        self
//    }
//
//    pub fn with_terminators(mut self, terminators: Vec<&str>) -> Self {
//        self.terminators = Some(terminators.into_iter().map(|s| s.to_string()).collect());
//        self
//    }
//
//    pub fn set_terminators(&mut self, terminators: Vec<&str>) {
//        self.terminators = Some(terminators.into_iter().map(|s| s.to_string()).collect());
//    }
//
//    pub fn generate(
//        &mut self,
//        prompt: String,
//        sample_len: Option<usize>,
//        channel: Option<std::sync::mpsc::Sender<String>>,
//    ) -> Result<String> {
//        let sample_len = sample_len.or(self.sample_len).unwrap_or(200);
//
//        let mut tokens = self
//            .tokenizer
//            .encode(prompt.as_str(), true)
//            .map_err(Error::msg)?
//            .get_ids()
//            .to_vec();
//
//        let mut response = String::new();
//        let mut cache = self.model.new_cache();
//
//        for index in 0..sample_len {
//            let start_pos = if index == 0 { 0 } else { tokens.len() - 1 };
//
//            let token_slice = &tokens[start_pos..];
//            let input: Tensor<Backend, 2, Int> = Tensor::<Backend, 2>::from_data(
//                TensorData::new(token_slice.to_vec(), [1, token_slice.len()]),
//                &self.device,
//            )
//                .int();
//
//            // Run the model forward pass
//            let logits = self.model.forward(input, start_pos, &mut cache);
//
//            // Get the logits for the last token
//            let next_token_logits = logits.slice([0..1, (logits.dims()[1] - 1)..]);
//
//            // Simple greedy sampling (argmax)
//            let next_token_id = next_token_logits.argmax(2).into_scalar() as u32;
//
//            if next_token_id == self.eos_token && index > 0 {
//                break;
//            }
//
//            tokens.push(next_token_id);
//
//            let token_vec = vec![next_token_id];
//            if let Ok(text) = self.tokenizer.decode(&token_vec, false) {
//                if let Some(terminators) = &self.terminators {
//                    if terminators.iter().any(|t| text.contains(t)) {
//                        break;
//                    }
//                }
//
//                let text = text.replace(" ", " ");
//                response.push_str(&text);
//
//                if let Some(channel) = &channel {
//                    channel.send(text.clone())?;
//                }
//                if let Some(channel) = &self.channel {
//                    channel.send(text)?;
//                }
//            } else {
//                println!("Failed to decode token {}", next_token_id);
//            }
//        }
//        Ok(response)
//    }
//
//    pub fn prompt(&mut self, prompt: String) -> Result<String> {
//        self.generate(prompt, None, None)
//    }
//}
//
//pub fn init_text_gen(
//    tx: Option<std::sync::mpsc::Sender<String>>,
//    terminators: Option<Vec<&str>>,
//) -> Result<Arc<Mutex<TextGenerator>>> {
//    // Using a smaller, compatible model for demonstration
//    let model_id = "core42/tinylama-1.1b-chat-v1.0-hf";
//    println!("Loading model '{}'...", model_id);
//
//    let mut all_eos = vec!["<|end|>", "</s>", "\n\n"];
//    if let Some(terminators) = terminators {
//        for t in &terminators {
//            all_eos.push(t);
//        }
//    }
//
//    let mut generator = TextGenerator::new(model_id, "main")?
//        .with_sample_len(50)
//        .with_terminators(all_eos);
//
//    if let Some(tx) = tx {
//        generator = generator.with_channel(tx);
//    }
//    Ok(Arc::new(Mutex::new(generator)))
//}
