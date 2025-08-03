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
        
        
        let max_new_tokens = std::cmp::min(sample_len, self.config.max_seq_len - tokens.len());
        let mut response = String::new();
        
        // Use smart sampling based on temperature (defaulting to greedy for consistency)
        let mut sampler = Sampler::Argmax;
        
        
        // Initial forward pass (prefill)
        let mut pos = 0;
        let token_data = TensorData::new(tokens.clone(), [1, tokens.len()]);
        let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
        
        
        // Run initial forward pass (debug enabled for first token)
        let mut out = self.model.forward_with_debug(token_tensor, pos, &mut self.cache, &self.rope, true);
        pos += tokens.len();

        
        // Extract logits for last position
        let seq_len = tokens.len();
        let logits = out.select(1, [seq_len - 1].into()).flatten(1, 2);
        
        // Get logits values for debugging
        let logits_data: Vec<f32> = logits.clone().into_data().to_vec().unwrap();
        let min_logit = logits_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_logit = logits_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Get top 10 tokens for comparison
        let mut indexed_logits: Vec<(usize, f32)> = logits_data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        for i in 0..10.min(indexed_logits.len()) {
            let (token_id, value) = indexed_logits[i];
            let token_text = self.tokenizer.decode(&[token_id as u32], false).unwrap();
        }
        
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        let greedy_token_text = self.tokenizer.decode(&[next_token], false).unwrap();
        
        // Decode first generated token
        let token_text = self.tokenizer.decode(&[next_token], false).unwrap();
        response.push_str(&token_text);
        
        // Send to channel if provided
        if let Some(channel) = &channel {
            if let Err(_) = channel.send(token_text.clone()) {
                // Channel receiver has disconnected, but continue generation
            }
        }
        
        tokens.push(next_token as i32);
        
        
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