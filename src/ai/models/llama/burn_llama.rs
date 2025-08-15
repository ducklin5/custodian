use std::path::PathBuf;

use anyhow::{Context, Result};

use hf_hub::api::Progress;
use serde::Deserialize;

use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    module::Module,
    tensor::{Int, Tensor, TensorData},
};
use burn::record::{HalfPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

use tokenizers::Tokenizer as GTokenizer;

use anyhow::Error;


// Define the backend. We use Wgpu with optimized tensor operations.
// WGPU backend provides good cross-platform GPU performance.

use crate::ai::{models::llama::TextGenerator, operators::sampling::Sampler};
use crate::ai::models::transformer::{KeyValueCache, Transformer, TransformerConfig, TransformerRecord};
use crate::ai::models::custom_rope::{CustomRotaryEncoding, CustomRotaryEncodingConfig};

#[derive(Deserialize, Debug, Clone)]
pub struct RopeScalingConfig {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
enum EosTokenId {
    Single(u32),
    Array(Vec<u32>),
}

impl EosTokenId {
    pub fn contains(&self, id: u32) -> bool {
        match self {
            EosTokenId::Single(x) => *x == id,
            EosTokenId::Array(xs) => xs.contains(&id),
        }
    }
}

#[derive(Deserialize, Debug, Clone)]
pub struct LlamaConfig {
    #[serde(rename = "_name_or_path")]
    pub name_or_path: Option<String>,
    pub architectures: Vec<String>,
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f64,
    pub bos_token_id: u32,
    pub eos_token_id: EosTokenId,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    #[serde(rename = "max_position_embeddings")]
    pub max_seq_len: usize,
    #[serde(default)]
    pub mlp_bias: bool,
    pub model_type: String,
    pub num_attention_heads: usize,
    #[serde(rename = "num_hidden_layers")]
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: Option<RopeScalingConfig>,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: Option<String>,
    pub use_cache: bool,
    pub vocab_size: usize,
}

impl LlamaConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let file =
            std::fs::File::open(path).context(format!("Failed to open config file: {}", path))?;
        let reader = std::io::BufReader::new(file);
        let config: Self = serde_json::from_reader(reader)
            .context(format!("Failed to parse config file: {}", path))?;
        Ok(config)
    }
}

pub type LBackend = Wgpu;
pub struct BurnLlama {
    config: LlamaConfig,
    tokenizer: GTokenizer,
    model: Transformer<LBackend>,
    rope: CustomRotaryEncoding<LBackend>,
    cache: Vec<KeyValueCache<LBackend>>,
    device: WgpuDevice,
    terminators: Vec<String>,
}

impl BurnLlama {
    fn get_file<P: Progress + Clone>(repo: &hf_hub::Repo, filename: &str, mut progress: P) -> Result<PathBuf> {
        let api =
            hf_hub::api::sync::Api::new().context("Failed to create Hugging Face API client")?;
        let api_repo = api.repo(repo.clone());
        println!("Getting file: {}", filename);
        progress.init(0, filename);
        let result = match hf_hub::Cache::default().repo(repo.clone()).get(filename) {
            Some(path) => Ok(path),
            None => api_repo.download_with_progress(filename, progress.clone()).map_err(Error::msg),
        };
        progress.finish();
        result
    }

    pub fn new<P: Progress + Clone>(model_id: &str, revision: &str, device: WgpuDevice, mut progress: P) -> Result<BurnLlama> {
        let repo = hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        );

        // Download necessary files from the Hub
        let config_filename = Self::get_file(&repo, "config.json", progress.clone()).context("Failed to get config.json")?;
        let tokenizer_filename = Self::get_file(&repo, "tokenizer.json", progress.clone()).context("Failed to get tokenizer.json")?;
        let model_filename = Self::get_file(&repo, "model.safetensors", progress.clone()).context("Failed to get model.safetensors")?;

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
        progress.init(1, "Loading model weights...");
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

        let record: TransformerRecord<_> = SafetensorsFileRecorder::<HalfPrecisionSettings>::default()
            .load(load_args, &device)
            .context("Should decode state successfully")?;

        // Create TransformerConfig with correct field names
        let transformer_config = TransformerConfig {
            vocab_size: config.vocab_size,
            n_layers: config.num_hidden_layers,
            d_model: config.hidden_size,
            hidden_size: config.intermediate_size,
            n_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            max_seq_len: config.max_seq_len,
            norm_eps: config.rms_norm_eps,
            tie_word_embeddings: config.tie_word_embeddings,
        };

        let model = transformer_config
            .init(&device)
            .load_record(record);
        
        progress.finish();

        // Defer KV cache allocation to generation time to avoid huge allocations for large max_seq_len
        let cache: Vec<KeyValueCache<_>> = Vec::new();

        let rope_config = CustomRotaryEncodingConfig::new(
            config.max_seq_len,
            config.hidden_size / config.num_attention_heads,
        )
        .with_theta(config.rope_theta as f32);

        let rope = rope_config.init(&device);

        Ok(BurnLlama {
            config,
            model,
            cache,
            rope,
            tokenizer,
            device,
            terminators: vec![],
        })
    }

    

    pub fn reset_cache(&mut self) {
        for cache in self.cache.iter_mut() {
            cache.reset();
        }
    }

    
}

impl TextGenerator for BurnLlama {
    fn add_terminator(&mut self, terminators: &str) {
        self.terminators.push(terminators.to_string());
    }       
    fn generate(
        &mut self,
        prompt: String,
        sample_len: Option<usize>,
        channel: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<String> {
        let sample_len = sample_len.unwrap_or(50); // Default to 50 tokens if not specified

        // Tokenize the prompt
        let tokens = self.tokenizer.encode(prompt.clone(), true).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        
        
        // Ensure at least one token for prefill to avoid seq_len underflow
        if tokens.is_empty() {
            tokens.push(self.config.bos_token_id as i32);
        }

        let max_new_tokens = std::cmp::min(sample_len, self.config.max_seq_len.saturating_sub(tokens.len()));

        // Rebuild KV caches sized to this request to avoid over-allocation
        let desired_kv_len = std::cmp::min(self.config.max_seq_len, tokens.len() + max_new_tokens);
        self.cache = (0..self.config.num_hidden_layers)
            .map(|_| {
                KeyValueCache::new(
                    1,
                    self.config.num_key_value_heads,
                    desired_kv_len,
                    self.config.hidden_size / self.config.num_attention_heads,
                    &self.device,
                )
            })
            .collect::<Vec<_>>();
        let mut response = String::new();
        
        
        // Initial forward pass (prefill)
        let mut pos = 0;
        // Clamp prefill to the allocated KV capacity to avoid oversized allocations
        let prefill_len = tokens.len().min(desired_kv_len);
        let token_data = TensorData::new(tokens[..prefill_len].to_vec(), [1, prefill_len]);
        let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
        
        
        // Run initial forward pass using non-debug forward for performance
        let mut out = self.model.forward(token_tensor, pos, &mut self.cache, &self.rope);
        pos += prefill_len;

        
        // Extract logits for last position (guard against zero-length)
        let seq_len = prefill_len.max(1);
        let last_idx = (seq_len - 1).max(0);
        let logits = out.select(1, [last_idx].into()).flatten(1, 2);

        // Use smart sampling based on temperature (defaulting to greedy for consistency)
        let mut sampler = Sampler::Argmax;
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        
        
        // Prepare a reusable [1,1] tensor for decode steps to avoid per-step allocations
        let mut step_ids = Tensor::<LBackend, 2, Int>::zeros([1, 1], &self.device);
        // Accumulate token ids and decode in batches to reduce tokenizer overhead
        const DECODE_BATCH: usize = 8;
        let mut pending_ids: Vec<u32> = Vec::with_capacity(32);
        // Precompute terminator token sequences for robust detection across batch boundaries
        let terminator_token_seqs: Vec<Vec<u32>> = self
            .terminators
            .iter()
            .map(|t| {
                let enc = self.tokenizer.encode(t.as_str(), false).unwrap();
                enc.get_ids().iter().map(|&x| x as u32).collect::<Vec<u32>>()
            })
            .collect();
        let mut terminated = false;

        // Continue normal generation for remaining tokens
        for _i in 0..max_new_tokens {
            if self.config.eos_token_id.contains(next_token) {
                break;
            }
            
            // Accumulate token ids
            pending_ids.push(next_token);

            // Token-level terminator detection: check if pending tail matches any terminator token seq
            if !terminator_token_seqs.is_empty() {
                let mut matched = false;
                for term_seq in &terminator_token_seqs {
                    let tlen = term_seq.len();
                    if tlen > 0 && pending_ids.len() >= tlen {
                        if &pending_ids[pending_ids.len() - tlen..] == term_seq.as_slice() {
                            // Flush everything before the terminator
                            let flush_len = pending_ids.len() - tlen;
                            if flush_len > 0 {
                                if let Ok(chunk_text) = self.tokenizer.decode(&pending_ids[..flush_len], false) {
                                    if !chunk_text.is_empty() {
                                        response.push_str(&chunk_text);
                                        if let Some(channel) = &channel { let _ = channel.send(chunk_text.clone()); }
                                    }
                                }
                            }
                            pending_ids.clear();
                            terminated = true;
                            matched = true;
                            break;
                        }
                    }
                }
                if matched { break; }
            }

            let should_flush = pending_ids.len() >= DECODE_BATCH;

            if should_flush {
                if let Ok(mut chunk_text) = self.tokenizer.decode(&pending_ids, false) {
                    let mut found_terminator = false;
                    for terminator in self.terminators.iter() {
                        if let Some(idx) = chunk_text.find(terminator) {
                            chunk_text.truncate(idx);
                            found_terminator = true;
                            break;
                        }
                    }

                    if !chunk_text.is_empty() {
                        response.push_str(&chunk_text);
                        if let Some(channel) = &channel {
                            let _ = channel.send(chunk_text.clone());
                        }
                    }

                    pending_ids.clear();

                    if found_terminator {
                        terminated = true;
                        break;
                    }
                }
            }

            tokens.push(next_token as i32);

            // Reuse step_ids tensor by overwriting its single value
            step_ids = step_ids.clone().slice_assign(
                [0..1, 0..1],
                Tensor::from_ints([[next_token as i32]], &self.device),
            );

            // Non-debug forward for performance
            out = self.model.forward(step_ids.clone(), pos, &mut self.cache, &self.rope);
            pos += 1;
            
            let logits = out.flatten(1, 2);
            next_token = sampler.sample(logits).into_scalar() as u32;
        }
        // Flush any remaining pending tokens with terminator handling
        if !terminated && !pending_ids.is_empty() {
            if let Ok(mut chunk_text) = self.tokenizer.decode(&pending_ids, false) {
                let mut truncate_at: Option<usize> = None;
                for terminator in self.terminators.iter() {
                    if let Some(idx) = chunk_text.find(terminator) {
                        truncate_at = Some(truncate_at.map_or(idx, |min| min.min(idx)));
                    }
                }
                if let Some(idx) = truncate_at { chunk_text.truncate(idx); }

                if !chunk_text.is_empty() {
                    response.push_str(&chunk_text);
                    if let Some(channel) = &channel {
                        let _ = channel.send(chunk_text.clone());
                    }
                }
            }
        }
        
        Ok(response)
    }
}

mod test {
    use burn_wgpu::WgpuDevice;
    use anyhow::{Context, Result};
    use crate::ai::models::llama::BurnLlama;
    use crate::ai::models::llama::TextGenerator;


    #[test]
    fn test_generate_method() -> Result<()> {
        let device = WgpuDevice::default();
        let model_id = "HuggingFaceTB/SmolLM-135M";
        let revision = "main";
        
        println!("Loading model for generate test...");
        let mut llama = BurnLlama::new(model_id, revision, device, ())
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
