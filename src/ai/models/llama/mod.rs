use anyhow::{Context, Result};

use serde::Deserialize;

use burn::{
    backend::{wgpu::WgpuDevice, Wgpu},
    module::Module,
    tensor::{Int, Tensor, TensorData},
};
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

use tokenizers::Tokenizer as GTokenizer;

use anyhow::Error;


// Define the backend. We use Wgpu with optimized tensor operations.
// WGPU backend provides good cross-platform GPU performance.

use crate::ai::operators::sampling::Sampler;
use super::transformer::{KeyValueCache, Transformer, TransformerConfig, TransformerRecord};
use super::custom_rope::{CustomRotaryEncoding, CustomRotaryEncodingConfig};

#[derive(Deserialize, Debug, Clone)]
pub struct RopeScalingConfig {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
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
    pub eos_token_id: u32,
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
pub struct HFLlama {
    config: LlamaConfig,
    tokenizer: GTokenizer,
    model: Transformer<LBackend>,
    rope: CustomRotaryEncoding<LBackend>,
    cache: Vec<KeyValueCache<LBackend>>,
    device: WgpuDevice,
    terminators: Vec<String>,
}

impl HFLlama {
    pub fn new(model_id: &str, revision: &str, device: WgpuDevice) -> Result<HFLlama> {
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
            terminators: vec![],
        })
    }

    pub fn add_terminator(&mut self, terminators: &str) {
        self.terminators.push(terminators.to_string());
    }        

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
        
        
        // Run initial forward pass 
        let mut out = self.model.forward_with_debug(token_tensor, pos, &mut self.cache, &self.rope, false);
        pos += tokens.len();

        
        // Extract logits for last position
        let seq_len = tokens.len();
        let logits = out.select(1, [seq_len - 1].into()).flatten(1, 2);
        
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        
        
        // Continue normal generation for remaining tokens
        for _i in 0..max_new_tokens {
            if next_token == self.config.eos_token_id {
                break;
            }
            
            let mut token_text = self.tokenizer.decode(&[next_token], false).unwrap();

            let mut found_terminator = false;
            for terminator in self.terminators.iter() {
                if token_text.contains(terminator) {
                    token_text = token_text.split(terminator).next().unwrap().to_string();
                    found_terminator = true;
                    break;
                }
            }

            response.push_str(&token_text);
            
            // Send to channel if provided
            if let Some(channel) = &channel {
                if let Err(_) = channel.send(token_text.clone()) {
                    // Channel receiver has disconnected, but continue generation
                }
            }

            if found_terminator {
                break;
            }

            tokens.push(next_token as i32);

            let token_data = TensorData::new(vec![next_token as i32], [1, 1]);
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(token_data, &self.device);
            
            out = self.model.forward_with_debug(token_tensor, pos, &mut self.cache, &self.rope, false);
            pos += 1;
            
            let logits = out.flatten(1, 2);
            next_token = sampler.sample(logits).into_scalar() as u32;
        }
        
        Ok(response)
    }
}

mod test {
    use burn_wgpu::WgpuDevice;
    use anyhow::{Context, Result};
    use crate::ai::models::llama::HFLlama;


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
