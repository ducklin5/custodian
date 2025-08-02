use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
};
use tokenizers::Tokenizer as GTokenizer;

use anyhow::{Context, Error, Result};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};

use wgpu::{Instance, InstanceDescriptor, Backends, RequestAdapterOptions};
use pollster::block_on;

// Define the backend. We use Wgpu with f32 elements and i64 integers.
// Autodiff is not needed for inference.
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

    use crate::ai::operators::sampling::{Sampler, TopP};

    use super::*;
    use burn::tensor::{activation::softmax, Int, Tensor, TensorData};
    use num_traits::ToPrimitive;

    #[test]
    fn test_load_llama_from_hf() -> Result<()> {
        let max_tokens = 30;


        let device = WgpuDevice::default();
        println!("WgpuDevice selected: {:?}", device);
        let instance = Instance::new(InstanceDescriptor { backends: Backends::all(), ..Default::default() });
        let adapter = block_on(instance.request_adapter(&RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).unwrap();
        let info = adapter.get_info();
        print!("Selected device name: {}\n", info.name);
        let temperature = 0.7;
        let top_p = 0.9;
        let seed = 10;
        let model_id = "HuggingFaceTB/SmolLM-135M";
        let revision = "main";
        println!("Loading model '{}' from revision '{}'", model_id, revision);
        let llama =
            HFLlama::new(model_id, revision, device).context("Failed to load Llama model")?;

        let HFLlama {
            model,
            rope,
            mut cache,
            device,
            tokenizer,
            config,
        } = llama;

        let prompt = "The quick brown".to_string();
        let tokens = tokenizer.encode(prompt.clone(), true).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();

        let remaining = std::cmp::min(max_tokens, config.max_seq_len) - tokens.len();
        println!("Remaining: {}", remaining);
        println!("Initial tokens: {:?}", tokens);

        println!("\n\n\n");
        print!("{}", prompt);

        let mut sampler = Sampler::TopP(TopP::new(top_p, seed));

        let mut pos = 0;
        let mut token_tensor = Tensor::<LBackend, 2, Int>::from_data(TensorData::new(tokens.clone(), [1, tokens.len()]), &device);
        let mut out = model.forward(token_tensor, pos, &mut cache, &rope);
        pos += tokens.len();
        let logits = out.select(1, [tokens.len() - 1].into()).flatten(1, 2);
        let mut next_token = sampler.sample(logits).into_scalar() as u32;
        let token_text = tokenizer.decode(&[next_token], false).unwrap();
        print!("{}", token_text);
        std::io::stdout().flush().unwrap();
        tokens.push(next_token as i32);
        for _ in 0..(remaining - 1) {
            token_tensor = Tensor::<LBackend, 2, Int>::from_data(TensorData::new(vec![next_token as i32], [1, 1]), &device);
            out = model.forward(token_tensor, pos, &mut cache, &rope);
            pos += 1;
            let logits = out.flatten(1, 2);
            next_token = sampler.sample(logits).into_scalar() as u32;
            if next_token == config.eos_token_id { break; }
            let token_text = tokenizer.decode(&[next_token], false).unwrap();
            print!("{}", token_text);
            std::io::stdout().flush().unwrap();
            tokens.push(next_token as i32);
        }

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
