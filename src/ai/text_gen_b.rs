use burn::backend::{Wgpu, wgpu::WgpuDevice};
use safetensors::SafeTensors;
use tokenizers::Tokenizer as GTokenizer;

use memmap2::MmapOptions;

use anyhow::{Context, Error, Result};

// Define the backend. We use Wgpu with f32 elements and i64 integers.
// Autodiff is not needed for inference.
type LBackend = Wgpu;
type DType = f32;

use super::models::llama::{Llama, LlamaConfig};

struct HFLlama {
    config: LlamaConfig,
    model: Llama<LBackend>,
    tokenizer: GTokenizer,
}

fn load_llama_from_hf(model_id: &str, revision: &str, device: &WgpuDevice) -> Result<HFLlama> {
    let api = hf_hub::api::sync::Api::new().context("Failed to create Hugging Face API client")?;
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

    println!("Loading config...");
    let config_filename = config_filename
        .to_str()
        .context("Config filename is not valid UTF-8")?;
    let config =
        LlamaConfig::from_file(config_filename).context("Failed to load LlamaConfig from file")?;
    println!(
        "Loaded config: num_hidden_layers={:?}, num_attention_heads={:?}, num_key_value_heads={:?}",
        config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads
    );

    println!("Loading tokenizer...");
    let tokenizer = GTokenizer::from_file(tokenizer_filename).map_err(Error::msg)?;
    println!(
        "Loaded tokenizer with vocab size: {}",
        tokenizer.get_vocab_size(false)
    );

    println!("Loading model weights...");
    let file = std::fs::File::open(model_filename.clone()).unwrap();
    println!("Reading model weights from file: {:?}", model_filename);
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    println!("Deserializing model weights...");
    let tensors =
        SafeTensors::deserialize(&buffer).context("Failed to load model weights from file")?;
    println!("Loaded model weights with {} tensors", tensors.len());

    // Debug: Check if we have the expected tensors
    let tensor_names: Vec<&str> = tensors.names().into_iter().collect();
    println!("Available tensor names: {:?}", tensor_names);

    // Check for key tensors
    let has_embed_tokens = tensor_names
        .iter()
        .any(|&name| name == "model.embed_tokens.weight");
    let has_lm_head = tensor_names.iter().any(|&name| name == "lm_head.weight");
    let has_norm = tensor_names.iter().any(|&name| name == "model.norm.weight");

    println!(
        "Has embed_tokens: {}, has lm_head: {}, has norm: {}",
        has_embed_tokens, has_lm_head, has_norm
    );

    let device = WgpuDevice::default();
    let model = config.build_pretrained(&device, &tensors);

    Ok(HFLlama {
        config,
        model,
        tokenizer,
    })
}

mod test {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};
    use num_traits::ToPrimitive;
    use std::io::Read;
    use std::io::Write;
    //#[test]
    fn test_load_llama_from_hf() -> Result<()> {
        let device = WgpuDevice::default();
        println!("Using device: {:?}", device);
        //let model_id = "HuggingFaceM4/tiny-random-Llama3ForCausalLM";
        let model_id = "HuggingFaceTB/SmolLM-135M";
        let revision = "main";
        println!("Loading model '{}' from revision '{}'", model_id, revision);
        let llama = load_llama_from_hf(model_id, revision, &device)
            .context("Failed to load Llama model")?;
        let HFLlama {
            config,
            model,
            tokenizer,
        } = llama;

        let prompt = "Hello there,".to_string();
        let tokens = tokenizer.encode(prompt.clone(), false).unwrap();
        let mut tokens = tokens
            .get_ids()
            .iter()
            .map(|&x| x as i32)
            .collect::<Vec<_>>();
        let mut text = String::new();
        let mut generated_tokens = 0;
        let max_tokens = 30; // Shorter for better quality

        let remaining = std::cmp::min(max_tokens, config.max_seq_len) - tokens.len();
        println!("Remaining: {}", remaining);
        println!("Initial tokens: {:?}", tokens);

        println!("\n\n\n");
        print!("{}", prompt);

        for i in 0..remaining {
            // Create tensor using llama3-burn approach but adapted for your burn version
            let token_tensor = Tensor::<LBackend, 2, Int>::from_data(
                TensorData::new(tokens.iter().map(|&t| t as i32).collect(), [1, tokens.len()]),
                &device,
            );

            let out = model.forward(token_tensor);
            let [_n_batch, n_token, _n_dict] = out.dims();

            let last_row: Tensor<LBackend, 1> =
                out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);

            // Simple greedy decoding for now - nucleus sampling is complex with burn tensors
            let token_id = last_row.argmax(0).into_scalar().to_i32().unwrap();
            tokens.push(token_id);

            let token_text = tokenizer.decode(&[token_id as u32], true).unwrap();
            print!("{token_text}");
            std::io::stdout().flush().unwrap();

            text += &token_text;
        }
        println!();
        Ok(())
    }
}

pub struct TextGenerator {
    model: Llama<LBackend>,
    config: LlamaConfig,
    device: WgpuDevice,
    eos_token: u32,
    sample_len: Option<usize>,
    channel: Option<std::sync::mpsc::Sender<String>>,
    terminators: Option<Vec<String>>,
}

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
