mod loaders;

use burn::{
    module::{Module, Param},
    nn::{self, Embedding},
    tensor::{Int, Tensor, backend::Backend},
};
use safetensors::SafeTensors;

use anyhow::{Context, Result};

use serde::Deserialize;

use crate::ai::loaders::WgpuFloatTensorReader;
use crate::ai::models::parts::{
    ResidualDecoderAttentionBlock, RmsNorm, RotaryEncoding, RotaryEncodingConfig,
};

use super::parts::attn_decoder_mask;
use super::*;
use loaders::{load_rmsnorm, load_transformer_block};

#[derive(Module, Debug)]
pub struct Llama<B: Backend> {
    pub token_embedding: nn::Embedding<B>,
    pub rotary_encoding: RotaryEncoding<B>,
    pub blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    pub norm: RmsNorm<B>,
    pub lm_head: nn::Linear<B>,
    pub mask: Tensor<B, 2>,
    // pub n_vocab: usize,
    pub max_seq_len: usize,
}

impl<B: Backend> Llama<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.max_seq_len,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.max_seq_len
        );

        let x = self.token_embedding.forward(x);

        let x = self.blocks.iter().fold(x, |acc, block| {
            block.forward(acc, &self.rotary_encoding, self.mask.clone())
        });

        self.lm_head.forward(self.norm.forward(x))
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
    pub rope_scaling: Option<RopeScaling>,
    pub rope_theta: f64,
    pub tie_word_embeddings: bool,
    pub torch_dtype: Option<String>,
    pub transformers_version: Option<String>,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Deserialize, Debug, Clone)]
pub struct RopeScaling {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[allow(dead_code)]
impl LlamaConfig {
    pub fn from_file(path: &str) -> Result<Self> {
        let file =
            std::fs::File::open(path).context(format!("Failed to open config file: {}", path))?;
        let reader = std::io::BufReader::new(file);
        let config: Self = serde_json::from_reader(reader)
            .context(format!("Failed to parse config file: {}", path))?;
        Ok(config)
    }
    pub fn build_pretrained<B: Backend>(
        &self,
        device: &B::Device,
        safetensors: &SafeTensors,
    ) -> Llama<B> {
        let mut blocks: Vec<ResidualDecoderAttentionBlock<B>> =
            Vec::with_capacity(self.num_hidden_layers);
        for i in 0..self.num_hidden_layers {
            println!(
                "loading transformer layer {} of {}",
                i, self.num_hidden_layers
            );
            let transformer_block = load_transformer_block::<B>(
                safetensors,
                self,
                &format!("model.layers.{}", i),
                device,
            );
            blocks.push(transformer_block);
        }

        let embed_tokens = safetensors.read_full_float::<B, 2>("model.embed_tokens.weight", device);

        let [_n_vacab, n_state] = embed_tokens.dims();
        let n_heads = self.num_attention_heads;
        let _n_kv_heads = self.num_key_value_heads;
        let head_dim = n_state / n_heads;
        let token_embedding = Embedding {
            weight: Param::from_tensor(embed_tokens),
        };
        let rotary_encoding =
            RotaryEncodingConfig::new(self.max_seq_len, head_dim, self.rope_theta).init(device);
        let norm = load_rmsnorm::<B>(safetensors, self, "model.norm", device);
        // sometimes lm_head is also called "output"
        let lm_head_key = if safetensors.names().contains(&"lm_head.weight") {
            "lm_head.weight"
        } else {
            "model.embed_tokens.weight"
        };
        let lm_head = safetensors.read_full_float::<B, 2>(lm_head_key, device);
        let lm_head = nn::Linear {
            weight: Param::from_tensor(lm_head.transpose()),
            bias: None,
        };
        let mask = attn_decoder_mask::<B>(self.max_seq_len, device);
        let _norm_eps = norm.eps;

        Llama {
            token_embedding,
            rotary_encoding,
            blocks,
            norm,
            lm_head,
            mask,
            max_seq_len: self.max_seq_len,
        }
    }
}
