mod cache;

use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    tensor::{Bool, Device, Int, Tensor, activation::softmax, backend::Backend},
};

use super::custom_rope::CustomRotaryEncoding;
use super::parts::{Mlp, MlpConfig};

use cache::AutoregressiveCache;
use md5;

// Helper function to compute tensor hash for debugging - concatenate all elements as strings and MD5 hash
pub fn tensor_hash<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> u64 {
    let data: Vec<f32> = tensor.clone().into_data().to_vec().unwrap();

    // Round each element to 3 decimal places and concatenate to string
    let tensor_string: String = data
        .iter()
        .map(|&val| format!("{:.3}", val))
        .collect::<Vec<String>>()
        .join("");

    // MD5 hash the concatenated string
    let hash_bytes = md5::compute(tensor_string.as_bytes());

    // Convert first 8 bytes to u64 for display (little-endian to match Python)
    u64::from_le_bytes([
        hash_bytes[0],
        hash_bytes[1],
        hash_bytes[2],
        hash_bytes[3],
        hash_bytes[4],
        hash_bytes[5],
        hash_bytes[6],
        hash_bytes[7],
    ])
}

/// Configuration to create a Llama [decoder-only transformer](Transformer).
#[derive(Config)]
pub struct TransformerConfig {
    /// The size of the vocabulary.
    pub vocab_size: usize,
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// Maximum token sequence length.
    #[config(default = "512")]
    pub max_seq_len: usize,
    /// RMSNorm epsilon.
    #[config(default = "1e-5")]
    pub norm_eps: f64,
    pub tie_word_embeddings: bool,
}

impl TransformerConfig {
    /// Initialize a new [decoder-only transformer](Transformer).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Transformer<B> {
        let tok_embeddings = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);
        let layers = (0..self.n_layers)
            .map(|_| {
                TransformerBlockConfig::new(
                    self.n_layers,
                    self.d_model,
                    self.hidden_size,
                    self.n_heads,
                    self.n_kv_heads,
                    self.norm_eps,
                )
                .init(device)
            })
            .collect::<Vec<_>>();
        let norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        let output = if self.tie_word_embeddings {
            None
        } else {
            Some(
                LinearConfig::new(self.d_model, self.vocab_size)
                    .with_bias(false)
                    .init(device),
            )
        };

        Transformer {
            tok_embeddings,
            layers,
            norm,
            output,
        }
    }
}

/// Llama decoder-only transformer.
#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    tok_embeddings: Embedding<B>,
    layers: Vec<TransformerBlock<B>>,
    norm: RmsNorm<B>,
    output: Option<Linear<B>>,
    // NOTE: Starting with Llama 3.2, the weights of the output layer are tied with the embedding
}

impl<B: Backend> Transformer<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 2, Int>,
        _pos: usize,
        cache: &mut Vec<KeyValueCache<B>>,
        rope: &CustomRotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        self.forward_with_debug(input, _pos, cache, rope, false)
    }

    pub fn forward_with_debug(
        &self,
        input: Tensor<B, 2, Int>,
        _pos: usize,
        cache: &mut Vec<KeyValueCache<B>>,
        rope: &CustomRotaryEncoding<B>,
        debug: bool,
    ) -> Tensor<B, 3> {
        if debug {
            println!("=== TRANSFORMER DEBUG ===");
        }

        let mut h = self.tok_embeddings.forward(input);
        if debug {
            println!(
                "After embeddings: hash={:016x}, shape={:?}",
                tensor_hash(&h),
                h.dims()
            );
        }

        for (i, (layer, c)) in self.layers.iter().zip(cache.into_iter()).enumerate() {
            if debug && i < 1 {
                let h_before = h.clone();
                h = layer.forward_with_debug(h, c, rope, debug);
                if debug {
                    println!(
                        "After layer {}: hash={:016x}, shape={:?}",
                        i,
                        tensor_hash(&h),
                        h.dims()
                    );

                    // Check if layer output is very different from input
                    let h_diff = h.clone() - h_before;
                    println!("  Layer {} diff hash: {:016x}", i, tensor_hash(&h_diff));
                }
            } else {
                h = layer.forward(h, c, rope);
            }
        }

        let h = self.norm.forward(h);
        if debug {
            println!(
                "After final norm: hash={:016x}, shape={:?}",
                tensor_hash(&h),
                h.dims()
            );
        }

        let final_output = if let Some(output) = &self.output {
            let result = output.forward(h);
            if debug {
                println!(
                    "After output projection: hash={:016x}, shape={:?}",
                    tensor_hash(&result),
                    result.dims()
                );
            }
            result
        } else {
            let embedding_weights = self.tok_embeddings.weight.val();
            if debug {
                println!(
                    "Embedding weights: hash={:016x}, shape={:?}",
                    tensor_hash(&embedding_weights),
                    embedding_weights.dims()
                );
            }

            let output_weights = embedding_weights.transpose().unsqueeze::<3>();
            if debug {
                println!(
                    "Output weights (transposed): hash={:016x}, shape={:?}",
                    tensor_hash(&output_weights),
                    output_weights.dims()
                );
            }

            let result = h.matmul(output_weights);
            if debug {
                println!(
                    "After tied embedding matmul: hash={:016x}, shape={:?}",
                    tensor_hash(&result),
                    result.dims()
                );
            }
            result
        };

        final_output
    }
}

/// Configuration to create a [decoder-only transformer block](TransformerBlock).
#[derive(Config)]
pub struct TransformerBlockConfig {
    /// The number of transformer blocks.
    pub n_layers: usize,
    /// The size of the model.
    pub d_model: usize,
    /// The size of the feed-forward hidden inner features.
    pub hidden_size: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f64,
}

impl TransformerBlockConfig {
    /// Initialize a new [decoder-only transformer block](TransformerBlock).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> TransformerBlock<B> {
        let attention =
            MultiHeadAttentionConfig::new(self.d_model, self.n_heads, self.n_kv_heads).init(device);
        let feed_forward =
            FeedForwardConfig::new(self.d_model, self.hidden_size, false).init(device);
        let attention_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);
        let ffn_norm = RmsNormConfig::new(self.d_model)
            .with_epsilon(self.norm_eps)
            .init(device);

        TransformerBlock {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        }
    }
}

/// Decoder-only transformer block.
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention.
    attention: MultiHeadAttention<B>,
    /// Feed-forward transformation.
    feed_forward: FeedForward<B>,
    /// Attention pre-normalization.
    attention_norm: RmsNorm<B>,
    /// Feed-forward pre-normalization.
    ffn_norm: RmsNorm<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &CustomRotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        self.forward_with_debug(input, cache, rope, false)
    }

    pub fn forward_with_debug(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &CustomRotaryEncoding<B>,
        debug: bool,
    ) -> Tensor<B, 3> {
        if debug {
            println!("    Block input: hash={:016x}", tensor_hash(&input));
        }

        // Optimize: Reduce unnecessary tensor cloning in residual connections
        let normalized_input = self.attention_norm.forward(input.clone());
        if debug {
            println!(
                "    After attn norm: hash={:016x}",
                tensor_hash(&normalized_input)
            );
        }

        let attention_output =
            self.attention
                .forward_with_debug(normalized_input, cache, rope, debug);
        if debug {
            println!(
                "    After attention: hash={:016x}",
                tensor_hash(&attention_output)
            );
        }

        let h = input + attention_output;
        if debug {
            println!("    After attn residual: hash={:016x}", tensor_hash(&h));
        }

        let normalized_h = self.ffn_norm.forward(h.clone());
        if debug {
            println!(
                "    After ffn norm: hash={:016x}",
                tensor_hash(&normalized_h)
            );
        }

        let ffn_output = self.feed_forward.forward_with_debug(normalized_h, debug);
        if debug {
            println!("    After ffn: hash={:016x}", tensor_hash(&ffn_output));
        }

        let final_output = h + ffn_output;
        if debug {
            println!(
                "    After ffn residual: hash={:016x}",
                tensor_hash(&final_output)
            );
        }

        final_output
    }
}

/// Configuration to create a [feed-forward transformation network](FeedForward).
#[derive(Config)]
pub struct FeedForwardConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub hidden_size: usize,
    /// Whether to use bias in the linear layers.
    pub bias: bool,
}

impl FeedForwardConfig {
    /// Initialize a new [feed-forward transformation network](FeedForward).
    pub fn init<B: Backend>(&self, device: &Device<B>) -> FeedForward<B> {
        let mlp = MlpConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.d_model,
            bias: self.bias,
        }
        .init(device);

        FeedForward { mlp }
    }
}

/// Feed-forward transformation network using LLaMA MLP architecture.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    /// LLaMA MLP with gate_proj, up_proj, and down_proj.
    mlp: Mlp<B>,
}

impl<B: Backend> FeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.forward_with_debug(input, false)
    }

    pub fn forward_with_debug(&self, input: Tensor<B, 3>, debug: bool) -> Tensor<B, 3> {
        // Delegate to the MLP's forward_with_debug method
        self.mlp.forward_with_debug(input, debug)
    }
}

/// Key-value cache for autoregressive models.
pub struct KeyValueCache<B: Backend> {
    key: AutoregressiveCache<B>,
    value: AutoregressiveCache<B>,
}

impl<B: Backend> KeyValueCache<B> {
    /// Create a new [key-value cache](KeyValueCache).
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            key: AutoregressiveCache::new(max_batch_size, num_heads, max_seq_len, d_model, device),
            value: AutoregressiveCache::new(
                max_batch_size,
                num_heads,
                max_seq_len,
                d_model,
                device,
            ),
        }
    }

    /// Computes the complete keys and values.
    pub fn forward(
        &mut self,
        key: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let k = self.key.forward(key);
        let v = self.value.forward(value);
        (k, v)
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        // We can assume key and value have the same length
        self.key.len()
    }

    /// Reset key-value cache.
    /// Use between different contexts (i.e., for each new prompt).
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }
}

/// Configuration to create a [multi-head attention](MultiHeadAttention) module.
#[derive(Config)]
pub struct MultiHeadAttentionConfig {
    /// The size of the model.
    pub d_model: usize,
    /// The number of heads.
    pub n_heads: usize,
    /// The number of key-value heads.
    pub n_kv_heads: usize,
}

impl MultiHeadAttentionConfig {
    /// Initialize a new [multi-head attention](MultiHeadAttention) module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MultiHeadAttention<B> {
        let head_dim = self.d_model / self.n_heads;

        let wq = LinearConfig::new(self.d_model, self.n_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wk = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wv = LinearConfig::new(self.d_model, self.n_kv_heads * head_dim)
            .with_bias(false)
            .init(device);
        let wo = LinearConfig::new(self.n_heads * head_dim, self.d_model)
            .with_bias(false)
            .init(device);

        MultiHeadAttention {
            wq,
            wk,
            wv,
            wo,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadAttention<B: Backend> {
    /// Query projection.
    wq: Linear<B>,
    /// Key projection.
    wk: Linear<B>,
    /// Value projection.
    wv: Linear<B>,
    /// Output projection.
    wo: Linear<B>,

    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl<B: Backend> MultiHeadAttention<B> {
    /// Applies the forward pass on the input tensors.
    ///
    /// # Shapes
    ///
    /// - query: `[batch_size, seq_length_1, d_model]`
    /// - key: `[batch_size, seq_length_2, d_model]`
    /// - value: `[batch_size, seq_length_2, d_model]`
    /// - output: `[batch_size, seq_length_1, d_model]`
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &CustomRotaryEncoding<B>,
    ) -> Tensor<B, 3> {
        self.forward_with_debug(input, cache, rope, false)
    }

    pub fn forward_with_debug(
        &self,
        input: Tensor<B, 3>,
        cache: &mut KeyValueCache<B>,
        rope: &CustomRotaryEncoding<B>,
        debug: bool,
    ) -> Tensor<B, 3> {
        if debug {
            println!("      Attn input: hash={:016x}", tensor_hash(&input));
        }

        let device = input.device();
        let [batch_size, seq_len, hidden_size] = input.dims();

        // Optimize: Reduce tensor cloning overhead
        let q = self.wq.forward(input.clone());
        let k = self.wk.forward(input.clone());
        let v = self.wv.forward(input);

        if debug {
            println!("      After Q proj: hash={:016x}", tensor_hash(&q));
            println!("      After K proj: hash={:016x}", tensor_hash(&k));
            println!("      After V proj: hash={:016x}", tensor_hash(&v));
        }

        // [batch_size, num_heads, seq_len, head_dim]
        let q = q
            .reshape([batch_size, seq_len, self.n_heads, self.head_dim])
            .swap_dims(1, 2);
        let k = k
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = v
            .reshape([batch_size, seq_len, self.n_kv_heads, self.head_dim])
            .swap_dims(1, 2);

        if debug {
            println!(
                "      After reshape/transpose Q: hash={:016x}",
                tensor_hash(&q)
            );
            println!(
                "      After reshape/transpose K: hash={:016x}",
                tensor_hash(&k)
            );
            println!(
                "      After reshape/transpose V: hash={:016x}",
                tensor_hash(&v)
            );
        }

        // Apply RoPE using our custom implementation that matches HuggingFace exactly
        let start_position = cache.len();
        let (q, k) = rope.apply_rope(q, k, start_position);

        if debug {
            println!("      After RoPE Q: hash={:016x}", tensor_hash(&q));
            println!("      After RoPE K: hash={:016x}", tensor_hash(&k));
        }

        // Key-value caching
        let (k, v) = cache.forward(k, v);

        if debug {
            println!(
                "      After KV cache: K hash={:016x}, V hash={:016x}",
                tensor_hash(&k),
                tensor_hash(&v)
            );
        }

        // Repeat key/value heads if num_kv_heads < num_heads
        let k = self.repeat_kv(k);
        let v = self.repeat_kv(v);

        if debug {
            println!(
                "      After repeat_kv: K hash={:016x}, V hash={:016x}",
                tensor_hash(&k),
                tensor_hash(&v)
            );
        }

        // Attention scores - optimize by pre-computing scale
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut scores = q.matmul(k.swap_dims(2, 3)) * scale;

        if debug {
            println!(
                "      After attention scores: hash={:016x}",
                tensor_hash(&scores)
            );
        }

        // Apply causal masking - revert to original working approach for now
        if seq_len > 1 {
            let cache_seq_len = cache.len();
            let mask = Tensor::<B, 2, Bool>::tril_mask(
                [seq_len, cache_seq_len],
                (cache_seq_len - seq_len) as i64,
                &device,
            );
            scores = scores.mask_fill(mask.unsqueeze::<4>(), f32::NEG_INFINITY);
            if debug {
                println!("      After masking: hash={:016x}", tensor_hash(&scores));
            }
        }

        // Apply softmax with explicit float32 precision like HuggingFace
        // HuggingFace: nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        let scores = softmax(scores, 3);
        if debug {
            println!("      After softmax: hash={:016x}", tensor_hash(&scores));
        }

        // Output [batch_size, num_heads, seq_len, head_dim]
        let output = scores.matmul(v);
        if debug {
            println!("      After scores*V: hash={:016x}", tensor_hash(&output));
        }

        let output = output
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, hidden_size]);
        if debug {
            println!("      After reshape: hash={:016x}", tensor_hash(&output));
        }

        let final_output = self.wo.forward(output);
        if debug {
            println!(
                "      After output proj: hash={:016x}",
                tensor_hash(&final_output)
            );
        }

        final_output
    }

    /// Repeats a key or value tensor for grouped query attention.
    fn repeat_kv(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let n_rep = self.n_heads / self.n_kv_heads;
        if n_rep == 1 {
            x
        } else {
            let [batch_size, num_kv_heads, seq_len, head_dim] = x.dims();

            x.unsqueeze_dim::<5>(2)
                .expand([batch_size, num_kv_heads, n_rep, seq_len, head_dim])
                .reshape([batch_size, num_kv_heads * n_rep, seq_len, head_dim])
        }
    }
}

#[cfg(test)]
#[cfg(any(feature = "cuda", feature = "tch-gpu"))]
mod tests {
    use super::*;
    use crate::tests::*;

    use burn::tensor::TensorData;

    #[test]
    fn test_rms_norm() {
        let device = Default::default();

        let rms = RmsNormConfig::new(4).with_epsilon(1e-5).init(&device);
        let input = TestTensor::<3>::from([[
            [0.0025997162, 0.0030002594, -0.006000519, 0.006000519],
            [0.0010004044, 0.00080013275, 0.0015001297, -0.01600647],
        ]]);

        let output = rms.forward(input);
        let expected = TensorData::from([[
            [0.45996094, 0.5307617, -1.0615234, 1.0615234],
            [0.11553955, 0.09240723, 0.17321777, -1.8486328],
        ]]);

        output.into_data().assert_approx_eq(&expected, 3);
    }
}
