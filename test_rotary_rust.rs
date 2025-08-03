use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    module::Module,
    nn::{RotaryEncoding, RotaryEncodingConfig},
    tensor::{Int, Tensor, TensorData},
};
use tokenizers::Tokenizer as GTokenizer;
use anyhow::Result;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::safetensors::{LoadArgs, SafetensorsFileRecorder};
use md5;

type LBackend = Wgpu;

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

#[cfg(test)]
mod test {
    use super::*;
    
    #[test]
    fn test_rotary_encoding_comparison() -> Result<()> {
        println!("=== RUST ROTARY ENCODING DEBUG ===");
        
        let device = WgpuDevice::default();
        
        // Load tokenizer to match Python exactly
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            "HuggingFaceTB/SmolLM-135M".to_string(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ));
        
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer = GTokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        
        // Tokenize the same prompt
        let prompt = "The quick brown";
        let tokens = tokenizer.encode(prompt, true).unwrap();
        let tokens: Vec<i32> = tokens.get_ids().iter().map(|&x| x as i32).collect();
        
        println!("Input tokens: {:?}", tokens);
        
        // Model configuration (SmolLM-135M)
        let vocab_size = 49152;
        let hidden_size = 576;  
        let num_attention_heads = 9;
        let num_key_value_heads = 3;
        let head_dim = hidden_size / num_attention_heads;  // 64
        let rope_theta = 10000.0;
        let max_seq_len = 2048;
        
        println!("Head dim: {}", head_dim);
        println!("Rope theta: {}", rope_theta);
        
        // Create rotary encoding to match Python
        let rope_config = RotaryEncodingConfig::new(max_seq_len, head_dim)
            .with_theta(rope_theta as f32);
        let rope = rope_config.init(&device);
        
        // Create dummy embeddings and projections to test rotary encoding
        // We'll use some simple test data that we can manually verify
        let batch_size = 1;
        let seq_len = tokens.len();
        
        // Create test Q and K tensors with known values
        // Let's create ascending values to make debugging easier
        let q_data: Vec<f32> = (0..batch_size * num_attention_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let k_data: Vec<f32> = (0..batch_size * num_key_value_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.01 + 100.0)  // Offset K values
            .collect();
            
        let q = Tensor::<LBackend, 4>::from_data(
            TensorData::new(q_data, [batch_size, num_attention_heads, seq_len, head_dim]),
            &device
        );
        let k = Tensor::<LBackend, 4>::from_data(
            TensorData::new(k_data, [batch_size, num_key_value_heads, seq_len, head_dim]),
            &device
        );
        
        println!("Q before RoPE: hash={:016x}, shape={:?}", tensor_hash(&q), q.dims());
        println!("K before RoPE: hash={:016x}, shape={:?}", tensor_hash(&k), k.dims());
        
        // Apply rotary encoding at position 0 (start of sequence)
        let start_position = 0;
        let q_rope = rope.apply(q.clone(), start_position);
        let k_rope = rope.apply(k.clone(), start_position);
        
        println!("Q after RoPE: hash={:016x}, shape={:?}", tensor_hash(&q_rope), q_rope.dims());
        println!("K after RoPE: hash={:016x}, shape={:?}", tensor_hash(&k_rope), k_rope.dims());
        
        // Extract and print some values for manual verification
        let q_vals: Vec<f32> = q.clone().into_data().to_vec().unwrap();
        let q_rope_vals: Vec<f32> = q_rope.clone().into_data().to_vec().unwrap();
        let k_vals: Vec<f32> = k.clone().into_data().to_vec().unwrap();
        let k_rope_vals: Vec<f32> = k_rope.clone().into_data().to_vec().unwrap();
        
        println!("\nFirst few Q values before RoPE: {:?}", &q_vals[0..5]);
        println!("First few Q values after RoPE: {:?}", &q_rope_vals[0..5]);
        println!("First few K values before RoPE: {:?}", &k_vals[0..5]);
        println!("First few K values after RoPE: {:?}", &k_rope_vals[0..5]);
        
        // Test with position > 0 to see rotation effect
        let start_position_1 = 1;
        let q_rope_1 = rope.apply(q.clone(), start_position_1);
        let k_rope_1 = rope.apply(k.clone(), start_position_1);
        
        println!("\n=== POSITION 1 TEST ===");
        println!("Q after RoPE (pos=1): hash={:016x}", tensor_hash(&q_rope_1));
        println!("K after RoPE (pos=1): hash={:016x}", tensor_hash(&k_rope_1));
        
        let q_rope_1_vals: Vec<f32> = q_rope_1.clone().into_data().to_vec().unwrap();
        println!("First few Q values after RoPE (pos=1): {:?}", &q_rope_1_vals[0..5]);
        
        // Manual rotary computation verification
        // Extract first token, first head data
        let offset = 0; // batch=0, head=0, token=0
        let q_first_token = &q_vals[offset..offset+head_dim];
        let q_rope_first_token = &q_rope_vals[offset..offset+head_dim];
        
        println!("\n=== MANUAL ROTARY VERIFICATION ===");
        println!("Original Q (first token, first head): {:?}", &q_first_token[0..8]);
        println!("RoPE Q (first token, first head): {:?}", &q_rope_first_token[0..8]);
        
        // Compute expected rotary encoding manually
        // For theta=10000, head_dim=64, position=0
        // cos(m * theta^(-2i/d)) where m=0, so cos(0) = 1, sin(0) = 0
        // So for position 0, RoPE should be identity (no rotation)
        let diff: Vec<f32> = q_first_token.iter()
            .zip(q_rope_first_token.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();
        let max_diff = diff.iter().fold(0.0f32, |a, &b| a.max(b));
        println!("Max difference for pos=0 (should be ~0): {:.10}", max_diff);
        
        Ok(())
    }
}

fn main() {
    // Run the test
    if let Err(e) = test::test_rotary_encoding_comparison() {
        eprintln!("Test failed: {}", e);
    }
}