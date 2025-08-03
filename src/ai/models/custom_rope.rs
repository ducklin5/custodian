use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Device, Tensor},
};

/// Configuration for custom rotary positional encoding to match HuggingFace implementation
#[derive(Config, Debug)]
pub struct CustomRotaryEncodingConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Dimension of each head
    pub head_dim: usize,
    /// RoPE theta parameter
    #[config(default = "10000.0")]
    pub theta: f32,
}

impl CustomRotaryEncodingConfig {
    /// Initialize custom rotary encoding
    pub fn init<B: Backend>(&self, device: &Device<B>) -> CustomRotaryEncoding<B> {
        CustomRotaryEncoding::new(self.max_seq_len, self.head_dim, self.theta, device)
    }
}

/// Custom rotary positional encoding that exactly matches HuggingFace implementation
#[derive(Module, Debug)]
pub struct CustomRotaryEncoding<B: Backend> {
    cos_cache: Tensor<B, 2>,
    sin_cache: Tensor<B, 2>,
    max_seq_len: usize,
    head_dim: usize,
}

impl<B: Backend> CustomRotaryEncoding<B> {
    /// Create a new custom rotary encoding
    pub fn new(max_seq_len: usize, head_dim: usize, theta: f32, device: &Device<B>) -> Self {
        let (cos_cache, sin_cache) = Self::precompute_freqs_cis(head_dim, max_seq_len, theta, device);
        
        Self {
            cos_cache,
            sin_cache,
            max_seq_len,
            head_dim,
        }
    }
    
    /// Precompute cos and sin values for all possible positions
    fn precompute_freqs_cis(
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
        device: &Device<B>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Create frequency tensor: 1.0 / (theta^(2i/head_dim)) for i in [0, head_dim/2)
        let half_dim = head_dim / 2;
        let freqs_data: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(2.0 * (i as f32) / (head_dim as f32)))
            .collect();
        
        let freqs = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(freqs_data, [half_dim]), 
            device
        );
        
        // Create position tensor: [0, 1, 2, ..., max_seq_len-1]
        let positions_data: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::<B, 1>::from_data(
            burn::tensor::TensorData::new(positions_data, [max_seq_len]), 
            device
        );
        
        // Compute outer product: positions[i] * freqs[j] for all i, j
        let positions_expanded = positions.unsqueeze_dim::<2>(1); // [max_seq_len, 1]
        let freqs_expanded = freqs.unsqueeze_dim::<2>(0); // [1, half_dim]
        let angles = positions_expanded * freqs_expanded; // [max_seq_len, half_dim]
        
        // Compute cos and sin
        let cos_half = angles.clone().cos();
        let sin_half = angles.sin();
        
        (cos_half, sin_half)
    }
    
    /// Apply rotary encoding to Q and K tensors
    /// 
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_kv_heads, seq_len, head_dim]  
    /// * `start_position` - Starting position in the sequence (for KV caching)
    /// 
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn apply_rope(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        start_position: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch_size, _, seq_len, head_dim] = q.dims();
        
        // Create position IDs: [start_position, start_position+1, ..., start_position+seq_len-1]
        let position_ids: Vec<usize> = (start_position..start_position + seq_len).collect();
        
        // Extract cos and sin values for these positions
        let cos_values = self.extract_cos_sin_for_positions(&position_ids, head_dim);
        let sin_values = self.extract_cos_sin_for_positions_sin(&position_ids, head_dim);
        
        // Apply rotary encoding
        let q_rot = self.rotate_tensor(q, &cos_values, &sin_values);
        let k_rot = self.rotate_tensor(k, &cos_values, &sin_values);
        
        (q_rot, k_rot)
    }
    
    /// Extract cos values for given positions and expand to full head dimension
    fn extract_cos_sin_for_positions(&self, position_ids: &[usize], head_dim: usize) -> Tensor<B, 2> {
        let seq_len = position_ids.len();
        let half_dim = head_dim / 2;
        
        // Create indices tensor for gathering cos values
        let indices: Vec<usize> = position_ids.iter()
            .map(|&pos| pos.min(self.max_seq_len - 1))
            .collect();
        
        // Extract cos values by creating a new tensor with the selected rows
        let cos_cache_data: Vec<f32> = self.cos_cache.clone().into_data().to_vec().unwrap();
        let cos_cache_shape = self.cos_cache.dims();
        
        let mut cos_data: Vec<f32> = Vec::with_capacity(seq_len * half_dim);
        for &pos in &indices {
            let start_idx = pos * cos_cache_shape[1];
            let end_idx = start_idx + half_dim;
            cos_data.extend(&cos_cache_data[start_idx..end_idx]);
        }
        
        let cos_half = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(cos_data, [seq_len, half_dim]), 
            &self.cos_cache.device()
        );
        
        // Repeat to full head dimension: [seq_len, half_dim] -> [seq_len, head_dim]
        Tensor::cat(vec![cos_half.clone(), cos_half], 1)
    }
    
    /// Extract sin values for given positions and expand to full head dimension
    fn extract_cos_sin_for_positions_sin(&self, position_ids: &[usize], head_dim: usize) -> Tensor<B, 2> {
        let seq_len = position_ids.len();
        let half_dim = head_dim / 2;
        
        // Create indices tensor for gathering sin values
        let indices: Vec<usize> = position_ids.iter()
            .map(|&pos| pos.min(self.max_seq_len - 1))
            .collect();
        
        // Extract sin values by creating a new tensor with the selected rows
        let sin_cache_data: Vec<f32> = self.sin_cache.clone().into_data().to_vec().unwrap();
        let sin_cache_shape = self.sin_cache.dims();
        
        let mut sin_data: Vec<f32> = Vec::with_capacity(seq_len * half_dim);
        for &pos in &indices {
            let start_idx = pos * sin_cache_shape[1];
            let end_idx = start_idx + half_dim;
            sin_data.extend(&sin_cache_data[start_idx..end_idx]);
        }
        
        let sin_half = Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(sin_data, [seq_len, half_dim]), 
            &self.sin_cache.device()
        );
        
        // Repeat to full head dimension: [seq_len, half_dim] -> [seq_len, head_dim]
        Tensor::cat(vec![sin_half.clone(), sin_half], 1)
    }
    
    /// Apply rotation to a tensor using cos and sin values
    /// 
    /// # Arguments  
    /// * `x` - Input tensor [batch, num_heads, seq_len, head_dim]
    /// * `cos_values` - Cos values [seq_len, head_dim]
    /// * `sin_values` - Sin values [seq_len, head_dim]
    fn rotate_tensor(
        &self,
        x: Tensor<B, 4>,
        cos_values: &Tensor<B, 2>,
        sin_values: &Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, head_dim] = x.dims();
        
        // Expand cos and sin for broadcasting: [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        let cos_expanded = cos_values.clone().unsqueeze_dim::<4>(0).unsqueeze_dim::<4>(0);
        let sin_expanded = sin_values.clone().unsqueeze_dim::<4>(0).unsqueeze_dim::<4>(0);
        
        // Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        let x_rotated_half = self.rotate_half(x.clone());
        
        x * cos_expanded + x_rotated_half * sin_expanded
    }
    
    /// Rotate half of the tensor dimensions
    /// This swaps the first and second half and negates the first half: [-x2, x1]
    fn rotate_half(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, _num_heads, _seq_len, head_dim] = x.dims();
        let half_dim = head_dim / 2;
        
        // Split tensor into two halves
        let x1 = x.clone().slice([0..x.dims()[0], 0..x.dims()[1], 0..x.dims()[2], 0..half_dim]);
        let x2 = x.clone().slice([0..x.dims()[0], 0..x.dims()[1], 0..x.dims()[2], half_dim..head_dim]);
        
        // Concatenate [-x2, x1]
        Tensor::cat(vec![-x2, x1], 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    
    type TestBackend = Wgpu;
    
    #[test]
    fn test_custom_rotary_encoding() {
        let device = Default::default();
        let max_seq_len = 2048;
        let head_dim = 64;
        let theta = 10000.0;
        
        let rope = CustomRotaryEncoding::new(max_seq_len, head_dim, theta, &device);
        
        // Test with dummy Q and K tensors
        let batch_size = 1;
        let num_q_heads = 9;
        let num_kv_heads = 3;
        let seq_len = 3;
        
        let q_data: Vec<f32> = (0..batch_size * num_q_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();
        let k_data: Vec<f32> = (0..batch_size * num_kv_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.01 + 100.0)
            .collect();
            
        let q = Tensor::<TestBackend, 4>::from_data(
            burn::tensor::TensorData::new(q_data, [batch_size, num_q_heads, seq_len, head_dim]), 
            &device
        );
        let k = Tensor::<TestBackend, 4>::from_data(
            burn::tensor::TensorData::new(k_data, [batch_size, num_kv_heads, seq_len, head_dim]), 
            &device
        );
        
        // Apply rotation starting from position 0
        let (q_rot, k_rot) = rope.apply_rope(q, k, 0);
        
        // Check that shapes are preserved
        assert_eq!(q_rot.dims(), [batch_size, num_q_heads, seq_len, head_dim]);
        assert_eq!(k_rot.dims(), [batch_size, num_kv_heads, seq_len, head_dim]);
        
        println!("Custom RoPE test passed!");
    }
}