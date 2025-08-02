use burn::tensor::{backend::Backend, Device, Tensor};

pub(crate) struct AutoregressiveCache<B: Backend> {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: Tensor<B, 4>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: usize,
}

/// Optimized Key-Value cache for inference with separate K and V tensors
pub struct KeyValueCache<B: Backend> {
    k_cache: Tensor<B, 4>, // [batch, num_heads, max_seq, head_dim]
    v_cache: Tensor<B, 4>, // [batch, num_heads, max_seq, head_dim]  
    cur_seq_len: usize,
    max_seq_len: usize,
}

impl<B: Backend> KeyValueCache<B> {
    /// Create new optimized KV cache
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        head_dim: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            k_cache: Tensor::zeros([batch_size, num_heads, max_seq_len, head_dim], device),
            v_cache: Tensor::zeros([batch_size, num_heads, max_seq_len, head_dim], device),
            cur_seq_len: 0,
            max_seq_len,
        }
    }

    /// Update cache with new K,V and return the full cached tensors
    pub fn update(
        &mut self,
        k: Tensor<B, 4>, // [batch, num_heads, seq_len, head_dim]
        v: Tensor<B, 4>, // [batch, num_heads, seq_len, head_dim]
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch_size, num_heads, seq_len, head_dim] = k.dims();
        let new_seq_len = self.cur_seq_len + seq_len;
        
        // Update K cache
        self.k_cache = self.k_cache.clone().slice_assign(
            [0..batch_size, 0..num_heads, self.cur_seq_len..new_seq_len, 0..head_dim],
            k,
        );
        
        // Update V cache  
        self.v_cache = self.v_cache.clone().slice_assign(
            [0..batch_size, 0..num_heads, self.cur_seq_len..new_seq_len, 0..head_dim],
            v,
        );
        
        self.cur_seq_len = new_seq_len;
        
        // Return sliced cache up to current length
        let k_out = self.k_cache.clone().slice([
            0..batch_size, 
            0..num_heads, 
            0..self.cur_seq_len, 
            0..head_dim
        ]);
        let v_out = self.v_cache.clone().slice([
            0..batch_size, 
            0..num_heads, 
            0..self.cur_seq_len, 
            0..head_dim
        ]);
        
        (k_out, v_out)
    }
    
    /// Get current sequence length
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
    
    /// Reset the cache
    pub fn reset(&mut self) {
        self.cur_seq_len = 0;
        // No need to zero out tensors, just reset position
    }
}

impl<B: Backend> AutoregressiveCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            cache: Tensor::empty([max_batch_size, num_heads, max_seq_len, d_model], device),
            max_seq_len,
            cur_seq_len: 0,
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len = 0;
    }

    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let mut new_seq_len = self.cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            self.cur_seq_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                0..batch_size,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model],
                prev_slice,
            );
            new_seq_len = self.max_seq_len;
        }

        self.cache = self.cache.clone().slice_assign(
            [
                0..batch_size,
                0..num_heads,
                self.cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor,
        );

        self.cur_seq_len += seq_len;

        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..self.cur_seq_len, 0..d_model])
    }

    /// Returns the cached sequence length.
    pub fn len(&self) -> usize {
        self.cur_seq_len
    }
}
