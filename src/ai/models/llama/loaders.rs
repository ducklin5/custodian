use burn::{
    module::Param,
    nn::{self, Embedding},
    tensor::backend::Backend,
};
use super::{LlamaConfig, Llama};
use crate::ai::loaders::WgpuFloatTensorReader;
use crate::ai::models::parts::{Mlp, MultiHeadSelfAttention, ResidualDecoderAttentionBlock, RmsNorm};
use safetensors::SafeTensors;

use super::parts::*;

pub fn load_mlp<B: Backend>(
    safetensors: &SafeTensors,
    _config: &LlamaConfig,
    path: &str,
    device: &B::Device,
) -> Mlp<B>
{
    let gate_proj =
        safetensors.read_full_float::<B, 2>(&format!("{}.gate_proj.weight", path), device);
    let gate_proj = nn::Linear {
        weight: Param::from_tensor(gate_proj.transpose()),
        bias: None,
    };

    let up_proj = safetensors.read_full_float::<B, 2>(&format!("{}.up_proj.weight", path), device);
    let up_proj = nn::Linear {
        weight: Param::from_tensor(up_proj.transpose()),
        bias: None,
    };

    let down_proj =
        safetensors.read_full_float::<B, 2>(&format!("{}.down_proj.weight", path), device);
    let down_proj = nn::Linear {
        weight: Param::from_tensor(down_proj.transpose()),
        bias: None,
    };

    Mlp {
        gate_proj,
        up_proj,
        down_proj,
    }
}

pub fn load_rmsnorm<B: Backend>(
    safetensors: &SafeTensors,
    config: &LlamaConfig,
    path: &str,
    device: &B::Device,
) -> RmsNorm<B>
{
    let weight = safetensors.read_full_float::<B, 1>(&format!("{}.weight", path), device);
    let eps = config.rms_norm_eps;
    RmsNorm {
        weight: Param::from_tensor(weight),
        eps,
    }
}

pub fn load_attention<B: Backend>(
    safetensors: &SafeTensors,
    config: &LlamaConfig,
    path: &str,
    device: &B::Device,
) -> MultiHeadSelfAttention<B>
{
    let k_proj = safetensors.read_full_float::<B, 2>(&format!("{}.k_proj.weight", path), device);
    let q_proj = safetensors.read_full_float::<B, 2>(&format!("{}.q_proj.weight", path), device);
    let v_proj = safetensors.read_full_float::<B, 2>(&format!("{}.v_proj.weight", path), device);
    let o_proj = safetensors.read_full_float::<B, 2>(&format!("{}.o_proj.weight", path), device);

    let n_heads = config.num_attention_heads;
    let n_kv_heads = config.num_key_value_heads;
    MultiHeadSelfAttention {
        n_heads,
        n_kv_heads,
        query: nn::Linear {
            weight: Param::from_tensor(q_proj.transpose()),
            bias: None,
        },
        key: nn::Linear {
            weight: Param::from_tensor(k_proj.transpose()),
            bias: None,
        },
        value: nn::Linear {
            weight: Param::from_tensor(v_proj.transpose()),
            bias: None,
        },
        out: nn::Linear {
            weight: Param::from_tensor(o_proj.transpose()),
            bias: None,
        },
    }
}

pub fn load_transformer_block<B: Backend>(
    safetensors: &SafeTensors,
    config: &LlamaConfig,
    path: &str,
    device: &B::Device,
) -> ResidualDecoderAttentionBlock<B>
{
    let self_attn =
        load_attention::<B>(safetensors, config, &format!("{}.self_attn", path), device);
    let input_layernorm = load_rmsnorm::<B>(
        safetensors,
        config,
        &format!("{}.input_layernorm", path),
        device,
    );
    let mlp = load_mlp::<B>(safetensors, config, &format!("{}.mlp", path), device);
    let post_attn_layernorm = load_rmsnorm::<B>(
        safetensors,
        config,
        &format!("{}.post_attention_layernorm", path),
        device,
    );

    ResidualDecoderAttentionBlock {
        self_attn,
        input_layernorm,
        mlp,
        post_attn_layernorm,
    }
}
