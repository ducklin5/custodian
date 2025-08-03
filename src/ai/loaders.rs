use burn::tensor::{backend::Backend, bf16, f16, Bool, TensorData, Device, Element, Float, Int, Tensor};
use safetensors::{Dtype, SafeTensors};

pub trait SafeTensorsReader {
    fn read_burn_tensor_bool<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Bool>;

    fn read_burn_tensor_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;

    #[cfg(not(feature = "wgpu"))]
    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;

    #[cfg(feature = "wgpu")]
    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;

    #[cfg(not(feature = "wgpu"))]
    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;

    #[cfg(feature = "wgpu")]
    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;

    fn read_burn_tensor_i32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Int>;
}

impl SafeTensorsReader for SafeTensors<'_> {
    fn read_burn_tensor_bool<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Bool>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BOOL);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[bool] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data = TensorData::new(data.to_vec(), shape.to_vec());
        Tensor::<B, D, Bool>::from_data(data, device)
    }

    fn read_burn_tensor_f32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::F32);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[f32] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data = TensorData::new(data.to_vec(), shape.to_vec());
        Tensor::<B, D, Float>::from_data(data, device)
    }

    #[cfg(not(feature = "wgpu"))]
    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::F16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };

        let data: Vec<f16> = data
            .iter()
            .map(|&x| f16::from_bits(x))
            .collect::<Vec<f16>>();
        let data = TensorData::new(data, shape.to_vec());
        Tensor::<B, D, Float>::from_data(data, device)
    }

    #[cfg(not(feature = "wgpu"))]
    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BF16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Vec<bf16> = data
            .iter()
            .map(|&x| bf16::from_bits(x))
            .collect::<Vec<bf16>>();
        let data = TensorData::new(data, shape.to_vec());
        Tensor::<B, D, Float>::from_data(data, device)
    }

    #[cfg(feature = "wgpu")]
    fn read_burn_tensor_f16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::F16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Vec<f32> = data
            .iter()
            .map(|&x| f16::from_bits(x).to_f32())
            .collect::<Vec<f32>>();
        let data = TensorData::new(data, shape.to_vec());
        Tensor::<B, D, Float>::from_data(data, device)
    }

    /// When the `wgpu` feature is enabled, `bf16` is represented as `f32` in SafeTensors.
    #[cfg(feature = "wgpu")]
    fn read_burn_tensor_bf16<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::BF16);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[u16] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data: Vec<f32> = data
            .iter()
            .map(|&x| bf16::from_bits(x).to_f32())
            .collect::<Vec<f32>>();
        let data = TensorData::new(data, shape.to_vec());
        Tensor::<B, D, Float>::from_data(data, device)
    }

    fn read_burn_tensor_i32<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Int>
    {
        let safe_tensors = self.tensor(name).unwrap();
        let dtype = safe_tensors.dtype();
        assert_eq!(dtype, Dtype::I32);
        let shape = safe_tensors.shape();
        let count = shape.iter().product::<usize>();

        let data: &[i32] =
            unsafe { std::slice::from_raw_parts(safe_tensors.data().as_ptr() as *const _, count) };
        let data = TensorData::new(data.to_vec(), shape.to_vec());
        Tensor::<B, D, Int>::from_data(data, device)
    }
}

pub trait WgpuFloatTensorReader {
    fn read_full_float<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>;
}

impl WgpuFloatTensorReader for SafeTensors<'_> {
    fn read_full_float<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &Device<B>,
    ) -> Tensor<B, D, Float>
    {
        let dtype = self.tensor(name).unwrap().dtype();
        match dtype {
            Dtype::F32 => self.read_burn_tensor_f32::<B, D>(name, device),
            Dtype::F16 => self.read_burn_tensor_f16::<B, D>(name, device),
            Dtype::BF16 => self.read_burn_tensor_bf16::<B, D>(name, device),
            _ => panic!("Unsupported dtype: {:?}", dtype),
        }
    }
}

//#[test]
fn test_serialize() {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use safetensors::{serialize, tensor::TensorView, Dtype, SafeTensors};
    use std::collections::HashMap;

    let raw = vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let data: Vec<u8> = raw
        .clone()
        .into_iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    let shape = vec![1, 1, 2, 3];
    let attn_0 = TensorView::new(Dtype::F32, shape.clone(), &data).unwrap();
    let metadata: HashMap<String, TensorView> =
            // Smaller string to force misalignment compared to previous test.
            [("attn0".to_string(), attn_0)].into_iter().collect();
    let binary = serialize(&metadata, None).unwrap();

    // ====================
    let parsed = SafeTensors::deserialize(&binary).unwrap();
    let device = WgpuDevice::default();
    let tensor = parsed.read_full_float::<Wgpu, 4>("attn0", &device);

    assert_eq!(shape, tensor.shape().dims);
}
