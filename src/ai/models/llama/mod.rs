mod burn_llama;
mod cpp_llama;

use anyhow::Result;

pub trait TextGenerator: Send {
    fn add_terminator(&mut self, terminators: &str);
    fn generate(
        &mut self,
        prompt: String,
        sample_len: Option<usize>,
        channel: Option<std::sync::mpsc::Sender<String>>,
    ) -> Result<String>;
}


pub use burn_llama::BurnLlama;
pub use cpp_llama::CppLlama;