use dioxus::signals::{Signal, SyncStorage};
use dioxus::prelude::*;
use hf_hub::api::Progress;

#[derive(Clone)]
pub struct UiProgress {
    pub progress: usize,
    pub total: usize,
    pub stage: String,
    pub stages_total: usize,
    pub stages_done: usize,
}

impl UiProgress {
    pub fn new(stage: &str, stages_total: usize) -> Self {
        Self { progress: 0, total: 0, stage: stage.to_string(), stages_total, stages_done: 0 }
    }
}

#[derive(Clone)]
pub struct UiProgressSignal(pub Signal<UiProgress, SyncStorage>);

impl Progress for UiProgressSignal {
    fn init(&mut self, size: usize, filename: &str) {
        println!("Progress init: {} {}", size, filename);
        let mut progress = self.0.read().clone();
        progress.stage = filename.to_string();
        progress.total = size;
        progress.progress = 0;
        self.0.set(progress);
    }
    fn update(&mut self, size: usize) {
        println!("Progress update: {}", size);
        let mut progress = self.0.read().clone();
        progress.progress += size;
        self.0.set(progress);
    }
    fn finish(&mut self) {
        println!("Progress finish");
        let mut progress = self.0.read().clone();
        progress.progress = progress.total;
        progress.stages_done += 1;
        self.0.set(progress);
    }
}