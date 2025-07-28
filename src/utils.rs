use std::sync::{Arc, Mutex};

pub struct AsyncPtrProp<T> {
    inner: Arc<Mutex<T>>,
}

impl<T> AsyncPtrProp<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value)),
        }
    }
    
    pub fn lock(&self) -> Result<std::sync::MutexGuard<T>, std::sync::PoisonError<std::sync::MutexGuard<T>>> {
        self.inner.lock()
    }
}

impl<T> Clone for AsyncPtrProp<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> PartialEq for AsyncPtrProp<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T> Eq for AsyncPtrProp<T> {} 