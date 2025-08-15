pub mod progress;
pub mod organizer;

use std::sync::{Arc, Mutex};


pub struct AsyncPtrProp<T: ?Sized> {
    inner: Arc<Mutex<T>>,
}

impl<T> AsyncPtrProp<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Arc::new(Mutex::new(value)),
        }
    }
}

impl<T: ?Sized> AsyncPtrProp<T> {
    pub fn lock(&self) -> Result<std::sync::MutexGuard<T>, std::sync::PoisonError<std::sync::MutexGuard<T>>> {
        self.inner.lock()
    }
}

impl<T : ?Sized> From<Arc<Mutex<T>>> for AsyncPtrProp<T> {
    fn from(value: Arc<Mutex<T>>) -> Self {
        Self {
            inner: value,
        }
    }
}

impl<T: ?Sized> Clone for AsyncPtrProp<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: ?Sized> PartialEq for AsyncPtrProp<T> {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: ?Sized> Eq for AsyncPtrProp<T> {} 
