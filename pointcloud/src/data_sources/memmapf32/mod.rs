/*
* Licensed to Elasticsearch B.V. under one or more contributor
* license agreements. See the NOTICE file distributed with
* this work for additional information regarding copyright
* ownership. Elasticsearch B.V. licenses this file to you under
* the Apache License, Version 2.0 (the "License"); you may
* not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

//! A cross-platform Rust API for memory mapped buffers. Modified to be f32 specific.

#![doc(html_root_url = "https://docs.rs/memmap/0.7.0")]
#![allow(warnings)]
#[cfg(windows)]
extern crate winapi;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows::MmapInner;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
use self::unix::MmapInner;

use std::fmt;
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::mem::size_of;
use std::ops::{Deref, DerefMut};
use std::slice;
use std::usize;

#[derive(Clone, Debug, Default)]
pub struct MmapOptionsf32 {
    offset: u64,
    len: Option<usize>,
    stack: bool,
}

impl MmapOptionsf32 {
    pub fn new() -> MmapOptionsf32 {
        MmapOptionsf32::default()
    }

    pub fn offset(&mut self, offset: u64) -> &mut Self {
        self.offset = offset;
        self
    }

    pub fn len(&mut self, len: usize) -> &mut Self {
        self.len = Some(len);
        self
    }

    /// Returns the configured length, or the length of the provided file.
    fn get_len(&self, file: &File) -> Result<usize> {
        self.len.map(Ok).unwrap_or_else(|| {
            let len = file.metadata()?.len() - self.offset;
            if len > (usize::MAX as u64) {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "memory map length overflows usize",
                ));
            }
            Ok(len as usize)
        })
    }

    pub fn stack(&mut self) -> &mut Self {
        self.stack = true;
        self
    }

    pub unsafe fn map(&self, file: &File) -> Result<Mmapf32> {
        MmapInner::map(self.get_len(file)?, file, self.offset).map(|inner| Mmapf32 { inner })
    }

    pub unsafe fn map_exec(&self, file: &File) -> Result<Mmapf32> {
        MmapInner::map_exec(self.get_len(file)?, file, self.offset).map(|inner| Mmapf32 { inner })
    }

    pub unsafe fn map_mut(&self, file: &File) -> Result<MmapMutf32> {
        MmapInner::map_mut(self.get_len(file)?, file, self.offset).map(|inner| MmapMutf32 { inner })
    }

    pub unsafe fn map_copy(&self, file: &File) -> Result<MmapMutf32> {
        MmapInner::map_copy(self.get_len(file)?, file, self.offset)
            .map(|inner| MmapMutf32 { inner })
    }

    pub fn map_anon(&self) -> Result<MmapMutf32> {
        MmapInner::map_anon(self.len.unwrap_or(0), self.stack).map(|inner| MmapMutf32 { inner })
    }
}

pub struct Mmapf32 {
    inner: MmapInner,
}

impl Mmapf32 {
    pub unsafe fn map(file: &File) -> Result<Mmapf32> {
        MmapOptionsf32::new().map(file)
    }

    pub fn make_mut(mut self) -> Result<MmapMutf32> {
        self.inner.make_mut()?;
        Ok(MmapMutf32 { inner: self.inner })
    }
}

impl Deref for Mmapf32 {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.inner.ptr() as *const f32, self.inner.len()) }
    }
}

impl AsRef<[f32]> for Mmapf32 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.deref()
    }
}

impl fmt::Debug for Mmapf32 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("Mmapf32")
            .field("ptr", &self.as_ptr())
            .field("len", &self.len())
            .finish()
    }
}

pub struct MmapMutf32 {
    inner: MmapInner,
}

impl MmapMutf32 {
    pub unsafe fn map_mut(file: &File) -> Result<MmapMutf32> {
        MmapOptionsf32::new().map_mut(file)
    }

    pub fn map_anon(length: usize) -> Result<MmapMutf32> {
        MmapOptionsf32::new().len(length).map_anon()
    }

    pub fn flush(&self) -> Result<()> {
        let len = self.len();
        self.inner.flush(0, len)
    }

    pub fn flush_async(&self) -> Result<()> {
        let len = self.len();
        self.inner.flush_async(0, len)
    }

    pub fn flush_range(&self, offset: usize, len: usize) -> Result<()> {
        self.inner.flush(offset, len)
    }

    pub fn flush_async_range(&self, offset: usize, len: usize) -> Result<()> {
        self.inner.flush_async(offset, len)
    }

    pub fn make_read_only(mut self) -> Result<Mmapf32> {
        self.inner.make_read_only()?;
        Ok(Mmapf32 { inner: self.inner })
    }

    pub fn make_exec(mut self) -> Result<Mmapf32> {
        self.inner.make_exec()?;
        Ok(Mmapf32 { inner: self.inner })
    }
}

impl Deref for MmapMutf32 {
    type Target = [f32];

    #[inline]
    fn deref(&self) -> &[f32] {
        let f32_size = size_of::<f32>();
        unsafe {
            slice::from_raw_parts(self.inner.ptr() as *const f32, self.inner.len() / f32_size)
        }
    }
}

impl DerefMut for MmapMutf32 {
    #[inline]
    fn deref_mut(&mut self) -> &mut [f32] {
        let f32_size = size_of::<f32>();
        unsafe {
            slice::from_raw_parts_mut(
                self.inner.mut_ptr() as *mut f32,
                self.inner.len() / f32_size,
            )
        }
    }
}

impl AsRef<[f32]> for MmapMutf32 {
    #[inline]
    fn as_ref(&self) -> &[f32] {
        self.deref()
    }
}

impl AsMut<[f32]> for MmapMutf32 {
    #[inline]
    fn as_mut(&mut self) -> &mut [f32] {
        self.deref_mut()
    }
}

impl fmt::Debug for MmapMutf32 {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("MmapMut")
            .field("ptr", &self.as_ptr())
            .field("len", &self.len())
            .finish()
    }
}
