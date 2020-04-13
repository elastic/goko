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
extern crate protoc_rust;

#[cfg(not(feature = "docs-only"))]
use std::env;

#[cfg(not(feature = "docs-only"))]
fn main() {
    if env::var("TRAVIS_RUST_VERSION").is_err() {
        protoc_rust::Codegen::new()
        .out_dir("src")
        .include("protos")
        .input("protos/tree_file_format.proto")
        .run()
        .expect("protoc");
    }
    println!("cargo:rerun-if-changed=protos/tree_file_format.proto");
}

#[cfg(feature = "docs-only")]
fn main() {
    println!("NOT Building proto");
}