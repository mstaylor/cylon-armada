// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Cylon Armada — context-based cost optimization for multi-agent LLM workflows.
//!
//! This crate provides the Arrow-native ContextTable for embedding storage,
//! SIMD similarity search, and distributed context sharing. It depends on
//! the core `cylon` crate for SIMD primitives and communication.

pub mod context;