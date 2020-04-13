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

extern crate protobuf;
extern crate rand;
extern crate yaml_rust;
use std::path::Path;
#[allow(dead_code)]
extern crate grandma;
extern crate pointcloud;
use grandma::utils::*;
use grandma::CoverTreeWriter;
use pointcloud::*;

fn build_tree() -> CoverTreeWriter<L2> {
    let file_name = "../data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }
    cover_tree_from_yaml(&path).unwrap()
}

fn main() {
    for _i in 0..1 {
        let mut ct = build_tree();
        //ct.cluster().unwrap();
        ct.refresh();
        let ct_reader = ct.reader();
        println!("Tree has {} nodes", ct_reader.node_count());
        for scale_index in ct_reader.scale_range() {
            println!(
                "Layer {} has {} nodes in scale {}",
                scale_index,
                ct_reader.layer(scale_index).node_count(),
                ct_reader.scale(scale_index)
            );
        }
    }
    /*
    println!("===== Parameters =====");
    let (sb, co, re) = ct.parameters();
    println!(
        "{{\"scale_base\":{},\"cutoff\":{},\"resolution\":{}}}",
        sb, co, re
    );
    println!("===== Schema =====");
    println!("{}", ct.metadata_schema());
    println!("===== KNN =====");
    let query1 = ct.knn_query(&zeros, 5).unwrap();
    println!("{:?}", query1);
    println!("===== Center KNN =====");
    let query1 = ct.center_knn_query(&zeros, 5).unwrap();
    println!("{:?}", query1);
    println!("===== Trace =====");
    assert!(query1.len() == 5);
    let query2 = ct.insert_trace(&zeros).unwrap();
    let trace_report: String = query2
        .iter()
        .map(|node| node.report_json())
        .collect::<Vec<String>>()
        .join(",");
    println!("[{}]", trace_report);
    println!("===== Saving =====");
    */
}
