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

use std::sync::Arc;
fn build_tree() -> CoverTreeWriter<L2> {
    let file_name = "../data/mnist_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }
    cover_tree_from_yaml(&path).unwrap()
}
/*
#[test]
fn load_tree_and_query() {
    let pc = PointCloud::<L2>::from_file(Path::new("data/mnist.yml")).unwrap();
    let zeros = Arc::new(vec![0.0; 784]);
    let ct_loaded = load_tree(Path::new("data/mnist.tree"), pc).unwrap();
    let ct_reader = ct_loaded.reader();
    let query = ct_reader.knn(&zeros, 5).unwrap();
    println!("(array([3.56982747, 3.65066243, 3.83593169, 3.84857365, 3.86859321]), array([17664, 21618, 51468,  8080, 37920]))");
    assert!(query[0].1 == 17664);
    assert!(query[1].1 == 21618);
    assert!(query[2].1 == 51468);
    assert!(query[3].1 == 8080);
    assert!(query[4].1 == 37920);
    assert!(query.len() == 5);
}
*/
//Cover tree on MNIST builds and is queryable
#[test]
fn run_knn_query() {
    let ct = build_tree();
    save_tree(Path::new("../data/mnist.tree"), &ct).unwrap();
    let ct_reader = ct.reader();
    let zeros = Arc::new(vec![0.0; 784]);
    let query = ct_reader.knn(&zeros, 5).unwrap();
    println!("{:#?}", query);
    println!("Expected: (array([3.56982747, 3.65066243, 3.83593169, 3.84857365, 3.86859321]), array([17664, 21618, 51468,  8080, 37920]))");
    assert!(query[0].1 == 17664);
    assert!(query[1].1 == 21618);
    assert!(query[2].1 == 51468);
    assert!(query[3].1 == 8080);
    assert!(query[4].1 == 37920);
    assert!(query.len() == 5);
}
