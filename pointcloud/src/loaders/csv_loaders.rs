

use crate::pc_errors::*;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use csv::Reader;


use crate::label_sources::*;

/// Opens a CSV and reads a single column from it as a integer label. Negative labels are treated as unlabeled and are masked.
pub fn open_int_csv<P: AsRef<Path> + std::fmt::Debug>(path: &P,index:usize) -> PointCloudResult<SmallIntLabels> {
    if !path.as_ref().exists() {
        panic!("CSV file {:?} does not exist", path);
    }

    match File::open(&path) {
        Ok(file) => {
            if path.as_ref().extension().unwrap() == "gz" {
                read_csv(index,path,Reader::from_reader(GzDecoder::new(file)))
            } else {
                read_csv(index,path,Reader::from_reader(file))
            }
        }
        Err(e) => panic!("Unable to open csv file {:#?}", e),
    }
}

fn read_csv<P: AsRef<Path> + std::fmt::Debug,R: Read>(
        index:  usize,
        path: &P,
        mut rdr: Reader<R>,
    ) -> PointCloudResult<SmallIntLabels> {

    let mut labels = Vec::new();
    let mut mask = Vec::new();

    for result in rdr.records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result.expect("Unable to read a record from the label CSV");
        match record.get(index) {
            Some(val) => {
                let val = val.parse::<i64>().map_err(|_| {
                    PointCloudError::ParsingError(
                                            ParsingError::CSVReadError {
                                                file_name: path.as_ref().to_string_lossy().to_string(),
                                                line_number: record.position().unwrap().line() as usize,
                                                key: format!("Unable to read u64 from {:?}",record).to_string(),
                                            },
                                        )})?;
                if 0 < val {
                    mask.push(true);
                } else {
                    mask.push(false);
                }
                labels.push(val as u64);
            }
            None => {
                labels.push(0);
                mask.push(false);
            }
        }
    }
    if mask.iter().any(|f|!f) {
        Ok(SmallIntLabels::new(labels, Some(mask)))
    } else {
        Ok(SmallIntLabels::new(labels, None))
    }
}
