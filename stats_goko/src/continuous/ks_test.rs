use std::cmp::Ordering;

pub fn ks_test<T: Ord + Clone>(x: &[T], y: &[T]) -> f64 {
    let mut x_sorted: Vec<T> = Vec::from(x);
    x_sorted.sort();
    let mut y_sorted: Vec<T> = Vec::from(y);
    y_sorted.sort();
    let x_inc = 1.0 / (x.len() as f64);
    let y_inc = 1.0 / (y.len() as f64);
    let mut curr_diff = 0.0;
    let mut max_cdf_diff = 0.0;
    let mut x_iter = x_sorted.iter();
    let mut y_iter = y_sorted.iter();

    let mut x_val = x_iter.next();
    let mut y_val = y_iter.next();
    loop {
        match (x_val, y_val) {
            (None, None) => return max_cdf_diff,
            (Some(_), None) => {
                curr_diff += x_inc;
                max_cdf_diff = max_cdf_diff.max(curr_diff.abs());
                x_val = x_iter.next();
            }
            (None, Some(_)) => {
                curr_diff -= y_inc;
                max_cdf_diff = max_cdf_diff.max(curr_diff.abs());
                y_val = y_iter.next();
            }
            (Some(x_1), Some(y_1)) => {
                match x_1.cmp(y_1) {
                    Ordering::Equal => {
                        curr_diff += x_inc;
                        x_val = x_iter.next();
                        curr_diff -= y_inc;
                        y_val = y_iter.next();
                    }
                    Ordering::Greater => {
                        curr_diff -= y_inc;
                        y_val = y_iter.next();
                    }
                    Ordering::Less => {
                        curr_diff += x_inc;
                        x_val = x_iter.next();
                    }
                }
                max_cdf_diff = max_cdf_diff.max(curr_diff.abs());
            }
        }
    }
}
