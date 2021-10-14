//! Trait to bitpack a u64 with both the point index and the scale index.
//! We store the scale index in the top 9bits. It is offset by 64, so we can handle scales in [-64, 447]
//! The point index is stored in the rest, so we can handle 2^65, or about 3.6e19 points. 

/// The bitpacked type 
pub type NodeAddress = u64;

/// The bitpacking trait
pub trait NodeAddressBase {
    /// Creates a new address. This is very well checked.
    fn new(scale_index: i32, point_index: usize) -> Self;
    /// Gets the point index back out
    fn point_index(&self) -> usize;
    /// Gets the scale index back out
    fn scale_index(&self) -> i32;
    /// The None address. This is the max (scale_index 447, point index ~ 3.6e9). Useful for 
    /// situations where you need a none, but 
    fn singleton() -> Self;
}

impl NodeAddressBase for NodeAddress {
    fn new(scale_index: i32, point_index: usize) -> Self {
        assert!(scale_index >= -64);
        let scale_index_u64 = (scale_index + 64) as u64;
        assert!(scale_index_u64.leading_zeros() > 55);
        assert!(point_index.leading_zeros() > 9);
        scale_index_u64 << 55 | point_index as u64
    }
    #[inline]
    fn point_index(&self) -> usize {
        let mask = 0b0000000001111111111111111111111111111111111111111111111111111111;
        (*self & mask) as usize
    }
    #[inline]
    fn scale_index(&self) -> i32 {
        let mask = 0b1111111111000000000000000000000000000000000000000000000000000000;
        ((self & mask) >> 55) as i32 - 64
    }
    #[inline]
    fn singleton() -> Self {
        0b1111111111111111111111111111111111111111111111111111111111111111
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn reconstruction() {
        let scale_index = 10;
        let point_index = 12345;
        let na = NodeAddress::new(scale_index, point_index);
        println!("{:#066b}", na);
        assert_eq!(12, na.vec_scale_index());
        assert_eq!(scale_index, na.scale_index());
        assert_eq!(point_index, na.point_index());
    }

    #[test]
    fn reconstruction_array() {
        for point_index in 1234..12345 {
            for scale_index in -32..32 {
                let na = NodeAddress::new(scale_index, point_index);
                println!("{:#066b}", na);
                assert_eq!(scale_index, na.scale_index());
                assert_eq!(point_index, na.point_index());
            }
        }
    }
}