//! Trait to bitpack a u64 with both the point index and the scale index.
//! We store the scale index in the top 9bits. It is offset by 64, so we can handle scales in [-64, 447]
//! The point index is stored in the rest, so we can handle 2^55, or about 3.6e19 points.
use serde::{Deserialize, Serialize};
use std::convert::From;
use std::fmt;
/// The bitpacked type
#[derive(Copy, Clone, Serialize, Deserialize, PartialEq, Eq, std::hash::Hash, PartialOrd, Ord)]
pub struct NodeAddress {
    na: u64,
}

impl From<(i32, usize)> for NodeAddress {
    fn from(n: (i32, usize)) -> NodeAddress {
        assert!(n.0 >= -64);
        let scale_index_u64 = (n.0 + 64) as u64;
        assert!(scale_index_u64.leading_zeros() > 55);
        assert!(n.1.leading_zeros() > 9);
        NodeAddress {
            na: scale_index_u64 << 55 | n.1 as u64,
        }
    }
}

impl From<NodeAddress> for (i32, usize) {
    fn from(n: NodeAddress) -> (i32, usize) {
        n.unpack()
    }
}

impl fmt::Display for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        write!(f, "({}, {})", self.scale_index(), self.point_index())
    }
}

impl fmt::Debug for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        f.debug_struct("NodeAddress")
            .field("scale_index", &self.scale_index())
            .field("point_index", &self.point_index())
            .finish()
    }
}

impl fmt::Binary for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        write!(f, "{:#066b}", self.na)
    }
}

impl NodeAddress {
    const SINGLETON_U64: u64 = 0b1111111111111111111111111111111111111111111111111111111111111111;
    /// Creates a new node address with unchecked math and no asserts.
    /// Requires you know the scale index and point index are within bounds.
    pub unsafe fn new_unchecked(scale_index: i32, point_index: usize) -> Self {
        let scale_index_u64 = (scale_index + 64) as u64;
        NodeAddress {
            na: scale_index_u64 << 55 | point_index as u64,
        }
    }
    /// The underlying u64
    pub fn raw(&self) -> u64 {
        self.na
    }
    /// Borrowed unpack into the
    pub fn unpack(&self) -> (i32, usize) {
        (self.scale_index(), self.point_index())
    }
    /// The point index of the address
    #[inline]
    pub fn point_index(&self) -> usize {
        let mask = 0b0000000001111111111111111111111111111111111111111111111111111111;
        (self.na & mask) as usize
    }
    /// The scale index of the address
    #[inline]
    pub fn scale_index(&self) -> i32 {
        let mask = 0b1111111111000000000000000000000000000000000000000000000000000000;
        ((self.na & mask) >> 55) as i32 - 64
    }
    /// The "none address", or maximum. This is a point at (447,2^55-1)
    #[inline]
    pub const fn singleton() -> Self {
        NodeAddress {
            na: NodeAddress::SINGLETON_U64,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn reconstruction() {
        let scale_index = 10;
        let point_index = 12345;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(scale_index, na.scale_index());
        assert_eq!(point_index, na.point_index());
        let na_unsafe = unsafe { NodeAddress::new_unchecked(scale_index, point_index) };
        assert_eq!(na, na_unsafe);
    }

    #[test]
    fn reconstruction_array() {
        for point_index in 1234..12345 {
            for scale_index in -32..32 {
                let na = NodeAddress::from((scale_index, point_index));
                println!("{:#066b}", na);
                assert_eq!(scale_index, na.scale_index());
                assert_eq!(point_index, na.point_index());
                let na_unsafe = unsafe { NodeAddress::new_unchecked(scale_index, point_index) };
                assert_eq!(na, na_unsafe);
            }
        }
    }
}
