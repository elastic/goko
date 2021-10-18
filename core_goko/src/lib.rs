#![feature(unchecked_math)]
//! Trait to bitpack a u64 with both the point index and the scale index.
//! We store the scale index in the top 9bits. It is offset by 64, so we can handle scales in [-64, 447]
//! The point index is stored in the rest, so we can handle 2^55, or about 3.6e19 points.
use serde::{Deserialize, Serialize};
use std::convert::From;
use std::fmt;

const SMASK_U64: u64 = 0b1111111110000000000000000000000000000000000000000000000000000000;
const PMASK_U64: u64 = 0b0000000001111111111111111111111111111111111111111111111111111111;

/// The bitpacked type
#[derive(Copy, Clone, Serialize, Deserialize, PartialEq, Eq, std::hash::Hash, PartialOrd, Ord)]
pub struct NodeAddress {
    na: u64,
}

impl From<(i32, usize)> for NodeAddress {
    fn from(n: (i32, usize)) -> NodeAddress {
        assert!(n.0 >= -64);
        let scale_index_u64 = (n.0 + 64) as u64;
        assert!(
            scale_index_u64.leading_zeros() >= 55,
            "{:#066b}",
            scale_index_u64
        );
        assert!(n.1.leading_zeros() >= 9, "{:#066b}", n.1);
        let na = scale_index_u64 << 55 | n.1 as u64;
        assert!(na != PMASK_U64);
        NodeAddress { na }
    }
}

impl From<NodeAddress> for (i32, usize) {
    fn from(n: NodeAddress) -> (i32, usize) {
        n.unpack()
    }
}

impl fmt::Display for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.scale_index(), self.point_index())
    }
}

impl fmt::Debug for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("NodeAddress")
            .field("scale_index", &self.scale_index())
            .field("point_index", &self.point_index())
            .finish()
    }
}

impl fmt::Binary for NodeAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#066b}", self.na)
    }
}

impl NodeAddress {
    /// The "none address", or maximum. This is a point at (-30,)
    pub const SINGLETON_U64: u64 = PMASK_U64;
    /// Creates a new node address with unchecked math and no asserts.
    /// Requires you know the scale index and point index are within bounds.
    /// 
    /// # Safety
    /// You have to know that scale index is within [-64, 447] and point index is smaller than 2^54, and both aren't (-64, 2^54 - 1)
    #[inline]
    pub unsafe fn new_unchecked(scale_index: i32, point_index: usize) -> Self {
        debug_assert!(scale_index >= -64);
        let scale_index_u64 = scale_index.unchecked_add(64) as u64;
        debug_assert!(
            scale_index_u64.leading_zeros() >= 55,
            "{:#066b}",
            scale_index_u64
        );
        debug_assert!(point_index.leading_zeros() >= 9, "{:#066b}", point_index);
        let scale_index_u64 = scale_index.unchecked_add(64) as u64;
        NodeAddress {
            na: scale_index_u64.unchecked_shl(55) | point_index as u64,
        }
    }
    /// The underlying u64
    pub fn raw(&self) -> u64 {
        self.na
    }
    /// coverts a u64 into a node address, there's one invalid u64, the singleton address, which does not correspond to  
    #[inline]
    pub fn from_u64(na: u64) -> Option<NodeAddress> {
        if na == NodeAddress::SINGLETON_U64 {
            None
        } else {
            Some(NodeAddress { na })
        }
    }
    /// Borrowed unpack into the
    pub fn unpack(&self) -> (i32, usize) {
        (self.scale_index(), self.point_index())
    }
    /// The point index of the address
    #[inline]
    pub fn point_index(&self) -> usize {
        (self.na & PMASK_U64) as usize
    }
    /// The scale index of the address
    #[inline]
    pub fn scale_index(&self) -> i32 {
        ((self.na & SMASK_U64) >> 55) as i32 - 64
    }
    /// The "none address", or maximum. This is a point at (447,2^55-1)
    #[inline]
    pub const fn singleton() -> Self {
        NodeAddress {
            na: NodeAddress::SINGLETON_U64,
        }
    }
}

/// Helper trait that cleans up code. 
pub trait AddressesToVec {
    fn to_point_indexes(&self) -> Vec<usize>;
    fn to_scale_indexes(&self) -> Vec<i32>;
}

impl AddressesToVec for [NodeAddress] {
    fn to_point_indexes(&self) -> Vec<usize> {
        self.iter().map(|na| na.point_index()).collect()
    }
    fn to_scale_indexes(&self) -> Vec<i32> {
        self.iter().map(|na| na.scale_index()).collect()
    }
}

pub trait TuplesToAddresses {
    fn to_addresses(&self) -> Vec<NodeAddress>;
    /// # Safety
    /// You have to know that all the scale indexes are within [-64, 447] and point index is smaller than 2^54.
    unsafe fn to_addresses_unchecked(&self) -> Vec<NodeAddress>;
}

impl TuplesToAddresses for [(i32, usize)] {
    fn to_addresses(&self) -> Vec<NodeAddress> {
        self.iter().map(|t| (*t).into()).collect()
    }

    unsafe fn to_addresses_unchecked(&self) -> Vec<NodeAddress> {
        self.iter().map(|(si,pi)| NodeAddress::new_unchecked(*si,*pi)).collect()
    }
}

pub trait SliceToAddresses {
    fn to_addresses(&self) -> Vec<Option<NodeAddress>>;
}

impl SliceToAddresses for [u64] {
    fn to_addresses(&self) -> Vec<Option<NodeAddress>> {
        self.iter().map(|t| NodeAddress::from_u64(*t)).collect()
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
    fn reconstruction_minimum_scale_min_point() {
        let scale_index = -64;
        let point_index = 0;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(na.na, 0);
        assert_eq!(scale_index, na.scale_index());
        assert_eq!(point_index, na.point_index());
        let na_unsafe = unsafe { NodeAddress::new_unchecked(scale_index, point_index) };
        assert_eq!(na, na_unsafe);
    }

    #[test]
    fn reconstruction_maximum_scale_min_point() {
        let scale_index = 511 - 64;
        let point_index = 0;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(SMASK_U64, na.na, "{:b}", na);
        assert_eq!(scale_index, na.scale_index());
        assert_eq!(point_index, na.point_index());
        let na_unsafe = unsafe { NodeAddress::new_unchecked(scale_index, point_index) };
        assert_eq!(na, na_unsafe);
    }

    #[test]
    fn reconstruction_minimum_scale_max_point() {
        // If this was -64 it'd form the invalid point.
        let scale_index = -62;
        let point_index = (2 << 54) - 1;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(
            na.na,
            0b0000000101111111111111111111111111111111111111111111111111111111
        );
        assert_eq!(scale_index, na.scale_index());
        assert_eq!(point_index, na.point_index());
        let na_unsafe = unsafe { NodeAddress::new_unchecked(scale_index, point_index) };
        assert_eq!(na, na_unsafe);
    }
    #[test]
    fn reconstruction_maximum_scale_max_point() {
        let scale_index = 511 - 64;
        let point_index = (2 << 54) - 1;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(
            0b1111111111111111111111111111111111111111111111111111111111111111, na.na,
            "{:b}",
            na
        );
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

    #[test]
    fn singleton_in_correct_spot() {
        let s = NodeAddress::singleton();
        println!("{:#066b}", ((2u64 << 54) - 1));
        println!("{:#034b}", -64i32);
        println!("{:b}", s);
        assert_eq!(-64, s.scale_index());
        assert_eq!((2 << 54) - 1, s.point_index());
    }
}
