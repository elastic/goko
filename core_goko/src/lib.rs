#![feature(unchecked_math)]
//! Trait to bitpack a u64 with both the point index and the scale index.
//! We store the scale index in the top 9bits. It is offset by 64, so we can handle scales in [-63, 447]
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

impl From<&(i32, usize)> for NodeAddress {
    fn from(n: &(i32, usize)) -> NodeAddress {
        (*n).into()
    }
}

impl From<Option<(i32, usize)>> for NodeAddress {
    fn from(n: Option<(i32, usize)>) -> NodeAddress {
        if let Some(n) = n {
            n.into()
        } else {
            NodeAddress{ na: NodeAddress::SINGLETON_U64 }
        }
    }
}

impl From<Option<&(i32, usize)>> for NodeAddress {
    fn from(n: Option<&(i32, usize)>) -> NodeAddress {
        if let Some(n) = n {
            n.into()
        } else {
            NodeAddress{ na: NodeAddress::SINGLETON_U64 }
        }
    }
}

impl From<NodeAddress> for Option<(i32, usize)> {
    fn from(n: NodeAddress) -> Option<(i32, usize)> {
        n.to_tuple()
    }
}

impl From<&NodeAddress> for Option<(i32, usize)> {
    fn from(n: &NodeAddress) -> Option<(i32, usize)> {
        n.to_tuple()
    }
}

impl From<u64> for NodeAddress {
    fn from(na: u64) -> NodeAddress {
        NodeAddress {na}
    }
}

impl From<&u64> for NodeAddress {
    fn from(na: &u64) -> NodeAddress {
        NodeAddress {na: *na}
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
    pub const SINGLETON: NodeAddress = NodeAddress{na: Self::SINGLETON_U64};
    /// Creates a new node address with unchecked math and no asserts.
    /// Requires you know the scale index and point index are within bounds.
    /// 
    /// # Safety
    /// You have to know that scale index is within [-64, 447] and point index is smaller than 2^54, and both aren't (-64, 2^54 - 1)
    #[inline]
    pub unsafe fn new_unchecked(scale_index: i32, point_index: usize) -> Self {
        debug_assert!(scale_index >= -63);
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
    /// Borrowed to_tuple into the
    pub fn to_tuple(&self) -> Option<(i32, usize)> {
        let si = self.scale_index();
        if si == -64 {
            None
        } else {
            Some((self.scale_index(), self.point_index()))
        }
    }
    /// The point index of the address
    #[inline]
    pub fn point_index(&self) -> usize {
        (self.na & PMASK_U64) as usize
    }
    /// The scale index of the address. If this is -64 then the node is a singleton.
    #[inline]
    pub fn scale_index(&self) -> i32 {
        ((self.na & SMASK_U64) >> 55) as i32 - 64
    }
    /// The "none address", or maximum. This is a point at (447,2^55-1)
    #[inline]
    pub fn singleton(&self) -> bool {
        self.na & PMASK_U64 == self.na 
    }
}

/// Helper trait that cleans up code. 
pub trait AddressesToVec {
    fn to_point_indexes(&self) -> Vec<usize>;
    fn to_scale_indexes(&self) -> Vec<i32>;
    fn to_tuples(&self) -> Vec<Option<(i32, usize)>>;
    fn to_valid_tuples(&self) -> Vec<(i32, usize)>;
}

impl AddressesToVec for [NodeAddress] {
    fn to_point_indexes(&self) -> Vec<usize> {
        self.iter().map(|na| na.point_index()).collect()
    }
    fn to_scale_indexes(&self) -> Vec<i32> {
        self.iter().map(|na| na.scale_index()).collect()
    }
    fn to_tuples(&self) -> Vec<Option<(i32, usize)>> {
        self.iter().map(|na| na.to_tuple()).collect()
    }
    fn to_valid_tuples(&self) -> Vec<(i32, usize)> {
        self.iter().filter_map(|na| na.to_tuple()).collect()
    }
}

/// Helper trait that cleans up code. 
pub trait PairAddressesToVec<T> {
    fn to_point_indexes(&self) -> Vec<(usize, T)>;
    fn to_scale_indexes(&self) -> Vec<(i32, T)>;
    fn to_tuples(&self) -> Vec<(Option<(i32, usize)>, T)>;
    fn to_valid_tuples(&self) -> Vec<((i32, usize), T)>;
}

impl<T: Clone> PairAddressesToVec<T> for [(NodeAddress, T)] {
    fn to_point_indexes(&self) -> Vec<(usize, T)> {
        self.iter().map(|na| (na.0.point_index(), na.1.clone())).collect()
    }
    fn to_scale_indexes(&self) -> Vec<(i32, T)> {
        self.iter().map(|na| (na.0.scale_index(), na.1.clone())).collect()
    }
    fn to_tuples(&self) -> Vec<(Option<(i32, usize)>, T)> {
        self.iter().map(|na| (na.0.to_tuple(), na.1.clone())).collect()
    }
    fn to_valid_tuples(&self) -> Vec<((i32, usize), T)> {
        self.iter().filter_map(|na| (na.0.to_tuple().map(|n| (n,na.1.clone())))).collect()
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

impl TuplesToAddresses for [Option<(i32, usize)>] {
    fn to_addresses(&self) -> Vec<NodeAddress> {
        self.iter().map(|t| (*t).into()).collect()
    }

    unsafe fn to_addresses_unchecked(&self) -> Vec<NodeAddress> {
        self.iter().map(|n| n.map(|(si,pi)| NodeAddress::new_unchecked(si,pi)).unwrap_or(NodeAddress::SINGLETON) ).collect()
    }
}

pub trait SliceToAddresses {
    fn to_addresses(&self) -> Vec<NodeAddress>;
}

impl SliceToAddresses for [u64] {
    fn to_addresses(&self) -> Vec<NodeAddress> {
        self.iter().map(|t| t.into()).collect()
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
        let scale_index = -63;
        let point_index = 0;
        let na = NodeAddress::from((scale_index, point_index));
        println!("{:#066b}", na);
        assert_eq!(na.na, 1 << 55);
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
        let s = NodeAddress::SINGLETON;
        println!("{:#066b}", ((2u64 << 54) - 1));
        println!("{:#034b}", -64i32);
        println!("{:b}", s);
        assert_eq!(-64, s.scale_index());
        assert_eq!((2 << 54) - 1, s.point_index());
    }
}
