#![allow(non_snake_case)]
use bellpepper_core::{
    ConstraintSystem, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::{
    traits::{Engine, circuit::SpartanCircuit},
};
use std::{marker::PhantomData};

use generic_array::typenum::{U1, U2};
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::circuit::path_verify_circuit;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};

#[derive(Clone, Debug)]
pub struct MerkleTreeMembershipCircuit<Scalar: PrimeField + PrimeFieldBits, const HEIGHT: usize> {
    pub leaves: Vec<Scalar>,
    pub tree: MerkleTree<Scalar, HEIGHT, U1, U2>,
}

impl<Scalar: PrimeField + PrimeFieldBits, const HEIGHT: usize> MerkleTreeMembershipCircuit<Scalar, HEIGHT> {
    pub fn new(leaves: Vec<Scalar>) -> Self {
        let merkle_leaves = leaves
            .iter()
            .map(|&val| Leaf {
                val: vec![val],
                _arity: PhantomData::<U1>,
            })
            .collect();

        let tree = MerkleTree::from_vec(merkle_leaves, vanilla_tree::tree::Leaf::default());

        Self { leaves, tree }
    }
}

impl<E: Engine, const HEIGHT: usize> SpartanCircuit<E> for MerkleTreeMembershipCircuit<E::Scalar, HEIGHT> {
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        // root should be public
        Ok(vec![self.tree.root])
    }

    fn shared<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        // No shared variables in this circuit
        Ok(vec![])
    }

    fn precommitted<CS: ConstraintSystem<E::Scalar>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<E::Scalar>], // shared variables, if any
    ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
        let root_var =
            AllocatedNum::alloc(cs.namespace(|| format!("tree root")), || Ok(self.tree.root))?;

        let input = AllocatedNum::alloc_input(cs.namespace(|| format!("public input")), || {
            Ok(self.tree.root)
        })?;
        cs.enforce(
            || "enforce input == root",
            |lc| lc + root_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + input.get_variable(),
        );

        for index in 0..100 {
            let index_scalar = E::Scalar::from(index as u64);

            let index_bits = idx_to_bits(HEIGHT, index_scalar);

            let index_bits_var = index_bits
                .clone()
                .into_iter()
                .enumerate()
                .map(|(i, b)| {
                    AllocatedBit::alloc(
                        cs.namespace(|| format!("index {index} preimage bit {i}")),
                        Some(b),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let siblings_path = self.tree.get_siblings_path(index_bits.clone());

            let val = self.leaves[index];
            let leaf = Leaf {
                val: vec![val],
                _arity: PhantomData::<U1>,
            };
            assert!(
                siblings_path.verify(index_bits, &leaf, self.tree.root),
                "leaf should verify"
            );

            let siblings_var = siblings_path
                .siblings
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    AllocatedNum::alloc(
                        cs.namespace(|| format!("index {index} sibling {i}")),
                        || Ok(s),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let input_var = leaf
                .val
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    AllocatedNum::alloc(
                        cs.namespace(|| format!("index {index} leaf val {i}")),
                        || Ok(s),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let is_valid = Boolean::from(path_verify_circuit::<E::Scalar, U1, U2, HEIGHT, _>(
                &mut cs.namespace(|| format!("valid {index}")),
                root_var.clone(),
                input_var,
                index_bits_var,
                siblings_var,
            )?);

            Boolean::enforce_equal(
                cs.namespace(|| format!("enforce true {index}")),
                &is_valid,
                &Boolean::constant(true),
            )?;
        }

        Ok(vec![])
    }

    fn num_challenges(&self) -> usize {
        // circuit does not expect any challenges
        0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<E::Scalar>],
        _: &[AllocatedNum<E::Scalar>],
        _: Option<&[E::Scalar]>,
    ) -> Result<(), SynthesisError> {
        Ok(())
    }
}
