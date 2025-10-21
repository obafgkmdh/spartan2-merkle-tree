#![allow(non_snake_case)]
use bellpepper_core::{
    ConstraintSystem, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::traits::{Engine, circuit::SpartanCircuit};
use std::marker::PhantomData;

use generic_array::typenum::{Const, ToUInt, U, U2};
use merkle_trees::hash::circuit::hash_circuit;
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::circuit::path_verify_circuit;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};

use neptune::poseidon::Arity;

#[derive(Clone)]
pub struct AggregationCircuit<
    Scalar: PrimeField + PrimeFieldBits,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> where
    Const<BATCH_SIZE>: ToUInt,
    U<BATCH_SIZE>: Arity<Scalar>,
{
    pub raw_logs: Vec<[Scalar; BATCH_SIZE]>, // Raw logs, batched
    pub hashes: Vec<Scalar>,                 // Hashes of batched logs
    pub tree: MerkleTree<Scalar, HEIGHT, U<BATCH_SIZE>, U2>, // Previous tree
                                             // TODO: Add fields for modified and inserted logs, and new tree
}

impl<Scalar: PrimeField + PrimeFieldBits, const HEIGHT: usize, const BATCH_SIZE: usize>
    AggregationCircuit<Scalar, HEIGHT, BATCH_SIZE>
where
    Const<BATCH_SIZE>: ToUInt,
    U<BATCH_SIZE>: Arity<Scalar>,
{
    pub fn new(raw_logs: Vec<[Scalar; BATCH_SIZE]>) -> Self {
        let merkle_leaves: Vec<_> = raw_logs
            .iter()
            .map(|&batch| Leaf {
                val: batch.to_vec(),
                _arity: PhantomData::<U<BATCH_SIZE>>,
            })
            .collect();

        let tree = MerkleTree::from_vec(merkle_leaves.clone(), vanilla_tree::tree::Leaf::default());

        let hashes = merkle_leaves
            .iter()
            .map(|leaf| leaf.hash_leaf(&tree.leaf_hash_params))
            .collect();

        Self {
            raw_logs,
            hashes,
            tree,
        }
    }
}

impl<E: Engine, const HEIGHT: usize, const BATCH_SIZE: usize> SpartanCircuit<E>
    for AggregationCircuit<E::Scalar, HEIGHT, BATCH_SIZE>
where
    Const<BATCH_SIZE>: ToUInt,
    U<BATCH_SIZE>: Arity<E::Scalar> + Sync + Send,
{
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        // Previous tree root is public, along with leaf hashes
        let mut public_values = vec![self.tree.root];
        public_values.extend(&self.hashes);
        Ok(public_values)
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

        let root_input =
            AllocatedNum::alloc_input(cs.namespace(|| format!("public root")), || {
                Ok(self.tree.root)
            })?;
        cs.enforce(
            || "enforce tree root == root input",
            |lc| lc + root_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + root_input.get_variable(),
        );

        // Check that all the leaves hash into the public hash values
        let leaf_hash_params = &self.tree.leaf_hash_params;
        for (idx, &batch) in self.raw_logs.iter().enumerate() {
            let hash_input = AllocatedNum::alloc_input(
                cs.namespace(|| format!("public leaf hash {idx}")),
                || Ok(self.hashes[idx]),
            )?;

            let hash_var =
                AllocatedNum::alloc(cs.namespace(|| format!("hash index {idx}")), || {
                    Ok(self.hashes[idx])
                })?;

            cs.enforce(
                || format!("enforce hash {idx} == hash input"),
                |lc| lc + hash_var.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + hash_input.get_variable(),
            );

            let leaf_var = batch
                .iter()
                .enumerate()
                .map(|(i, &log)| {
                    AllocatedNum::alloc(
                        cs.namespace(|| format!("leaf index {idx}, log {i}")),
                        || Ok(log),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let hashed_leaf = hash_circuit(
                &mut cs.namespace(|| format!("leaf hash {idx}")),
                leaf_var,
                &leaf_hash_params,
            )
            .unwrap();

            let hash_is_equal = Boolean::from(AllocatedBit::alloc(
                cs.namespace(|| format!("leaf hash {idx} == hash {idx}")),
                Some(hash_var.get_value() == hashed_leaf.get_value()),
            )?);

            Boolean::enforce_equal(
                cs.namespace(|| format!("enforce leaf hash {idx}")),
                &hash_is_equal,
                &Boolean::constant(true),
            )?;
        }

        // Membership check
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

            let val = self.raw_logs[index];
            let leaf = Leaf {
                val: val.to_vec(),
                _arity: PhantomData::<U<BATCH_SIZE>>,
            };
            assert!(
                siblings_path.verify(index_bits, &leaf, self.tree.root),
                "leaf {index} should verify"
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

            let is_valid = Boolean::from(path_verify_circuit::<
                E::Scalar,
                U<BATCH_SIZE>,
                U2,
                HEIGHT,
                _,
            >(
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

        // TODO: build updated merkle tree

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
