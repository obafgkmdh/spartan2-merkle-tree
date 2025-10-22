#![allow(non_snake_case)]
use bellpepper_core::{
    ConstraintSystem, LinearCombination, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use spartan2::traits::{Engine, circuit::SpartanCircuit};
use std::marker::PhantomData;

use generic_array::typenum::{Const, ToUInt, U, U1, U2};
use merkle_trees::hash::circuit::hash_circuit;
use merkle_trees::hash::vanilla::hash;
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::circuit::path_verify_circuit;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};

use neptune::Strength;
use neptune::poseidon::Arity;
use neptune::sponge::vanilla::{Sponge, SpongeTrait};

use std::collections::HashMap;
use std::collections::hash_map::Entry;

#[derive(Clone, Debug)]
pub struct Log<Scalar> {
    pub flow_id: i32,
    pub hop_cnt: Scalar,
}

#[derive(Clone, Debug)]
pub struct CompressedLog<Scalar> {
    pub merkle_idx: usize,
    pub hop_cnt: Scalar,
}

#[derive(Clone, Debug)]
pub struct AggregationCircuit<
    Scalar: PrimeField + PrimeFieldBits,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> where
    Const<BATCH_SIZE>: ToUInt,
    U<BATCH_SIZE>: Arity<Scalar>,
{
    pub raw_logs: Vec<[Log<Scalar>; BATCH_SIZE]>, // New raw logs from routers, batched
    pub hashes: Vec<Scalar>,                      // Hashes of batched logs
    pub old_compressed_logs: HashMap<i32, CompressedLog<Scalar>>, // Old compressed logs, one per flow id
    pub prev_tree: MerkleTree<Scalar, HEIGHT, U1, U2>,            // Previous tree
    pub new_tree: MerkleTree<Scalar, HEIGHT, U1, U2>,             // Updated tree
}

impl<Scalar: PrimeField + PrimeFieldBits, const HEIGHT: usize, const BATCH_SIZE: usize>
    AggregationCircuit<Scalar, HEIGHT, BATCH_SIZE>
where
    Const<BATCH_SIZE>: ToUInt,
    U<BATCH_SIZE>: Arity<Scalar>,
{
    pub fn new(raw_logs: Vec<Log<Scalar>>, num_new_batches: usize) -> Self {
        // Split the new logs into batches
        let new_idx = raw_logs.len() - (BATCH_SIZE * num_new_batches);
        let (batched_logs, rem) = raw_logs[new_idx..].as_chunks::<BATCH_SIZE>();
        assert_eq!(
            rem.len(),
            0,
            "Number of inputs ({}) was not a multiple of the batch size ({})",
            rem.len(),
            0
        );
        let batched_logs = batched_logs.to_vec();

        // Compute hashes of batched logs
        let log_hash_constants = Sponge::<Scalar, U<BATCH_SIZE>>::api_constants(Strength::Standard);
        let hashes = batched_logs
            .iter()
            .map(|batch| {
                let logs = batch
                    .iter()
                    .map(|log| {
                        // Combine fields into a single Scalar. This is hacky and possibly
                        // unsound (since there's no size checks).
                        // TODO: Make this a hash call or just flatten all the fields
                        let flow_id = Scalar::from_u128(log.flow_id as u128);
                        (flow_id * Scalar::from_u128(1 << 64)) + log.hop_cnt
                    })
                    .collect();
                hash(logs, &log_hash_constants)
            })
            .collect();

        // Compress all the old logs
        let mut old_compressed_logs: HashMap<_, CompressedLog<_>> = HashMap::new();
        let mut merkle_idx = 0;
        for log in raw_logs[..new_idx].into_iter() {
            match old_compressed_logs.entry(log.flow_id) {
                Entry::Occupied(clog) => {
                    clog.into_mut().hop_cnt += log.hop_cnt;
                }
                Entry::Vacant(entry) => {
                    let clog = CompressedLog {
                        merkle_idx,
                        hop_cnt: log.hop_cnt,
                    };
                    entry.insert(clog);
                    merkle_idx += 1;
                }
            }
        }

        // Build prev_tree from old compressed logs
        let mut merkle_leaves: Vec<Option<_>> = Vec::new();
        merkle_leaves.resize(merkle_idx, None);
        for clog in old_compressed_logs.values() {
            merkle_leaves[clog.merkle_idx] = Some(clog);
        }

        let merkle_leaves: Vec<_> = merkle_leaves
            .iter()
            .map(|clog| Leaf {
                val: vec![clog.unwrap().hop_cnt],
                _arity: PhantomData::<U1>,
            })
            .collect();

        let prev_tree = MerkleTree::from_vec(merkle_leaves, vanilla_tree::tree::Leaf::default());

        // Compress the rest of the logs
        let mut compressed_logs = old_compressed_logs.clone();
        for log in raw_logs[new_idx..].into_iter() {
            match compressed_logs.entry(log.flow_id) {
                Entry::Occupied(clog) => {
                    clog.into_mut().hop_cnt += log.hop_cnt;
                }
                Entry::Vacant(entry) => {
                    let clog = CompressedLog {
                        merkle_idx,
                        hop_cnt: log.hop_cnt,
                    };
                    entry.insert(clog);
                    merkle_idx += 1;
                }
            }
        }

        // Build new_tree from new compressed logs
        let mut merkle_leaves: Vec<Option<_>> = Vec::new();
        merkle_leaves.resize(merkle_idx, None);
        for clog in compressed_logs.values() {
            merkle_leaves[clog.merkle_idx] = Some(clog);
        }

        let merkle_leaves: Vec<_> = merkle_leaves
            .iter()
            .map(|clog| Leaf {
                val: vec![clog.unwrap().hop_cnt],
                _arity: PhantomData::<U1>,
            })
            .collect();

        let new_tree = MerkleTree::from_vec(merkle_leaves, vanilla_tree::tree::Leaf::default());

        Self {
            raw_logs: batched_logs,
            hashes,
            old_compressed_logs,
            prev_tree,
            new_tree,
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
        let mut public_values = vec![self.prev_tree.root, self.new_tree.root];
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
        let prev_root_var =
            AllocatedNum::alloc(cs.namespace(|| format!("prev tree root")), || {
                Ok(self.prev_tree.root)
            })?;

        let prev_root_input =
            AllocatedNum::alloc_input(cs.namespace(|| format!("public prev root")), || {
                Ok(self.prev_tree.root)
            })?;
        cs.enforce(
            || "enforce prev tree root == root input",
            |lc| lc + prev_root_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + prev_root_input.get_variable(),
        );

        let new_root_var = AllocatedNum::alloc(cs.namespace(|| format!("new tree root")), || {
            Ok(self.new_tree.root)
        })?;

        let new_root_input =
            AllocatedNum::alloc_input(cs.namespace(|| format!("public new root")), || {
                Ok(self.new_tree.root)
            })?;
        cs.enforce(
            || "enforce new tree root == root input",
            |lc| lc + new_root_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + new_root_input.get_variable(),
        );

        let mut modified_flows = HashMap::<i32, CompressedLog<LinearCombination<E::Scalar>>>::new();
        let mut new_clogs = HashMap::<i32, CompressedLog<E::Scalar>>::new();
        let mut merkle_idx = self.old_compressed_logs.len();

        for (idx, batch) in self.raw_logs.iter().enumerate() {
            // Check that all the new raw logs hash into the public hash values
            let hash_input = AllocatedNum::alloc_input(
                cs.namespace(|| format!("public batch hash {idx}")),
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

            let mut batch_vars = Vec::new();
            for (i, log) in batch.iter().enumerate() {
                let flow_id = E::Scalar::from_u128(log.flow_id as u128);
                let hop_cnt = log.hop_cnt;

                let flow_id_var = AllocatedNum::alloc(
                    cs.namespace(|| format!("batch index {idx}, flow_id {i}")),
                    || Ok(flow_id),
                )
                .unwrap();
                let hop_cnt_var = AllocatedNum::alloc(
                    cs.namespace(|| format!("batch index {idx}, hop_cnt {i}")),
                    || Ok(hop_cnt),
                )
                .unwrap();
                let batch_var = AllocatedNum::alloc(
                    cs.namespace(|| format!("batch index {idx}, var {i}")),
                    || Ok(flow_id * E::Scalar::from_u128(1 << 64) + hop_cnt),
                )
                .unwrap();

                // Keep track of new/modified flows
                match modified_flows.get_mut(&log.flow_id) {
                    Some(clog) => {
                        clog.hop_cnt = clog.hop_cnt.clone() + hop_cnt_var.get_variable();
                        new_clogs.get_mut(&log.flow_id).unwrap().hop_cnt += hop_cnt;
                    }
                    None => {
                        let (idx, old_hop_cnt) = match self.old_compressed_logs.get(&log.flow_id) {
                            Some(clog_old) => (clog_old.merkle_idx, clog_old.hop_cnt),
                            None => {
                                let idx = merkle_idx;
                                merkle_idx += 1;
                                (idx, E::Scalar::ZERO)
                            }
                        };
                        modified_flows.insert(
                            log.flow_id,
                            CompressedLog {
                                merkle_idx: idx,
                                hop_cnt: LinearCombination::from_coeff(
                                    CS::one(),
                                    old_hop_cnt,
                                ) + hop_cnt_var.get_variable(),
                            },
                        );
                        new_clogs.insert(
                            log.flow_id,
                            CompressedLog {
                                merkle_idx: idx,
                                hop_cnt: old_hop_cnt + hop_cnt,
                            },
                        );
                    }
                };

                let lincomb = LinearCombination::from_coeff(
                    flow_id_var.get_variable(),
                    E::Scalar::from_u128(1 << 64),
                ) + hop_cnt_var.get_variable();
                cs.enforce(
                    || format!("enforce batch index {idx}, var {i} == (flow_id << 64) + hop_cnt"),
                    |lc| lc + &lincomb,
                    |lc| lc + CS::one(),
                    |lc| lc + batch_var.get_variable(),
                );

                batch_vars.push(batch_var);
            }

            let log_hash_constants =
                Sponge::<E::Scalar, U<BATCH_SIZE>>::api_constants(Strength::Standard);
            let hashed_batch = hash_circuit(
                &mut cs.namespace(|| format!("batch hash {idx}")),
                batch_vars,
                &log_hash_constants,
            )
            .unwrap();

            let hash_is_equal = Boolean::from(AllocatedBit::alloc(
                cs.namespace(|| format!("batch hash {idx} == hash {idx}")),
                Some(hash_var.get_value() == hashed_batch.get_value()),
            )?);

            Boolean::enforce_equal(
                cs.namespace(|| format!("enforce batch hash {idx}")),
                &hash_is_equal,
                &Boolean::constant(true),
            )?;
        }

        // Membership check for compressed logs
        let mut flow_ids = modified_flows.keys().collect::<Vec<_>>();
        flow_ids.sort();
        for flow_id in flow_ids.iter() {
            let new_clog_lc = modified_flows.get(&flow_id).unwrap();
            let index = new_clog_lc.merkle_idx;
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

            match self.old_compressed_logs.get(&flow_id) {
                Some(clog) => {
                    // Verify membership of old compressed log
                    let siblings_path = self.prev_tree.get_siblings_path(index_bits.clone());

                    let leaf = Leaf {
                        val: vec![clog.hop_cnt],
                        _arity: PhantomData::<U1>,
                    };
                    assert!(
                        siblings_path.verify(index_bits.clone(), &leaf, self.prev_tree.root),
                        "compressed log {index} should verify"
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

                    let leaf_var = leaf
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

                    let is_valid =
                        Boolean::from(path_verify_circuit::<E::Scalar, U1, U2, HEIGHT, _>(
                            &mut cs.namespace(|| format!("valid {index}")),
                            prev_root_var.clone(),
                            leaf_var,
                            index_bits_var.clone(),
                            siblings_var,
                        )?);

                    Boolean::enforce_equal(
                        cs.namespace(|| format!("enforce true {index}")),
                        &is_valid,
                        &Boolean::constant(true),
                    )?;
                }
                None => {}
            }

            // Verify membership of new compressed log
            let new_clog = new_clogs.get(&flow_id).unwrap();

            let siblings_path = self.new_tree.get_siblings_path(index_bits.clone());

            let leaf = Leaf {
                val: vec![new_clog.hop_cnt],
                _arity: PhantomData::<U1>,
            };
            assert!(
                siblings_path.verify(index_bits.clone(), &leaf, self.new_tree.root),
                "new compressed log {index} should verify"
            );

            let siblings_var = siblings_path
                .siblings
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    AllocatedNum::alloc(
                        cs.namespace(|| format!("index {index} new sibling {i}")),
                        || Ok(s),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let new_lc = &new_clog_lc.hop_cnt;
            let leaf_var = leaf
                .val
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    let leaf_var = AllocatedNum::alloc(
                        cs.namespace(|| format!("index {index} new leaf val {i}")),
                        || Ok(s),
                    )
                    .unwrap();
                    cs.enforce(
                        || format!("enforce leaf var {index}, val {i} == updated clog"),
                        |lc| lc + new_lc,
                        |lc| lc + CS::one(),
                        |lc| lc + leaf_var.get_variable(),
                    );
                    leaf_var
                })
                .collect();

            let is_valid = Boolean::from(path_verify_circuit::<E::Scalar, U1, U2, HEIGHT, _>(
                &mut cs.namespace(|| format!("valid new {index}")),
                new_root_var.clone(),
                leaf_var,
                index_bits_var.clone(),
                siblings_var,
            )?);

            Boolean::enforce_equal(
                cs.namespace(|| format!("enforce true new {index}")),
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
