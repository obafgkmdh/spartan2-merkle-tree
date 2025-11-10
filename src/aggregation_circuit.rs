#![allow(non_snake_case)]
use bellpepper::gadgets::boolean::field_into_allocated_bits_le;
use bellpepper::gadgets::multipack::pack_bits;
use bellpepper_core::{
    ConstraintSystem, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::traits::{Engine, circuit::SpartanCircuit};
use std::marker::PhantomData;

use generic_array::typenum::{U1, U2};
use merkle_trees::hash::circuit::hash_circuit;
use merkle_trees::hash::vanilla::hash;
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::circuit::path_computed_root;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};

use neptune::Strength;
use neptune::sponge::vanilla::{Sponge, SpongeTrait};

use std::cmp::Ord;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::iter::zip;

#[derive(Clone, Debug)]
pub struct Log<T> {
    pub id: T,
    pub flow_id: T,
    pub src: T,
    pub dst: T,
    pub pred: T,
    pub packet_size: T,
    pub hop_cnt: T,
}

const ENTRY_SIZE: usize = 32;
const LOG_OFFSETS: Log<(usize, usize)> = Log {
    // This defines the packing of Log fields into a Scalar. Each field takes up 32 bits in the resulting Scalar.
    id: (ENTRY_SIZE * 0, ENTRY_SIZE),
    flow_id: (ENTRY_SIZE * 1, ENTRY_SIZE),
    src: (ENTRY_SIZE * 2, ENTRY_SIZE),
    dst: (ENTRY_SIZE * 3, ENTRY_SIZE),
    pred: (ENTRY_SIZE * 4, ENTRY_SIZE),
    packet_size: (ENTRY_SIZE * 5, ENTRY_SIZE),
    hop_cnt: (ENTRY_SIZE * 6, ENTRY_SIZE),
};

impl<T> Log<T> {
    fn fields(&self) -> Vec<&T> {
        // List of all fields that are hashed
        vec![
            &self.id,
            &self.flow_id,
            &self.src,
            &self.dst,
            &self.pred,
            &self.packet_size,
            &self.hop_cnt,
        ]
    }
}

impl<T: Copy + Into<u64>> Log<T> {
    fn to_scalar_log<Scalar: PrimeField + PrimeFieldBits>(&self) -> Log<Scalar> {
        Log {
            id: Scalar::from(self.id.into()),
            flow_id: Scalar::from(self.flow_id.into()),
            src: Scalar::from(self.src.into()),
            dst: Scalar::from(self.dst.into()),
            pred: Scalar::from(self.pred.into()),
            packet_size: Scalar::from(self.packet_size.into()),
            hop_cnt: Scalar::from(self.hop_cnt.into()),
        }
    }
}

pub fn update_clogs<T: Eq + Hash + Copy + Into<u64>, Scalar: PrimeField + PrimeFieldBits>(
    compressed_logs: &mut HashMap<T, CompressedLog<Scalar>>,
    raw_log: &Log<T>,
) -> CompressedLog<Scalar> {
    let scalar_log = raw_log.to_scalar_log::<Scalar>();
    let len = compressed_logs.len();
    match compressed_logs.entry(raw_log.flow_id) {
        Entry::Occupied(clog) => {
            let clog = clog.into_mut();
            clog.hop_cnt += scalar_log.hop_cnt;
            (*clog).clone()
        }
        Entry::Vacant(entry) => {
            let clog = CompressedLog::from_idx_log(len, &scalar_log);
            entry.insert(clog.clone());
            clog
        }
    }
}

impl<Scalar: PrimeField + PrimeFieldBits> Log<Scalar> {
    fn pack(&self) -> Scalar {
        // Combine fields into a single Scalar.
        let mut packed = Scalar::ZERO;
        let TWO = Scalar::from(2);
        for (field, (offset, _nbits)) in zip(self.fields(), LOG_OFFSETS.fields()) {
            packed += *field * TWO.pow(std::slice::from_ref(&(*offset as u64)));
        }
        packed
    }
}

#[derive(Clone, Debug)]
pub struct CompressedLog<Scalar> {
    pub merkle_idx: usize,
    pub id: Scalar,
    pub flow_id: Scalar,
    pub src: Scalar,
    pub dst: Scalar,
    pub packet_size: Scalar,
    pub hop_cnt: Scalar,
}

const CLOG_OFFSETS: CompressedLog<(usize, usize)> = CompressedLog {
    // Each CompressedLog field takes up 32 bits in the resulting Scalar.
    merkle_idx: 0,
    id: (ENTRY_SIZE * 0, ENTRY_SIZE),
    flow_id: (ENTRY_SIZE * 1, ENTRY_SIZE),
    src: (ENTRY_SIZE * 2, ENTRY_SIZE),
    dst: (ENTRY_SIZE * 3, ENTRY_SIZE),
    packet_size: (ENTRY_SIZE * 4, ENTRY_SIZE),
    hop_cnt: (ENTRY_SIZE * 5, ENTRY_SIZE),
};

impl<T> CompressedLog<T> {
    fn fields(&self) -> Vec<&T> {
        // List of all fields that are hashed
        vec![
            &self.id,
            &self.flow_id,
            &self.src,
            &self.dst,
            &self.packet_size,
            &self.hop_cnt,
        ]
    }
}

impl<Scalar: PrimeField + PrimeFieldBits> CompressedLog<Scalar> {
    fn to_leaf(&self) -> Leaf<Scalar, U1> {
        Leaf {
            val: vec![self.pack()],
            _arity: PhantomData::<U1>,
        }
    }

    fn from_idx_log(merkle_idx: usize, log: &Log<Scalar>) -> Self {
        CompressedLog {
            merkle_idx,
            id: log.id,
            flow_id: log.flow_id,
            src: log.src,
            dst: log.dst,
            packet_size: log.packet_size,
            hop_cnt: log.hop_cnt,
        }
    }

    fn pack(&self) -> Scalar {
        // Combine fields into a single Scalar.
        let mut packed = Scalar::ZERO;
        let TWO = Scalar::from(2);
        for (field, (offset, _nbits)) in zip(self.fields(), CLOG_OFFSETS.fields()) {
            packed += *field * TWO.pow(std::slice::from_ref(&(*offset as u64)));
        }
        packed
    }
}

#[derive(Clone, Debug)]
pub struct AggregationCircuit<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> {
    pub raw_logs: Vec<[Log<K>; BATCH_SIZE]>, // New raw logs from routers, batched
    pub hashes: Vec<Scalar>,                 // Hashes of batched logs
    pub old_compressed_logs: HashMap<K, CompressedLog<Scalar>>, // Old compressed logs, one per flow id
    pub prev_tree: MerkleTree<Scalar, HEIGHT, U1, U2>,          // Previous tree
    pub merkle_roots: Vec<Scalar>,                              // Merkle roots for each log update
}

impl<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> AggregationCircuit<Scalar, K, HEIGHT, BATCH_SIZE>
{
    pub fn new(
        old_compressed_logs: HashMap<K, CompressedLog<Scalar>>,
        raw_logs: Vec<Log<K>>,
    ) -> Self {
        // Split the new logs into batches
        let (batched_logs, rem) = raw_logs.as_chunks::<BATCH_SIZE>();
        assert_eq!(
            rem.len(),
            0,
            "Number of inputs ({}) was not a multiple of the batch size ({})",
            raw_logs.len(),
            BATCH_SIZE
        );
        let batched_logs = batched_logs.to_vec();

        // Compute hashes of batched logs
        let log_hash_constants = Sponge::<Scalar, U2>::api_constants(Strength::Standard);
        let hashes = batched_logs
            .iter()
            .map(|batch| {
                let logs = batch.iter().map(|log| log.to_scalar_log().pack()).collect();
                hash(logs, &log_hash_constants)
            })
            .collect();

        // Build prev_tree from old compressed logs
        let mut merkle_leaves: Vec<Option<_>> = Vec::new();
        merkle_leaves.resize(old_compressed_logs.len(), None);
        for clog in old_compressed_logs.values() {
            merkle_leaves[clog.merkle_idx] = Some(clog);
        }

        let merkle_leaves: Vec<_> = merkle_leaves
            .iter()
            .map(|clog| clog.unwrap().to_leaf())
            .collect();

        let prev_tree =
            MerkleTree::from_vec(merkle_leaves.clone(), vanilla_tree::tree::Leaf::default());

        let mut new_tree = prev_tree.clone();
        let mut merkle_roots = vec![new_tree.root];

        // Compress the new logs
        let mut compressed_logs = old_compressed_logs.clone();
        for log in raw_logs.into_iter() {
            let clog = update_clogs(&mut compressed_logs, &log);
            new_tree.insert(
                idx_to_bits(HEIGHT, Scalar::from(clog.merkle_idx as u64)),
                &clog.to_leaf(),
            );
            merkle_roots.push(new_tree.root);
        }

        Self {
            raw_logs: batched_logs,
            hashes,
            old_compressed_logs,
            prev_tree,
            merkle_roots,
        }
    }
}

impl<
    E: Engine,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> SpartanCircuit<E> for AggregationCircuit<E::Scalar, K, HEIGHT, BATCH_SIZE>
{
    fn public_values(&self) -> Result<Vec<<E as Engine>::Scalar>, SynthesisError> {
        // Previous and last tree roots are public, along with public batch hashes
        let mut public_values = vec![
            self.merkle_roots[0].clone(),
            self.merkle_roots.last().unwrap().clone(),
        ];
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
        // Get merkle tree roots
        let merkle_roots_inputs = self
            .merkle_roots
            .iter()
            .enumerate()
            .map(|(i, root)| {
                if i == 0 || i == self.merkle_roots.len() - 1 {
                    AllocatedNum::alloc_input(
                        cs.namespace(|| format!("public merkle root {i}")),
                        || Ok(*root),
                    )
                } else {
                    AllocatedNum::alloc(cs.namespace(|| format!("merkle root {i}")), || Ok(*root))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut cur_tree = self.prev_tree.clone();
        let mut root_idx = 0;
        let mut compressed_logs = self.old_compressed_logs.clone();

        for (batch_idx, batch) in self.raw_logs.iter().enumerate() {
            // Allocate variables
            let mut batch_vars = Vec::new();
            for (log_idx, log) in batch.iter().enumerate() {
                let scalar_log = log.to_scalar_log();
                let packed = scalar_log.pack();

                // This creates as many variables as the field size in bits, but we only use some
                // of them. Not sure if the unused ones get eliminated
                let unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: bit decomposition")),
                    Some(packed),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let packed_log_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: packed log")),
                    &unpacked_bits,
                )?;

                batch_vars.push(packed_log_var);

                // Extract hop count from bit decomposition
                let (hop_cnt_offset, hop_cnt_sz) = LOG_OFFSETS.hop_cnt;
                let hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: hop_cnt")),
                    &unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz],
                )?;

                // Keep track of new/modified flows
                let old_leaf = match compressed_logs.get(&log.flow_id) {
                    Some(clog) => clog.to_leaf(),
                    None => vanilla_tree::tree::Leaf::default(),
                };
                let new_clog = update_clogs(&mut compressed_logs, log);

                let index_bits = idx_to_bits(HEIGHT, E::Scalar::from(new_clog.merkle_idx as u64));

                let index_bits_var = index_bits
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(j, b)| {
                        AllocatedBit::alloc(
                            cs.namespace(|| format!("log {batch_idx}-{log_idx}: index bit {j}")),
                            Some(b),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let siblings_path = cur_tree.get_siblings_path(index_bits.clone());
                let siblings_var = siblings_path
                    .siblings
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(j, s)| {
                        AllocatedNum::alloc(
                            cs.namespace(|| format!("log {batch_idx}-{log_idx}: sibling {j}")),
                            || Ok(s),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Extract old hop count from bit decomposition
                let old_unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| {
                        format!("log {batch_idx}-{log_idx}: old clog bit decomposition")
                    }),
                    Some(old_leaf.val[0]),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let old_packed_clog_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: old packed clog")),
                    &old_unpacked_bits,
                )?;

                let (hop_cnt_offset, hop_cnt_sz) = CLOG_OFFSETS.hop_cnt;
                let old_hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: old clog hop_cnt")),
                    &old_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz],
                )?;

                // Verify membership of old compressed log
                assert!(
                    siblings_path.verify(index_bits.clone(), &old_leaf, cur_tree.root),
                    "log {batch_idx}-{log_idx}: old compressed log should verify"
                );
                let old_computed_root_var = path_computed_root::<E::Scalar, U1, U2, HEIGHT, _>(
                    &mut cs.namespace(|| format!("valid old {batch_idx}-{log_idx}")),
                    vec![old_packed_clog_var.clone()],
                    index_bits_var.clone(),
                    siblings_var.clone(),
                )?;
                cs.enforce(
                    || {
                        format!(
                            "log {batch_idx}-{log_idx}: enforce old computed root == public root"
                        )
                    },
                    |lc| lc + merkle_roots_inputs[root_idx].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + old_computed_root_var.get_variable(),
                );

                // Update cur_tree by inserting new log
                let new_leaf = new_clog.to_leaf();
                cur_tree.insert(index_bits.clone(), &new_leaf);
                root_idx += 1;

                // Extract new hop count from bit decomposition
                let new_unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| {
                        format!("log {batch_idx}-{log_idx}: new clog bit decomposition")
                    }),
                    Some(new_clog.pack()),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let (hop_cnt_offset, hop_cnt_sz) = CLOG_OFFSETS.hop_cnt;
                let new_hop_cnt_bits =
                    &new_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz];
                let new_hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: new clog hop_cnt")),
                    new_hop_cnt_bits,
                )?;
                let new_packed_clog_var = pack_bits(
                    cs.namespace(|| format!("log {batch_idx}-{log_idx}: new packed clog")),
                    &new_unpacked_bits,
                )?;

                // Verify that new hop count is related to the old hop count
                cs.enforce(
                    || format!("log {batch_idx}-{log_idx}: enforce new hop_cnt == old hop_cnt + hop_cnt"),
                    |lc| lc + old_hop_cnt_var.get_variable() + hop_cnt_var.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + new_hop_cnt_var.get_variable(),
                );

                // Reconstruct new leaf by updating hop_cnt of old leaf
                let mut recons_unpacked_bits = old_unpacked_bits.clone();
                recons_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz]
                    .clone_from_slice(new_hop_cnt_bits);

                let recons_packed_clog_var = pack_bits(
                    cs.namespace(|| {
                        format!("log {batch_idx}-{log_idx}: reconstructed packed clog")
                    }),
                    &recons_unpacked_bits,
                )?;

                // Clog is updated, in which case it should equal the reconstructed Clog, or it's
                // new, in which case the old packed Clog should be 0 (default leaf value)
                cs.enforce(
                    || format!("log {batch_idx}-{log_idx}: leaf is updated or new"),
                    |lc| {
                        lc + new_packed_clog_var.get_variable()
                            - recons_packed_clog_var.get_variable()
                    },
                    |lc| lc + old_packed_clog_var.get_variable(),
                    |lc| lc,
                );

                // Verify membership of new compressed log
                assert!(
                    siblings_path.verify(index_bits.clone(), &new_leaf, cur_tree.root),
                    "log {batch_idx}-{log_idx}: new compressed log should verify"
                );
                let new_computed_root_var = path_computed_root::<E::Scalar, U1, U2, HEIGHT, _>(
                    &mut cs.namespace(|| format!("valid new {batch_idx}-{log_idx}")),
                    vec![new_packed_clog_var],
                    index_bits_var.clone(),
                    siblings_var.clone(),
                )?;
                cs.enforce(
                    || {
                        format!(
                            "log {batch_idx}-{log_idx}: enforce new computed root == public root"
                        )
                    },
                    |lc| lc + merkle_roots_inputs[root_idx].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + new_computed_root_var.get_variable(),
                );
            }

            // Check that this batch of logs matches the public hash value
            let hash_input = AllocatedNum::alloc_input(
                cs.namespace(|| format!("public batch hash {batch_idx}")),
                || Ok(self.hashes[batch_idx]),
            )?;

            let log_hash_constants = Sponge::<E::Scalar, U2>::api_constants(Strength::Standard);
            let hashed_batch = hash_circuit(
                &mut cs.namespace(|| format!("batch hash {batch_idx}")),
                batch_vars,
                &log_hash_constants,
            )?;

            cs.enforce(
                || format!("enforce batch hash {batch_idx} == hash {batch_idx}"),
                |lc| lc + hash_input.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + hashed_batch.get_variable(),
            );
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
