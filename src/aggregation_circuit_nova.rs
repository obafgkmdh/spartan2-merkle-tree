#![allow(non_snake_case)]
use ff::{PrimeField, PrimeFieldBits};
use nova_snark::frontend::{
    AllocatedBit, Boolean, ConstraintSystem, Elt, PoseidonConstants, SpongeCircuit, SynthesisError,
    gadgets::{
        boolean::field_into_allocated_bits_le,
        poseidon::{IOPattern, Simplex, Sponge, SpongeAPI, SpongeOp, SpongeTrait, Strength},
    },
    num::{AllocatedNum, Num},
};
use nova_snark::traits::circuit::StepCircuit;
use std::marker::PhantomData;

use generic_array::typenum::{U1, U2};
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};

use std::cmp::Ord;
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::Hash;
use std::iter::zip;

// Annoyingly, nova_snark::frontend::num::AllocatedNum and bellpepper::gadgets::num::AllocatedNum
// seem to be incompatible. Hence we reproduce various functions from other crates below.
pub fn hash_U2<F: PrimeField>(input: Vec<F>, p: &PoseidonConstants<F, U2>) -> F {
    let parameter = IOPattern(vec![
        SpongeOp::Absorb(input.len() as u32),
        SpongeOp::Squeeze(1),
    ]);
    let mut sponge = Sponge::new_with_constants(p, Simplex);
    let acc = &mut ();

    sponge.start(parameter, None, acc);
    SpongeAPI::absorb(&mut sponge, input.len() as u32, &input, acc);

    let output = SpongeAPI::squeeze(&mut sponge, 1, acc);
    assert_eq!(output.len(), 1);

    sponge.finish(acc).unwrap();

    output[0]
}
pub fn hash_circuit_U1<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    input: Vec<AllocatedNum<F>>,
    p: &PoseidonConstants<F, U1>,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut sponge = SpongeCircuit::<F, U1, _>::new_with_constants(p, Simplex);

    let mut ns = cs.namespace(|| "ns");

    let val_var: Vec<Elt<F>> = input
        .clone()
        .into_iter()
        .map(|s| Elt::Allocated(s))
        .collect();
    assert_eq!(val_var.len(), input.len());

    let acc = &mut ns;
    let parameter = IOPattern(vec![
        SpongeOp::Absorb(input.len() as u32),
        SpongeOp::Squeeze(1),
    ]);

    sponge.start(parameter, None, acc);

    SpongeAPI::absorb(&mut sponge, input.len() as u32, val_var.as_slice(), acc);

    let calc_node = SpongeAPI::squeeze(&mut sponge, 1, acc);

    assert_eq!(calc_node.len(), 1);

    sponge.finish(acc).unwrap();

    calc_node[0].ensure_allocated(acc, true)
}
// This crate is honestly just incredibly toxic. nova_snark doesn't expose the internal Arity trait
// so we have to write separate implementations for U1 and U2
pub fn hash_circuit_U2<F: PrimeField, CS: ConstraintSystem<F>>(
    cs: &mut CS,
    input: Vec<AllocatedNum<F>>,
    p: &PoseidonConstants<F, U2>,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut sponge = SpongeCircuit::<F, U2, _>::new_with_constants(p, Simplex);

    let mut ns = cs.namespace(|| "ns");

    let val_var: Vec<Elt<F>> = input
        .clone()
        .into_iter()
        .map(|s| Elt::Allocated(s))
        .collect();
    assert_eq!(val_var.len(), input.len());

    let acc = &mut ns;
    let parameter = IOPattern(vec![
        SpongeOp::Absorb(input.len() as u32),
        SpongeOp::Squeeze(1),
    ]);

    sponge.start(parameter, None, acc);

    SpongeAPI::absorb(&mut sponge, input.len() as u32, val_var.as_slice(), acc);

    let calc_node = SpongeAPI::squeeze(&mut sponge, 1, acc);

    assert_eq!(calc_node.len(), 1);

    sponge.finish(acc).unwrap();

    calc_node[0].ensure_allocated(acc, true)
}
pub fn pack_bits<Scalar, CS>(
    mut cs: CS,
    bits: &[Boolean],
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
{
    let mut num = Num::<Scalar>::zero();
    let mut coeff = Scalar::ONE;
    for bit in bits.iter().take(Scalar::CAPACITY as usize) {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
    }

    let alloc_num = AllocatedNum::alloc(cs.namespace(|| "input"), || {
        num.get_value().ok_or(SynthesisError::AssignmentMissing)
    })?;

    // num * 1 = input
    cs.enforce(
        || "packing constraint",
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + alloc_num.get_variable(),
    );

    Ok(alloc_num)
}

pub fn path_computed_root<
    F: PrimeField + PrimeFieldBits,
    const N: usize,
    CS: ConstraintSystem<F>,
>(
    cs: &mut CS,
    val_var: Vec<AllocatedNum<F>>,
    mut idx_var: Vec<AllocatedBit>,
    siblings_var: Vec<AllocatedNum<F>>,
) -> Result<AllocatedNum<F>, SynthesisError> {
    let node_hash_params = Sponge::<F, U2>::api_constants(Strength::Standard);
    let leaf_hash_params = Sponge::<F, U1>::api_constants(Strength::Standard);
    let mut cur_hash_var = hash_circuit_U1(
        &mut cs.namespace(|| "hash num -1 :"),
        val_var,
        &leaf_hash_params,
    )
    .unwrap();

    idx_var.reverse(); // Going from leaf to root

    for (i, sibling) in siblings_var.clone().into_iter().rev().enumerate() {
        let (lc, rc) = AllocatedNum::conditionally_reverse(
            &mut cs.namespace(|| format!("rev num {} :", i)),
            &cur_hash_var,
            &sibling,
            &Boolean::from(idx_var[i].clone()),
        )
        .unwrap();
        cur_hash_var = hash_circuit_U2(
            &mut cs.namespace(|| format!("hash num {} :", i)),
            vec![lc, rc],
            &node_hash_params,
        )
        .unwrap();
    }

    Ok(cur_hash_var)
}
// end reproduced functions

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
pub struct Batch<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const BATCH_SIZE: usize,
> {
    pub raw_logs: [Log<K>; BATCH_SIZE], // Single batch of raw logs from router
    pub idxs: [usize; BATCH_SIZE],
    pub siblings: [Vec<Scalar>; BATCH_SIZE],
    pub old_clogs: [Option<CompressedLog<Scalar>>; BATCH_SIZE], // Old compressed logs
}

#[derive(Clone, Debug)]
pub struct AggregationCircuit<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> {
    pub batches: Vec<Batch<Scalar, K, BATCH_SIZE>>,
    pub step_count: Scalar,
}

impl<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> AggregationCircuit<Scalar, K, HEIGHT, BATCH_SIZE>
{
    // Outputs a vector of circuits, and also the expected (public) final state
    pub fn new_circuits(
        old_compressed_logs: &HashMap<K, CompressedLog<Scalar>>,
        raw_logs: Vec<Log<K>>,
        batches_per_step: usize,
    ) -> (Vec<Self>, (Scalar, Scalar, Scalar, Scalar)) {
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

        let step_logs_iter = batched_logs.chunks_exact(batches_per_step);
        assert_eq!(
            step_logs_iter.remainder().len(),
            0,
            "Number of batches ({}) was not a multiple of batches_per_step ({})",
            batched_logs.len(),
            batches_per_step
        );

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

        let prev_tree: MerkleTree<Scalar, HEIGHT, U1, U2> =
            MerkleTree::from_vec(merkle_leaves.clone(), vanilla_tree::tree::Leaf::default());

        // Compress the new logs
        let mut compressed_logs = old_compressed_logs.clone();

        let mut new_tree = prev_tree.clone();

        let log_hash_constants = Sponge::<Scalar, U2>::api_constants(Strength::Standard);
        let mut hash_chain = Scalar::ZERO;

        // Create circuits
        let circuits: Vec<_> = step_logs_iter
            .enumerate()
            .map(|(step, batches)| {
                let circuit_batches = batches
                    .iter()
                    .map(|batch| {
                        let mut idxs: Vec<usize> = Vec::new();
                        let mut siblings: Vec<Vec<Scalar>> = Vec::new();
                        let mut old_clogs: Vec<Option<CompressedLog<Scalar>>> = Vec::new();

                        for log in batch.into_iter() {
                            let old_clog = compressed_logs.get(&log.flow_id).cloned();
                            let clog = update_clogs(&mut compressed_logs, &log);
                            let idx = clog.merkle_idx;
                            let idx_bits = idx_to_bits(HEIGHT, Scalar::from(idx as u64));
                            new_tree.insert(idx_bits.clone(), &clog.to_leaf());
                            let siblings_path = new_tree.get_siblings_path(idx_bits);
                            siblings.push(siblings_path.siblings);
                            idxs.push(idx);
                            old_clogs.push(old_clog);
                        }

                        let scalar_logs =
                            batch.iter().map(|log| log.to_scalar_log().pack()).collect();
                        let batch_hash = hash_U2(scalar_logs, &log_hash_constants);
                        hash_chain = hash_U2(vec![hash_chain, batch_hash], &log_hash_constants);

                        Batch {
                            raw_logs: batch.clone(),
                            idxs: idxs.try_into().unwrap(),
                            siblings: siblings.try_into().unwrap(),
                            old_clogs: old_clogs.try_into().unwrap(),
                        }
                    })
                    .collect();
                Self {
                    batches: circuit_batches,
                    step_count: Scalar::from(step as u64),
                }
            })
            .collect::<Vec<_>>();
        let n_steps = circuits.len();
        (
            circuits,
            (
                prev_tree.root,
                new_tree.root,
                hash_chain,
                Scalar::from(n_steps as u64),
            ),
        )
    }
}

impl<
    Scalar: PrimeField + PrimeFieldBits,
    K: Ord + Hash + Copy + Into<u64> + Sync + Send,
    const HEIGHT: usize,
    const BATCH_SIZE: usize,
> StepCircuit<Scalar> for AggregationCircuit<Scalar, K, HEIGHT, BATCH_SIZE>
{
    fn arity(&self) -> usize {
        4
    }

    fn synthesize<CS: ConstraintSystem<Scalar>>(
        &self,
        cs: &mut CS,
        z_in: &[AllocatedNum<Scalar>],
    ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
        let [initial_root, prev_root, hash_chain, step_count] = match z_in {
            [a, b, c, d] => [a, b, c, d],
            _ => panic!("Expected 4 elements"),
        };

        let step_count_inv = self.step_count.invert().unwrap_or(Scalar::ONE);
        let step_count_inv_var =
            AllocatedNum::alloc(cs.namespace(|| "step count inverse"), || Ok(step_count_inv))?;
        step_count_inv_var.assert_nonzero(cs.namespace(|| "step count inverse is invertible"))?;

        let step_count_invertible = step_count.mul(
            cs.namespace(|| "1 if step_count invertible, else 0"),
            &step_count_inv_var,
        )?;

        cs.enforce(
            || "step_count_invertible is a boolean",
            |lc| lc + step_count_invertible.get_variable(),
            |lc| lc + step_count_invertible.get_variable() - CS::one(),
            |lc| lc,
        );

        cs.enforce(
            || "step_count != 0 OR initial_root == prev_root",
            |lc| lc + prev_root.get_variable() - initial_root.get_variable(),
            |lc| lc + step_count_invertible.get_variable() - CS::one(),
            |lc| lc,
        );

        cs.enforce(
            || "step_count != 0 OR hash_chain == 0",
            |lc| lc + hash_chain.get_variable(),
            |lc| lc + step_count_invertible.get_variable() - CS::one(),
            |lc| lc,
        );

        let unpacked_step_bits: Vec<_> = field_into_allocated_bits_le(
            cs.namespace(|| format!("step count bit decomposition")),
            Some(self.step_count),
        )?
        .iter()
        .map(|bit| Boolean::from(bit.clone()))
        .collect();

        let packed_step_var = pack_bits(
            cs.namespace(|| format!("step count packed, 128 bits")),
            &unpacked_step_bits[..128],
        )?;

        cs.enforce(
            || "step_count < 2^128",
            |lc| lc + packed_step_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + step_count.get_variable(),
        );

        let mut cur_root = prev_root.clone();
        let mut new_hash_chain = hash_chain.clone();

        for batch in self.batches.iter() {
            let mut batch_vars = Vec::new();
            for log_idx in 0..BATCH_SIZE {
                let idx = batch.idxs[log_idx];
                let siblings = &batch.siblings[log_idx];
                let old_clog = &batch.old_clogs[log_idx];

                let scalar_log = batch.raw_logs[log_idx].to_scalar_log();
                let packed = scalar_log.pack();

                // This creates as many variables as the field size in bits, but we only use some
                // of them. Not sure if the unused ones get eliminated
                let unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| format!("log {log_idx}: bit decomposition")),
                    Some(packed),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let packed_log_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: packed log")),
                    &unpacked_bits,
                )?;

                batch_vars.push(packed_log_var);

                // Extract hop count from bit decomposition
                let (hop_cnt_offset, hop_cnt_sz) = LOG_OFFSETS.hop_cnt;
                let hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: hop_cnt")),
                    &unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz],
                )?;

                // Keep track of new/modified flows
                let (old_leaf, new_clog) = match old_clog {
                    Some(clog) => {
                        let mut new_clog = clog.clone();
                        new_clog.hop_cnt += scalar_log.hop_cnt;
                        (clog.to_leaf(), new_clog)
                    }
                    None => (
                        vanilla_tree::tree::Leaf::default(),
                        CompressedLog::from_idx_log(idx, &scalar_log),
                    ),
                };

                let index_bits = idx_to_bits(HEIGHT, Scalar::from(idx as u64));

                let index_bits_var = index_bits
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(j, b)| {
                        AllocatedBit::alloc(
                            cs.namespace(|| format!("log {log_idx}: index bit {j}")),
                            Some(b),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let siblings_var = siblings
                    .clone()
                    .into_iter()
                    .enumerate()
                    .map(|(j, s)| {
                        AllocatedNum::alloc(
                            cs.namespace(|| format!("log {log_idx}: sibling {j}")),
                            || Ok(s),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Extract old hop count from bit decomposition
                let old_unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| format!("log {log_idx}: old clog bit decomposition")),
                    Some(old_leaf.val[0]),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let old_packed_clog_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: old packed clog")),
                    &old_unpacked_bits,
                )?;

                let (hop_cnt_offset, hop_cnt_sz) = CLOG_OFFSETS.hop_cnt;
                let old_hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: old clog hop_cnt")),
                    &old_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz],
                )?;

                // Verify membership of old compressed log
                let old_computed_root_var = path_computed_root::<Scalar, HEIGHT, _>(
                    &mut cs.namespace(|| format!("valid old {log_idx}")),
                    vec![old_packed_clog_var.clone()],
                    index_bits_var.clone(),
                    siblings_var.clone(),
                )?;
                cs.enforce(
                    || format!("log {log_idx}: enforce current root == old computed root"),
                    |lc| lc + cur_root.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + old_computed_root_var.get_variable(),
                );

                // Extract new hop count from bit decomposition
                let new_unpacked_bits: Vec<_> = field_into_allocated_bits_le(
                    cs.namespace(|| format!("log {log_idx}: new clog bit decomposition")),
                    Some(new_clog.pack()),
                )?
                .iter()
                .map(|bit| Boolean::from(bit.clone()))
                .collect();

                let (hop_cnt_offset, hop_cnt_sz) = CLOG_OFFSETS.hop_cnt;
                let new_hop_cnt_bits =
                    &new_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz];
                let new_hop_cnt_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: new clog hop_cnt")),
                    new_hop_cnt_bits,
                )?;
                let new_packed_clog_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: new packed clog")),
                    &new_unpacked_bits,
                )?;

                // Verify that new hop count is related to the old hop count
                cs.enforce(
                    || format!("log {log_idx}: enforce new hop_cnt == old hop_cnt + hop_cnt"),
                    |lc| lc + old_hop_cnt_var.get_variable() + hop_cnt_var.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + new_hop_cnt_var.get_variable(),
                );

                // Reconstruct new leaf by updating hop_cnt of old leaf
                let mut recons_unpacked_bits = old_unpacked_bits.clone();
                recons_unpacked_bits[hop_cnt_offset..hop_cnt_offset + hop_cnt_sz]
                    .clone_from_slice(new_hop_cnt_bits);

                let recons_packed_clog_var = pack_bits(
                    cs.namespace(|| format!("log {log_idx}: reconstructed packed clog")),
                    &recons_unpacked_bits,
                )?;

                // Clog is updated, in which case it should equal the reconstructed Clog, or it's
                // new, in which case the old packed Clog should be 0 (default leaf value)
                cs.enforce(
                    || format!("log {log_idx}: leaf is updated or new"),
                    |lc| {
                        lc + new_packed_clog_var.get_variable()
                            - recons_packed_clog_var.get_variable()
                    },
                    |lc| lc + old_packed_clog_var.get_variable(),
                    |lc| lc,
                );

                // Compute root for new compressed log
                let new_computed_root_var = path_computed_root::<Scalar, HEIGHT, _>(
                    &mut cs.namespace(|| format!("log {log_idx}: new computed root")),
                    vec![new_packed_clog_var],
                    index_bits_var.clone(),
                    siblings_var.clone(),
                )?;
                // Update current root
                cur_root = new_computed_root_var.clone();
            }

            // Compute hash for this batch of logs
            let log_hash_constants = Sponge::<Scalar, U2>::api_constants(Strength::Standard);
            let hashed_batch = hash_circuit_U2(
                &mut cs.namespace(|| format!("batch hash")),
                batch_vars,
                &log_hash_constants,
            )?;

            new_hash_chain = hash_circuit_U2(
                &mut cs.namespace(|| format!("hash chain")),
                vec![new_hash_chain.clone(), hashed_batch],
                &log_hash_constants,
            )?;
        }

        let new_step_count = AllocatedNum::alloc(cs.namespace(|| "step counter"), || {
            Ok(self.step_count + Scalar::ONE)
        })?;

        cs.enforce(
            || format!("enforce new step count == old step count + 1"),
            |lc| lc + step_count.get_variable() + CS::one(),
            |lc| lc + CS::one(),
            |lc| lc + new_step_count.get_variable(),
        );

        Ok(vec![
            initial_root.clone(),
            cur_root.clone(),
            new_hash_chain,
            new_step_count,
        ])
    }
}
