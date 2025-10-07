#![allow(non_snake_case)]
use bellpepper_core::{
    ConstraintSystem, SynthesisError,
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
};
use ff::{PrimeField, PrimeFieldBits};
use spartan2::{
    provider::T256HyraxEngine,
    spartan::SpartanSNARK,
    traits::{Engine, circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::{marker::PhantomData, time::Instant};
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

use generic_array::typenum::{U1, U2};
use merkle_trees::vanilla_tree;
use merkle_trees::vanilla_tree::circuit::path_verify_circuit;
use merkle_trees::vanilla_tree::tree::{Leaf, MerkleTree, idx_to_bits};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

type E = T256HyraxEngine;
const HEIGHT: usize = 20;

#[derive(Clone, Debug)]
struct MerkleTreeCircuit<Scalar: PrimeField + PrimeFieldBits> {
    inputs: Vec<Scalar>,
    tree: MerkleTree<Scalar, HEIGHT, U1, U2>,
}

impl<Scalar: PrimeField + PrimeFieldBits> MerkleTreeCircuit<Scalar> {
    fn new(inputs: Vec<Scalar>) -> Self {
        let mut tree = MerkleTree::new(vanilla_tree::tree::Leaf::default());

        for i in 0..inputs.len() {
            let idx = Scalar::from(i as u64);
            let idx_in_bits = idx_to_bits(HEIGHT, idx);
            let val = Leaf {
                val: vec![inputs[i]],
                _arity: PhantomData::<U1>,
            };
            tree.insert(idx_in_bits.clone(), &val);
        }

        Self { inputs, tree }
    }
}

impl<E: Engine> SpartanCircuit<E> for MerkleTreeCircuit<E::Scalar> {
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
        let index = 5;
        let index_scalar = E::Scalar::from(index as u64);

        let index_bits = idx_to_bits(HEIGHT, index_scalar);

        let index_bits_var = index_bits
            .clone()
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), Some(b))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let root_var =
            AllocatedNum::alloc(cs.namespace(|| format!("tree root")), || Ok(self.tree.root))?;

        let input = AllocatedNum::alloc_input(cs.namespace(|| format!("public input")), || Ok(self.tree.root))?;
        cs.enforce(
            || "enforce input == root",
            |lc| lc + root_var.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + input.get_variable(),
        );

        let siblings_path = self.tree.get_siblings_path(index_bits.clone());

        let val = self.inputs[index];
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
            .map(|(i, s)| AllocatedNum::alloc(cs.namespace(|| format!("sibling {i}")), || Ok(s)))
            .collect::<Result<Vec<_>, _>>()?;

        let input_var = leaf.val
            .into_iter()
            .enumerate()
            .map(|(i, s)| AllocatedNum::alloc(cs.namespace(|| format!("leaf val {i}")), || Ok(s)))
            .collect::<Result<Vec<_>, _>>()?;

        let is_valid = Boolean::from(path_verify_circuit::<E::Scalar, U1, U2, HEIGHT, _>(
            cs,
            root_var,
            input_var,
            index_bits_var,
            siblings_var,
        )?);

        Boolean::enforce_equal(
            cs.namespace(|| format!("enforce true")),
            &is_valid,
            &Boolean::constant(true),
        )?;

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

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(true) // no bold colour codes
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let mut rng = SmallRng::seed_from_u64(1);
    let input = (0..(1 << 15))
        .map(|_| <E as Engine>::Scalar::from(rng.random::<u64>()))
        .collect();

    let circuit = MerkleTreeCircuit::<<E as Engine>::Scalar>::new(input);

    let n_inputs = circuit.inputs.len();
    let root_span = info_span!("bench", n_inputs).entered();
    info!("======= n_inputs={} =======", n_inputs);

    // SETUP
    let t0 = Instant::now();
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms, "setup");

    // PREPARE
    let t0 = Instant::now();
    let prep_snark =
        SpartanSNARK::<E>::prep_prove(&pk, circuit.clone(), false).expect("prep_prove failed");
    let prep_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prep_ms, "prep_prove");

    // PROVE
    let t0 = Instant::now();
    let proof =
        SpartanSNARK::<E>::prove(&pk, circuit.clone(), &prep_snark, false).expect("prove failed");
    let prove_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = prove_ms, "prove");

    // VERIFY
    let t0 = Instant::now();
    proof.verify(&vk).expect("verify errored");
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");

    // Summary
    info!(
        "SUMMARY n_inputs={}, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
        n_inputs, setup_ms, prep_ms, prove_ms, verify_ms
    );
    drop(root_span);
}
