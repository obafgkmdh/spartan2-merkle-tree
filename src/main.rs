#![allow(non_snake_case)]
use spartan2::{
    provider::T256HyraxEngine,
    spartan::SpartanSNARK,
    traits::{Engine, snark::R1CSSNARKTrait},
};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

mod aggregation_circuit;

type E = T256HyraxEngine;
const HEIGHT: usize = 15;
const BATCH_SIZE: usize = 8;

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(true) // no bold colour codes
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let mut rng = SmallRng::seed_from_u64(1);
    let raw_logs: Vec<_> = (0..1000)
        .map(|_| aggregation_circuit::Log {
            flow_id: rng.random_range(0..=500),
            hop_cnt: <E as Engine>::Scalar::from(rng.random_range(0..=50)),
        })
        .collect();

    // Create circuit
    let circuit =
        aggregation_circuit::AggregationCircuit::<<E as Engine>::Scalar, HEIGHT, BATCH_SIZE>::new(
            raw_logs, 20,
        );

    let n_new_batches = circuit.raw_logs.len();
    let n_clogs = circuit.old_compressed_logs.len();
    let root_span = info_span!("bench", HEIGHT, n_new_batches, BATCH_SIZE, n_clogs).entered();
    info!(
        "======= height={}, n_new_batches={}, batch_size={}, n_clogs={} =======",
        HEIGHT, n_new_batches, BATCH_SIZE, n_clogs
    );

    // SETUP
    let t0 = Instant::now();
    let (pk, vk) = SpartanSNARK::<E>::setup(circuit.clone()).expect("setup failed");
    let setup_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = setup_ms, "setup");
    info!("Constraint count is: {}", pk.sizes()[0]);

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
        "SUMMARY n_new_batches={}, batch_size={}, setup={} ms, prep_prove={} ms, prove={} ms, verify={} ms",
        n_new_batches, BATCH_SIZE, setup_ms, prep_ms, prove_ms, verify_ms
    );
    drop(root_span);
}
