#![allow(non_snake_case)]
use flate2::{Compression, write::ZlibEncoder};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    provider::{Bn256EngineKZG, GrumpkinEngine},
    traits::{Group, snark::RelaxedR1CSSNARKTrait},
};
use spartan2::{provider::T256HyraxEngine, spartan::SpartanSNARK, traits::snark::R1CSSNARKTrait};
use std::time::Instant;
use tracing::{info, info_span};
use tracing_subscriber::EnvFilter;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use std::collections::HashMap;
use std::env;

mod aggregation_circuit;
mod aggregation_circuit_nova;

const HEIGHT: usize = 15;
const BATCH_SIZE: usize = 10;
const BATCHES_PER_STEP: usize = 5;

fn run_spartan_circuit(n_new_logs: u32) {
    let mut rng = SmallRng::seed_from_u64(1);
    type E = T256HyraxEngine;
    // Generate old raw logs
    let old_raw_logs: Vec<_> = (0..1000)
        .map(|id| aggregation_circuit::Log::<u32> {
            id: id,
            flow_id: rng.random_range(0..=500),
            src: rng.random::<u32>(),
            dst: rng.random::<u32>(),
            pred: rng.random::<u32>(),
            packet_size: rng.random_range(0..=65000),
            hop_cnt: rng.random_range(0..=50),
        })
        .collect();

    // Compress old raw logs
    let mut old_compressed_logs: HashMap<_, aggregation_circuit::CompressedLog<_>> = HashMap::new();
    for log in old_raw_logs.into_iter() {
        aggregation_circuit::update_clogs(&mut old_compressed_logs, &log);
    }

    // Generate new raw logs
    let new_raw_logs: Vec<_> = (1000..1000 + n_new_logs)
        .map(|id| aggregation_circuit::Log::<u32> {
            id: id,
            flow_id: rng.random_range(0..=500),
            src: rng.random::<u32>(),
            dst: rng.random::<u32>(),
            pred: rng.random::<u32>(),
            packet_size: rng.random_range(0..=65000),
            hop_cnt: rng.random_range(0..=50),
        })
        .collect();

    // Create circuit
    let circuit = aggregation_circuit::AggregationCircuit::<
        <E as spartan2::traits::Engine>::Scalar,
        _,
        HEIGHT,
        BATCH_SIZE,
    >::new(old_compressed_logs, new_raw_logs);

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

fn run_nova_circuit(n_new_logs: u32) {
    type E1 = Bn256EngineKZG;
    type E2 = GrumpkinEngine;
    type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
    type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
    type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
    type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK
    type Scalar = <<E1 as nova_snark::traits::Engine>::GE as Group>::Scalar;
    type C = aggregation_circuit_nova::AggregationCircuit<
        Scalar,
        u32,
        HEIGHT,
        BATCH_SIZE,
        BATCHES_PER_STEP,
    >;

    let mut rng = SmallRng::seed_from_u64(1);
    // Generate old raw logs
    let old_raw_logs: Vec<_> = (0..1000)
        .map(|id| aggregation_circuit_nova::Log::<u32> {
            id: id,
            flow_id: rng.random_range(0..=500),
            src: rng.random::<u32>(),
            dst: rng.random::<u32>(),
            pred: rng.random::<u32>(),
            packet_size: rng.random_range(0..=65000),
            hop_cnt: rng.random_range(0..=50),
        })
        .collect();

    // Compress old raw logs
    let mut old_compressed_logs: HashMap<_, aggregation_circuit_nova::CompressedLog<_>> =
        HashMap::new();
    for log in old_raw_logs.into_iter() {
        aggregation_circuit_nova::update_clogs(&mut old_compressed_logs, &log);
    }

    // Generate new raw logs
    let new_raw_logs: Vec<_> = (1000..1000 + n_new_logs)
        .map(|id| aggregation_circuit_nova::Log::<u32> {
            id: id,
            flow_id: rng.random_range(0..=500),
            src: rng.random::<u32>(),
            dst: rng.random::<u32>(),
            pred: rng.random::<u32>(),
            packet_size: rng.random_range(0..=65000),
            hop_cnt: rng.random_range(0..=50),
        })
        .collect();

    // Create circuits
    let (circuits, (pub_prev_root, pub_cur_root, pub_hash_chain, pub_n_steps)) =
        C::new_circuits(&old_compressed_logs, new_raw_logs);

    let t0 = Instant::now();
    let pp =
        PublicParams::<E1, E2, _>::setup(&circuits[0], &*S1::ck_floor(), &*S2::ck_floor()).unwrap();
    let pp_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = pp_ms, "public_params");

    let initial_state = &[pub_prev_root, pub_prev_root, Scalar::zero(), Scalar::zero()];

    let n_steps = circuits.len();
    let n_clogs = old_compressed_logs.len();
    let root_span = info_span!("bench", HEIGHT, n_steps, BATCH_SIZE, n_clogs).entered();
    info!(
        "======= height={}, n_steps={}, batches_per_step={}, batch_size={}, n_clogs={} =======",
        HEIGHT, n_steps, BATCHES_PER_STEP, BATCH_SIZE, n_clogs
    );

    // Create recursive SNARK
    let mut recursive_snark: RecursiveSNARK<E1, E2, C> =
        RecursiveSNARK::<E1, E2, C>::new(&pp, &circuits[0], initial_state).unwrap();

    // PROVE
    let t0 = Instant::now();
    for circuit in circuits.iter() {
        let res = recursive_snark.prove_step(&pp, circuit);
        assert!(res.is_ok());
    }
    let prove_ms = t0.elapsed().as_millis();
    let step_ms = prove_ms / (n_steps as u128);
    info!(elapsed_ms = step_ms, "prove_step");
    info!(elapsed_ms = prove_ms, "prove");

    // VERIFY
    let t0 = Instant::now();
    let res = recursive_snark.verify(&pp, n_steps, initial_state);
    let verify_ms = t0.elapsed().as_millis();
    info!(elapsed_ms = verify_ms, "verify");
    assert!(res.is_ok());

    // Create compressed SNARK
    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp).unwrap();

    // COMPRESSED PROVE
    let t0 = Instant::now();
    let res = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
    let compressed_prove_ms = t0.elapsed().as_millis();
    assert!(res.is_ok());
    info!(elapsed_ms = compressed_prove_ms, "compressed_prove");

    let compressed_snark = res.unwrap();
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        &compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )
    .expect("Failed to serialize compressed SNARK");
    let compressed_snark_encoded = encoder.finish().unwrap();
    info!(
        "CompressedSNARK::len {:?} bytes",
        compressed_snark_encoded.len()
    );

    // COMPRESSED VERIFY
    let t0 = Instant::now();
    let res = compressed_snark.verify(&vk, n_steps, initial_state);
    let compressed_verify_ms = t0.elapsed().as_millis();
    assert!(res.is_ok());
    info!(elapsed_ms = compressed_verify_ms, "compressed_verify");

    // Verify final state
    let final_state = res.unwrap();
    match &final_state[..] {
        [a, b, c, d] => {
            assert!(*a == pub_prev_root);
            assert!(*b == pub_cur_root);
            assert!(*c == pub_hash_chain);
            assert!(*d == pub_n_steps);
        }
        _ => panic!("Expected 4 elements"),
    };

    // Summary
    info!(
        concat!(
            "SUMMARY n_steps={}, batches_per_step={}, batch_size={}, ",
            "pub_params={} ms, prove_step={} ms, prove={} ms, verify={} ms, comp_prove={} ms, comp_verify={} ms",
        ),
        n_steps,
        BATCHES_PER_STEP,
        BATCH_SIZE,
        pp_ms,
        step_ms,
        prove_ms,
        verify_ms,
        compressed_prove_ms,
        compressed_verify_ms
    );
    drop(root_span);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n_new_logs: u32 = args[1].parse().unwrap();

    tracing_subscriber::fmt()
        .with_target(false)
        .with_ansi(true) // no bold colour codes
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // This should be a command line switch... I'll implement it later
    if false {
        run_spartan_circuit(n_new_logs);
    } else {
        run_nova_circuit(n_new_logs);
    }
}
