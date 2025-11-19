use std::convert::TryInto;
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use halo2_gadgets::poseidon::{
    Hash as PoseidonHash, Pow5Chip, Pow5Config,
    primitives::{self as poseidon, ConstantLength, P128Pow5T3},
};
use halo2_proofs::plonk::Advice;
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    pasta::{EqAffine, Fp, group::ff::PrimeField},
    plonk::{
        Circuit, Column, ConstraintSystem, Error, Instance, SingleVerifier, create_proof,
        keygen_pk, keygen_vk, verify_proof,
    },
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct WatermarkConfig {
    poseidon: Pow5Config<Fp, 3, 2>,
    state_columns: [Column<Advice>; 3],
    public_commitment: Column<Instance>,
}

#[derive(Clone, Default)]
struct WatermarkCircuit {
    secret: Value<Fp>,
    anchor: Value<Fp>,
}

impl Circuit<Fp> for WatermarkCircuit {
    type Config = WatermarkConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            secret: Value::unknown(),
            anchor: Value::unknown(),
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
        let state: [Column<Advice>; 3] = (0..3)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let partial_sbox = meta.advice_column();
        let rc_a: [Column<_>; 3] = (0..3)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let rc_b: [Column<_>; 3] = (0..3)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        meta.enable_constant(rc_b[0]);

        let poseidon = Pow5Chip::configure::<P128Pow5T3>(meta, state, partial_sbox, rc_a, rc_b);

        let instance = meta.instance_column();
        meta.enable_equality(instance);
        for column in state.iter() {
            meta.enable_equality(*column);
        }

        WatermarkConfig {
            poseidon,
            state_columns: state,
            public_commitment: instance,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fp>,
    ) -> Result<(), Error> {
        let chip = Pow5Chip::construct(config.poseidon.clone());
        let message = layouter.assign_region(
            || "load inputs",
            |mut region| {
                let secret_cell = region.assign_advice(
                    || "secret",
                    config.state_columns[0],
                    0,
                    || self.secret,
                )?;
                let anchor_cell = region.assign_advice(
                    || "anchor",
                    config.state_columns[1],
                    0,
                    || self.anchor,
                )?;
                Ok([secret_cell, anchor_cell])
            },
        )?;

        let hasher = PoseidonHash::<_, _, P128Pow5T3, ConstantLength<2>, 3, 2>::init(
            chip,
            layouter.namespace(|| "init hasher"),
        )?;
        let output = hasher.hash(layouter.namespace(|| "hash message"), message)?;

        layouter.constrain_instance(output.cell(), config.public_commitment, 0)?;
        Ok(())
    }
}

#[derive(Clone)]
struct CliOptions {
    secret: u64,
    anchor: u64,
    secret_hex: Option<String>,
    anchor_hex: Option<String>,
    prove: bool,
    k: u32,
    verify_dir: Option<PathBuf>,
}

#[derive(Serialize, Deserialize)]
struct PublicInputsFile {
    inputs: Vec<String>,
    anchor_hex: String,
    poseidon_hex: String,
}

fn fp_to_hex(value: Fp) -> String {
    format!("0x{}", hex::encode(value.to_repr().as_ref()))
}

fn fp_from_hex(s: &str) -> Result<Fp, Box<dyn std::error::Error>> {
    let trimmed = s.trim();
    let hex_str = trimmed.strip_prefix("0x").unwrap_or(trimmed);
    let bytes = hex::decode(hex_str)?;
    let mut repr = <Fp as PrimeField>::Repr::default();
    let repr_bytes = repr.as_mut();
    if bytes.len() != repr_bytes.len() {
        return Err(format!("invalid field element length: {}", bytes.len()).into());
    }
    repr_bytes.copy_from_slice(&bytes);
    Fp::from_repr(repr)
        .into_option()
        .ok_or_else(|| "invalid field element representation".into())
}

fn parse_cli() -> CliOptions {
    let mut opts = CliOptions {
        secret: 12345,
        anchor: 6789,
        secret_hex: None,
        anchor_hex: None,
        prove: false,
        k: 9,
        verify_dir: None,
    };
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--secret" => {
                opts.secret = args
                    .next()
                    .expect("--secret requires a value")
                    .parse()
                    .expect("invalid secret");
            }
            "--secret-hex" => {
                opts.secret_hex = Some(args.next().expect("--secret-hex requires a value"));
            }
            "--anchor" => {
                opts.anchor = args
                    .next()
                    .expect("--anchor requires a value")
                    .parse()
                    .expect("invalid anchor");
            }
            "--anchor-hex" => {
                opts.anchor_hex = Some(args.next().expect("--anchor-hex requires a value"));
            }
            "--prove" => opts.prove = true,
            "--k" => {
                opts.k = args
                    .next()
                    .expect("--k requires a value")
                    .parse()
                    .expect("invalid k");
            }
            "--verify" => {
                let dir = args.next().expect("--verify requires a directory path");
                opts.verify_dir = Some(PathBuf::from(dir));
            }
            _ => {}
        }
    }
    opts
}

fn prove_and_persist(
    circuit: WatermarkCircuit,
    public: Fp,
    opts: CliOptions,
    anchor_hex: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let params: Params<EqAffine> = Params::new(opts.k);
    let vk = keygen_vk(&params, &circuit)?;
    let pk = keygen_pk(&params, vk.clone(), &circuit)?;

    let instances = vec![vec![public]];
    let instance_slices: Vec<&[Fp]> = instances.iter().map(|v| v.as_slice()).collect();
    let instance_refs: Vec<&[&[Fp]]> = vec![instance_slices.as_slice()];

    let mut transcript = Blake2bWrite::<_, EqAffine, Challenge255<_>>::init(vec![]);
    create_proof(
        &params,
        &pk,
        &[circuit.clone()],
        &instance_refs,
        &mut OsRng,
        &mut transcript,
    )?;
    let proof = transcript.finalize();

    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    verify_proof(
        &params,
        pk.get_vk(),
        strategy,
        &instance_refs,
        &mut transcript,
    )?;

    let out_dir = PathBuf::from("outputs");
    fs::create_dir_all(&out_dir)?;

    let mut params_writer = BufWriter::new(File::create(out_dir.join("params.bin"))?);
    params.write(&mut params_writer)?;

    let mut proof_file = File::create(out_dir.join("proof.bin"))?;
    proof_file.write_all(&proof)?;

    let poseidon_hex = fp_to_hex(public);
    let public_json = PublicInputsFile {
        inputs: vec![poseidon_hex.clone()],
        anchor_hex,
        poseidon_hex,
    };
    serde_json::to_writer_pretty(
        File::create(out_dir.join("public_inputs.json"))?,
        &public_json,
    )?;

    println!(
        "Proof artifacts written to {} (params.bin, proof.bin, public_inputs.json)",
        out_dir.display()
    );
    Ok(())
}

fn verify_from_dir(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut params_reader = BufReader::new(File::open(dir.join("params.bin"))?);
    let params = Params::<EqAffine>::read(&mut params_reader)?;

    let vk = keygen_vk(&params, &WatermarkCircuit::default())?;

    let mut proof_bytes = vec![];
    File::open(dir.join("proof.bin"))?.read_to_end(&mut proof_bytes)?;

    let public_file: PublicInputsFile =
        serde_json::from_reader(File::open(dir.join("public_inputs.json"))?)?;
    let public_inputs: Vec<Fp> = public_file
        .inputs
        .iter()
        .map(|hex| fp_from_hex(hex))
        .collect::<Result<_, _>>()?;

    let instances = vec![public_inputs];
    let instance_refs: Vec<&[Fp]> = instances.iter().map(|v| v.as_slice()).collect();
    let instance_refs_refs: Vec<&[&[Fp]]> = vec![instance_refs.as_slice()];

    let strategy = SingleVerifier::new(&params);
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof_bytes[..]);
    verify_proof(&params, &vk, strategy, &instance_refs_refs, &mut transcript)?;
    println!("Verification succeeded for {}", dir.display());
    Ok(())
}

fn main() {
    let opts = parse_cli();

    if let Some(dir) = opts.verify_dir.as_ref() {
        verify_from_dir(dir).expect("verification from files failed");
        return;
    }

    let secret = Fp::from(opts.secret);
    let anchor = Fp::from(opts.anchor);
    let secret = if let Some(hex) = opts.secret_hex.as_ref() {
        fp_from_hex(hex).expect("invalid secret hex")
    } else {
        secret
    };
    let anchor = if let Some(hex) = opts.anchor_hex.as_ref() {
        fp_from_hex(hex).expect("invalid anchor hex")
    } else {
        anchor
    };

    let public =
        poseidon::Hash::<Fp, P128Pow5T3, ConstantLength<2>, 3, 2>::init().hash([secret, anchor]);

    let circuit = WatermarkCircuit {
        secret: Value::known(secret),
        anchor: Value::known(anchor),
    };

    let public_inputs = vec![public];
    let mock_prover = MockProver::run(opts.k, &circuit, vec![public_inputs]).expect("prover run");
    assert_eq!(mock_prover.verify(), Ok(()));
    println!("Poseidon(secret || anchor) = {:?}", public);

    if opts.prove {
        let anchor_hex = fp_to_hex(anchor);
        prove_and_persist(circuit, public, opts, anchor_hex).expect("proof generation failed");
    }
}
