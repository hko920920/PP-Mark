from __future__ import annotations

from pathlib import Path
import json
from typing import Optional

import typer

from .config import ProviderConfig
from .provider import ProviderRuntime, create_provider_config
from .prover import ProverService
from .verifier import VerifierService


app = typer.Typer(help="PP-Mark v0.2 Typer CLI.")


@app.command()
def setup(
    out: Path = typer.Option(
        Path("artifacts/provider_setup.json"),
        "--out",
        "-o",
        help="Path where the provider configuration JSON will be written.",
    ),
    halo2_dir: Path = typer.Option(
        Path("halo2_prover"),
        "--halo2-dir",
        help="Path to the Halo2 prover workspace.",
    ),
    base_image: Path = typer.Option(
        Path("assets/processed/coco/coco_val_street_market.png"),
        "--base-image",
        help="Default base image used for demo generations.",
    ),
) -> None:
    """Stage 1 — create provider configuration + master secret."""
    artifacts = create_provider_config(
        output_path=out,
        halo2_dir=halo2_dir,
        base_image=base_image,
    )
    typer.echo(f"Wrote provider config to {artifacts.config_path}")
    typer.echo(f"Master secret hex: {artifacts.secret_hex}")


@app.command()
def prover(
    config: Path = typer.Option(..., "--config", "-c", help="Provider config JSON."),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt stored in metadata."),
    output_dir: Path = typer.Option(..., "--output-dir", "-o", help="Where to store image/proof outputs."),
    base_image: Optional[Path] = typer.Option(None, "--base-image", help="Override default base image."),
    noise_seed: int = typer.Option(0, "--noise-seed", help="Noise seed recorded in metadata."),
) -> None:
    """Stage 2 — run the prover and emit outputs."""
    provider_cfg = ProviderConfig.load(config)
    runtime = ProviderRuntime(provider_cfg)
    service = ProverService(runtime)
    outputs = service.generate(
        prompt=prompt,
        output_dir=output_dir,
        base_image=base_image,
        noise_seed=noise_seed,
    )
    typer.echo(f"Watermarked image: {outputs.image_path}")
    typer.echo(f"Metadata JSON: {outputs.metadata_path}")
    typer.echo(f"Proof: {outputs.proof_path}")
    typer.echo(f"Public inputs: {outputs.public_inputs_path}")


@app.command()
def verify(
    config: Path = typer.Option(..., "--config", "-c", help="Provider config JSON."),
    image: Path = typer.Option(..., "--image", "-i", help="Image to verify."),
    metadata: Path = typer.Option(..., "--metadata", "-m", help="Metadata JSON describing h/C."),
    proof_dir: Optional[Path] = typer.Option(None, "--proof-dir", help="Directory containing proof.bin/public_inputs.json."),
) -> None:
    """Stage 3 — verify payload + Halo2 proof."""
    provider_cfg = ProviderConfig.load(config)
    runtime = ProviderRuntime(provider_cfg)
    with metadata.open("r", encoding="utf-8") as handle:
        claimed = json.load(handle)
    dir_for_proof = proof_dir or metadata.parent
    service = VerifierService(runtime)
    result = service.verify(image, dir_for_proof, claimed)
    typer.echo(f"Verification status: {result.reason}")
    typer.echo(f"h_claimed={result.h_claimed}")
    typer.echo(f"h_extracted={result.h_extracted}")
    typer.echo(f"C_claimed={result.C_claimed}")
    typer.echo(f"C_extracted={result.C_extracted}")
    typer.echo(f"Extractor variant={result.extractor_variant}")
    typer.echo(f"Halo2 proof ok? {result.proof_ok}")
    if not result.is_valid:
        raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
