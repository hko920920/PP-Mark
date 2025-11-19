# PP-Mark v2.1 — Semantic Anchor & Dual Watermark Prototype

`pp_mark_v2.1`는 PP-Mark v2.0 실험 체계를 기반으로 **품질 우선 LiteVAE(L0) + Stable Signature(L1) + ROBIN(L2) + Halo2 증명(L3)** 아키텍처를 재현하기 위한 작업 공간입니다. 레거시 v0.2 도구도 함께 보존되어 있으나, 최신 워크플로는 아래 v2.1 CLI와 스크립트를 기준으로 합니다.

## 핵심 구성

- **Semantic Anchor** `h = H(prompt‖seed‖model_id‖timestamp‖parent_hash?)`
- **Poseidon Payload** `C = Poseidon(k‖h)` → L1/L2 삽입 + L3 공개 입력
- **L1**: LiteVAE 디코더 미세조정 + 타일링 추출기(크롭/압축 내성)
- **L2**: ROBIN 중간 단계 패턴 주입/은닉 — `MidTrajectoryInjector`가 `t_injection` 시점의 블러/노이즈(latent)를 시뮬레이션하여 패턴을 심고, 가이던스 스케일로 원본과 다시 혼합합니다. `l2_trace`(t, guidance, hiding prompt, payload checksum)가 메타데이터에 포함됩니다.
- **L3**: Halo2 (Poseidon v1 Pow5 기준) ZKP + ECDSA 기반 메타데이터 무결성 — 현재 회로는 Pow5 칩을 사용하고, 향후 Poseidon2 칩을 구현해 교체할 계획입니다. 기본 서명 스킴은 `signature_scheme="ecdsa-p256"`이며, 필요 시 HMAC-SHA256 등 다른 스킴으로 교체할 수 있습니다.

배경과 세부 설계는 `docs/v21_summary.md`, `docs/v21_attack_plan.md`, `docs/v21_attack_steps.md`, `docs/seed_guidance_v21.md`에서 확인할 수 있습니다.

## 빠른 시작

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# 선택: 학습/추가 기능
pip install -e .[train]
```

### Typer CLI (v2.1)

```bash
# 1) Provider 설정
ppmark-v21 setup \
  --out artifacts/provider_setup_v21.json \
  --halo2-dir halo2_prover \
  --base-image assets/processed/coco/coco_val_street_market.png

# 2) Prover
ppmark-v21 prover \
  --config artifacts/provider_setup_v21.json \
  --prompt "PP-Mark LiteVAE demo" \
  --seed 9f3b6c2f4e7d1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d5e4f3a2b1c0d9e8f7a6 \
  --model-id demo-model \
  --timestamp 2025-11-18T12:10:30Z \
  --output-dir results/v21_demo

# 3) Verifier
ppmark-v21 verify \
  --config artifacts/provider_setup_v21.json \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json
```

### 원샷 데모

```bash
python scripts/run_full_demo_v21.py \
  --prompt "pp-mark v2.1 demo" \
  --seed 9f3...f7a6 \
  --model-id demo-model \
  --timestamp 2025-11-18T12:10:30Z \
  --output-dir results/v21_demo \
  --skip-verify   # 필요 시
```

### C2PA 임베딩

```bash
python scripts/embed_c2pa_v21.py \
  --image results/v21_demo/watermarked.png \
  --metadata results/v21_demo/metadata.json \
  --out results/v21_demo/watermarked.c2pa.png \
  --unsigned
```

스크립트는 `c2patool`이 설치되어 있으면 자동으로 호출하고, 없으면 메타데이터를 포함한 사이드카를 남깁니다.

## 학습/튜닝 스크립트

아직 v0.2 경로(`ppmark_v02`)를 사용하지만, LiteVAE/패턴/추출기 튜닝을 위한 스크립트는 그대로 활용 가능합니다.

```bash
python scripts/train_extractor.py --config artifacts/provider_setup.json \
  --data-root assets/processed/coco --output runs/extractor/latest.pt

python scripts/tune_litevae.py --config artifacts/provider_setup.json \
  --data-root assets/processed/coco --output runs/litevae_strength.json

python scripts/train_pattern.py --config artifacts/provider_setup.json \
  --data-root assets/processed/coco --output runs/pattern_selection.json
```

훈련 산출물은 `ProviderConfig`의 `extractor_weights`, `litevae_weights`, `pattern_weights` 필드에 지정하여 런타임이 로드하도록 합니다.

## 공격/평가 스크립트

- `scripts/run_attack_matrix_v21.py`: JPEG/블러/크롭/스케일/플립 + 옵션(`--stress`) 회전·엘라스틱
- `scripts/run_regen_surrogate_v21.py`: CPU 기반 Zhao surrogate
- `scripts/run_regen_attack_v21.py`: GPU+WatermarkAttacker
- `scripts/run_mueller_attack.py`: Müller 위조 실행기(외부 레포 필요)
- `scripts/eval_image_quality.py`: 원본 대비 워터마크/공격 이미지를 비교(PSNR·SSIM·LPIPS)하는 품질 평가 도구

## 레거시 v0.2

기존 v0.2 흐름(`ppmark_v02` CLI, `scripts/run_full_demo.py`, `docs/architecture.md`)도 그대로 남아 있습니다. 과거 실험을 재현해야 할 때만 사용하고, 새로운 워크로드는 v2.1 경로를 따라주세요.
