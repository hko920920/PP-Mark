# PP-Mark Prototype Workspace

PP-Mark 실험을 재현하기 위한 워터마킹 CLI, 로컬 공격 하네스, Halo2 증명기, 그리고 Müller·Diffusion 재생성 공격용 GPU 워크로드 래퍼가 포함돼 있습니다. 아래 순서대로 실행하면 로컬에서 워터마크를 만들고 검증한 뒤, H100 서버에서 바로 무거운 공격을 돌릴 수 있습니다.

---

## 1. 로컬 파이썬 환경

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

샘플 이미지 준비 + 워터마크 삽입 + 기본 공격 평가까지 한 번에 돌리고 싶다면:

```bash
./scripts/run_full_pipeline.sh "PP-Mark Rocks"
```

수동 실행시에는 아래 순서를 따르면 됩니다.

### 1.1 샘플 에셋 (`scripts/prepare_samples.py`)
- COCO train/val 3장과 CelebAMask-HQ demo 이미지를 다운로드해 `assets/raw`/`assets/processed`에 정규화 저장하고, `assets/metadata.json`을 생성합니다.
  ```bash
  python3 scripts/prepare_samples.py
  ```
- 기본 입력 이미지는 `assets/processed/coco/coco_val_street_market.png`.

### 1.2 워터마크 CLI (`dct_watermark.py`)
- Pairwise mid-frequency DCT 임베딩 + RS(96) + repeat-factor 11 + 영역별 셔플 구조.
- 사용 예시:
  ```bash
  python3 dct_watermark.py embed --input assets/processed/coco/coco_val_street_market.png \
      --output results/watermarked.png --message "PP-Mark Rocks"
  python3 dct_watermark.py extract --input results/watermarked.png
  ```

### 1.3 공격 하네스 (`attack_matrix.py`)
- clean / JPEG q50 / Gaussian σ=12 / centered crop (ratio 0.8) 공격을 돌리고 성공 여부를 `results/attack_matrix.json`에 기록합니다.
  ```bash
  python3 attack_matrix.py --input assets/processed/coco/coco_val_street_market.png \
      --workdir results --message "PP-Mark Rocks"
  ```
- Müller attack 항목은 GPU 파이프라인이 필요한 관계로 placeholder 상태지만, 아래 2장에서 래퍼 스크립트로 바로 실행할 수 있습니다.

### 1.4 Halo2 Poseidon 회로 (`halo2_prover`)
```bash
cd halo2_prover
cargo run -- --secret 123 --anchor 456                 # MockProver
cargo run -- --prove --secret 123 --anchor 456 --k 9   # Proof 생성 + 검증
cargo run -- --verify outputs                          # 저장된 증거만 재검증
```

---

## 2. GPU 서버에서 바로 실행하는 무거운 공격

H100/A100 서버에서 아래 순서를 따르면 Müller et al. 블랙박스 공격(semantic-forgery)과 Diffusion 재생성 공격(WatermarkAttacker)을 즉시 돌릴 수 있습니다.

### 2.1 공통 준비
1. 이 저장소를 푸시한 후 GPU 서버에서 `git pull`.
2. Hugging Face 모델을 쓰므로 서버에서 `huggingface-cli login`을 한 번 실행합니다.

### 2.2 Müller et al. (CVPR 2025) 공격
1. `external/semantic-forgery` 폴더에서 안내한 대로 별도 conda 환경을 설정합니다.
   ```bash
   conda create -n semantic-forgery python=3.10
   conda activate semantic-forgery
   pip install -r external/semantic-forgery/requirements.txt
   ```
2. 워터마크를 특정 커버 이미지에 위조하려면:
   ```bash
   python scripts/run_mueller_attack.py \
       --cover-image results/watermarked.png \
       --target-prompt "cat standing on a rock in front of a crowd of cats" \
       --steps 80
   ```
   - 기본값으로 `GS` 워터마크, SDXL 타깃, SD2.1 공격자 구성을 사용합니다.
   - `--extra --scheduler_target DDIM ...` 처럼 필요한 옵션을 `--extra` 뒤에 그대로 붙이면 `run_imprint_forgery.py`로 전달됩니다.
   - `--dry-run`으로 실제 실행 전 명령만 확인 가능.
3. 결과물은 `results/mueller_imprint/cover_image_name=...` 아래에 저장됩니다.

### 2.3 Diffusion 재생성/스무딩 공격 (WatermarkAttacker 기반)
1. 동일한 GPU 환경(또는 별도 venv)에 diffusers 의존성을 설치했다면:
   ```bash
   python scripts/run_diffusion_regen.py \
       --input results/watermarked.png \
       --output results/regen_attack.png \
       --noise-step 80 \
       --prompt "empty prompt for unconditional regen" \
       --enable-xformers
   ```
   - `--model runwayml/stable-diffusion-v1-5`가 기본이며 다른 Hugging Face 모델로 교체 가능.
   - `--noise-step`을 키울수록 워터마크 제거력이 강해지지만 품질이 무너질 수 있습니다.
   - WatermarkAttacker 리포지터리(`external/WatermarkAttacker`)를 그대로 import하므로 추가 설정 없이 바로 동작합니다.

---

## 3. External Repos 참고
- `external/semantic-forgery` (commit 10d6a3d): CVPR 2025 Müller et al. 블랙박스 공격.
- `external/WatermarkAttacker` (commit 2637dd2): NeurIPS 2024 Diffusion smoothing/regen 공격.
- 상위 폴더의 `tree-ring-watermark-main`은 semantic-forgery가 필요로 하는 surrogate Diffusion 모델 실험에 사용 가능합니다.

## 4. TODO 메모
1. Crop/Diffusion-smoothing 탐지율 확장을 위한 multi-scale 앵커 실험.
2. Müller et al. 공격 결과의 CPR(cryptographic proof rate) 로그 자동화.
