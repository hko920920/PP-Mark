# External Attack Repos

## semantic-forgery (Müller et al., CVPR 2025)
- Cloned from https://github.com/and-mill/semantic-forgery.git (commit 10d6a3d).
- Requires PyTorch + diffusers + Stable Diffusion 2.1 weights + GPU (see repo README) to rerun Imprint-Forgery/Removal/Reprompting.
- GPU 서버에서 재현 시:
  1. `cd pp_mark/external/semantic-forgery`
  2. Set up conda env per upstream instructions (`pip install -r requirements.txt`).
  3. 준비물: reference watermarked image (`dct_watermark.py embed` 결과) + clean cover image.
  4. `python run_imprint_forgery.py --reference_image ... --cover_image ... --output_image ... --watermark_type TR` 실행 후, 산출물은 `pp_mark/dct_watermark.py extract` → Halo2 prover로 연결.

## tree-ring-watermark-main
- 이미 `/mnt/c/Users/SOGANG/Downloads/ICML_materials/tree-ring-watermark-main` 에 존재하며, semantic-forgery에서 TR surrogate 실험 시 참조 가능합니다.

## WatermarkAttacker (Invisible Image Watermarks Are Provably Removable Using Generative AI, NeurIPS 2024)
- Cloned from https://github.com/XuandongZhao/WatermarkAttacker.git (commit 2637dd2).
- Implements diffusion-smoothing style regenerations and text-to-image attack scripts described in the paper.
- GPU 서버 셋업:
  1. `cd pp_mark/external/WatermarkAttacker`
  2. Follow README (conda env, `pip install -r requirements.txt`, download SD1.5 checkpoints).
  3. Provide our watermarked image as `input_image`, run provided regeneration scripts, 그리고 결과물을 `pp_mark/dct_watermark.py extract` 및 Halo2 prover로 연결해 CPR 측정.
