python -m basic.generate \
  --checkpoint basic/artifacts/model/result_L24-H16-D1024-ctx512-bs24-lr0.001-wd0.01-feb384eb_owt_1_3146.pth \
  --vocab-file basic/artifacts/vocab_32k.json \
  --merge-file basic/artifacts/merges_32k.txt \
  --config configs/train_owt_medium.yaml \
  --prompt "Once upon a time" \
  --seed 93 \
  --max-tokens 500
