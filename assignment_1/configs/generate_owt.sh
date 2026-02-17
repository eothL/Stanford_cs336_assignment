python -m basic.generate \
  --checkpoint basic/artifacts/model/result_L36-H20-D1280-ctx512-bs3-lr0.001-wd0.01-3b53f084_owt_1_5000.pth\
  --vocab-file basic/artifacts/vocab_32k.json \
  --merge-file basic/artifacts/merges_32k.txt \
  --config configs/train_owt.yaml \
  --prompt "Once upon a time" \
  --batch-size 3 \
  --seed 93 \
  --max-tokens 10000 
