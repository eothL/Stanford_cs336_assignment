python -m basic.generate \
  --checkpoint basic/artifacts/model/result_tinystories_example_ts_1_4874.pth \
  --vocab-file basic/artifacts/vocab_10k.json \
  --merge-file basic/artifacts/merges_10k.txt \
  --config configs/train_ts_exp.yaml \
  --prompt "Once upon a time" \
  --seed 93 \
  --max-tokens 10000 
