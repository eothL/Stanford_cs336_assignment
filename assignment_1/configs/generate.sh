python -m basic.generate \
  --checkpoint basic/artifacts/model/result_tinystories_example_ts_1_4874.pth \
  --vocab-file basic/artifacts/vocab_10k.json \ 
  --merge-file basic/artifacts/merge_10k.tkt \
  --config configs/train_ts.yaml \
  --prompt "Once upon a time"
