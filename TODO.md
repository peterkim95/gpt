* Sequence length 35 leads to out of GPU memory - what's vram of g4dn.xlarge - maybe smaller batch sizes - is single gpu doomed to fail for word language models? - pytorch wlm example does repackaaging - wtf?
* Validation loss multiples by tokens in batch, why?? Examine CrossEntropyLoss
