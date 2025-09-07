# Quantization notes and quick commands

## bitsandbytes (bnb) 4-bit serving (GPU)
- In Transformers, load with:
  model = AutoModelForCausalLM.from_pretrained(path, load_in_4bit=True, device_map='auto', bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')

## AutoGPTQ conversion (fast inference / CPU-friendly)
- Install: pip install auto-gptq
- Convert (example):
  python -m auto_gptq.convert --model /path/to/student --outfile /out/quant_student --wbits 4 --act-order
- Serve with TGI or AutoGPTQ runtime

## Verification checklist after quantization
- Ensure numeric price token is still present in generated outputs (run 10k samples).
- Run latency tests on target hardware (p50, p95).
- Run human eval on a sample set to ensure cultural quality remains acceptable.
