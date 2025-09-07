# Distillation plan (teacher -> student)

1) Use the fine-tuned 13B LoRA teacher to generate a large dataset of (prompt, response) pairs.
   - Use deterministic generation (temp=0.2, do_sample=False) for consistency.
   - Filter outputs to ensure price tokens are preserved.

2) Train a 7B student on the generated dataset (SFT / LoRA).
   - Candidate bases: Mistral-7B, Zephyr-7B, LLaMA-2-7B.
   - Use LoRA on 7B if GPU memory limited, or full fine-tune if resources permit.

3) Validate student against holdout: ensure price-token correctness and human eval on cultural fit.

4) If successful, quantize the student (AutoGPTQ / bitsandbytes) and deploy.
