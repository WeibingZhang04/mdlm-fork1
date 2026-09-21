# Causal follow-up: does training-cache precision explain the PPL regression?

The completed startup replay proved a numerical training difference, not its downstream PPL effect. This follow-up tests that effect directly. It is a controlled diagnostic, not a replacement for any existing result or a complete replication of the eight fresh arms.

## Predeclared protocol

- Basic DD; same historical 6,000-update checkpoint and restored optimizer for both arms. Restart the same training stream with seed 1; 2,000 further updates per arm to 8,000. All original objective weights, learning rate, batch size and architecture stay unchanged.
- Only intervention: initialize the positional rotation cache under BF16 versus FP32 before running identical monitoring. This preserves random state and model parameters. BF16 emulates historical startup; FP32 emulates the new monitored startup. Assert the chosen cache dtype before every batch.
- Run both arms sequentially in one generic GPU allocation, with no H200 or other GPU-model pin. First run a three-update smoke for each; verify identical initial head, first three input/attention batches and CPU/CUDA/corruption/topology random states. Repeat these checks in the full continuations.
- Evaluate the before-6k checkpoint and both after-8k checkpoints in that same allocation. 100 samples per cell; fresh seed 240001; length 1024; 8 and 32 denoising steps, original NFE accounting. Record both joint and independent-marginal sampling, and an MDLM baseline. Same FP32 positional cache at generation, asserted at runtime. Same GPT2-large revision, FP32 scorer, EOS rule and raw sample retention.
- Primary comparisons: BF16-trained versus FP32-trained after-8k joint PPL at 8 and 32 denoising steps. Report both, regardless of outcome. Marginal sampling, repetition, diversity and scored lengths are supporting diagnostics. Use paired sample resampling for descriptive uncertainty. One checkpoint/training seed; no broad architecture claims.

This test changes numerical execution, not graph construction, model parameters/architecture or the scoring rule. No existing jobs or historical outputs are canceled or overwritten. A positive result supports the precision-change explanation for this continuation; it does not automatically explain every fresh 15k run. A null/negative result must be retained and further causes investigated.
