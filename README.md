# CSE 554

The [Systems for Machine Learning](https://courses.cs.washington.edu/courses/cse554/) course at the University of Washington (Group 14, Winter 2026).

## Setup

```bash
# UV settings
export UV_LINK_MODE=copy

# Set up UV
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt --cache-dir /tmp/uv-cache
```

## Schedule

| Date | Topic | Slides & Reading | Assignments |
| :---- | :---- | :---- | :---- |
| Jan 5, 2026 | GPU Architecture | [[PPTX](slides/cse554-l01_gpu_arch_intro.pptx)] [[PDF](slides/cse554-l01_gpu_arch_intro.pdf)] [[CUDA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)] | |
| Jan 7, 2026 | GPU kernel optimizations - 1 | [[PPTX](slides/cse554-l02_gpu_kernel_opt1.pptx)] [[PDF](slides/cse554-l02_gpu_kernel_opt1.pdf)] | |
| Jan 12, 2026 | GPU kernel optimizations - 2 | [[PPTX](slides/cse554-l03_gpu_kernel_opt2.pptx)] [[PDF](slides/cse554-l03_gpu_kernel_opt2.pdf)] | HW1 Out |
| Jan 14, 2026 | Transformer architecture | [[PPTX](slides/cse554-l04_transformer_arch.pptx)] [[PDF](slides/cse554-l04_transformer_arch.pdf)] | |
| Jan 19, 2026 | Martin Luther King Jr. Day | | |
| Jan 21, 2026 | Transformer implementation 1 | [[PPTX](slides/cse554-l06_transformer_impl.pptx)] [[PDF](slides/cse554-l06_transformer_impl.pdf)] | |
| Jan 26, 2026 | Transformer Implementation 2, Performance Modeling 1 | [[PPTX](slides/cse554-l06_transformer_impl.pptx)] [[PDF](slides/cse554-l06_transformer_impl.pdf)] | |
| Jan 28, 2026 | Performance Modeling 2 | [[PPTX](slides/cse554-l07_perf_modeling.pptx)] [[PDF](slides/cse554-l07_perf_modeling.pdf)] [[Nanoflow](https://arxiv.org/abs/2408.12757)] | HW1 Due, HW2 Out |
| Feb 2, 2026 | Quantization | [[PPTX](slides/cse554-l08_quantization.pptx)] [[PDF](slides/cse554-l08_quantization.pdf)] [[LLM.int8()](https://arxiv.org/abs/2208.07339)] | |
| Feb 4, 2026 | Memory management | [[PPTX](slides/cse554-l09_memory_management.pptx)] [[PDF](slides/cse554-l09_memory_management.pdf)] [[PagedAttention](https://arxiv.org/abs/2309.06180)] [[FlashAttention](https://arxiv.org/abs/2205.14135)] | |
| Feb 9, 2026 | NVIDIA Blackwell Architecture | | |
| Feb 11, 2026 | Sparsity and Pruning | [[PPTX](slides/cse554-l10_sparsity_pruning.pptx)] [[PDF](slides/cse554-l10_sparsity_pruning.pdf)] | HW2 Due, HW3 Out |
| Feb 16, 2026 | Presidents' Day | | |
| Feb 18, 2026 | MoE | [[PPTX](slides/cse554-l11_moe.pptx)] [[PDF](slides/cse554-l11_moe.pdf)] | |
| Feb 23, 2026 | Speculative Decoding | [[PPTX](slides/cse554-l12_speculative_decoding.pptx)] [[PDF](slides/cse554-l12_speculative_decoding.pdf)] | |
| Feb 25, 2026 | Batching | [[PPTX](slides/cse554-l13_batching.pptx)] [[PDF](slides/cse554-l13_batching.pdf)] | |
| Mar 2, 2026 | Parallelism-1 | [[PPTX](slides/cse554-l14_parallelism1.pptx)] [[PDF](slides/cse554-l14_parallelism1.pdf)] [[Megatron-LM](https://arxiv.org/pdf/2104.04473)] [[Zero Bubble Pipeline Parallel](https://arxiv.org/pdf/2401.10241)] | HW3 Due, HW4 Out |
| Mar 4, 2026 | Parallelism-2 | [[PPTX](slides/cse554-l15_parallelism2.pptx)] [[PDF](slides/cse554-l15_parallelism2.pdf)] [[Megatron-LM, Sequence Parallel](https://arxiv.org/pdf/2205.05198)] [[DeepSpeed-Ulysses](https://arxiv.org/pdf/2309.14509)] | |
| Mar 9, 2026 | RL for LLMs | [[PPTX](slides/cse554-l16_rl.pptx)] [[PDF](slides/cse554-l16_rl.pdf)] | |
| Mar 11, 2026 | Agentic Systems | [[PPTX](slides/cse554-l17_agents.pptx)] [[PDF](slides/cse554-l17_agents.pdf)] | |
| Mar 13, 2026 | | | HW4 Due |
