`forward_pass` runs on Apple GPUs: `forward.device` accepts `mps`, and `auto` now picks `mps` on a Mac without CUDA, where it used to pick the CPU.
