# vllm-benchmark-helper
Helper scripts to organize and run vllm benchmark

1. Source the venv which can kick off the vllm benchmark tool
```
source ~/workplaces/vllm/.venv/bin/activate 
```
2. Setup configurations you want to experiment with in ```run_benchmark.py```
3. Kick off the scrip, better in a screen or tmux session
```
python run_benchmark.py
```