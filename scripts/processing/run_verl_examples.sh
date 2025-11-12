#!/bin/bash
#
# VERL Data Processing - Example Commands
#
# This script provides ready-to-use example commands for processing VERL datasets
# with various configurations. Copy and modify these examples for your use case.
#
# Usage:
#   # Run a specific example (uncomment the desired example below)
#   bash scripts/processing/run_verl_examples.sh
#
#   # Or execute individual commands directly:
#   bash -c "<command from below>"
#

set -e  # Exit on error

# ==============================================================================
# Example 1: Basic Math Dataset Processing
# ==============================================================================
# Process math dataset with default settings (10 responses per prompt)
echo "=== Example 1: Basic Math Dataset ===" echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/math-verl/train.parquet \
    --output-dir output/math-processed \
    --model-name-or-path meta-llama/Llama-3-8B
EOF
echo ""
echo "This will:"
echo "  - Read VERL math data from data/math-verl/train.parquet"
echo "  - Generate 10 responses per prompt (default)"
echo "  - Use local vLLM server with meta-llama/Llama-3-8B"
echo "  - Save processed results to output/math-processed/"
echo "  - Use temperature 0.7 (default) for diverse responses"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/math-verl/train.parquet \
#     --output-dir output/math-processed \
#     --model-name-or-path meta-llama/Llama-3-8B


# ==============================================================================
# Example 2: Custom Response Generation Settings
# ==============================================================================
# Generate more responses with higher temperature
echo "=== Example 2: Custom Response Generation ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/math-verl/train.parquet \
    --output-dir output/math-custom \
    --model-name-or-path meta-llama/Llama-3-8B \
    --num-responses-per-prompt 20 \
    --sampling-temperature 0.9 \
    --max-tokens-per-response 4096
EOF
echo ""
echo "This will:"
echo "  - Generate 20 responses per prompt (instead of default 10)"
echo "  - Use higher temperature 0.9 for more diversity"
echo "  - Allow up to 4096 tokens per response (instead of default 2048)"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/math-verl/train.parquet \
#     --output-dir output/math-custom \
#     --model-name-or-path meta-llama/Llama-3-8B \
#     --num-responses-per-prompt 20 \
#     --sampling-temperature 0.9 \
#     --max-tokens-per-response 4096


# ==============================================================================
# Example 3: Code Dataset with Sandbox Fusion
# ==============================================================================
# Process code dataset with execution-based scoring
echo "=== Example 3: Code Dataset with Sandbox Fusion ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/codecontests.parquet \
    --output-dir output/code-processed \
    --model-name-or-path deepseek-ai/deepseek-coder-7b \
    --num-responses-per-prompt 5 \
    --sandbox-fusion-url http://localhost:5000 \
    --max-concurrent-scoring 20
EOF
echo ""
echo "This will:"
echo "  - Process code generation dataset (codecontests)"
echo "  - Use deepseek-coder model optimized for code"
echo "  - Generate 5 responses per prompt"
echo "  - Score responses via Sandbox Fusion server at localhost:5000"
echo "  - Limit concurrent scoring to 20 requests (prevents sandbox overload)"
echo ""
echo "Prerequisites:"
echo "  - Sandbox Fusion server must be running on localhost:5000"
echo "  - Start with: docker run -p 5000:5000 sandbox-fusion:latest"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/codecontests.parquet \
#     --output-dir output/code-processed \
#     --model-name-or-path deepseek-ai/deepseek-coder-7b \
#     --num-responses-per-prompt 5 \
#     --sandbox-fusion-url http://localhost:5000 \
#     --max-concurrent-scoring 20


# ==============================================================================
# Example 4: Remote vLLM Server
# ==============================================================================
# Use external vLLM server instead of spawning local one
echo "=== Example 4: Remote vLLM Server ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/large-dataset.parquet \
    --output-dir output/remote-processed \
    --model-name-or-path meta-llama/Llama-3-70B \
    --inference-server-type vllm-remote \
    --remote-vllm-endpoint http://vllm-cluster.example.com:8000 \
    --num-responses-per-prompt 15 \
    --max-concurrent-inference 200
EOF
echo ""
echo "This will:"
echo "  - Connect to external vLLM server (no local GPU required)"
echo "  - Use large 70B model running on remote cluster"
echo "  - Generate 15 responses per prompt"
echo "  - Allow 200 concurrent inference requests (high throughput)"
echo ""
echo "Use case:"
echo "  - Processing on machine without GPU"
echo "  - Using centralized vLLM cluster for multiple users"
echo "  - Large models that don't fit on local GPU"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/large-dataset.parquet \
#     --output-dir output/remote-processed \
#     --model-name-or-path meta-llama/Llama-3-70B \
#     --inference-server-type vllm-remote \
#     --remote-vllm-endpoint http://vllm-cluster.example.com:8000 \
#     --num-responses-per-prompt 15 \
#     --max-concurrent-inference 200


# ==============================================================================
# Example 5: Production Settings (High Parallelism)
# ==============================================================================
# Process large dataset with maximum parallelism and custom paths
echo "=== Example 5: Production Settings ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/production-dataset \
    --output-dir output/production \
    --model-name-or-path Qwen/Qwen2.5-Math-7B \
    --num-responses-per-prompt 20 \
    --sampling-temperature 0.8 \
    --max-tokens-per-response 4096 \
    --num-parallel-tasks 50 \
    --num-concurrent-workers 10 \
    --checkpoint-frequency 1000 \
    --checkpoint-dir checkpoints/production \
    --log-dir logs/production \
    --stats-output-dir stats/production \
    --max-concurrent-inference 150 \
    --max-concurrent-scoring 100
EOF
echo ""
echo "This will:"
echo "  - Process large dataset directory (multiple parquet files)"
echo "  - Use Qwen2.5-Math model optimized for mathematical reasoning"
echo "  - Generate 20 responses per prompt with temperature 0.8"
echo "  - Run 50 parallel tasks for data sharding"
echo "  - Use 10 concurrent workers per task"
echo "  - Save checkpoint every 1000 documents (faster resumption)"
echo "  - Collect comprehensive statistics in stats/production/"
echo "  - Allow 150 concurrent inference + 100 concurrent scoring"
echo ""
echo "Recommended for:"
echo "  - Large-scale dataset processing (>100K examples)"
echo "  - Multi-GPU servers"
echo "  - Production RLHF data generation"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/production-dataset \
#     --output-dir output/production \
#     --model-name-or-path Qwen/Qwen2.5-Math-7B \
#     --num-responses-per-prompt 20 \
#     --sampling-temperature 0.8 \
#     --max-tokens-per-response 4096 \
#     --num-parallel-tasks 50 \
#     --num-concurrent-workers 10 \
#     --checkpoint-frequency 1000 \
#     --checkpoint-dir checkpoints/production \
#     --log-dir logs/production \
#     --stats-output-dir stats/production \
#     --max-concurrent-inference 150 \
#     --max-concurrent-scoring 100


# ==============================================================================
# Example 6: Resume from Checkpoint
# ==============================================================================
# Automatically resume from checkpoint if processing was interrupted
echo "=== Example 6: Resume from Checkpoint ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/large-dataset.parquet \
    --output-dir output/resumed \
    --model-name-or-path meta-llama/Llama-3-8B \
    --num-responses-per-prompt 10 \
    --checkpoint-dir checkpoints/resume
EOF
echo ""
echo "This will:"
echo "  - Automatically detect existing checkpoints in checkpoints/resume/"
echo "  - Resume processing from last saved checkpoint"
echo "  - Skip already processed documents"
echo "  - Continue with same configuration"
echo ""
echo "How it works:"
echo "  1. First run: processes documents 0-500, saves checkpoint"
echo "  2. Process interrupted at document 300"
echo "  3. Re-run same command: resumes from document 300"
echo "  4. Continues until completion"
echo ""
echo "Note: Use the SAME --checkpoint-dir path to enable resumption"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/large-dataset.parquet \
#     --output-dir output/resumed \
#     --model-name-or-path meta-llama/Llama-3-8B \
#     --num-responses-per-prompt 10 \
#     --checkpoint-dir checkpoints/resume


# ==============================================================================
# Example 7: Incremental Response Generation (Append Mode)
# ==============================================================================
# Generate additional responses for already-processed data
echo "=== Example 7: Incremental Response Generation ==="
echo ""
echo "Scenario: You want to add more responses to existing processed data"
echo ""
echo "Step 1 - Initial processing (generates 10 responses):"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/math.parquet \
    --output-dir output/math-incremental \
    --model-name-or-path meta-llama/Llama-3-8B \
    --num-responses-per-prompt 10
EOF
echo "  Result: output/math-incremental/*.parquet (10 responses per prompt)"
echo ""
echo "Step 2 - Add more responses (input = previous output):"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data output/math-incremental \
    --output-dir output/math-20-responses \
    --model-name-or-path meta-llama/Llama-3-8B \
    --num-responses-per-prompt 10
EOF
echo "  Result: output/math-20-responses/*.parquet (20 responses per prompt)"
echo ""
echo "This works because:"
echo "  - Script detects existing responses in extra_info.responses"
echo "  - New responses are APPENDED (not replaced)"
echo "  - Statistics are recalculated from all responses (old + new)"
echo ""
echo "Use cases:"
echo "  - Increase response diversity without re-generating existing ones"
echo "  - Add responses at different temperatures"
echo "  - Scale up dataset incrementally"
echo ""
# Uncomment to run step 1:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/math.parquet \
#     --output-dir output/math-incremental \
#     --model-name-or-path meta-llama/Llama-3-8B \
#     --num-responses-per-prompt 10
#
# Uncomment to run step 2:
# python scripts/processing/verl_data_processing.py \
#     --input-data output/math-incremental \
#     --output-dir output/math-20-responses \
#     --model-name-or-path meta-llama/Llama-3-8B \
#     --num-responses-per-prompt 10


# ==============================================================================
# Example 8: SGLang Server (Alternative to vLLM)
# ==============================================================================
# Use SGLang inference server instead of vLLM
echo "=== Example 8: SGLang Server ==="
echo ""
echo "Command:"
cat << 'EOF'
python scripts/processing/verl_data_processing.py \
    --input-data data/dataset.parquet \
    --output-dir output/sglang-processed \
    --model-name-or-path meta-llama/Llama-3-8B \
    --inference-server-type sglang \
    --num-responses-per-prompt 10
EOF
echo ""
echo "This will:"
echo "  - Use SGLang server instead of vLLM"
echo "  - Automatically spawn local SGLang server"
echo "  - Process with same configuration as vLLM examples"
echo ""
echo "When to use SGLang:"
echo "  - Better performance for certain model architectures"
echo "  - Advanced features like RadixAttention"
echo "  - Specific model compatibility requirements"
echo ""
# Uncomment to run:
# python scripts/processing/verl_data_processing.py \
#     --input-data data/dataset.parquet \
#     --output-dir output/sglang-processed \
#     --model-name-or-path meta-llama/Llama-3-8B \
#     --inference-server-type sglang \
#     --num-responses-per-prompt 10


# ==============================================================================
# Help and Documentation
# ==============================================================================
echo ""
echo "=== Getting Help ==="
echo ""
echo "For full parameter list:"
echo "  python scripts/processing/verl_data_processing.py --help"
echo ""
echo "For implementation details:"
echo "  See: examples/verl_data_processing.py"
echo ""
echo "For dataset-specific scoring requirements:"
echo "  Math datasets: No additional setup"
echo "  Code datasets: Requires Sandbox Fusion server"
echo "  CodeV (Verilog): Requires Sandbox Fusion + iverilog"
echo "  ToolRL: No additional setup"
echo ""
echo "Common issues:"
echo "  - CUDA out of memory: Reduce --num-concurrent-inference or --num-responses-per-prompt"
echo "  - Sandbox timeout: Reduce --max-concurrent-scoring"
echo "  - vLLM startup failure: Check GPU availability and model compatibility"
echo ""
