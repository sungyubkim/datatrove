#!/bin/bash
#
# Batch Standardize All Code Datasets
#
# This script processes all 8 code datasets from the VERL Code Datasets collection,
# applying instruction format standardization with appropriate presets.
#
# Usage:
#   bash scripts/processing/batch_standardize_all_code_datasets.sh
#
# Output:
#   - output/<dataset>-standardized/train.parquet
#   - output/<dataset>-standardized/standardization_report.txt
#   - output/<dataset>-standardized/standardization_stats.json
#

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_header() {
    echo -e "${BLUE}${'='*70}${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}${'='*70}${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Start time
SCRIPT_START_TIME=$(date +%s)

print_header "Batch Code Instruction Standardization"
echo "Started at: $(date)"
echo ""

# Dataset definitions: dataset_id:preset:expected_size
# Order: smallest to largest for testing purposes
DATASETS=(
    "sungyub/codev-r1-verl:verilog-hdl:2960"
    "sungyub/code-contests-plus-verl:competitive:6540"
    "sungyub/skywork-or1-code-verl:python-code:14100"
    "sungyub/eurus-2-code-verl:competitive:25100"
    "sungyub/acecode-87k-verl:python-code:87100"
    "sungyub/rstar-coder-verl:python-code:345000"
    "sungyub/kodcode-v1-verl:python-code:435000"
)

# Track statistics
TOTAL_DATASETS=${#DATASETS[@]}
PROCESSED_COUNT=0
FAILED_COUNT=0
SUCCESS_LIST=()
FAILED_LIST=()

# Process each dataset
for dataset_info in "${DATASETS[@]}"; do
    # Parse dataset info
    IFS=':' read -r dataset_id preset expected_size <<< "$dataset_info"
    dataset_name=$(basename "$dataset_id")

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))

    print_header "[$PROCESSED_COUNT/$TOTAL_DATASETS] Processing: $dataset_name"
    print_info "Dataset: $dataset_id"
    print_info "Preset:  $preset"
    print_info "Expected size: ~${expected_size} samples"
    echo ""

    # Set output paths
    output_dir="output/${dataset_name}-standardized"
    output_file="${output_dir}/train.parquet"
    report_file="${output_dir}/standardization_report.txt"

    # Create output directory
    mkdir -p "$output_dir"

    # Start processing
    dataset_start_time=$(date +%s)

    print_info "Running standardization..."
    if python scripts/processing/standardize_instructions.py \
        --input "$dataset_id" \
        --output "$output_file" \
        --preset "$preset" \
        --report-file "$report_file"; then

        # Calculate duration
        dataset_end_time=$(date +%s)
        duration=$((dataset_end_time - dataset_start_time))

        # Check output file size
        if [ -f "$output_file" ]; then
            file_size=$(du -h "$output_file" | cut -f1)
            row_count=$(python -c "import pyarrow.parquet as pq; print(pq.read_metadata('$output_file').num_rows)")

            print_success "Processing completed!"
            print_success "Output: $output_file"
            print_success "Rows: $row_count"
            print_success "Size: $file_size"
            print_success "Duration: ${duration}s ($(($duration / 60))m)"

            SUCCESS_LIST+=("$dataset_name ($row_count rows, ${duration}s)")
        else
            print_error "Output file not created: $output_file"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_LIST+=("$dataset_name (output file missing)")
        fi
    else
        print_error "Processing failed for $dataset_name"
        FAILED_COUNT=$((FAILED_COUNT + 1))
        FAILED_LIST+=("$dataset_name (script error)")
    fi

    echo ""
    echo ""
done

# Final summary
SCRIPT_END_TIME=$(date +%s)
TOTAL_DURATION=$((SCRIPT_END_TIME - SCRIPT_START_TIME))
SUCCESS_COUNT=$((TOTAL_DATASETS - FAILED_COUNT))

print_header "Batch Processing Summary"
echo "Completed at: $(date)"
echo "Total duration: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)"
echo ""
echo "Total datasets:   $TOTAL_DATASETS"
print_success "Successfully processed: $SUCCESS_COUNT"
if [ $FAILED_COUNT -gt 0 ]; then
    print_error "Failed: $FAILED_COUNT"
fi
echo ""

if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    echo "Successfully processed datasets:"
    for item in "${SUCCESS_LIST[@]}"; do
        print_success "  $item"
    done
    echo ""
fi

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed datasets:"
    for item in "${FAILED_LIST[@]}"; do
        print_error "  $item"
    done
    echo ""
fi

# Output summary file
summary_file="output/batch_standardization_summary.txt"
cat > "$summary_file" << EOF
Batch Code Instruction Standardization Summary
================================================

Completed at: $(date)
Total duration: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)

Statistics:
- Total datasets: $TOTAL_DATASETS
- Successfully processed: $SUCCESS_COUNT
- Failed: $FAILED_COUNT

Successfully processed:
$(for item in "${SUCCESS_LIST[@]}"; do echo "  ✓ $item"; done)

$(if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo "Failed:"
    for item in "${FAILED_LIST[@]}"; do echo "  ✗ $item"; done
fi)
EOF

print_success "Summary saved to: $summary_file"
echo ""

# Exit with appropriate code
if [ $FAILED_COUNT -eq 0 ]; then
    print_success "All datasets standardized successfully!"
    exit 0
else
    print_error "Some datasets failed to process"
    exit 1
fi
