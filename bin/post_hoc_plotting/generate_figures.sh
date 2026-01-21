#!/bin/bash
#
# Generate main publication figure for evaluation summary
#
# Usage: bash bin/generate_figures.sh [organism]
#   organism: mmus (default), human, or all
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Output directory
OUTDIR="${PROJECT_DIR}/figures"
mkdir -p "$OUTDIR"

# Select organism
ORGANISM="${1:-mmus}"

echo "Generating main figure for: $ORGANISM"
echo "Output directory: $OUTDIR"
echo ""

# =============================================================================
# Mouse (mmus_new)
# =============================================================================
generate_mouse_figure() {
    echo "=== Generating Mouse Main Figure ==="

    BASE="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mmus_new/100/dataset_id/SCT/gap_false"
    MODEL_DIR="$BASE/weighted_models/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method"
    WEIGHTED_F1="$BASE/aggregated_results/weighted_f1_results.tsv"

    python bin/plot_main_figure.py \
        --cutoff_effects "$MODEL_DIR/subclass/files/method_cutoff_effects.tsv" \
        --reference_emmeans "$MODEL_DIR/subclass/files/reference_method_emmeans_summary.tsv" \
        --weighted_f1 "$WEIGHTED_F1" \
        --emmeans_dir "$MODEL_DIR" \
        --key subclass \
        --outdir "$OUTDIR" \
        --output_prefix main_figure_mouse

    echo "Mouse figure saved to: $OUTDIR/main_figure_mouse.pdf"
}

# =============================================================================
# Human (homo_sapiens)
# =============================================================================
generate_human_figure() {
    echo "=== Generating Human Main Figure ==="

    BASE="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false"
    MODEL_DIR="$BASE/weighted_models/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_disease_state_+_sex_+_region_match_+_method:cutoff_+_reference:method"
    WEIGHTED_F1="$BASE/aggregated_results/files/weighted_f1_results.tsv"

    python bin/plot_main_figure.py \
        --cutoff_effects "$MODEL_DIR/subclass/files/method_cutoff_effects.tsv" \
        --reference_emmeans "$MODEL_DIR/subclass/files/reference_method_emmeans_summary.tsv" \
        --weighted_f1 "$WEIGHTED_F1" \
        --emmeans_dir "$MODEL_DIR" \
        --key subclass \
        --outdir "$OUTDIR" \
        --output_prefix main_figure_human

    echo "Human figure saved to: $OUTDIR/main_figure_human.pdf"
}

# =============================================================================
# Main
# =============================================================================
case "$ORGANISM" in
    mmus|mouse)
        generate_mouse_figure
        ;;
    human|homo_sapiens)
        generate_human_figure
        ;;
    all)
        generate_mouse_figure
        echo ""
        generate_human_figure
        ;;
    *)
        echo "Usage: $0 [mmus|human|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
