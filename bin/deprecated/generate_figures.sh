#!/bin/bash
#
# Generate publication figures for evaluation summary
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
ORGANISM="${1:-human}"

echo "Generating publication figures for: $ORGANISM"
echo "Output directory: $OUTDIR"
echo ""

# =============================================================================
# Mouse (mus_musculus)
# =============================================================================
generate_mouse_figure() {
    echo "=== Generating Mouse Publication Figures ==="

    BASE="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus_tabulamuris/100/dataset_id/SCT/gap_false"
    MODEL_DIR="$BASE/weighted_models/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method"

    python bin/plot_pub_figures.py \
        --cutoff_effects "$MODEL_DIR/subclass/files/method_cutoff_effects.tsv" \
        --reference_emmeans "$MODEL_DIR/subclass/files/reference_method_emmeans_summary.tsv" \
        --emmeans_dir "$MODEL_DIR" \
        --key subclass \
        --organism mus_musculus \
        --outdir "$OUTDIR" \
        --output_prefix pub_figure_mouse

    echo "Mouse figures saved to: $OUTDIR/pub_figure_mouse*.png"
}

# =============================================================================
# Human (homo_sapiens)
# =============================================================================
generate_human_figure() {
    echo "=== Generating Human Publication Figures ==="

    BASE="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false"
    MODEL_DIR="$BASE/weighted_models/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_disease_state_+_sex_+_region_match_+_method:cutoff_+_reference:method"

    python bin/plot_pub_figures.py \
        --cutoff_effects "$MODEL_DIR/subclass/files/method_cutoff_effects.tsv" \
        --reference_emmeans "$MODEL_DIR/subclass/files/reference_method_emmeans_summary.tsv" \
        --emmeans_dir "$MODEL_DIR" \
        --key subclass \
        --organism homo_sapiens \
        --outdir "$OUTDIR" \
        --output_prefix pub_figure_human

    echo "Human figures saved to: $OUTDIR/pub_figure_human*.png"
}

# =============================================================================
# Main
# =============================================================================
case "$ORGANISM" in
    mmus|mouse|mus_musculus)
        generate_mouse_figure
        ;;
    human|homo_sapiens)
        generate_human_figure
        ;;
    all)
        generate_human_figure
        echo ""
        generate_mouse_figure
        ;;
    *)
        echo "Usage: $0 [mmus|human|all]"
        exit 1
        ;;
esac

echo ""
echo "Done!"
