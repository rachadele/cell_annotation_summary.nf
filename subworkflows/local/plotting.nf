/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    SUBWORKFLOW: PLOTTING
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Handles all visualization and plotting tasks
----------------------------------------------------------------------------------------
*/

include { PLOT_CUTOFF       } from '../../modules/local/plot_cutoff/main'
include { PLOT_COMPTIME     } from '../../modules/local/plot_comptime/main'
include { PLOT_LABEL_DIST   } from '../../modules/local/plot_label_dist/main'
include { PLOT_CONTRASTS    } from '../../modules/local/plot_contrasts/main'
include { PLOT_CONTINUOUS_CONTRAST } from '../../modules/local/plot_continuous_contrast/main'

workflow PLOTTING {

    take:
    ch_weighted_f1              // channel: weighted F1 results
    ch_label_f1                 // channel: label F1 results
    ch_emmeans_all              // channel: [ key, contrast, emmeans_estimates, emmeans_summary ]
    ch_continuous_effects_map   // channel: [ key, mode, continuous_effects ]
    results_dir                 // val: results directory path

    main:
    PLOT_CUTOFF(ch_weighted_f1, ch_label_f1)
    PLOT_LABEL_DIST(ch_label_f1)
    PLOT_COMPTIME(results_dir)
    PLOT_CONTRASTS(ch_emmeans_all, ch_weighted_f1)
    // PLOT_CONTINUOUS_CONTRAST(ch_continuous_effects_map)

    emit:
    cutoff_plots    = PLOT_CUTOFF.out
    label_dist      = PLOT_LABEL_DIST.out
    comptime        = PLOT_COMPTIME.out
    contrast_plots  = PLOT_CONTRASTS.out
}
