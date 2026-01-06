process PLOT_CONTINUOUS_CONTRAST {
    tag "${key}_${mode}"
    label 'process_single'
    conda '/home/rschwartz/anaconda3/envs/scanpyenv'

    input:
    tuple val(key), val(mode), path(continuous_effects)

    output:
    path "**png"

    script:
    """
    python ${projectDir}/bin/plot_continuous_contrasts.py --key ${key} --contrast ${continuous_effects}
    """
}
