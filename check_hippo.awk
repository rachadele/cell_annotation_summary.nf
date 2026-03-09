BEGIN { FS="\t" }
NR==1 { for(i=1;i<=NF;i++) col[$i]=i; next }
$(col["label"])=="Hippocampal neuron" && $(col["key"])=="class" && $(col["cutoff"])=="0.0" && $(col["study"])!="GSE185454" {
    print $(col["study"]), $(col["reference_acronym"]), $(col["method"]), $(col["f1_score"]), $(col["precision"]), $(col["recall"]), $(col["support"])
}
