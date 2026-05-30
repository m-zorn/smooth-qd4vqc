[![DOI](https://zenodo.org/badge/1254311417.svg)](https://doi.org/10.5281/zenodo.20465733)

# On Representation Smoothness for Robust Architecture Discovery of Variational Quantum Circuits

Implementation of the smooth encoding variants in [0] for optimizing variational quantum circuits (VQC), as well as the base encoding from [1]. The quality diversity (QD) optimization part is based on the `pyribs` library [2], the solutions of which (after conversion) are simulated as `pennylane` [3] quantum circuits. The generated erdos-reny graphs for combinatorial optimization tasks with `14` vertices can be found in the `data/` directory as pickle files for replication.


### Usage:
```
    # Make virtual env with:
    conda create -n smooth-qd4vqc python=3.10
    conda activate smooth-qd4vqc
    pip install -r requirements.txt

    # Run the main experiments
    python src/runnables/qd_experiments.py

    # Run the ablation study
    python src/runnables/qd_experiments_ablation.py
``` 

### References:
[0] Zorn et al., On Representation Smoothness for Robust Architecture Discovery of Variational Quantum Circuits, KDD 2026 (to appear)

[1] Zorn et al., Quality Diversity for Variational Quantum Circuit Optimization, ICAPS 2025

[2] Pyribs QD-optimization (pyribs.org/)

[3] Pennylane quantum simulator (pennylane.ai/)
