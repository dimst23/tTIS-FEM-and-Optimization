# Description

In the code folder, the main software can be found. The folder contains the Core part which implements the meshing and solution for all models. The Scripts folder contains all the necessary scripts to produce a solved model and most of these scripts call functions from the Core folder.

The scripts, included in the unanimous folder, are named based on the stage at which they are used and this naming follows the pattern of `[Stage-Name]_Short_Description`.

## Short explanation of the pipeline

For a detailed explanation of the software and the pipeline [check here](http://dx.doi.org/10.13140/RG.2.2.20020.99202).

* At first the electrode positions for the desired models have to be calculated using [Mesh2EEG](https://engineering.dartmouth.edu/multimodal/mesh2eeg.html), producing a file which contains the electrode names and their corresponding position for the particular model.
* Before doing any mesh manipulation work (including meshing the model), it is suggested to run all meshes through [MeshFix](https://github.com/MarcoAttene/MeshFix-V2.1), thus avoiding any errors during the next steps. An automation script for MeshFix can be found in the _Utilities_ folder, named `mesh_fixing.py`.
* The next step is to add the electrodes on the skin surface for each model. This is done using the `[Meshing]_Model_POLY.py` script, generating a `.poly` file to be used later by TetGen.
* After the successful generation of the `.poly` file(s) for the desired model(s), [TetGen](https://wias-berlin.de/software/index.jsp?id=TetGen) has to be called for each model. A utility script can be found in the _Utilities_ folder, named `tetgen_meshing.sh`.
* Having the tetrahedralized meshes, the next step is to solve the models. A script solving all models for all electrode combinations using one electrode as a reference, is provided in the script named `[Solution]_Electrode_Combinations.py`. The solver part of the code can be adapted to solve only one model, selecting the desired electrode pairs and injection currents.
* With all models solved the optimizer can run. The script named `[Optimization]_Gentic_Algorithm.py` contains the genetic algorithm and the corresponding settings required for running the optimizer.
* The results of the optimization include the objective function value and the optimized electrode pairs. To save the currents that optimized the electrode pairs, the scripts named `[Post-process]_Result_Validation.py` and located in `Data Analysis` has to be called.
*Lastly, all scripts found in the `Nifti Operations` folder are used in the Nifti generation and manipulations, after all the above have been successfully completed.

In some scripts included in the pipeline described above, the `sim_settings.yml` file is used. This file serves as a settings container for the electrode names and indices, as well as the mesh location and the output directory of the solver.
