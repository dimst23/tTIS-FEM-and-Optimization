# Transcranial Temporal Interference Stimulation

This project is part of the Bachelor thesis during the studies at the Physics Department of the Aristotle University of Thessaloniki, Greece.

## Short description

Simulations of the temporally interfering electric field distribution are conducted on the human brain using a _Simple Anthropomorphic model_ **(SAM)** and realistic brain models from the [PHM](https://itis.swiss/virtual-population/regional-human-models/phm-repository/) repository. The different material conductivities are drawn from the [IT'IS Virtual Population Tissue Properties](https://itis.swiss/virtual-population/tissue-properties/database/low-frequency-conductivity/) or from different papers, referenced accordingly.

## Further Information

In this repository there [is a Wiki](https://gitlab.com/dimst23/tacs-temporal-interference/-/wikis/0.-Home) in which you can find detailed information about the code and the approaches taken to solve the problems.

## Structure of the Repository

* [CAD/](/CAD)
  * [SAM/](/CAD/SAM)
  * [Sphere/](/CAD/Sphere)
    * [spheres_brain.stl](/CAD/Sphere/spheres_brain.stl)
    * [spheres_csf.stl](/CAD/Sphere/spheres_csf.stl)
    * [spheres_outer.stl](/CAD/Sphere/spheres_outer.stl)
    * [spheres_skin.stl](/CAD/Sphere/spheres_skin.stl)
    * [spheres_skull.stl](/CAD/Sphere/spheres_skull.stl)
* [Jupyter Notebooks/](/Jupyter Notebooks)
  * [Test Bench/](/Jupyter Notebooks/Test Bench)
    * [modulation_envelope_tests.ipynb](/Jupyter Notebooks/Test Bench/modulation_envelope_tests.ipynb)
    * [README.md](/Jupyter Notebooks/Test Bench/README.md)
  * [fem_simulation_analysis.ipynb](/Jupyter Notebooks/fem_simulation_analysis.ipynb)
  * [modulation_envelope.ipynb](/Jupyter Notebooks/modulation_envelope.ipynb)
  * [sim_analysis.ipynb](/Jupyter Notebooks/sim_analysis.ipynb)
* [MATLAB Workspaces and Scripts/](/MATLAB Workspaces and Scripts)
  * [Workspaces/](/MATLAB Workspaces and Scripts/Workspaces)
    * [BaseFrequency4Layer_Smaller.mat](/MATLAB Workspaces and Scripts/Workspaces/BaseFrequency4Layer_Smaller.mat)
    * [DeltaFrequency4Layer_Smaller.mat](/MATLAB Workspaces and Scripts/Workspaces/DeltaFrequency4Layer_Smaller.mat)
  * [arrange_elements.m](/MATLAB Workspaces and Scripts/arrange_elements.m)
  * [grid_points.m](/MATLAB Workspaces and Scripts/grid_points.m)
* [Scripts/](/Scripts)
  * [Archive/](/Scripts/Archive)
    * [FEM.py](/Scripts/Archive/FEM.py)
    * [meshing.py](/Scripts/Archive/meshing.py)
    * [simple.py](/Scripts/Archive/simple.py)
    * [simple_meshing.py](/Scripts/Archive/simple_meshing.py)
  * [FEM/](/Scripts/FEM)
    * [Good Files/](/Scripts/FEM/Good Files)
    * [real_head.py](/Scripts/FEM/real_head.py)
    * [real_head_10-20.py](/Scripts/FEM/real_head_10-20.py)
    * [sim_settings.yml](/Scripts/FEM/sim_settings.yml)
    * [sphere.py](/Scripts/FEM/sphere.py)
  * [GMSH/](/Scripts/GMSH)
    * [spheres.geo](/Scripts/GMSH/spheres.geo)
  * [Meshing/](/Scripts/Meshing)
    * [electrode_operations.py](/Scripts/Meshing/electrode_operations.py)
    * [gmsh_write.py](/Scripts/Meshing/gmsh_write.py)
    * [mesh_operations.py](/Scripts/Meshing/mesh_operations.py)
    * [modulation_envelope.py](/Scripts/Meshing/modulation_envelope.py)
    * [phm_model_meshing.py](/Scripts/Meshing/phm_model_meshing.py)
  * [Utils/](/Scripts/Utils)
    * [mesh_fixing.py](/Scripts/Utils/mesh_fixing.py)
  * [10-20_elec.mat](/Scripts/10-20_elec.mat)
  * [main.py](/Scripts/main.py)
  * [sphere_meshing.py](/Scripts/sphere_meshing.py)
* [LICENSE](/LICENSE)
* [README.md](/README.md)
