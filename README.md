# Sequences of Earthquakes and Aseismic Slip (seqeas) Python package

Code for the forward modeling part of the studies

> Köhne, T., Mallick, R. & Simons, M.
> Probabilistic estimation of rheological properties in subduction zones
> using sequences of earthquakes and aseismic slip.
> Earth Planets Space 77, 3 (2025). <https://doi.org/10.1186/s40623-024-02121-5>

and

> Köhne, T., Mallick, R., Ragon, T., Zhu, L. & Simons, M.
> Frictional Properties of the Northern Japanese Subduction Zone from
> Probabilistic Earthquake Cycle Inversions.
> (submitted).

The first study concerns the theoretical framework and the synthetic tests in 2D,
whereas the second one applies the framework in 3D to Northern Japan.

## Installation

```bash
git clone https://github.com/tobiscode/seqeas-public
conda create --name seqeas --channel conda-forge --override-channels --file seqeas-public/requirements.txt
conda activate seqeas
pip install ./seqeas-public
```

## Documentation

An API documentation is generated into the `docs` folder. It is hosted on GitHub publicly
at [tobiscode.github.io/seqeas-public](https://tobiscode.github.io/seqeas-public), but you can
also read it locally, e.g., by running `python -m http.server 8080 --bind 127.0.0.1`
from with the documentation folder and then opening a browser. It is created using
`pdoc ./seqeas --math -o docs -d numpy`.

## Examples

All commands assume the user is in the `examples` folder.

### 2D: Power-law viscosity

These are the commands used to generate the target synthetic data for the different test
cases. Running `python 2d_generate_powerlaw_cases.py` without any arguments shows an explanation
of the script arguments.

#### Case (1)

```bash
python 2d_generate_powerlaw_cases.py ./2d_pl_case1/ 200 400 5 -25 -0.0034223 9130 0.01
```

#### Case (2)

```bash
python 2d_generate_powerlaw_cases.py ./2d_pl_case2/ --no-plot-faultvels --no-plot-faultslip --no-plot-eqvels --no-plot-fault --no-plot-phases --no-plot-viscosity --no-plot-viscosity_ts 200 400 5 -10.809 -0.0034223 3947 0.01
```

#### Case (3)

```bash
python 2d_generate_powerlaw_cases.py ./2d_pl_case3/ --no-plot-faultvels --no-plot-faultslip --no-plot-eqvels --no-plot-fault --no-plot-phases --no-plot-viscosity --no-plot-viscosity_ts 200 400 5 -10.77067 -0.0034223 3933 0.01
```

(The horizontal observations are still generated here, but then later ignored by the inversion step.)

#### Case (4)

Uses the data from case (3) with the modified `./2d_pl_case4/fault_subduction.ini` for the inversion models.

#### Case (5)

Uses the data from case (1) with the modified `./2d_pl_case5/fault_subduction.ini` for the inversion models.

### 2D: Rate-dependent friction

Uses the data from case (1) with the modified `./2d_rd/fault_subduction.ini` for the inversion models.

### 3D: Rate-dependent friction in Northern Japan

The scripts related to the 3D study all follow the pattern `3d_step*.py`.

#### Step 0: Data

In addition to the scripts, there are some datasets that need to be downloaded or created,
for which we didn't ask for redistribution rights. These are:

- The [JPL GEONET Japan GPS position timeseries (2024)](https://doi.org/10.48577/jpl.NAHL2B)
  (extracted)
- A maintenance table for the GEONET stations - we reformatted the ones provided
  by GSI directly as part of their RINEX products
- The MIDAS velocity dataset from [their website](https://geodesy.unr.edu/PlugNPlayPortal.php),
  we used [the IGS14 one](https://geodesy.unr.edu/velocities/midas.IGS14.txt).
- Table S1 from [Loveless and Meade (2016)](https://doi.org/10.1016/j.epsl.2015.12.033)

#### Step 1: Post-processing of JPL's GEONET dataset

Edit the `3d_step1_process_timeseries.py` script and enter the necessary data paths,
then run it.

#### Step 2: Reference frame correction

Edit the `3d_step2_correct_japan_euler_poles.py` script and enter the necessary data
path. Change into the `3dmesh` folder (which contains the mesh files) and **then**
run the script with `python ../3d_step2_correct_japan_euler_poles.py`.
Change back out of the folder before continuing to the next step.

#### Step 3: Reformat timeseries to AlTar-compatible format

Run the `3d_step3_convert_synth_to_obs.py` script with appropriate input options
to convert from the NumPy format files from the previous steps into the format
expected by AlTar.

#### Step 4: Create the nonuniform coseismic slip distributions

This step is insofar optional as it is tied to the mesh used and the way the different
slip models are combined. The output files `final_nonuniform_slip_hon.npy` and
`eq_setup_hon.csv` are already provided in the `3dmesh/` subfolder.

For reference, the script `3d_step4_create_final_nonuniform_slip.py` contains the
processing. It requires all the slip models (as listed in the 3D paper supplementary
material) to be downloaded into a folder structure with a
`index_asperityname/year/*.{mat,geojson}` pattern (the root folder of which has
to be defined in the script).

#### Step 5: Run

The `3dmesh/fault_subduction_honshu.ini` configuration file contains all the necessary
inputs to run a SEQEAS 3D forward model based on the provided mesh and the files created
in the previous steps. The following is a minimal example of its CPU-based usage:

```python
from seqeas.subduction3d import SubductionSimulation3D, RateStateSteadyLogarithmic2D, Fault3D
# define paths
CONFIG_FILE = "fault_subduction_honshu.ini"
# create base objects
rheo_dict, fault_dict, sim_dict = \
    SubductionSimulation3D.read_config_file(CONFIG_FILE)
rheo = RateStateSteadyLogarithmic2D(**rheo_dict)
fault = Fault3D.from_cfg_and_files(fault_dict)
sim = SubductionSimulation3D.from_cfg_objs_and_files(sim_dict, rheo, fault)
sim_state, obs_state, surf_disps_sim, surf_disps_locked, \
    surf_disps_outer, surf_disps_lower = sim.run()
```

AlTar contains a CUDA-based implementation of the forward model, see below.

## AlTar integration

To run the inverse model (i.e., estimating the rheological parameters that were used to create the target synthetic
data), any Markov chain Monte Carlo sampler can be used. In our case, we use the
[AlTar Framework](https://github.com/lijun99/altar) which we have extended to incorporate the model
of this Python package. This code can be found in the `seas-devel` branch, with more
detailed links in the subsections below. Before running the next steps, make sure AlTar
is installed into the same environment as seqeas.

### 2D

AlTar links:

- [starting executable script](https://github.com/lijun99/altar/blob/b00a8194cf9c7d1137b25b0242aefaaf7216d2a3/models/seas/bin/SEAS)
- [model definition](https://github.com/lijun99/altar/blob/b00a8194cf9c7d1137b25b0242aefaaf7216d2a3/models/seas/seas/SEAS.py)

For the 2D case, an example `seas.pfg` configuration file can be found in the folder of
Example 1, and it can be used (once AlTar is installed) like this:

```bash
cd ./2d_pl_case1/
OMP_NUM_THREADS=1 SEAS --config=seas.pfg
```

### 3D

AlTar links:

- [starting executable script](https://github.com/lijun99/altar/blob/f35d16071ca68e97183ca949c01e67b2a4927835/models/seas/bin/SEAS3D)
- [model definition](https://github.com/lijun99/altar/blob/f35d16071ca68e97183ca949c01e67b2a4927835/models/seas/seas/cuda/SEAS3D.py)

Make sure to specify the timeseries file created in Step 3 in the `3dmesh/seas.pfg`
configuration file under `dataobs: data_file`. Since we used SLURM and CUDA GPUs to run
this forward and inverse model, there is no simple one-liner as in the 2D case
to get AlTar to run it. We provide our example script in `3dmesh/run.sh`, which should
only be used as a starting point.
