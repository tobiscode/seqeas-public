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
