# standard imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path

# import the earthquake cycle simulator
from seqeas.subduction2d import SubductionSimulation

# make synthetic data
if __name__ == "__main__":

    # make argument parser
    prs = ArgumentParser(description="Script to prepare SEAS AlTar input")
    prs.add_argument("case", type=str, help="Target folder that contains the "
                     "'fault_subduction.ini' configuration file")
    prs.add_argument("x_obs_min", type=float,
                     help="Minimum x_1 coordinate of observers [km]")
    prs.add_argument("x_obs_max", type=float,
                     help="Maximum x_1 coordinate of observers [km]")
    prs.add_argument("n_stations", type=int,
                     help="Number of observers [-]")
    prs.add_argument("t_obs_min", type=float,
                     help="Start of observation period [a]")
    prs.add_argument("t_obs_max", type=float,
                     help="End of observation period [a]")
    prs.add_argument("n_obs", type=int,
                     help="Number of observations in observation period [-]")
    prs.add_argument("cd_std", type=float,
                     help="Standard deviation [m] of observation noise")
    prs.add_argument("--no_obs_period", type=str, nargs="*", default=[],
                     help="Remove the specified time periods from the observation timestamps "
                          "(argument format 'from_year,to_year')")
    prs.add_argument("--no-plot-surfdisp", action="store_true",
                     help="Skip observed surface displacements plot")
    prs.add_argument("--no-plot-faultvels", action="store_true",
                     help="Skip fault velocities plot")
    prs.add_argument("--no-plot-faultslip", action="store_true",
                     help="Skip fault slip deficit plot")
    prs.add_argument("--no-plot-eqvels", action="store_true",
                     help="Skip earthquake velocity change plot")
    prs.add_argument("--no-plot-fault", action="store_true",
                     help="Skip fault mesh and observer locations plot")
    prs.add_argument("--no-plot-phases", action="store_true",
                     help="Skip cumulative slip phases plot")
    prs.add_argument("--no-plot-viscosity", action="store_true",
                     help="Skip viscosity structure plot")
    prs.add_argument("--no-plot-viscosity_ts", action="store_true",
                     help="Skip viscosity timeseries plot")
    prs.add_argument("--format", type=str, default="png", choices=["png", "pdf"],
                     help="Plot file format")
    prs.add_argument("--show", action="store_true", help="Show plots after saving")

    # print help if wrongly called
    if len(sys.argv) == 1:
        prs.print_help()
        exit()

    # get case folder and configuration path
    args = prs.parse_args()
    casedir = Path(args.case)
    config_file = casedir / "fault_subduction.ini"
    assert os.path.isdir(casedir), \
        f"Target folder {casedir} does not exist."
    assert os.path.isfile(config_file), \
        f"Target folder {casedir} does not contain a 'fault_subduction.ini' file."

    # create observers
    pts_surf = np.linspace(args.x_obs_min * 1e3, args.x_obs_max * 1e3, num=args.n_stations)
    t_obs = np.linspace(args.t_obs_min, args.t_obs_max, num=args.n_obs)
    for no_obs_period in args.no_obs_period:
        try:
            skip_from, skip_until = [float(v) for v in no_obs_period.split(",")]
        except ValueError as e:
            raise ValueError(f"Can't extract start and end time from {no_obs_period} "
                             "for 'no_obs_period'") from e
        else:
            t_obs = t_obs[np.logical_or(t_obs < skip_from, t_obs > skip_until)]

    # read main configuration
    cfg = SubductionSimulation.read_config_file(config_file)

    # initialize simulation
    print("Running simulation... ", end="", flush=True)
    # create simulation object
    sim = SubductionSimulation.from_config_dict(cfg, t_obs, pts_surf)
    # run simulations and get observations
    full_state, obs_state, surf_disps = sim.run()
    obs_zeroed = sim.zero_obs_at_eq(surf_disps)
    print("done", flush=True)

    # add noise
    obs_noisy = obs_zeroed + np.random.randn(*obs_zeroed.shape) * args.cd_std

    # write observer locations
    try:
        obs_loc_file = casedir / "obs_loc.csv"
        pd.DataFrame(data={"Trench Distances": pts_surf},
                     index=[f"S{n:03d}" for n in range(args.n_stations)]) \
            .rename_axis("Station") \
            .to_csv(obs_loc_file)
    except BaseException as e:
        raise IOError("Error writing observer locations file.") from e
    else:
        print(f"Saved {args.n_stations} observer locations to {obs_loc_file}.")

    # write timestamps
    try:
        t_obs_file = casedir / "t_obs.npy"
        np.save(t_obs_file, t_obs)
    except BaseException as e:
        raise IOError("Error writing timestamps file.") from e
    else:
        print(f"Saved {args.n_obs} timestamps to {t_obs_file}.")

    # write observations
    try:
        obsdata_file = casedir / "surfdisps.txt"
        np.savetxt(obsdata_file, obs_noisy.ravel())
    except BaseException as e:
        raise IOError("Error writing observations file.") from e
    else:
        print(f"Saved {obs_noisy.size} observations to {obsdata_file}.")

    # plot surface displacements
    if not args.no_plot_surfdisp:
        print("Plotting surface displacements... ", end="", flush=True)
        fig1, ax1 = sim.plot_surface_displacements(obs_zeroed, obs_noisy)
        fig1file = casedir / f"surfdisp.{args.format}"
        fig1.savefig(fig1file, dpi=300)
        print(f"saved to {fig1file}.", flush=True)

    # plot fault velocities
    if not args.no_plot_faultvels:
        print("Plotting fault velocities... ", end="", flush=True)
        fig2, ax2 = sim.plot_fault_velocities(full_state)
        fig2file = casedir / f"faultvels.{args.format}"
        fig2.savefig(fig2file, dpi=300)
        print(f"saved to {fig2file}.", flush=True)

    # plot fault slip deficit
    if not args.no_plot_faultslip:
        print("Plotting fault slip deficit... ", end="", flush=True)
        fig3, ax3 = sim.plot_fault_slip(full_state, include_deep=False)
        fig3file = casedir / f"faultslip.{args.format}"
        fig3.savefig(fig3file, dpi=300)
        print(f"saved to {fig3file}.", flush=True)

    # plot earthquake velocities
    if not args.no_plot_eqvels:
        print("Plotting earthquake velocity changes... ", end="", flush=True)
        fig4, ax4 = sim.plot_eq_velocities(full_state)
        fig4file = casedir / f"eqvels.{args.format}"
        fig4.savefig(fig4file, dpi=300)
        print(f"saved to {fig4file}.", flush=True)

    # plot fault mesh and observers
    if not args.no_plot_fault:
        print("Plotting fault setup... ", end="", flush=True)
        fig5, ax5 = sim.plot_fault()
        fig5file = casedir / f"fault.{args.format}"
        fig5.savefig(fig5file, dpi=300)
        print(f"saved to {fig5file}.", flush=True)

    # plot slip phases
    if not args.no_plot_phases:
        print("Plotting slip phases... ", end="", flush=True)
        fig6, ax6 = sim.plot_slip_phases(full_state)
        fig6file = casedir / f"slipphases.{args.format}"
        fig6.savefig(fig6file, dpi=300)
        print(f"saved to {fig6file}.", flush=True)

    # plot viscosity structure
    if not args.no_plot_viscosity:
        print("Plotting viscosity structure... ", end="", flush=True)
        fig7, ax7 = sim.plot_viscosity(full_state)
        fig7file = casedir / f"viscosity.{args.format}"
        fig7.savefig(fig7file, dpi=300)
        print(f"saved to {fig7file}.", flush=True)

    # plot effective viscosity timeseries
    if not args.no_plot_viscosity_ts:
        print("Plotting viscosity timseries... ", end="", flush=True)
        fig8, ax8 = sim.plot_viscosity_timeseries(full_state)
        fig8file = casedir / f"viscosity_timeseries.{args.format}"
        fig8.savefig(fig8file, dpi=300)
        print(f"saved to {fig8file}.", flush=True)

    # show
    if args.show:
        plt.show()
    else:
        plt.close("all")
