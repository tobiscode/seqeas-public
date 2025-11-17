"""
Convert the synthetic observations as output by `process_japan_f5.py` or
the `seqeas` package to an h5 file suitable for AlTar.
"""

import sys
import json
import h5py
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":

    # make argument parser
    prs = ArgumentParser(description="Convert synthetic to noisy observations")
    prs.add_argument("input_file", type=str,
                     help="Input file path to `.npy` file with accompanying `.json` file")
    prs.add_argument("--factor", type=float, default=1.0,
                     help="Rescale the output data by this factor")
    prs.add_argument("--reference-stations", type=str, default=None,
                     help=("JSON-formatted list of strings of reference stations "
                           "to make the output data relative to"))
    prs.add_argument("--obs-loc-file", type=str, default=None,
                     help=("File path of `.csv` file necessary when using "
                           "'--reference-stations'"))
    prs.add_argument("--noise-sd", type=float, default=0.0,
                     help="Noise standard deviation to add to data (after potential rescaling)")
    prs.add_argument("--mask-first-n-obs", type=int, default=0,
                     help="Set all the first n observations at earthquakes to invalid")
    prs.add_argument("--mask-after-n-obs", type=int, default=None,
                     help="Set all the n observations after an earthquake to invalid")
    prs.add_argument("--mask-preseismic", action="store_true",
                     help="Set all observations before the first event to invalid")
    prs.add_argument("--mask-verticals", action="store_true",
                     help="Mask out all vertical observations")
    prs.add_argument("--overwrite", action="store_true",
                     help="Ignore an existing output file and overwrite it")
    prs.add_argument("--output-name", type=str, default=None,
                     help="Output name (without suffix), defaults to input name")

    # print help if wrongly called
    if len(sys.argv) == 1:
        prs.print_help()
        exit()

    # parse arguments
    args = prs.parse_args()

    # get required paths
    data_file = Path(args.input_file)
    info_file = Path(data_file.parent / f"{data_file.stem}.json")
    assert info_file.exists() and info_file.is_file(), f"'{info_file}' not found"
    if args.output_name is None:
        h5_file = Path(data_file.parent / f"{data_file.stem}.h5")
    else:
        h5_file = Path(data_file.parent / f"{args.output_name}.h5")
    assert args.overwrite or (not h5_file.exists()), \
        f"'{h5_file}' already exists and should not be overwritten"

    # load files
    data = np.load(data_file)
    with open(info_file, mode="rt") as f:
        info = json.load(f)
    assert info["ts_shape"] == list(data.shape), \
        f"Data file shape ({data.shape}) mismatch with info file content ({info['ts_shape']})"
    assert max(info["i_eq"]) < data.shape[2], \
        f"Slip indices {info['i_eq']} larger than data file dimension ({data.shape[2]})"
    assert info["i_eq"][0] == 0, \
        f"Slip indices list has to start with 0, got {info['i_eq']}"
    print(f"Loaded info file '{info_file}', matching data file of shape {data.shape}")

    # get data availability mask
    data_mask = np.isfinite(data)

    # manually set earthquake observations to invalid
    new_mask_indices = []
    if args.mask_first_n_obs > 0:
        new_mask_indices.extend([ii for ri in info['i_eq'][1:]
                                 for ii in range(ri, ri + args.mask_first_n_obs)])
    # manually set late observations to invalid
    if args.mask_after_n_obs is not None:
        periods_to = info["i_eq"][1:] + [data.shape[2]]
        new_mask_indices.extend([ii for itieq in range(len(periods_to) - 1)
                                 for ii in range(periods_to[itieq] + args.mask_after_n_obs,
                                                 periods_to[itieq + 1])])
    # manually set preseismic observations to invalid
    if args.mask_preseismic:
        new_mask_indices.extend(list(range(info["i_eq"][1])))
    # apply masked indices
    if len(new_mask_indices) > 0:
        for ii in new_mask_indices:
            data_mask[:, :, ii] = False
        data[~data_mask] = np.nan
        print(f"Masked observations as invalid at time indices {new_mask_indices}")

    # mask out verticals
    if args.mask_verticals:
        data_mask[:, 2, :] = False
        data[:, 2, :] = np.nan
        print("Masked out all vertical observations")

    # print missing data status
    missing_data = data_mask.sum() < data_mask.size
    if missing_data:
        print(f"{(~data_mask).sum()} out of {data_mask.size} observations are missing")

    # make all observations relative to the mean motion of the reference stations
    data_rel = data.copy()
    if args.reference_stations and args.obs_loc_file:
        # get indices of reference stations
        obs_loc = pd.read_csv(args.obs_loc_file, index_col=0)
        names_all = obs_loc.index.astype(str).tolist()
        names_ref = json.loads(args.reference_stations)
        assert len(names_all) == data.shape[0] > 0
        indices_ref = [names_all.index(rn) for rn in names_ref]
        # subset the data
        data_ref = data_rel[indices_ref, :, :].reshape(-1, 3, data.shape[2])
        assert np.all(np.isfinite(data_ref)), \
            "There are NaNs in the reference station timeseries"
        # remove the mean
        data_ref_mean = np.mean(data_ref, axis=0, keepdims=True)
        data_rel -= data_ref_mean
        print(f"Removed reference motion calculated from stations {names_ref} "
              f"(indices {indices_ref})")

    # reset the observations to zero at the supplied indices (0 and all earthquakes)
    # or the first observation after an index but before the next
    data_zeroed = data_rel.copy()
    if not missing_data:
        # all the data is present, can use vectorized subtraction
        for i in info["i_eq"]:
            data_zeroed[:, :, i:] -= data_zeroed[:, :, i][:, :, None]
    else:
        # we have missing data that could be anywhere
        # rather than optimize the ones that are complete, use the same loop
        # logic as the CUDA code has to use
        periods_from = info["i_eq"]
        periods_to = info["i_eq"][1:] + [data.shape[2]]
        for i_stat in range(data_zeroed.shape[0]):
            for i_comp in range(data_zeroed.shape[1]):
                # iterate over all periods between bounds or earthquakes
                for i_from, i_to in zip(periods_from, periods_to):
                    offset = None
                    for j in range(i_from, i_to):
                        # skip if data is masked
                        if data_mask[i_stat, i_comp, j]:
                            if offset is None:
                                # save new offset for this period
                                offset = data_zeroed[i_stat, i_comp, j]
                            # apply current offset
                            data_zeroed[i_stat, i_comp, j] -= offset
    print(f"Reset observations to zero at indices {info['i_eq']}")

    # rescale
    if args.factor != 1:
        data_zeroed *= args.factor
        if args.reference_stations and args.obs_loc_file:
            data_ref_mean *= args.factor
        print(f"Rescaled data by factor = {args.factor}")

    # add white noise
    if args.noise_sd > 0:
        data_zeroed_noisy = data_zeroed + \
            np.random.default_rng().normal(scale=args.noise_sd, size=data_zeroed.shape)
        print(f"Added white noise with standard deviation = {args.noise_sd}")
    else:
        data_zeroed_noisy = data_zeroed

    # transpose to match GPU shape
    data_zeroed_noisy = data_zeroed_noisy.T
    data_mask = data_mask.T

    # write H5 file for AlTar
    with h5py.File(h5_file, "w") as f:
        f.create_dataset("input", data=data)
        f.create_dataset("intermediate", data=data_zeroed)
        f.create_dataset("output", data=data_zeroed_noisy)
        f.create_dataset("flat", data=data_zeroed_noisy.ravel())
        f.create_dataset("shape", data=np.asarray(data_zeroed_noisy.shape))
        f.create_dataset("reset_indices", data=np.asarray(info["i_eq"]))
        f.create_dataset("factor", data=np.asarray(args.factor))
        f.create_dataset("noise_sd", data=np.asarray(args.noise_sd))
        if args.reference_stations and args.obs_loc_file:
            f.create_dataset("indices_ref", data=np.asarray(indices_ref))
            f.create_dataset("names_ref", data=np.asarray(names_ref, dtype=np.bytes_))
            f.create_dataset("reference", data=data_ref_mean)
        else:
            f.create_dataset("indices_ref", data=np.asarray([]))
            f.create_dataset("names_ref", data=np.asarray([]))
            f.create_dataset("reference", data=np.asarray([]))
        if missing_data:
            f.create_dataset("mask", data=data_mask.ravel())
        else:
            f.create_dataset("mask", data=np.asarray([]))
    print(f"Wrote '{h5_file}' with {data_zeroed_noisy.size} observations")
