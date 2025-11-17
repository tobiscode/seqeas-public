"""
Simple script to show and postprocess the JPL timeseries for Japan
"""

# general imports
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from pathlib import Path
from matplotlib.patches import Polygon
from matplotlib.path import Path as MPath
from sklearn.linear_model import MultiTaskLassoCV
from multiprocessing import Pool
from scipy.spatial import ConvexHull
from disstans.tools import eulerpole2rotvec
from pyproj import Transformer, CRS

# from cmcrameri import cm
# from itertools import product

# DISSTANS
import disstans

# warnings.filterwarnings("error", category=RuntimeWarning)

# define mode of operation
RUN_DISSTANS = True
RUN_DOWNSAMPLING = True
MAKE_PLOTS = True

# define input paths
NETWORK_DATA_PATH = None  # set path to folder that contains the Moore (2024) dataset
MAINT_PATH = None  # set path to a maintenance file
MIDAS_FNAME = None  # set to the file path of a downloaded MIDAS velocity file
LM16_FNAME = None  # set to the file path of the downloaded Loveless & Meade Table S1
assert NETWORK_DATA_PATH is not None, "'NETWORK_DATA_PATH' must be set"
assert MAINT_PATH is not None, "'MAINT_PATH' must be set"
assert MIDAS_FNAME is not None, "'MIDAS_FNAME' must be set"
assert LM16_FNAME is not None, "'LM16_FNAME' must be set"
NETWORK_LOC_FILE = "data/nominalPositions.llh"
CATALOG_PATH = "data/locations_eq_japan.csv"
# define output paths (will be used to reload if desired, to ssave time)
NETWORK_FNAME = None  # set filename where to save/load processed Network object to
assert NETWORK_FNAME is not None, "'NETWORK_FNAME' must be set"
SEC_VEL_DATA_BASENAME = "out/japan/japan_jpl_secular"
SEC_VEL_FIG_IGS_FNAME = "out/japan/japan_jpl_secular_igs.png"
SEC_VEL_FIG_OK_FNAME = "out/japan/japan_jpl_secular_ok.png"
TOH_STEPS_DATA_FNAME = "out/japan/japan_jpl_tohoku.csv"
OBS_LOC_FNAME = "out/japan/obs_loc_jpl.csv"
T_EVAL_BASENAME = "out/japan/japan_jpl_ds_t_eval"
SURF_DISPS_BASENAME = "out/japan/japan_jpl_ds_timeseries"
SEAS_STATIONS_FNAME = "out/japan/japan_jpl_ds_stations.json"
SEAS_TIME_BASENAME = "out/japan/japan_jpl_final/times"
SEAS_DATA_BASENAME = "out/japan/japan_jpl_final/seas"
NONSEAS_DATA_BASENAME = "out/japan/japan_jpl_final/nonseas"

# define other settings
NJAPAN_POLY = np.array([[34.5, 138.5], [34.5, 147], [46, 147], [46, 138.5]])
ITOHOKU = 1
PS_THRESHOLDS = [0.5, 0.1]
ONE_YEAR = pd.Timedelta(365, "D")
UTC_JST_DIFF = pd.Timedelta(9, "h")
HONSHU_PATH = Polygon(
    np.array(
        [
            [141.8795803162128, 34.2150684494421],
            [144.51023414944223, 41.02029264657463],
            [143.03169148406232, 41.826581931782755],
            [141.64915808265295, 42.26861050724369],
            [141.26512102666948, 42.14059651777063],
            [140.36263394521433, 42.19752364698729],
            [139.63296353891639, 44.924043069107256],
            [137.9432004927101, 44.610489817173715],
            [132.64348912063258, 39.62964904361161],
            [138.03920975673122, 37.11608332386878],
            [140.76587285393128, 35.772288288287214],
            [141.8795803162128, 34.2150684494421],
        ]
    )
).get_path()
UTMZONE = 54
# ITRF2014 plate motion model: https://doi.org/10.1093/gji/ggx136
ROT_EU = np.deg2rad(np.array([-0.0235, -0.1476, 0.2140])) / 1e6  # [rad/a]
ROT_PA = np.deg2rad(np.array([-0.1135, 0.2907, -0.6025])) / 1e6  # [rad/a]
ROT_OK = eulerpole2rotvec(np.deg2rad([-94.36, 26.84, 0.194])) / 1e6  # [rad/a]
# map projections
CRS_GUI = ccrs.Mercator()
CRS_PC = ccrs.PlateCarree()
CRS_LLA = ccrs.Geodetic()
CRS_XYZ = ccrs.Geocentric()
CRS_WGS = CRS("WGS84")
CRS_UTM = CRS.from_epsg(32654)


def durbin_watson(e):
    return np.nansum(np.diff(e, axis=0) ** 2, axis=0) / np.nansum(e**2, axis=0)


def wrap_polyfitl1(x, y, degree=5):
    if x.size == 0:
        return None
    else:
        return PolyFitL1(x, y, degree)


class PolyFitL1:
    def __init__(self, x, y, degree=5):
        # input check
        nobs = x.size
        assert y.shape[0] == nobs
        assert y.ndim == 2
        self.y = y.copy()
        # quick return if not enough for a robust fit
        if nobs < 5:
            self.fit = np.nanmedian(self.y, axis=0, keepdims=True)
            return
        # check for NaNs
        self.isfinite = np.all(np.isfinite(self.y), axis=1)
        xfin = x[self.isfinite]
        yfin = y[self.isfinite, :]
        self.degree = min(int(degree), xfin.size)
        # get bounds of x
        xmin, xmax = xfin.min(), xfin.max()
        self.xdelta = xmax - xmin
        self.xcenter = (xmax + xmin) / 2
        # get mapping matrix
        self.G = self.get_G(xfin)
        # solve
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            try:
                self.fit = MultiTaskLassoCV(fit_intercept=False).fit(self.G, yfin)
            except ConvergenceWarning:
                self.fit = np.nanmedian(self.y, axis=0, keepdims=True)

    def transform(self, xorig):
        return (xorig - self.xcenter) / (self.xdelta / 2)

    def get_G(self, x):
        return self.transform(x).reshape(-1, 1) ** np.arange(self.degree + 1).reshape(
            1, -1
        )

    def get_fit_model(self):
        try:
            model = np.full_like(self.y, np.nan)
            model[self.isfinite, :] = self.fit.predict(self.G)
            return model
        except AttributeError:
            return np.full_like(self.y, self.fit)

    def get_fit_residual(self):
        return self.y - self.get_fit_model()

    def predict(self, x):
        try:
            return self.fit.predict(self.get_G(x))
        except AttributeError:
            return np.full((x.size, self.fit.shape[1]), self.fit)


if __name__ == "__main__":

    # use multiprocessing
    os.environ["OMP_NUM_THREADS"] = "1"
    disstans.defaults["general"]["num_threads"] = 30

    # load data
    if not RUN_DISSTANS:
        with open(NETWORK_FNAME, mode="rb") as f:
            net = pickle.load(f)
        njapan_path = Polygon(np.array(NJAPAN_POLY)).get_path()
        station_loc_df = net.station_locations
        catalog = pd.read_csv(CATALOG_PATH)
        catalog["Origin_Time(JST)"] -= UTC_JST_DIFF
        assert catalog["Date"].is_monotonic_increasing
        eq_dict_emp = disstans.earthquakes.empirical_prior(net, CATALOG_PATH)
        for k, v in eq_dict_emp.items():
            eq_dict_emp[k] = [vv - UTC_JST_DIFF for vv in v]
        check_times = sorted(list(set([t for d in eq_dict_emp.values() for t in d])))
        t_min = min([stat["final"].time.min() for stat in net])
        t_max = max([stat["final"].time.max() for stat in net])

    # run full model
    else:

        # create raw JPL network file from scratch
        print("create network")
        net = disstans.Network(
            "JPL Hon/Hok",
            default_location_path=NETWORK_LOC_FILE,
            auto_add=True,
            auto_add_filter=NJAPAN_POLY,
        )
        for sta_name in net.station_names:
            tspath = Path(NETWORK_DATA_PATH) / f"{sta_name}.series"
            if tspath.is_file():
                ts = disstans.timeseries.GipsyXTimeseries(tspath)
                if (ts.reliability < 0.5) or (ts.length < ONE_YEAR):
                    del net[sta_name]
                    print(f"Deleted {sta_name} because of quality of data.")
                else:
                    # remove day of Tohoku earthquake since it's inconsistent
                    # when EQ effect is visible
                    ts.cut(
                        t_min="2011-03-11 00:00:00",
                        t_max="2011-03-12 00:00:00",
                        keep_inside=False,
                    )
                    net[sta_name]["raw"] = ts
            else:
                del net[sta_name]
                print(f"Deleted {sta_name} because of missing data.")
        station_loc_df = net.station_locations
        sta_in_honshu = HONSHU_PATH.contains_points(station_loc_df.values[:, [1, 0]])
        honshu_sta_names = sorted(station_loc_df.index[sta_in_honshu].tolist())

        # legacy stuff
        njapan_path = Polygon(NJAPAN_POLY).get_path()
        inside_njapan = njapan_path.contains_points(
            net.station_locations.iloc[:, :2].values
        )
        # manual adjustments
        net["0562"]["raw"].cut(t_max="2010-01-01")

        # # show
        # net.gui()

        # low-pass filter using a median function, then clean the timeseries
        # (using the settings in the config file)
        # compute reference
        net.call_func_ts_return("median", ts_in="raw", ts_out="raw_filt", kernel_size=7)
        # remove outliers
        net.call_func_no_return(
            "clean",
            ts_in="raw",
            reference="raw_filt",
            ts_out="raw_clean",
            clean_kw_args={"std_bad": 1000},
        )
        # get the residual for each station
        net.math("raw_filt_res", "raw_clean", "-", "raw_filt")
        # remove obsolete timeseries
        net.remove_timeseries("raw_filt")

        # estimate the common mode, either with a visualization of the result or not
        # (same underlying function)
        # net.graphical_cme(ts_in="raw_filt_res", ts_out="common", method="ica")
        # calculate common mode
        net.call_netwide_func(
            "decompose", ts_in="raw_filt_res", ts_out="common", method="ica"
        )
        # now remove the common mode, call it the "intermed" timeseries,
        for station in net:
            station.add_timeseries(
                "intermed",
                station["raw_clean"] - station["common"],
                override_data_cols=station["raw"].data_cols,
            )
        # remove obsolete timeseries
        net.remove_timeseries("common", "raw_clean")

        # clean again
        net.call_func_ts_return(
            "median", ts_in="intermed", ts_out="intermed_filt", kernel_size=7
        )
        net.call_func_no_return(
            "clean", ts_in="intermed", reference="intermed_filt", ts_out="final"
        )
        net.remove_timeseries("intermed", "intermed_filt", "raw_filt_res")

        # use uncertainty data from raw timeseries
        net.copy_uncertainties(origin_ts="raw", target_ts="final")

        # reset observations to zero at beginning
        # (fix for non-convergence: 0511 and 0512 have huge absolute values)
        for station in net:
            station["final"].data -= station["final"].data.iloc[0, :]

        # do empirical check for SEAS-modeled earthquakes
        catalog = pd.read_csv(CATALOG_PATH)
        catalog["Origin_Time(JST)"] -= UTC_JST_DIFF
        assert catalog["Date"].is_monotonic_increasing
        eq_dict_emp = disstans.earthquakes.empirical_prior(net, CATALOG_PATH)
        for k, v in eq_dict_emp.items():
            eq_dict_emp[k] = [vv - UTC_JST_DIFF for vv in v]
        check_times = sorted(list(set([t for d in eq_dict_emp.values() for t in d])))

        # get trend changes for multi-logarithmic modeling
        trend_change_noT = net.get_trend_change(
            ts_description="final",
            spatial_mean=20,
            check_times=check_times[:ITOHOKU],
            window_size=28,
            return_signs=False,
        )
        trend_change_Toh = net.get_trend_change(
            ts_description="final",
            check_times=[check_times[ITOHOKU]],
            window_size=280,
            return_signs=False,
        )
        trend_change = {
            sta_name: trend_change_noT[sta_name] + trend_change_Toh[sta_name]
            for sta_name in eq_dict_emp.keys()
        }
        # need to mask out the dates that weren't in eq_dict_emp for any station
        for sta_name, eq_emp in eq_dict_emp.items():
            for i, ct in enumerate(check_times):
                if ct not in eq_emp:
                    trend_change[sta_name][i] = [None] * len(trend_change[sta_name][i])
        # stack as array
        tc_arr = np.stack([tc for tc in trend_change.values()]).astype(float)

        # # make a plot for each tested earthquake showing trend change in 3D
        # CRS_GUI = ccrs.Mercator()
        # CRS_PC = ccrs.PlateCarree()
        # station_locs = station_loc_df.values
        # for ieq, eqcat in catalog.iterrows():
        #     fig_map = plt.figure()
        #     ax_map = fig_map.add_subplot(projection=CRS_GUI)
        #     ax_map.gridlines(draw_labels=True, zorder=-1)
        #     ax_map.set_extent([138, 152, 34, 47])
        #     q = ax_map.quiver(station_locs[:, 1], station_locs[:, 0],
        #                       tc_arr[:, ieq, 0], tc_arr[:, ieq, 1],
        #                       scale=100, transform=CRS_PC)
        #     ax_map.quiverkey(q, 0.85, 0.85, 10, r"10 $\Delta$ horizontal [mm/a]")
        #     pts = ax_map.scatter(station_locs[:, 1], station_locs[:, 0],
        #                          c=tc_arr[:, ieq, 2], vmin=-1, vmax=1,
        #                          cmap=cm.vik, transform=CRS_PC)
        #     ax_map.scatter(eqcat["Longitude(°)"], eqcat["Latitude(°)"],
        #                    color=cm.bam(0.9), s=500, marker="*",
        #                    zorder=-1, transform=CRS_PC)
        #     ax_map.annotate(f"$M_w$ {eqcat['MT_Magnitude(Mw)']}",
        #                     (eqcat["Longitude(°)"], eqcat["Latitude(°)"]),
        #                     xytext=(20, 0), textcoords="offset pixels",
        #                     transform=CRS_PC, annotation_clip=True)
        #     fig_map.colorbar(pts, label=r"$\Delta$ up [mm/a]")
        #     fig_map.suptitle(f"Trend change for EQ on {check_times[ieq]}")

        # add simple models
        models = {
            "Annual": {
                "type": "Sinusoid",
                "kw_args": {"period": 365.25, "t_reference": "2000-01-01"},
            },
            "Biannual": {
                "type": "Sinusoid",
                "kw_args": {"period": 365.25 / 2, "t_reference": "2000-01-01"},
            },
            "Linear": {
                "type": "Polynomial",
                "kw_args": {"order": 1, "t_reference": "2000-01-01", "time_unit": "Y"},
            },
        }
        net.add_local_models(models=models, ts_description="final")

        # first fit
        net.fitevalres(
            "final",
            solver="linear_regression",
            output_description="model_1",
            residual_description="resid_1",
            use_data_covariance=False,
        )

        # load maintenance table
        # file has columns: [site;year;month;day;code;comment]
        maint_table, maint_dict = disstans.tools.parse_maintenance_table(
            csvpath=MAINT_PATH,
            sitecol=0,
            datecols=[1, 2, 3],
            delimiter=";",
            codecol=4,
            siteformatter=lambda site: site[-4:],
            verbose=True,
        )

        # run step detector
        stepdet = disstans.processing.StepDetector(kernel_size=61, kernel_size_min=21)
        step_table, _ = stepdet.search_network(net, "final")
        step_table_90varred = step_table[step_table["varred"] > 0.9]

        # # show map
        # net.gui(timeseries="final",
        #         annotate_stations=False,
        #         rms_on_map={"ts": "resid_1", "c_max": 100},
        #         mark_events=step_table_90varred)

        # go through the detected steps and add them as models
        # use different models for the EQs we care about and everything else
        STEP_MDL_NAMES = ["BigEQ", "SmallEQ", "Maintenance"]
        for sta_name, steptimes in step_table_90varred.groupby("station").agg("time"):
            # initialize different target step models
            big_steps = []
            small_steps = []
            maint_steps = []
            # loop over detected steps for this station
            for st in steptimes:
                if any(
                    [
                        abs((st_big - st).ceil("D")) < pd.Timedelta(3, "D")
                        for st_big in check_times[:ITOHOKU]
                    ]
                ):
                    # it's one of the non-Tohoku EQs we're tracking
                    eqix = np.argmax(
                        [
                            abs((st_big - st).ceil("D")) < pd.Timedelta(5, "D")
                            for st_big in check_times[:ITOHOKU]
                        ]
                    )
                    big_steps.append(check_times[eqix])
                elif (
                    abs((check_times[ITOHOKU] - st).ceil("D")) < pd.Timedelta(3, "D")
                ) or (
                    (net[sta_name]["final"].time[-1] >= check_times[ITOHOKU])
                    and (
                        st
                        == net[sta_name]["final"].time[
                            net[sta_name]["final"].time >= check_times[ITOHOKU]
                        ][0]
                    )
                ):
                    # it's the first acquisition after the Tohoku M9 EQ
                    big_steps.append(check_times[ITOHOKU])
                elif st in maint_dict[sta_name]:
                    # it's probably a maintenance event
                    maint_steps.append(st)
                else:
                    # it's something else
                    small_steps.append(st)
            # create the step models
            for mdl_desc, steplist in zip(
                STEP_MDL_NAMES, [big_steps, small_steps, maint_steps]
            ):
                if len(steplist) > 0:
                    mdl = disstans.models.Step(steplist)
                    net[sta_name].add_local_model("final", mdl_desc, mdl)

        # # add a model for the postseismic transients for each big earthquake
        # if it exceeds a certain threshold
        for sta_name, tc in trend_change.items():
            for i, thresh in enumerate(PS_THRESHOLDS):
                tci = np.array(tc[i]).astype(float)
                if (not np.any(np.isnan(tci))) and (np.linalg.norm(tci) >= thresh):
                    multilog = disstans.models.Logarithmic(
                        [10, 100, 1000], t_reference=check_times[i]
                    )  # ,
                    # sign_constraint=np.sign(tci).astype(int).tolist())
                    net[sta_name].add_local_model("final", f"BigPost{i}", multilog)
                    # # add a linear as well
                    # postlin = disstans.models.Polynomial(order=1, min_exponent=1,
                    #                                      t_reference=tref, t_start=tref,
                    #                                      zero_before=True, time_unit="Y")
                    # net[sta_name].add_local_model("final", f"BigLin{i}", postlin)
            # i = 2
            # thresh = PS_THRESHOLDS[i]
            # tci = np.array(tc[i]).astype(float)
            # if (not np.any(np.isnan(tci))) and (np.linalg.norm(tci.astype(float)) >= thresh):
            #     # add a decaying spline set to model postseismic deformation
            #     net[sta_name].add_local_model_kwargs(
            #         "final",
            #         {"BigPost2": {"type": "DecayingSplineSet",
            #                       "kw_args": {"degree": 2,
            #                                   "t_center_start": check_times[ITOHOKU],
            #                                   "t_start": check_times[ITOHOKU],
            #                                   "time_unit": "D",
            #                                   "list_scales": [3, 10, 30, 100, 300, 1000],
            #                                   "list_num_splines": 2}}})

        # find the area where a postseismic transient or big eq step was added, and then add
        # that transient or step to all stations within it that don't have it yet
        # (e.g., Tohoku postseimic for stations that didn't exist in 2011)
        for i in range(len(PS_THRESHOLDS)):
            # transient
            sta_has_post = [
                sta.name for sta in net if f"BigPost{i}" in sta.models["final"]
            ]
            if len(sta_has_post) == 0:
                continue
            sta_has_post_locs = station_loc_df.loc[sta_has_post, :]
            chull_has_post = ConvexHull(sta_has_post_locs.values[:, :2])
            path_has_post = MPath(chull_has_post.points[chull_has_post.vertices, :])
            for sta in net:
                if (
                    path_has_post.contains_point(sta.location[:2])
                    and f"BigPost{i}" not in sta.models["final"]
                ):
                    multilog = disstans.models.Logarithmic(
                        [10, 100, 1000], t_reference=check_times[i]
                    )
                    sta.add_local_model("final", f"BigPost{i}", multilog)
            # plt.figure()
            # plt.scatter(station_loc_df.iloc[:, 1], station_loc_df.iloc[:, 0])
            # plt.scatter(sta_has_post_locs.iloc[:, 1], sta_has_post_locs.iloc[:, 0])
            # plt.title(f"{i} - BigPost{i} pre")
            # sta_has_post = [sta.name for sta in net if f"BigPost{i}" in sta.models["final"]]
            # sta_has_post_locs = station_loc_df.loc[sta_has_post, :]
            # plt.figure()
            # plt.scatter(station_loc_df.iloc[:, 1], station_loc_df.iloc[:, 0])
            # plt.scatter(sta_has_post_locs.iloc[:, 1], sta_has_post_locs.iloc[:, 0])
            # plt.title(f"{i} - BigPost{i} post")
        for i in range(len(PS_THRESHOLDS)):
            # step
            sta_has_step = [
                sta.name
                for sta in net
                if "BigEQ" in sta.models["final"]
                if check_times[i] in sta.models["final"]["BigEQ"].timestamps
            ]
            if len(sta_has_step) == 0:
                continue
            sta_has_step_locs = station_loc_df.loc[sta_has_step, :]
            chull_has_step = ConvexHull(sta_has_step_locs.values[:, :2])
            path_has_step = MPath(chull_has_step.points[chull_has_step.vertices, :])
            for sta in net:
                if path_has_step.contains_point(sta.location[:2]):
                    if "BigEQ" in sta.models["final"]:
                        if (
                            check_times[i]
                            not in sta.models["final"]["BigEQ"].timestamps
                        ):
                            sta.models["final"]["BigEQ"].add_step(
                                check_times[i].isoformat()
                            )
                    else:
                        mdl = disstans.models.Step([check_times[i]])
                        sta.add_local_model("final", "BigEQ", mdl)
        #     plt.figure()
        #     plt.scatter(station_loc_df.iloc[:, 1], station_loc_df.iloc[:, 0])
        #     plt.scatter(sta_has_step_locs.iloc[:, 1], sta_has_step_locs.iloc[:, 0])
        #     plt.title(f"{i} - EQ {check_times[i]} pre")
        #     sta_has_step = [sta.name for sta in net
        #                     if "BigEQ" in sta.models["final"]
        #                     if check_times[i] in sta.models["final"]["BigEQ"].timestamps]
        #     sta_has_step_locs = station_loc_df.loc[sta_has_step, :]
        #     plt.figure()
        #     plt.scatter(station_loc_df.iloc[:, 1], station_loc_df.iloc[:, 0])
        #     plt.scatter(sta_has_step_locs.iloc[:, 1], sta_has_step_locs.iloc[:, 0])
        #     plt.title(f"{i} - EQ {check_times[i]} post")
        # plt.show()

        # run again
        net.fitevalres(
            "final",
            solver="linear_regression",
            output_description="model_2",
            residual_description="resid_2",
            use_data_covariance=False,
        )

        # run step detector again on residual
        step_table2, _ = stepdet.search_network(net, "resid_2")
        step_table2_50prob_50var0 = step_table2[
            (step_table2["probability"] > 50) & (step_table2["var0"] > 50)
        ]

        # # show map
        # net.gui(timeseries="final",
        #         annotate_stations=False,
        #         rms_on_map={"ts": "resid_2", "c_max": 50},
        #         mark_events=step_table2_50prob_50var0)

        # add new steps
        for sta_name, steptimes in step_table2_50prob_50var0.groupby("station").agg(
            "time"
        ):
            # initialize different target step models
            big_steps = []
            small_steps = []
            maint_steps = []
            # loop over detected steps for this station
            for st in steptimes:
                if any(
                    [
                        abs((st_big - st).ceil("D")) < pd.Timedelta(3, "D")
                        for st_big in check_times[:ITOHOKU]
                    ]
                ):
                    # it's one of the non-Tohoku EQs we're tracking
                    eqix = np.argmax(
                        [
                            abs((st_big - st).ceil("D")) < pd.Timedelta(5, "D")
                            for st_big in check_times[:ITOHOKU]
                        ]
                    )
                    big_steps.append(check_times[eqix])
                elif (
                    abs((check_times[ITOHOKU] - st).ceil("D")) < pd.Timedelta(3, "D")
                ) or (
                    (net[sta_name]["final"].time[-1] >= check_times[ITOHOKU])
                    and (
                        st
                        == net[sta_name]["final"].time[
                            net[sta_name]["final"].time >= check_times[ITOHOKU]
                        ][0]
                    )
                ):
                    # it's the first acquisition after the Tohoku M9 EQ
                    big_steps.append(check_times[ITOHOKU])
                elif sta_name in maint_dict and st in maint_dict[sta_name]:
                    # it's probably a maintenance event
                    maint_steps.append(st)
                else:
                    # it's something else
                    small_steps.append(st)
            # add or modify the existing step models
            for mdl_desc, steplist in zip(
                STEP_MDL_NAMES, [big_steps, small_steps, maint_steps]
            ):
                if len(steplist) > 0:
                    mdl = net[sta_name].models["final"]
                    if mdl_desc in mdl:
                        for eqt in steplist:
                            if eqt.isoformat() not in mdl[mdl_desc].steptimes:
                                mdl[mdl_desc].add_step(eqt.isoformat())
                    else:
                        mdl = disstans.models.Step(steplist)
                        net[sta_name].add_local_model("final", mdl_desc, mdl)

        # run again
        net.fitevalres(
            "final",
            solver="linear_regression",
            output_description="model_3",
            residual_description="resid_3",
            use_data_covariance=False,
        )

        # # show map
        # net.gui(timeseries="final",
        #         annotate_stations=False,
        #         rms_on_map={"ts": "resid_3", "c_max": 50})

        # # some manual adjustments
        # tref = check_times[ITOHOKU].to_pydatetime().date()
        # for sta_name in ["111184", "960594", "111185"]:
        #     net[sta_name].add_local_model(
        #         "final",
        #         "BigPost2",
        #         disstans.models.Logarithmic(
        #             [10, 100, 1000],
        #             t_reference=tref))
        #     # net[sta_name].add_local_model(
        #     #     "final",
        #     #     "BigLin2",
        #     #     disstans.models.Polynomial(order=1, min_exponent=1,
        #     #                                t_reference=tref, t_start=tref,
        #     #                                zero_before=True, time_unit="Y"))

        # add transient models
        t_min = min([stat["final"].time.min() for stat in net])
        t_max = max([stat["final"].time.max() for stat in net])
        new_models = {
            "Transient": {
                "type": "SplineSet",
                "kw_args": {
                    "degree": 2,
                    "t_center_start": t_min,
                    "t_center_end": t_max,
                    "time_unit": "D",
                    "list_scales": [30, 90],
                },
            },
            "AnnualDev": {
                "type": "AmpPhModulatedSinusoid",
                "kw_args": {
                    "period": 365.25,
                    "degree": 2,
                    "num_bases": t_max.year - t_min.year + 2,
                    "t_start": f"{t_min.year}-01-01",
                    "t_end": f"{t_max.year + 1}-01-01",
                },
            },
        }
        net.add_local_models(new_models, "final")

        # # gather stats
        # gridsearch_stats = {}

        # # do a basic grid search
        # for pen, penfactor, eps, scale, percentile, coupled in product([30, 100],
        #                                                                [3, 10],
        #                                                                [1e-5, 1e-6],
        #                                                                [1, 10],
        #                                                                [0.3, 0.5],
        #                                                                [True, False]):

        #     if (pen == 30) and (penfactor == 3):
        #         continue
        #     if (pen == 100) and (penfactor == 10) and (scale == 10) and (coupled is False):
        #         continue

        #     testname = (f"pen={pen}, penfactor={penfactor}, eps={eps:e}, "
        #                 f"scale={scale}, percentile={percentile}, coupled={coupled}")
        #     tqdm.write(f"\n{testname}\n")

        # run spatial fit for final timeseries
        pen, penfactor, eps, scale, percentile, coupled = 100, 10, 1e-6, 10, 0.3, True

        testname = (
            f"pen={pen}, penfactor={penfactor}, eps={eps:e}, "
            f"scale={scale}, percentile={percentile}, coupled={coupled}"
        )
        tqdm.write(f"\n{testname}\n")

        # perform fit
        net.remove_timeseries("model_srw", "resid_srw")
        rw_func = disstans.solvers.InverseReweighting(eps=eps, scale=scale)
        penalties = [pen, pen, pen * penfactor]
        stats = net.spatialfit(
            "final",
            penalty=penalties,
            spatial_l0_models=["Transient"],
            spatial_reweight_iters=10,
            spatial_reweight_percentile=percentile,
            spatial_reweight_max_rms=1,
            spatial_reweight_max_changed=1e-6,
            reweight_func=rw_func,
            dist_weight_min=100,
            formal_covariance=False,
            use_data_variance=True,
            use_data_covariance=False,
            local_reweight_coupled=coupled,
            verbose=True,
            no_pbar=False,
            extended_stats=True,
            keep_mdl_res_as=("model_srw", "resid_srw"),
        )

        # decompose velocities
        v_mod = np.stack(
            [
                net[sta_name].models["final"]["Linear"].par[1, :]
                for sta_name in honshu_sta_names
            ],
            axis=0,
        )
        v_fit_euler = net.euler_rot_field(
            "final", "Linear", subset_stations=honshu_sta_names, extrapolate=False
        )[0]
        v_fit_hom = net.hom_velocity_field(
            "final",
            "Linear",
            utmzone=UTMZONE,
            subset_stations=honshu_sta_names,
            extrapolate=False,
        )[0]
        rms_euler = np.sqrt(np.mean((v_fit_euler - v_mod[:, :2]) ** 2))
        rms_hom = np.sqrt(np.mean((v_fit_hom - v_mod[:, :2]) ** 2))
        # estimate ramp in verticals
        G = np.stack(
            [
                np.ones(v_mod.shape[0]),
                station_loc_df.loc[honshu_sta_names, "Longitude [°]"].values,
                station_loc_df.loc[honshu_sta_names, "Latitude [°]"].values,
            ],
            axis=1,
        )
        ramp = np.linalg.lstsq(G, v_mod[:, 2], rcond=None)[0]
        vert_fit = G @ ramp
        rms_vert = np.sqrt(np.mean((vert_fit - v_mod[:, 0]) ** 2))
        # get autocorrelation of residuals
        resid_df = net.export_network_ts(
            ts_description="resid_srw", subset_stations=honshu_sta_names
        )
        dw = np.array([durbin_watson(v.data.values).mean() for v in resid_df.values()])

        # save
        tqdm.write(f"\neuler={rms_euler}, homvel={rms_hom}, vert={rms_vert}, dw={dw}\n")
        # gridsearch_stats[testname] = (rms_euler, rms_hom, rms_vert, dw)

        # # show map
        # net.gui(timeseries="final",
        #         annotate_stations=False,
        #         sum_models=False,
        #         rms_on_map={"ts": "resid_srw", "c_max": 30},
        #         scalogram_kw_args={"ts": "final", "model": "Transient"})

        # # copy final timeseries for unregularized and L2-regularized fits using
        # # frozen spline set and with covariances
        # net.copy_timeseries("final", "final_unreg")
        # net.copy_timeseries("final", "final_l2")

        # # freeze spline models and run linear regression with covariances
        # net.freeze("final_unreg", ["Transient"], 0.1)
        # net.fitevalres("final_unreg",
        #                solver="linear_regression",
        #                output_description="model_unreg",
        #                residual_description="resid_unreg")

        # # calculate RMS as above
        # v_mod_unreg = np.stack([net[sta_name].models["final_unreg"]["Linear"].par[1, :]
        #                         for sta_name in honshu_sta_names], axis=0)
        # v_fit_euler_unreg = net.euler_rot_field("final_unreg", "Linear",
        #                                         subset_stations=honshu_sta_names,
        #                                         extrapolate=False)[0]
        # v_fit_hom_unreg = net.hom_velocity_field("final_unreg", "Linear",
        #                                          utmzone=UTMZONE,
        #                                          subset_stations=honshu_sta_names,
        #                                          extrapolate=False)[0]
        # rms_euler_unreg = np.sqrt(np.mean((v_fit_euler_unreg - v_mod_unreg[:, :2])**2))
        # rms_hom_unreg = np.sqrt(np.mean((v_fit_hom_unreg - v_mod_unreg[:, :2])**2))
        # ramp_unreg = np.linalg.lstsq(G, v_mod_unreg[:, 2], rcond=None)[0]
        # vert_fit_unreg = G @ ramp_unreg
        # rms_vert_unreg = np.sqrt(np.mean((vert_fit_unreg - v_mod_unreg[:, 0])**2))
        # tqdm.write(f"\neuler={rms_euler_unreg}, homvel={rms_hom_unreg}, vert={rms_vert_unreg}\n")

        # # freeze spline models and run ridge regression with covariances
        # net.freeze("final_l2", ["Transient"], 0.1)
        # net.fitevalres("final_l2",
        #                solver="ridge_regression",
        #                penalty=penalties,
        #                output_description="model_l2",
        #                residual_description="resid_l2")

        # # calculate RMS as above
        # v_mod_l2 = np.stack([net[sta_name].models["final_l2"]["Linear"].par[1, :]
        #                      for sta_name in honshu_sta_names], axis=0)
        # v_fit_euler_l2 = net.euler_rot_field("final_l2", "Linear",
        #                                      subset_stations=honshu_sta_names,
        #                                      extrapolate=False)[0]
        # v_fit_hom_l2 = net.hom_velocity_field("final_l2", "Linear",
        #                                       utmzone=UTMZONE,
        #                                       subset_stations=honshu_sta_names,
        #                                       extrapolate=False)[0]
        # rms_euler_l2 = np.sqrt(np.mean((v_fit_euler_l2 - v_mod_l2[:, :2])**2))
        # rms_hom_l2 = np.sqrt(np.mean((v_fit_hom_l2 - v_mod_l2[:, :2])**2))
        # ramp_l2 = np.linalg.lstsq(G, v_mod_l2[:, 2], rcond=None)[0]
        # vert_fit_l2 = G @ ramp_l2
        # rms_vert_l2 = np.sqrt(np.mean((vert_fit_l2 - v_mod_l2[:, 0])**2))
        # tqdm.write(f"\neuler={rms_euler_l2}, homvel={rms_hom_l2}, vert={rms_vert_l2}\n")

        # extract parts relevant to SEAS or not
        for station in net:
            # considering both Tohoku and Tokachi earthquakes
            ts_tect = station.sum_fits("final", ["BigPost0", "BigPost1", "Linear"])[0]
            ts_tect += station["resid_srw"].data.values
            ts_nontect = station.sum_fits(
                "final",
                [
                    "Annual",
                    "AnnualDev",
                    "Biannual",
                    "BigEQ",
                    "SmallEQ",
                    "Transient",
                    "Maintenance",
                ],
            )[0]
            station["SEAS_2EQ"] = disstans.Timeseries.from_array(
                station["final"].time,
                ts_tect,
                "modeled",
                station["final"].data_unit,
                station["final"].data_cols,
            )
            station["NonSEAS_2EQ"] = disstans.Timeseries.from_array(
                station["final"].time,
                ts_nontect,
                "modeled",
                station["final"].data_unit,
                station["final"].data_cols,
            )
            # considering only Tohoku
            ts_tect = station.sum_fits("final", ["BigPost1", "Linear"])[0]
            ts_tect += station["resid_srw"].data.values
            ts_nontect = station.sum_fits(
                "final",
                [
                    "Annual",
                    "AnnualDev",
                    "Biannual",
                    "BigEQ",
                    "SmallEQ",
                    "Transient",
                    "BigPost0",
                    "Maintenance",
                ],
            )[0]
            station["SEAS_1EQ"] = disstans.Timeseries.from_array(
                station["final"].time,
                ts_tect,
                "modeled",
                station["final"].data_unit,
                station["final"].data_cols,
            )
            station["NonSEAS_1EQ"] = disstans.Timeseries.from_array(
                station["final"].time,
                ts_nontect,
                "modeled",
                station["final"].data_unit,
                station["final"].data_cols,
            )

        # save
        with open(NETWORK_FNAME, mode="wb") as f:
            pickle.dump(net, f)

    # extract secular velocities
    poly_stat_names = [
        stat_name for stat_name in net.station_names if "final" in net[stat_name].models
    ]
    # skip the (co)variance estimates
    all_poly_pars_covs = np.concatenate(
        (
            np.stack(
                [
                    net[stat_name].models["final"]["Linear"].par.ravel()
                    for stat_name in poly_stat_names
                ],
                axis=0,
            ),
            np.zeros((len(poly_stat_names), 6)),
        ),
        axis=1,
    )
    all_poly_df = pd.DataFrame(
        data=all_poly_pars_covs,
        index=poly_stat_names,
        columns=[
            "off_e",
            "off_n",
            "off_u",
            "vel_e",
            "vel_n",
            "vel_u",
            "sig_vel_e",
            "sig_vel_n",
            "sig_vel_u",
            "corr_vel_en",
            "corr_vel_eu",
            "corr_vel_nu",
        ],
    )
    # make pretty and save
    all_poly_df.sort_index(inplace=True)
    all_poly_df.index.rename("station", inplace=True)
    all_poly_df.to_csv(f"{SEC_VEL_DATA_BASENAME}_igs.csv")

    # load MIDAS velocities
    v_mdl_midas = pd.read_csv(
        MIDAS_FNAME,
        header=0,
        delimiter=r"\s+",
        names=[
            "sta",
            "label",
            "t(1)",
            "t(m)",
            "delt",
            "m",
            "mgood",
            "n",
            "ve50",
            "vn50",
            "vu50",
            "sve",
            "svn",
            "svu",
            "xe50",
            "xn50",
            "xu50",
            "fe",
            "fn",
            "fu",
            "sde",
            "sdn",
            "sdu",
            "nstep",
            "lat",
            "lon",
            "alt",
        ],
    )
    v_mdl_midas.loc[v_mdl_midas["lon"] < 0, "lon"] += 360
    inside_njapan_midas = njapan_path.contains_points(
        v_mdl_midas[["lat", "lon"]].values
    )
    v_mdl_midas = v_mdl_midas.iloc[inside_njapan_midas, :]
    v_mdl_midas.set_index("sta", inplace=True, verify_integrity=True)
    v_mdl_midas.sort_index(inplace=True)

    # plot preparations
    station_locs = station_loc_df.loc[all_poly_df.index.tolist(), :].to_numpy()

    # make secular velocity map
    if MAKE_PLOTS:
        fig_map = plt.figure(figsize=(8, 5))
        ax_map = fig_map.add_subplot(projection=CRS_GUI)
        ax_map.gridlines(draw_labels=True, zorder=-1)
        q = ax_map.quiver(
            v_mdl_midas["lon"],
            v_mdl_midas["lat"],
            v_mdl_midas["ve50"] * 1e3,
            v_mdl_midas["vn50"] * 1e3,
            units="xy",
            scale=5e-4,
            width=2e3,
            color="C1",
            transform=CRS_PC,
            label="MIDAS",
        )
        q = ax_map.quiver(
            station_locs[:, 1],
            station_locs[:, 0],
            all_poly_df["vel_e"],
            all_poly_df["vel_n"],
            units="xy",
            scale=5e-4,
            width=2e3,
            color="k",
            transform=CRS_PC,
            label="DISSTANS",
        )
        ax_map.quiverkey(q, 0.85, 0.85, 10, "10 mm/a", color="0.3")
        ax_map.legend(loc="lower right", ncol=3, fontsize="small")
        ax_map.set_title("Secular Velocity (IGS)")
        fig_map.savefig(SEC_VEL_FIG_IGS_FNAME, dpi=300)
        plt.close(fig_map)

    # remove EU velocity from DISSTANS
    lons_d, lats_d, alts_d = station_loc_df.loc[
        all_poly_df.index, ["Longitude [°]", "Latitude [°]", "Altitude [m]"]
    ].values.T
    stat_xyz_d = CRS_XYZ.transform_points(CRS_LLA, lons_d, lats_d, alts_d)
    v_EU_xyz_d = np.cross(ROT_EU, stat_xyz_d)  # [m/a]
    v_EU_enu_d = pd.DataFrame(
        data=np.stack(
            [
                (disstans.tools.R_ecef2enu(lo, la) @ v_EU_xyz_d[i, :])
                for i, (lo, la) in enumerate(zip(lons_d, lats_d))
            ]
        ),
        index=all_poly_df.index,
        columns=["vel_e", "vel_n", "vel_u"],
    )
    v_mdl_DISSTANS_igs = (
        all_poly_df.loc[:, ["vel_e", "vel_n", "sig_vel_e", "sig_vel_n"]].copy() / 1000
    )
    v_mdl_DISSTANS_igs["corr_vel_en"] = all_poly_df.loc[
        :, ["corr_vel_en"]
    ].values.copy()
    v_mdl_DISSTANS_eu = (-v_EU_enu_d.iloc[:, :2]) + v_mdl_DISSTANS_igs[
        ["vel_e", "vel_n"]
    ].values
    v_mdl_DISSTANS_eu.iloc[:, :] *= 1000

    # remove OK velocity from DISSTANS
    v_OK_xyz_d = np.cross(ROT_OK, stat_xyz_d)  # [m/a]
    v_OK_enu_d = pd.DataFrame(
        data=np.stack(
            [
                (disstans.tools.R_ecef2enu(lo, la) @ v_OK_xyz_d[i, :])
                for i, (lo, la) in enumerate(zip(lons_d, lats_d))
            ]
        ),
        index=all_poly_df.index,
        columns=["vel_e", "vel_n", "vel_u"],
    )
    v_mdl_DISSTANS_ok = (-v_OK_enu_d.iloc[:, :2]) + v_mdl_DISSTANS_igs[
        ["vel_e", "vel_n"]
    ].values
    v_mdl_DISSTANS_ok.iloc[:, :] *= 1000

    # repeat for MIDAS
    # EU
    lons_m, lats_m, alts_m = v_mdl_midas.loc[:, ["lon", "lat", "alt"]].values.T
    stat_xyz_m = CRS_XYZ.transform_points(CRS_LLA, lons_m, lats_m, alts_m)
    v_EU_xyz_m = np.cross(ROT_EU, stat_xyz_m)  # [m/a]
    v_EU_enu_m = pd.DataFrame(
        data=np.stack(
            [
                (disstans.tools.R_ecef2enu(lo, la) @ v_EU_xyz_m[i, :])
                for i, (lo, la) in enumerate(zip(lons_m, lats_m))
            ]
        ),
        index=v_mdl_midas.index,
        columns=["vel_e", "vel_n", "vel_u"],
    )
    v_mdl_MIDAS_igs = pd.DataFrame(
        data=v_mdl_midas.loc[:, ["ve50", "vn50", "sve", "svn"]].values,
        columns=["vel_e", "vel_n", "sig_vel_e", "sig_vel_n"],
        index=v_mdl_midas.index,
    )
    v_mdl_MIDAS_igs["corr_vel_en"] = 0
    v_mdl_MIDAS_eu = (-v_EU_enu_m.iloc[:, :2]) + v_mdl_MIDAS_igs[
        ["vel_e", "vel_n"]
    ].values
    v_mdl_MIDAS_eu.iloc[:, :] *= 1000
    # OK
    v_OK_xyz_m = np.cross(ROT_OK, stat_xyz_m)  # [m/a]
    v_OK_enu_m = pd.DataFrame(
        data=np.stack(
            [
                (disstans.tools.R_ecef2enu(lo, la) @ v_OK_xyz_m[i, :])
                for i, (lo, la) in enumerate(zip(lons_m, lats_m))
            ]
        ),
        index=v_mdl_midas.index,
        columns=["vel_e", "vel_n", "vel_u"],
    )
    v_mdl_MIDAS_ok = (-v_OK_enu_m.iloc[:, :2]) + v_mdl_MIDAS_igs[
        ["vel_e", "vel_n"]
    ].values
    v_mdl_MIDAS_ok.iloc[:, :] *= 1000

    # load interseismic velocities from Loveless & Meade (2016)
    # https://doi.org/10.1016/j.epsl.2015.12.033
    v_mdl_LM16_ok = pd.read_csv(LM16_FNAME, index_col=0, delimiter=r"\s+", comment="#")
    inside_njapan_lm16 = njapan_path.contains_points(
        v_mdl_LM16_ok[["latitude", "longitude"]].values
    )
    v_mdl_LM16_ok = v_mdl_LM16_ok.iloc[inside_njapan_lm16, :]

    # calculate what the 100% PA plate convergence rate would be like in
    # the OK reference frame
    v_PA_xyz_d = np.cross(ROT_PA, stat_xyz_d)  # [m/a]
    v_PA_enu_d = pd.DataFrame(
        data=np.stack(
            [
                (disstans.tools.R_ecef2enu(lo, la) @ v_PA_xyz_d[i, :])
                for i, (lo, la) in enumerate(zip(lons_d, lats_d))
            ]
        ),
        index=all_poly_df.index,
        columns=["vel_e", "vel_n", "vel_u"],
    )
    v_PA_enu_d.iloc[:, :] /= 1000
    v_PA_rel_OK_d = (v_PA_enu_d - v_OK_enu_d) * 1000

    # as a quality check, calculate which extracted secular velocities are more
    # than 100% off the plate velocity, or which ones have been defined purely
    # using post-Tohoku data
    pa_res = v_PA_rel_OK_d.iloc[:, :2] - v_mdl_DISSTANS_ok
    pa_rms = np.linalg.norm(pa_res.values, axis=1)
    pa_rms_rel = pa_rms / np.linalg.norm(v_PA_rel_OK_d.values[:, :2], axis=1)
    pa_i_off = pa_rms_rel > 1
    sta_has_pre_toh_data = [
        np.any(
            np.isfinite(
                sta["final"].data.loc[sta["final"].time < pd.Timestamp("2011-03-10"), :]
            )
        )
        for sta in net
    ]
    i_sta_clean = np.logical_and(~pa_i_off, sta_has_pre_toh_data)
    sta_clean = v_PA_rel_OK_d.index[i_sta_clean].tolist()

    # save clean data
    v_mdl_DISSTANS_eu.iloc[i_sta_clean, :].to_csv(
        f"{SEC_VEL_DATA_BASENAME}_eu_clean.csv"
    )
    v_mdl_DISSTANS_ok.iloc[i_sta_clean, :].to_csv(
        f"{SEC_VEL_DATA_BASENAME}_ok_clean.csv"
    )

    # redo secular velocity plot in OK reference frame
    if MAKE_PLOTS:
        cols_pa_off = ["C0" if io else "C3" for io in i_sta_clean]
        fig_map = plt.figure(figsize=(8, 5))
        ax_map = fig_map.add_subplot(projection=CRS_GUI)
        ax_map.gridlines(draw_labels=True, zorder=-1)
        q = ax_map.quiver(
            station_locs[:, 1],
            station_locs[:, 0],
            v_PA_rel_OK_d["vel_e"],
            v_PA_rel_OK_d["vel_n"],
            units="xy",
            scale=5e-4,
            width=4e3,
            color="0.8",
            transform=CRS_PC,
            label="PA plate",
        )
        # q = ax_map.quiver(v_mdl_midas["lon"],
        #                   v_mdl_midas["lat"],
        #                   v_mdl_MIDAS_ok["vel_e"],
        #                   v_mdl_MIDAS_ok["vel_n"],
        #                   units="xy", scale=5e-4, width=2e3, color="C2",
        #                   transform=CRS_PC, label="MIDAS")
        q = ax_map.quiver(
            v_mdl_LM16_ok["longitude"],
            v_mdl_LM16_ok["latitude"],
            v_mdl_LM16_ok["east velocity (mm/yr)"],
            v_mdl_LM16_ok["north velocity (mm/yr)"],
            units="xy",
            scale=5e-4,
            width=2e3,
            color="C1",
            transform=CRS_PC,
            label="L&M 2016",
        )
        q = ax_map.quiver(
            station_locs[:, 1],
            station_locs[:, 0],
            v_mdl_DISSTANS_ok["vel_e"],
            v_mdl_DISSTANS_ok["vel_n"],
            units="xy",
            scale=5e-4,
            width=2e3,
            color=cols_pa_off,
            transform=CRS_PC,
            label="DISSTANS",
        )
        ax_map.quiverkey(q, 0.85, 0.85, 100, "10 cm/a", color="0.3")
        ax_map.legend(loc="lower right", ncol=1, fontsize="small")
        ax_map.set_title("Secular Velocity (OK)")
        fig_map.savefig(SEC_VEL_FIG_OK_FNAME, dpi=300)
        plt.close(fig_map)

    # extract Tohoku coseismic step
    toh_step_str = "2011-03-11T05:46:18.120000"
    tohoku_steps = np.array(
        [
            (
                net[n]
                .models["final"]["BigEQ"]
                .parameters[
                    net[n].models["final"]["BigEQ"].steptimes.index(toh_step_str), :
                ]
                if (
                    ("BigEQ" in net[n].models["final"])
                    and (toh_step_str in net[n].models["final"]["BigEQ"].steptimes)
                )
                else np.full(3, np.nan)
            )
            for n in sta_clean
        ]
    )
    tohoku_steps_df = pd.DataFrame(
        data=tohoku_steps, index=sta_clean, columns=["e", "n", "u"]
    )
    tohoku_steps_df.index.rename("station", inplace=True)
    tohoku_steps_df.to_csv(TOH_STEPS_DATA_FNAME)

    # # prepare subsampling
    # ct_plus = [t_min] + check_times + [t_max]
    # ct_diff = np.diff(pd.DatetimeIndex(ct_plus)).astype(float) / 1e9 / 86400 / 365.25
    # t_eval_rel = np.concatenate(
    #     [np.linspace(0, ct_diff[0], N_SAMPLES_PER_EQ, endpoint=False)] +
    #     [np.logspace(0, np.log10(1 + t), N_SAMPLES_PER_EQ, endpoint=False)
    #      - 1 + np.sum(ct_diff[:i + 1]) for i, t in enumerate(ct_diff[1:])])
    # t_eval_obs_2eq = t_min + t_eval_rel * pd.Timedelta(365.25, "D")
    # check_times_dates_2eq = [pd.Timestamp(ct.date()) for ct in check_times]
    # i_split_2eq = [int(np.argmax(t_eval_obs_2eq >= t)) if np.any(t_eval_obs_2eq >= t)
    #            else None for t in check_times_dates_2eq]
    # np.save(T_EVAL_FNAME, t_eval_obs_2eq)

    # define break times
    t_min = pd.Timestamp("1996-04-02")
    eq1 = pd.Timestamp("2003-09-26")
    eq2 = pd.Timestamp("2011-03-11")
    t_max = pd.Timestamp("2023-10-29")
    dt_start = 27  # start delta t in days from Francisco's thesis
    dt_log = 0.17607802  # spacing from Francisco's thesis
    t_eval_obs_2eq = np.sort(
        np.concatenate(
            [
                np.array(
                    [np.datetime64(t_min), np.datetime64(eq1), np.datetime64(eq2)]
                ),
                (
                    t_min
                    + np.cumsum(10 ** (np.log10(dt_start) + np.arange(9) * dt_log))
                    * pd.Timedelta(1, "D")
                ),
                (
                    eq1
                    + np.cumsum(10 ** (np.log10(dt_start) + np.arange(9) * dt_log))
                    * pd.Timedelta(1, "D")
                ),
                (
                    eq2
                    + np.cumsum(10 ** (np.log10(dt_start) + np.arange(11) * dt_log))
                    * pd.Timedelta(1, "D")
                ),
            ]
        )
    )
    i_split_2eq = [10, 20]
    check_times_dates_2eq = [pd.Timestamp(ct.date()) for ct in check_times]
    t_eval_obs_1eq = np.sort(
        np.concatenate(
            [
                np.array([np.datetime64(t_min), np.datetime64(eq2)]),
                (
                    t_min
                    + np.cumsum(10 ** (np.log10(dt_start) + np.arange(11) * dt_log))
                    * pd.Timedelta(1, "D")
                ),
                (
                    eq2
                    + np.cumsum(10 ** (np.log10(dt_start) + np.arange(11) * dt_log))
                    * pd.Timedelta(1, "D")
                ),
            ]
        )
    )
    i_split_1eq = [12]
    check_times_dates_1eq = [pd.Timestamp(check_times[ITOHOKU].date())]

    # subsample observations for SEAS inversion
    if not RUN_DOWNSAMPLING:
        obs_loc = pd.read_csv(OBS_LOC_FNAME, index_col=0)
    else:
        # save times
        np.save(f"{T_EVAL_BASENAME}_2eq.npy", t_eval_obs_2eq)
        np.save(f"{T_EVAL_BASENAME}_1eq.npy", t_eval_obs_1eq)
        # first for entire area
        assert not any([e is None for e in i_split_2eq])
        arr_split = np.split(t_eval_obs_2eq, i_split_2eq)
        arr_split_mid = [a[:-1] + (a[1:] - a[:-1]) / 2 for a in arr_split]
        with Pool(len(arr_split) + 1) as p:
            for station in tqdm(
                net, desc="Downsampling network timeseries (2 EQ)", unit="station"
            ):
                ts = station["SEAS_2EQ"]
                new_ts_time = []
                new_ts_data = []
                new_ts_sd = []
                for i, (a, a_mid) in enumerate(zip(arr_split, arr_split_mid)):
                    if i == 0:
                        sub_ix = ts.time < check_times_dates_2eq[0]
                    elif i < len(check_times_dates_2eq):
                        sub_ix = np.logical_and(
                            ts.time >= check_times_dates_2eq[i - 1],
                            ts.time < check_times_dates_2eq[i],
                        )
                    else:
                        sub_ix = ts.time >= check_times_dates_2eq[-1]
                    if not np.any(sub_ix):
                        continue
                    sub_ts = ts.data.loc[sub_ix, :]
                    sub_t_days = (
                        (sub_ts.index - sub_ts.index[0]) / pd.Timedelta(1, "D")
                    ).values
                    sub_t_eval_days = (
                        (pd.Series(a) - sub_ts.index[0]) / pd.Timedelta(1, "D")
                    ).values
                    sub_d = sub_ts.values
                    sub_i_split = [
                        (
                            np.argmax(sub_ts.index >= t)
                            if np.any(sub_ts.index >= t)
                            else None
                        )
                        for t in a_mid
                    ]
                    if any([e is None for e in sub_i_split]):
                        assert sum([e is None for e in sub_i_split]) == len(
                            sub_i_split
                        ) - sub_i_split.index(None)
                        sub_i_split = sub_i_split[: sub_i_split.index(None)]
                        assert not any([e is None for e in sub_i_split])
                    # TODO add variance information to the PolyFit1D routine
                    sub_t_split = np.split(sub_t_days, sub_i_split)
                    sub_d_split = np.split(sub_d, sub_i_split)
                    fit_series = list(
                        p.starmap(wrap_polyfitl1, zip(sub_t_split, sub_d_split))
                    )
                    fit_model = np.concatenate(
                        [
                            (
                                fssplit.predict(sub_t_eval_days[j])
                                if fssplit is not None
                                else np.full((1, sub_ts.shape[1]), np.nan)
                            )
                            for j, fssplit in enumerate(fit_series)
                        ],
                        axis=0,
                    )
                    fit_resid = np.concatenate(
                        [
                            (
                                fssplit.get_fit_residual()
                                if fssplit is not None
                                else np.full((sts.size, sub_ts.shape[1]), np.nan)
                            )
                            for sts, fssplit in zip(sub_t_split, fit_series)
                        ],
                        axis=0,
                    )
                    fit_resid_split = np.split(fit_resid, sub_i_split)
                    fit_valid = [len(frs) > 0 for frs in fit_resid_split]
                    fit_sd = np.stack(
                        [
                            np.std(frs, axis=0)
                            for frs, isvalid in zip(fit_resid_split, fit_valid)
                            if isvalid
                        ]
                    )
                    fit_model = fit_model[fit_valid, :]
                    fit_time = a[: len(sub_i_split) + 1][fit_valid]
                    assert fit_model.shape == fit_sd.shape
                    assert fit_time.size == fit_model.shape[0]
                    assert fit_time.size > 0
                    new_ts_time.append(fit_time)
                    new_ts_data.append(fit_model)
                    new_ts_sd.append(fit_sd)
                # only add if the new timeseries doesn't contain crazy values
                if not any([(np.abs(fm) > 1e4).sum() > 0 for fm in new_ts_data]):
                    new_ts_time = np.concatenate(new_ts_time)
                    new_ts_data = np.concatenate(new_ts_data)
                    new_ts_sd = np.concatenate(new_ts_sd)
                    new_ts = disstans.timeseries.Timeseries.from_array(
                        new_ts_time,
                        new_ts_data,
                        "downsampled_2eq",
                        ts.data_unit,
                        ts.data_cols,
                        var=new_ts_sd**2,
                        var_cols=ts.var_cols,
                    )
                    station["downsampled_2eq"] = new_ts
        # then just for Honshu
        assert not any([e is None for e in i_split_1eq])
        arr_split = np.split(t_eval_obs_1eq, i_split_1eq)
        arr_split_mid = [a[:-1] + (a[1:] - a[:-1]) / 2 for a in arr_split]
        with Pool(len(arr_split) + 1) as p:
            for station in tqdm(
                net, desc="Downsampling network timeseries (1 EQ)", unit="station"
            ):
                ts = station["SEAS_1EQ"]
                new_ts_time = []
                new_ts_data = []
                new_ts_sd = []
                for i, (a, a_mid) in enumerate(zip(arr_split, arr_split_mid)):
                    if i == 0:
                        sub_ix = ts.time < check_times_dates_1eq[0]
                    elif i < len(check_times_dates_1eq):
                        sub_ix = np.logical_and(
                            ts.time >= check_times_dates_1eq[i - 1],
                            ts.time < check_times_dates_1eq[i],
                        )
                    else:
                        sub_ix = ts.time >= check_times_dates_1eq[-1]
                    if not np.any(sub_ix):
                        continue
                    sub_ts = ts.data.loc[sub_ix, :]
                    sub_t_days = (
                        (sub_ts.index - sub_ts.index[0]) / pd.Timedelta(1, "D")
                    ).values
                    sub_t_eval_days = (
                        (pd.Series(a) - sub_ts.index[0]) / pd.Timedelta(1, "D")
                    ).values
                    sub_d = sub_ts.values
                    sub_i_split = [
                        (
                            np.argmax(sub_ts.index >= t)
                            if np.any(sub_ts.index >= t)
                            else None
                        )
                        for t in a_mid
                    ]
                    if any([e is None for e in sub_i_split]):
                        assert sum([e is None for e in sub_i_split]) == len(
                            sub_i_split
                        ) - sub_i_split.index(None)
                        sub_i_split = sub_i_split[: sub_i_split.index(None)]
                        assert not any([e is None for e in sub_i_split])
                    sub_t_split = np.split(sub_t_days, sub_i_split)
                    sub_d_split = np.split(sub_d, sub_i_split)
                    fit_series = list(
                        p.starmap(wrap_polyfitl1, zip(sub_t_split, sub_d_split))
                    )
                    fit_model = np.concatenate(
                        [
                            (
                                fssplit.predict(sub_t_eval_days[j])
                                if fssplit is not None
                                else np.full((1, sub_ts.shape[1]), np.nan)
                            )
                            for j, fssplit in enumerate(fit_series)
                        ],
                        axis=0,
                    )
                    fit_resid = np.concatenate(
                        [
                            (
                                fssplit.get_fit_residual()
                                if fssplit is not None
                                else np.full((sts.size, sub_ts.shape[1]), np.nan)
                            )
                            for sts, fssplit in zip(sub_t_split, fit_series)
                        ],
                        axis=0,
                    )
                    fit_resid_split = np.split(fit_resid, sub_i_split)
                    fit_valid = [len(frs) > 0 for frs in fit_resid_split]
                    fit_sd = np.stack(
                        [
                            np.std(frs, axis=0)
                            for frs, isvalid in zip(fit_resid_split, fit_valid)
                            if isvalid
                        ]
                    )
                    fit_model = fit_model[fit_valid, :]
                    fit_time = a[: len(sub_i_split) + 1][fit_valid]
                    assert fit_model.shape == fit_sd.shape
                    assert fit_time.size == fit_model.shape[0]
                    assert fit_time.size > 0
                    new_ts_time.append(fit_time)
                    new_ts_data.append(fit_model)
                    new_ts_sd.append(fit_sd)
                # only add if the new timeseries doesn't contain crazy values
                if not any([(np.abs(fm) > 1e4).sum() > 0 for fm in new_ts_data]):
                    new_ts_time = np.concatenate(new_ts_time)
                    new_ts_data = np.concatenate(new_ts_data)
                    new_ts_sd = np.concatenate(new_ts_sd)
                    new_ts = disstans.timeseries.Timeseries.from_array(
                        new_ts_time,
                        new_ts_data,
                        "downsampled_1eq",
                        ts.data_unit,
                        ts.data_cols,
                        var=new_ts_sd**2,
                        var_cols=ts.var_cols,
                    )
                    station["downsampled_1eq"] = new_ts

        # export downsampled timeseries
        ds_ts_2eq = net.export_network_ts("downsampled_2eq")
        # restrict to stations that are moving with PAC
        list_sta_2eq = ds_ts_2eq["east"].data.columns.tolist()
        # ds_values = {comp: np.array_split(ts.data.loc[:, list_sta_2eq].values, i_split_2eq)
        #              for comp, ts in ds_ts_2eq.items()}
        # ds_times = np.array_split(t_eval_obs_2eq.astype(float), i_split_2eq)
        observed_data_2eq = np.stack(
            [
                comp_ts.data.loc[:, list_sta_2eq].values.T
                for comp_ts in ds_ts_2eq.values()
            ],
            axis=1,
        )
        # make info dictionary
        surf_disp_info_2eq = {
            "i_eq": [0] + i_split_2eq,
            "ts_shape": observed_data_2eq.shape,
        }
        # save downsampled data with data gaps
        np.save(SURF_DISPS_BASENAME + "_2eq.npy", observed_data_2eq)
        with open(SURF_DISPS_BASENAME + "_2eq.json", mode="wt") as f:
            json.dump(surf_disp_info_2eq, f)
        # same for the Honshu subset, only reusing stations
        list_sta_1eq = sorted(
            [sta.name for sta in net if "downsampled_1eq" in sta.timeseries]
        )
        assert all([sta in list_sta_1eq for sta in list_sta_2eq])
        ds_ts_1eq = net.export_network_ts(
            "downsampled_1eq", subset_stations=list_sta_2eq
        )
        observed_data_1eq = np.stack(
            [
                comp_ts.data.loc[:, list_sta_2eq].values.T
                for comp_ts in ds_ts_1eq.values()
            ],
            axis=1,
        )
        surf_disp_info_1eq = {
            "i_eq": [0] + i_split_1eq,
            "ts_shape": observed_data_1eq.shape,
        }
        np.save(SURF_DISPS_BASENAME + "_1eq.npy", observed_data_1eq)
        with open(SURF_DISPS_BASENAME + "_1eq.json", mode="wt") as f:
            json.dump(surf_disp_info_1eq, f)
        # transform observer locations to ENU
        proj = Transformer.from_crs(CRS_WGS, CRS_UTM, always_xy=True)
        ds_locs = station_loc_df.loc[list_sta_2eq, :]
        pts_surf = np.stack(
            list(
                proj.transform(
                    ds_locs["Longitude [°]"].values, ds_locs["Latitude [°]"].values
                )
            )
            + [ds_locs["Altitude [m]"].values],
            axis=1,
        )
        # save observer locations
        obs_loc = pd.DataFrame(
            data=pts_surf, columns=["E", "N", "U"], index=ds_locs.index
        )
        obs_loc.index.names = ["Station"]
        obs_loc.to_csv(OBS_LOC_FNAME)

        # # fill data gaps
        # # select data from stations who had at least half of the data available in the first
        # # time period
        # i_keep = np.all(np.stack([np.isnan(li[0]).sum(axis=0) < 0.5 * li[0].shape[0]
        #                           for li in ds_values.values()], axis=0), axis=0)
        # sta_keep = ds_ts_2eq["east"].data.columns[i_keep].tolist()
        # ds_values = {comp: [a[:, i_keep] for a in li] for comp, li in ds_values.items()}
        # # fill the missing values period by period, component by component
        # filled_data = np.stack(
        #     [np.concatenate(
        #         [disstans.processing.decompose(
        #             a, method="pca", num_components=None, detrend=ds_times[i], impute=True)
        #          for i, a in enumerate(li)],
        #         axis=0)
        #      for li in ds_values.values()]).transpose(2, 0, 1)
        # # for the same stations, also stack the data with holes
        # observed_data_2eq = np.stack([comp_ts.data.values.T for comp_ts in ds_ts_2eq.values()],
        #                          axis=1)[i_keep, :, :]
        # assert filled_data.shape == observed_data_2eq.shape
        # # save downsampled and filled data
        # np.save(SURF_DISPS_BASENAME + "_filled.npy", filled_data)
        # with open(SURF_DISPS_BASENAME + "_filled.json", mode="wt") as f:
        #     json.dump(surf_disp_info_2eq,  f)
        # # get locations of stations
        # sta_keep_df = station_loc_df.loc[station_loc_df.index.isin(sta_keep), :]
        # # save observers
        # raise NotImplementedError("pts_surf needs to be recomputed for available stations")
        # obs_loc = pd.DataFrame(data=pts_surf, columns=["E", "N", "U"],
        #                        index=sta_keep_df.index)
        # obs_loc.index.names = ["Station"]
        # obs_loc.to_csv(OBS_LOC_FNAME)

    # save data in IGS reference frame
    avail_stas = obs_loc.index.values.astype(str).tolist()
    with open(SEAS_STATIONS_FNAME, mode="wt") as f:
        json.dump(avail_stas, f)
    # get SEAS data for downsampled stations
    for case in ["_2EQ", "_1EQ"]:
        net_seas_dict = net.export_network_ts(f"SEAS{case}", avail_stas)
        seas_data = np.stack(
            [v.data.values for v in net_seas_dict.values()], axis=1
        ).T.copy()
        net_nonseas_dict = net.export_network_ts(f"NonSEAS{case}", avail_stas)
        nonseas_data = np.stack(
            [v.data.values for v in net_nonseas_dict.values()], axis=1
        ).T.copy()
        seas_time = net_seas_dict["east"].time.values
        # save indices of earthquakes into JSON file
        cur_check_time = (
            check_times_dates_2eq if case == "_2EQ" else check_times_dates_1eq
        )
        j_split = [
            int(np.argmax(seas_time >= t)) if np.any(seas_time >= t) else None
            for t in cur_check_time
        ]
        # save data
        np.save(f"{SEAS_TIME_BASENAME}{case.lower()}.npy", seas_time)
        np.save(f"{SEAS_DATA_BASENAME}{case.lower()}.npy", seas_data)
        np.save(f"{NONSEAS_DATA_BASENAME}{case.lower()}.npy", nonseas_data)
        with open(f"{SEAS_DATA_BASENAME}{case.lower()}.json", mode="wt") as f:
            json.dump({"i_eq": [0] + j_split, "ts_shape": seas_data.shape}, f)
