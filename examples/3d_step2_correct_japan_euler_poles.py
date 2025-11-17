"""
Estimate better-fitting Euler poles for Northern Japan based on
an initial guess and precomputed stress & displacement kernels
"""

# imports
import shutil
import json
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from copy import deepcopy
from cmcrameri import cm
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from seqeas.subduction3d import (
    SubductionSimulation3D,
    RateStateSteadyLogarithmic2D,
    Fault3D,
    get_surface_displacements,
)

# input file path
LM16_FNAME = None
assert LM16_FNAME is not None, "'LM16_FNAME' must be set"
# constant parameters
SEC_PER_YEAR = 86400 * 365.25
CONFIG_FILE = "fault_subduction_uncor.ini"
K_INNER_ASP_FILE = "K_inner_asperities.npy"
K_INNER_INNER_FILE = "K_inner_inner.npy"
CROSS_SECTION_FILE = "cross_sections.pkl.gz"
G_SURF_FILE = "G_surf.npy"
DATA_ROOT = "../out/japan"
OBS_LOC_BASENAME = "obs_loc_jpl"
DATA_OUT_BASENAME = "japan_jpl_ds_timeseries_1eq"
DATA_IN_BASENAME = f"{DATA_ROOT}/{DATA_OUT_BASENAME}"
SEC_VEL_DATA_FNAME = f"{DATA_ROOT}/japan_jpl_secular_igs.csv"
UTMZONE = 54
# ROT_PAC_ITRF = np.array([-0.1135, 0.2907, -0.6025])  # [°/Ma]
ROT_PAC_ITRF = np.array(
    [-0.11096408, 0.29365791, -0.60432856]
)  # [°/Ma] PA plate from Kreemer
# ROT_EUR_ITRF = np.array([-0.0235, -0.1476, 0.2140])  # [°/Ma]
ROT_EUR_ITRF = np.array(
    [-0.01315961, -0.1725996, 0.08759111]
)  # [°/Ma] OK plate from Kreemer
ROT_PAC_EUR = ROT_PAC_ITRF - ROT_EUR_ITRF
PROJ_PC = ccrs.PlateCarree()
PROJ_GEOD = ccrs.Geodetic()
PROJ_UTM = ccrs.UTM(zone=UTMZONE)
CORRECTED_V_PLATE_FILE = "corrected_farfield_motion_ok.npy"
CORRECTIONS_FILE = "corrected_japan_euler_poles_ok.json"
# NEWTAB10 = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
#             "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]
NJAPAN_POLY = np.array([[34.5, 138.5], [34.5, 147], [46, 147], [46, 138.5]])
NJAPAN_PATH = Polygon(np.array(NJAPAN_POLY)).get_path()
# HONSHU_PATH = Polygon(np.array([
#     [141.8795803162128, 34.2150684494421],
#     [144.51023414944223, 41.02029264657463],
#     [143.03169148406232, 41.826581931782755],  # begin joint path
#     [141.64915808265295, 42.26861050724369],  #
#     [141.26512102666948, 42.14059651777063],  #
#     [140.36263394521433, 42.19752364698729],  #
#     [139.63296353891639, 44.924043069107256],  # end joint path
#     [137.9432004927101, 44.610489817173715],
#     [132.64348912063258, 39.62964904361161],
#     [138.03920975673122, 37.11608332386878],
#     [140.76587285393128, 35.772288288287214],
#     [141.8795803162128, 34.2150684494421]])).get_path()
# HONSHU_PATH = Polygon(np.array([
#     [139.4838304857616, 34.98869111435032],
#     [140.0563613533725, 34.6477189297741],
#     [142.4181131573214, 35.5737865141108],
#     [144.533049819588, 41.1790431103449],
#     [141.036580494293, 41.63759739857777],
#     [139.7142296644492, 41.03146286849424],
#     [138.4752004044225, 37.50963020606981],
#     [139.4838304857616, 34.98869111435032]])).get_path()
# HONSHU_PATH = Polygon(np.array([
#     [144.5662501343137, 41.16084291037269],
#     [141.0525910283028, 41.60298731863359],
#     [139.7621036345421, 41.03007851285377],
#     [138.7934951445023, 38.31180492559306],
#     [142.0143822392644, 37.01006606458038],
#     [144.5662501343137, 41.16084291037269]])).get_path()
HONSHU_PATH = Polygon(
    np.array(
        [
            [139.7742268964748, 41.04114877836916],
            [138.8795979327411, 38.26582786193165],
            [138.5212449338516, 37.42801616768181],
            [141.4963651052435, 35.21184391992918],
            [144.5616113997234, 41.14299750115369],
            [141.0551530613818, 41.62833088395637],
            [139.7742268964748, 41.04114877836916],
        ]
    )
).get_path()
# HOKKAIDO_PATH = Polygon(np.array([
#     [144.5107696445168, 41.03446982581286],
#     [148.7554841024168, 43.36085768859792],
#     [146.2811772122996, 44.899882616583],
#     [141.12637119120416, 46.894209860931284],
#     [139.64178705686413, 44.91448620924737],  # begin joint path
#     [140.3880610235754, 42.22485351919667],  #
#     [141.26174761875546, 42.14392967493404],  #
#     [141.64398550411386, 42.27201125621286],  #
#     [142.99091900501435, 41.83276244620885],  # end joint path
#     [144.5107696445168, 41.03446982581286]])).get_path()
HOKKAIDO_PATH = Polygon(
    np.array(
        [
            [139.7317253629787, 41.03203632738356],
            [141.0445125082474, 41.63572047889622],
            [144.5564226264133, 41.17552909637928],
            [147.4471287730532, 43.78634493472902],
            [146.274289829154, 44.91341427642622],
            [141.1087005967136, 45.89775438197605],
            [138.7982739452335, 43.17075611647431],
            [139.7317253629787, 41.03203632738356],
        ]
    )
).get_path()


def estimate_euler_pole(locations: np.ndarray, velocities: np.ndarray) -> tuple:
    r"""
    Estimate a best-fit Euler pole assuming all velocities lie on the same
    rigid plate on a sphere. The calculations are based on [goudarzi14]_.

    Parameters
    ----------
    locations
        Array of shape :math:`(\text{num_stations}, \text{num_components})`
        containing the locations of each station (observation), where
        :math:`\text{num_components}=2` and the locations are given by longitudes and
        latitudes [°].
    velocities
        Array of shape :math:`(\text{num_stations}, \text{num_components})`
        containing the velocities [m/time] at different stations (observations), where
        :math:`\text{num_components}=2` and the velocities are given in the
        East-North local geodetic reference frame.

    Returns
    -------
    rotation_vector
        Rotation vector [rad/time] containing the diagonals of the :math:`3 \times 3`
        rotation matrix specifying the Euler pole in cartesian, ECEF coordinates.
    rotation_covariance
        Formal :math:`3 \times 3` covariance matrix [rad^2/time^2] of the rotation vector.
    """
    # input checks
    assert (
        isinstance(locations, np.ndarray)
        and locations.ndim == 2
        and locations.shape[1] == 2
    ), "'locations' needs to be a NumPy Array with 2 columns."
    assert (
        isinstance(velocities, np.ndarray)
        and velocities.ndim == 2
        and velocities.shape[1] == 2
    ), "'velocities' needs to be a NumPy Array with 2 columns."
    assert locations.shape[0] == velocities.shape[0], (
        "Shape mismatch between "
        f"locations {locations.shape} and velocities {velocities.shape}."
    )
    num_stations, _ = velocities.shape
    # stack velocities
    d = velocities.reshape(-1, 1)
    # build mapping matrix
    lon, lat = np.deg2rad(locations[:, 0]), np.deg2rad(locations[:, 1])
    # stacking of eq. 11 (note difference row ordering to match input format)
    G = (
        np.stack(
            [
                -np.sin(lat) * np.cos(lon),
                -np.sin(lat) * np.sin(lon),
                np.cos(lat),
                np.sin(lon),
                -np.cos(lon),
                np.zeros(num_stations),
            ],
            axis=1,
        ).reshape(2 * num_stations, 3)
        * 6378137
    )
    # solve
    rotation_vector = sp.linalg.lstsq(G, d)[0].ravel()
    # calculate formal covariance
    rotation_covariance = sp.linalg.pinvh(G.T @ G)
    return rotation_vector, rotation_covariance


def rotvec2eulerpole(
    rotation_vector: np.ndarray, rotation_covariance: np.ndarray = None
) -> tuple:
    r"""
    Convert a rotation vector containing the diagonals of a :math:`3 \times 3`
    rotation matrix (and optionally, its formal covariance) into an Euler
    Pole and associated magnitude. Based on [goudarzi14]_.

    Parameters
    ----------
    rotation_vector
        Rotation vector [rad/time] containing the diagonals of the :math:`3 \times 3`
        rotation matrix specifying the Euler pole in cartesian, ECEF coordinates.
    rotation_covariance
        Formal :math:`3 \times 3` covariance matrix [rad^2/time^2] of the rotation vector.

    Returns
    -------
    euler_pole
        NumPy Array containing the longitude [rad], latitude [rad], and rotation
        rate [rad/time] of the Euler pole.
    euler_pole_covariance
        If ``rotation_covariance`` was given, a NumPy Array of the propagated uncertainty
        for the Euler Pole for all three components.

    See Also
    --------
    eulerpole2rotvec : Inverse function
    """
    # readability
    ω_x, ω_y, ω_z = rotation_vector
    ω_xy_mag = np.linalg.norm(rotation_vector[:2])
    ω_mag = np.linalg.norm(rotation_vector)
    # Euler pole, eq. 15
    euler_pole = np.array([np.arctan(ω_y / ω_x), np.arctan(ω_z / ω_xy_mag), ω_mag])
    # uncertainty, eq. 18
    if rotation_covariance is not None:
        jac = np.array(
            [
                [-ω_y / ω_xy_mag**2, ω_x / ω_xy_mag**2, 0],
                [
                    -ω_x * ω_z / (ω_xy_mag * ω_mag**2),
                    -ω_y * ω_z / (ω_xy_mag * ω_mag**2),
                    -ω_xy_mag / ω_mag**2,
                ],
                [ω_x / ω_mag, ω_y / ω_mag, ω_z / ω_mag],
            ]
        )
        euler_pole_covariance = jac @ rotation_covariance @ jac.T
    # return
    if rotation_covariance is not None:
        return euler_pole, euler_pole_covariance
    else:
        return euler_pole


def eulerpole2rotvec(
    euler_pole: np.ndarray, euler_pole_covariance: np.ndarray = None
) -> tuple:
    r"""
    Convert an Euler pole (and optionally, its formal covariance) into a rotation
    vector and associated covariance matrix. Based on [goudarzi14]_.

    Parameters
    ----------
    euler_pole
        NumPy Array containing the longitude [rad], latitude [rad], and rotation
        rate [rad/time] of the Euler pole.
    euler_pole_covariance
        If ``rotation_covariance`` was given, the propagated uncertainty for the Euler
        Pole for all three components.

    Returns
    -------
    rotation_vector
        Rotation vector [rad/time] containing the diagonals of the :math:`3 \times 3`
        rotation matrix specifying the Euler pole in cartesian, ECEF coordinates.
    rotation_covariance
        If ``euler_pole_covariance`` was given, formal :math:`3 \times 3` covariance
        matrix [rad^2/time^2] of the rotation vector.

    See Also
    --------
    rotvec2eulerpole : Inverse function
    """
    # readability
    Ω = euler_pole[2]
    sinΩlat, cosΩlat = np.sin(euler_pole[1]), np.cos(euler_pole[1])
    sinΩlon, cosΩlon = np.sin(euler_pole[0]), np.cos(euler_pole[0])
    # rotation vector, eq. 5 (no scaling)
    ω_x = Ω * cosΩlat * cosΩlon
    ω_y = Ω * cosΩlat * sinΩlon
    ω_z = Ω * sinΩlat
    rotation_vector = np.array([ω_x, ω_y, ω_z])
    # uncertainty, eq. 6 (no scaling)
    if euler_pole_covariance is not None:
        jac = np.array(
            [
                [-Ω * cosΩlat * sinΩlon, -Ω * sinΩlat * cosΩlon, cosΩlat * cosΩlon],
                [Ω * cosΩlat * cosΩlon, -Ω * sinΩlat * sinΩlon, cosΩlat * sinΩlon],
                [0, Ω * cosΩlat, sinΩlat],
            ]
        )
        rotation_covariance = jac @ euler_pole_covariance @ jac.T
    # return
    if euler_pole_covariance is not None:
        return rotation_vector, rotation_covariance
    else:
        return rotation_vector


def R_ecef2enu(lon: float, lat: float) -> np.ndarray:
    """
    Generate the rotation matrix used to express a vector written in ECEF (XYZ)
    coordinates as a vector written in local east, north, up (ENU) coordinates
    at the position defined by geodetic latitude and longitude. See Chapter 4
    and Appendix 4.A in [misraenge2010]_ for details.

    Parameters
    ----------
    lon
        Longitude [°] of vector position.
    lat
        Latitude [°] of vector position.

    Returns
    -------
        The 3-by-3 rotation matrix.

    References
    ----------

    .. [misraenge2010] Misra, P., & Enge, P. (2010),
       *Global Positioning System: Signals, Measurements, and Performance*,
       Lincoln, Mass: Ganga-Jamuna Press.

    """
    try:
        lon, lat = np.deg2rad(float(lon)), np.deg2rad(float(lat))
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Input longitude & latitude are not convertible to scalars "
            f"(got {lon} and {lat})."
        ).with_traceback(e.__traceback__) from e
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def vel_from_rotvec(rotvec: np.ndarray, pts_ecef: np.ndarray, pts_lla: np.ndarray):
    """
    Calculate velocities from a rotation vector and locations.

    Parameters
    ----------
    rotvec
        Rotation vector [°/time]
    pts_ecef
        ECEF locations [m]
    pts_lla
        Same locations in Longitude, Latitude [°]

    Returns
    -------
        Velocities [m/time]
    """
    # calculate rotation from rotation vector
    rotvec_rad = np.deg2rad(rotvec)
    centroids_vel_xyz = np.cross(rotvec_rad, pts_ecef)
    # project into local ENU
    return np.stack(
        [
            (R_ecef2enu(lo, la) @ centroids_vel_xyz[i, :])
            for i, (lo, la) in enumerate(zip(pts_lla[:, 0], pts_lla[:, 1]))
        ]
    )


def crossmat(x1: float, x2: float, x3: float) -> np.ndarray:
    """
    Create the matrix for an input vector that if premultiplied with another
    vector will give the same results as cross-multiplying the other vector
    with the input vector.

    Parameters
    ----------
    x1
        First element
    x2
        Second element
    x3
        Third element

    Returns
    -------
        Cross-product matrix
    """
    return np.array([[0, x3, -x2], [-x3, 0, x1], [x2, -x1, 0]])


# start script
if __name__ == "__main__":

    # load configuration
    rheo_dict, fault_dict, sim_dict = SubductionSimulation3D.read_config_file(
        CONFIG_FILE
    )
    # overwrite prior plate motion
    fault_dict["farfield_motion"] = ROT_PAC_EUR
    fault_dict["farfield_motion_type"] = 2
    # create objects
    rheo = RateStateSteadyLogarithmic2D(**rheo_dict)
    fault = Fault3D.from_cfg_and_files(
        fault_dict,
        K_inner_asperities_file=K_INNER_ASP_FILE,
        K_inner_inner_file=K_INNER_INNER_FILE,
        cross_sections_file=CROSS_SECTION_FILE,
        write_new_files=True,
        verbose=True,
    )
    sim = SubductionSimulation3D.from_cfg_objs_and_files(
        sim_dict,
        rheo,
        fault,
        G_surf_file=G_SURF_FILE,
        write_new_files=True,
        verbose=True,
    )
    n_patches = fault.inner_num_patches

    # load data used in the inversion
    # with h5py.File(DATA_OUT_BASENAME, "r") as f:
    #     raw = f["input"][()]
    #     intermediate = f["intermediate"][()]
    #     output = f["output"][()]
    #     flat = f["flat"][()]
    #     shape = f["shape"][()]
    #     reset_indices = f["reset_indices"][()]
    #     factor = f["factor"][()]
    #     noise_sd = f["noise_sd"][()]
    #     indices_ref = f["indices_ref"][()]
    #     names_ref = f["names_ref"][()]
    #     reference = f["reference"][()]
    #     mask = f["mask"][()]
    # mask = mask.reshape(output.shape)  # shape (observations, 3, stations)
    ts_obs_itrf = np.load(f"{DATA_IN_BASENAME}.npy") / 1000  # this is in IGS [m]
    mask = np.isfinite(ts_obs_itrf)
    n_stations, n_comps, n_observations = ts_obs_itrf.shape

    # load observer locations
    obs_loc = pd.read_csv(f"{DATA_ROOT}/{OBS_LOC_BASENAME}.csv", index_col=0)
    assert obs_loc.shape[0] == n_stations

    # properly-formatted time vector
    t_obs = sim_dict["t_obs"]

    # define coordinate systems
    crs_lla = ccrs.Geodetic()
    crs_xyz = ccrs.Geocentric()
    crs_utm = ccrs.UTM(zone=UTMZONE)
    # get locations of centroids in lon/lat and ECEF cartesian
    obs_loc_lla = crs_lla.transform_points(
        crs_utm, obs_loc.iloc[:, 0], obs_loc.iloc[:, 1]
    )
    obs_loc_xyz = crs_xyz.transform_points(
        crs_utm, obs_loc.iloc[:, 0], obs_loc.iloc[:, 1]
    )
    joint_centroids_lla = crs_lla.transform_points(
        crs_utm, fault.joint_centroids[:, 0], fault.joint_centroids[:, 1]
    )
    joint_centroids_xyz = crs_xyz.transform_points(
        crs_utm, fault.joint_centroids[:, 0], fault.joint_centroids[:, 1]
    )

    # convert IGS to PAC
    vel_pac_itrf = vel_from_rotvec(ROT_PAC_ITRF, obs_loc_xyz, obs_loc_lla) / 1e6
    ts_pac_itrf = ((t_obs - t_obs[0]).astype(float) / 1e9 / SEC_PER_YEAR)[
        None, None, :
    ] * vel_pac_itrf[:, :, None]
    ts_obs_pac = ts_obs_itrf - ts_pac_itrf

    # convert IGS to EUR
    vel_eur_itrf = vel_from_rotvec(ROT_EUR_ITRF, obs_loc_xyz, obs_loc_lla) / 1e6
    ts_eur_itrf = ((t_obs - t_obs[0]).astype(float) / 1e9 / SEC_PER_YEAR)[
        None, None, :
    ] * vel_eur_itrf[:, :, None]
    ts_obs_eur = ts_obs_itrf - ts_eur_itrf

    # # save PAC data
    # np.save(DATA_OUT_BASENAME + "_pac.npy", ts_obs_pac)
    # shutil.copy2(DATA_OUT_BASENAME + ".json", DATA_OUT_BASENAME + "_pac.json")

    # get average absolute velocity of the stations for the interseismic time
    # have to loop because of NaNs in data
    # vel_obs_pac = np.stack([[
    #     np.linalg.lstsq(
    #         Gt[mask[istat, iax, s_preseismic], :],
    #         ts_obs_pac[:, :, s_preseismic][istat, iax, mask[istat, iax, s_preseismic]],
    #         rcond=None
    #         )[0][1]
    #     for iax in range(n_comps)]
    #     for istat in range(n_stations)])
    # load from all_poly_df
    itrf_sec_vels = pd.read_csv(SEC_VEL_DATA_FNAME, index_col=0)
    vel_obs_itrf = itrf_sec_vels.loc[obs_loc.index, ["vel_e", "vel_n", "vel_u"]] / 1000
    vel_obs_pac = (vel_obs_itrf - vel_pac_itrf).values
    vel_obs_eur = (vel_obs_itrf - vel_eur_itrf).values

    # 0 = K(v-v_p) with v_p due to initial euler pole gives slip rate on fault
    # v=0 inside the asperities
    # no stress from farfield since we impose v=v_p
    # use these v to predict surface displacement due to late interseismic locking

    # add this velocity to observed surface velocity (should point in different directions)
    # to give residual that should be only due to plate motion

    # now estimate euler pole(s)

    # repeat

    # initial guess of euler pole derived surface velocity
    # this is EUR relative to PAC
    # vel_prior = vel_from_rotvec(-ROT_PAC_EUR, obs_loc_xyz, obs_loc_lla) / 1e6

    # we need polygons for Northern Honshu and Hokkaido to do Euler pole estimation
    # independently for the two areas
    # get station mask for the two areas
    sta_in_honshu = HONSHU_PATH.contains_points(obs_loc_lla[:, :2])
    sta_in_hokkaido = HOKKAIDO_PATH.contains_points(obs_loc_lla[:, :2])
    sta_outside = (~sta_in_honshu) & (~sta_in_hokkaido)
    assert not np.any(sta_in_honshu & sta_in_hokkaido)
    n_in_honshu = sta_in_honshu.sum()

    # for repeat, we need new v_plate at all patch centroids
    # first, find out which patch belongs to which block
    patch_in_honshu = HONSHU_PATH.contains_points(joint_centroids_lla[:, :2])
    patch_in_hokkaido = HOKKAIDO_PATH.contains_points(joint_centroids_lla[:, :2])
    patch_outside = (~patch_in_honshu) & (~patch_in_hokkaido)
    assert not np.any(patch_in_honshu & patch_in_hokkaido)
    # now, for the patches outside these blocks, find the closest patch that is inside one
    dists = np.linalg.norm(
        fault.joint_centroids[patch_outside, None, :]
        - fault.joint_centroids[None, ~patch_outside, :],
        axis=2,
    )
    i_closest = np.argmin(dists, axis=1)  # closest to hon or hok
    # assign that block to that patch
    closest_is_honshu = patch_in_honshu.copy()
    closest_is_honshu[patch_outside] = patch_in_honshu[~patch_outside][i_closest]
    closest_is_hokkaido = patch_in_hokkaido.copy()
    closest_is_hokkaido[patch_outside] = patch_in_hokkaido[~patch_outside][i_closest]
    # v_plate for boundary/infinity patches is unrealistic, need to assign
    # closest non-infinite velocity
    ix_inf = (
        np.flatnonzero(fault.ix_tri_inf)
        + fault.inner_num_patches
        + fault.asperities_num_patches
    ).tolist() + (
        np.arange(fault.lower_num_patches)[-fault.num_lower_extended :]
        + fault.inner_num_patches
        + fault.asperities_num_patches
        + fault.outer_num_patches
    ).tolist()
    ix_notinf = ~np.isin(np.arange(fault.joint_num_patches), ix_inf)

    # new method: one-step solution assuming backslip model

    # TODO: need to add tensile reprojection like so:
    # v_new = [ 1 0 0             ] [ strike ]
    #         [ 0 1 tan(dipangle) ] [ dip ]
    #         [ 0 0 0             ] [ tensile ]

    # # get mapping matrices from Euler pole to velocity
    # R_cross = [crossmat(x1, x2, x3)
    #            for x1, x2, x3 in joint_centroids_xyz]
    # v_u = np.array([0, 0, 1])
    # dipangles = np.array([np.arccos(np.dot(-sim.fault.R_joint_efcs_to_ddcs[i, 2, :], v_u))
    #                       for i in range(sim.fault.joint_num_patches)])
    # R_project = [np.array([[1, 0, 0], [0, 1, np.tan(da)], [0, 0, 0]])
    #              for da in dipangles]
    # R_ecef2ddcs = [sim.fault.R_joint_tdcs_to_ddcs[i, :, :]
    #                @ R_project[i]
    #                @ sim.fault.R_joint_efcs_to_tdcs[i, :, :]
    #                @ R_ecef2enu(lo, la)
    #                for i, (lo, la)
    #                in enumerate(joint_centroids_lla[:, :2])]
    # G_EP = np.stack([(R_ecef2ddcs[i] @ R_cross[i])[:2, :]
    #                  for i in range(sim.fault.joint_num_patches)])
    # # get mapping matrices for asperities
    # G_fault_asp = sim.G_surf[sta_in_honshu, :, sim.fault.s_asperities, :2]
    # n_asp = sim.fault.asperities_num_patches
    # G_asp = G_fault_asp.reshape(n_in_honshu * 3, n_asp * 2)
    # # get mapping matrices for inner patches
    # G_fault_inner = sim.G_surf[sta_in_honshu, :, sim.fault.s_inner, :2]
    # K_inner_inv = sim.fault.K_inner_inner_inv
    # K_ext = sim.fault.K_inner_asperities[:, :2, :, :2]
    # n_inner = sim.fault.inner_num_patches
    # G_inner = (G_fault_inner.reshape(n_in_honshu * 3, n_inner * 2)
    #            @ K_inner_inv.reshape(n_inner * 2, n_inner * 2)
    #            @ K_ext.reshape(n_inner * 2, n_asp * 2))
    # # combine everything
    # G_direct = (G_inner - G_asp) @ G_EP[sim.fault.s_asperities, :, :].reshape(n_asp * 2, 3)
    # # solve
    # rotvec_direct_3d = np.linalg.lstsq(G_direct,
    #                                    vel_obs_eur[sta_in_honshu, :].ravel(),
    #                                    rcond=None)[0]
    # rotvec_direct_2d = np.linalg.lstsq(G_direct[np.tile([True, True, False], n_in_honshu), :],
    #                                    vel_obs_eur[sta_in_honshu, :2].ravel(),
    #                                    rcond=None)[0]
    # euler_vel_3d = vel_from_rotvec(np.rad2deg(rotvec_direct_3d),
    #                                obs_loc_xyz[sta_in_honshu, :],
    #                                obs_loc_lla[sta_in_honshu, :])
    # euler_vel_2d = vel_from_rotvec(np.rad2deg(rotvec_direct_2d),
    #                                obs_loc_xyz[sta_in_honshu, :],
    #                                obs_loc_lla[sta_in_honshu, :])
    # temp = vel_from_rotvec(ROT_PAC_EUR / 5,
    #                        obs_loc_xyz[sta_in_honshu, :],
    #                        obs_loc_lla[sta_in_honshu, :]) / 1e6

    # # debugging
    # K_inner_red = sim.fault.K_inner_inner[:, :2, :, :2].reshape(n_inner * 2, n_inner * 2)
    # K_asp_red = sim.fault.K_inner_asperities[:, :2, :, :2].reshape(n_inner * 2, -1)
    # v_star = np.linalg.lstsq(
    #     K_inner_red,
    #     K_inner_red @ sim.v_plate_ddcs_proj_eff[fault.s_inner, :].ravel() +
    #     K_asp_red @ sim.v_plate_ddcs_proj_eff[fault.s_asperities, :].ravel(),
    #     rcond=None)[0].reshape(n_patches, 2)
    # contrib_theor = (G_fault_inner.reshape(n_in_honshu * 3, n_inner * 2)
    #                  @ (v_star - sim.v_plate_ddcs_proj_eff[fault.s_inner, :2]).ravel()
    #                  + G_asp @ sim.v_plate_ddcs_proj_eff[fault.s_asperities, :2].ravel()
    #                  ).reshape(n_in_honshu, -1) * SEC_PER_YEAR

    # s = 5e-2
    # fig = plt.figure()
    # ax = fig.add_subplot(projection=PROJ_PC)
    # q = ax.quiver(obs_loc.values[sta_in_honshu, 0],
    #               obs_loc.values[sta_in_honshu, 1],
    #               temp[:, 0],
    #               temp[:, 1],
    #               scale=s, angles='xy', scale_units='xy', color="C2",
    #               label="PAC r.t. EUR / 5", transform=PROJ_UTM)
    # _ = ax.quiver(obs_loc.values[sta_in_honshu, 0],
    #               obs_loc.values[sta_in_honshu, 1],
    #               vel_obs_eur[sta_in_honshu, 0],
    #               vel_obs_eur[sta_in_honshu, 1],
    #               scale=s, angles='xy', scale_units='xy', color="k",
    #               label="obs", transform=PROJ_UTM)
    # _ = ax.quiver(obs_loc.values[sta_in_honshu, 0],
    #               obs_loc.values[sta_in_honshu, 1],
    #               euler_vel_2d[:, 0],
    #               euler_vel_2d[:, 1],
    #               scale=s, angles='xy', scale_units='xy', color="C0",
    #               label="from 2D", transform=PROJ_UTM)
    # _ = ax.quiver(obs_loc.values[sta_in_honshu, 0],
    #               obs_loc.values[sta_in_honshu, 1],
    #               euler_vel_3d[:, 0],
    #               euler_vel_3d[:, 1],
    #               scale=s, angles='xy', scale_units='xy', color="C1",
    #               label="from 3D", transform=PROJ_UTM)
    # ax.legend()
    # ax.quiverkey(q, 0.8, 0.25, 0.01, label="10 mm/a", color="k")

    # old method: iterative and with intermediate steps

    # we need:
    # 1) v_p in DDCS on inner and asperity mesh
    # v_p_joint = sim.v_plate_ddcs_proj_eff.copy()  # this is just relative
    iter_sims = [sim]
    # 2) observed velocity [m/a] in PAC
    # 3) initial euler pole vel_prior from above

    # solve for best-fitting v_star: 0 = K_int (v_star*-v_p) - K_ext*v_p
    # <=> K_inner*v_star = K_inner*v_p + K_asp*v_p
    iter_faults = [fault]
    K_inner_red = fault.K_inner_inner[:, :2, :, :2].reshape(
        n_patches * 2, n_patches * 2
    )
    K_asp_red = fault.K_inner_asperities[:, :2, :, :2].reshape(n_patches * 2, -1)
    iter_v_star = [
        np.linalg.lstsq(
            K_inner_red,
            K_inner_red @ iter_sims[0].v_plate_ddcs_proj_eff[fault.s_inner, :].ravel()
            + K_asp_red
            @ iter_sims[0].v_plate_ddcs_proj_eff[fault.s_asperities, :].ravel(),
            rcond=None,
        )[0].reshape(n_patches, 2)
    ]

    # predict surface velocities
    # v_p needs to be JAP rel to PAC; prior guess is EUR rel to PAC
    # this needs to be G_int (vstar - v_p) - G_ext v_p
    iter_vel_inter = [
        (
            get_surface_displacements(
                (iter_v_star[0] - iter_sims[0].v_plate_ddcs_proj_eff[fault.s_inner, :])[
                    :, :, None
                ],
                iter_sims[0].G_surf[:, :, fault.s_inner, :],
            )
            - get_surface_displacements(
                iter_sims[0].v_plate_ddcs_proj_eff[fault.s_asperities, :][:, :, None],
                iter_sims[0].G_surf[:, :, fault.s_asperities, :],
            )
        ).squeeze()
        * SEC_PER_YEAR
    ]

    # difference is the contribution of the plate motion
    # vel_inter is JAP rel to PAC
    # vel_obs is in PAC (fixed)
    # vel_euler will be in JAP rel to PAC
    iter_vel_euler = [vel_obs_pac - iter_vel_inter[0]]

    # fit this euler velocity HON & HOK rel to PAC
    iter_new_rotvec_hon = [
        np.rad2deg(
            estimate_euler_pole(
                obs_loc_lla[sta_in_honshu, :2], iter_vel_euler[0][sta_in_honshu, :2]
            )[0]
        )
    ]
    # only estimate a single rotation vector based on Honshu data
    # iter_new_rotvec_hok = [np.rad2deg(
    #     estimate_euler_pole(obs_loc_lla[sta_in_hokkaido, :2],
    #                         iter_vel_euler[0][sta_in_hokkaido, :2])[0])]
    iter_new_rotvec_hok = [iter_new_rotvec_hon[0].copy()]

    # predict velocities
    iter_vel_prior_hon = [
        vel_from_rotvec(
            iter_new_rotvec_hon[0],
            obs_loc_xyz[sta_in_honshu, :],
            obs_loc_lla[sta_in_honshu, :],
        )
    ]
    iter_vel_prior_hok = [
        vel_from_rotvec(
            iter_new_rotvec_hok[0],
            obs_loc_xyz[sta_in_hokkaido, :],
            obs_loc_lla[sta_in_hokkaido, :],
        )
    ]
    iter_vel_prior_out_hon = [
        vel_from_rotvec(
            iter_new_rotvec_hon[0],
            obs_loc_xyz[sta_outside, :],
            obs_loc_lla[sta_outside, :],
        )
    ]
    iter_vel_prior_hok_hon = [
        vel_from_rotvec(
            iter_new_rotvec_hon[0],
            obs_loc_xyz[sta_in_hokkaido, :],
            obs_loc_lla[sta_in_hokkaido, :],
        )
    ]

    # plot
    s = 1e5
    # fig = plt.figure()
    # ax = fig.add_subplot(projection=PROJ_PC)
    # ax.quiver(obs_loc.values[sta_in_honshu, 0],
    #           obs_loc.values[sta_in_honshu, 1],
    #           iter_vel_prior_hon[0][:, 0],
    #           iter_vel_prior_hon[0][:, 1],
    #           scale=s, angles='xy', scale_units='xy', color=cm.roma(0.9), transform=PROJ_UTM)
    # ax.quiver(obs_loc.values[sta_outside, 0],
    #           obs_loc.values[sta_outside, 1],
    #           iter_vel_prior_out_hon[0][:, 0],
    #           iter_vel_prior_out_hon[0][:, 1],
    #           scale=s, angles='xy', scale_units='xy', color=cm.roma(0.7), transform=PROJ_UTM)
    # ax.quiver(obs_loc.values[sta_in_hokkaido, 0],
    #           obs_loc.values[sta_in_hokkaido, 1],
    #           iter_vel_prior_hok_hon[0][:, 0],
    #           iter_vel_prior_hok_hon[0][:, 1],
    #           scale=s, angles='xy', scale_units='xy', color=cm.roma(0.8), transform=PROJ_UTM)
    # ax.quiver(obs_loc.values[sta_in_hokkaido, 0],
    #           obs_loc.values[sta_in_hokkaido, 1],
    #           iter_vel_prior_hok[0][:, 0],
    #           iter_vel_prior_hok[0][:, 1],
    #           scale=s, angles='xy', scale_units='xy', color=cm.roma(0.1), transform=PROJ_UTM)
    # ax.quiver(obs_loc.values[:, 0],
    #           obs_loc.values[:, 1],
    #           vel_prior[:, 0],
    #           vel_prior[:, 1],
    #           scale=s, angles='xy', scale_units='xy', color="0.5", transform=PROJ_UTM)
    # ax.quiver(obs_loc.values[:, 0],
    #           obs_loc.values[:, 1],
    #           iter_vel_euler[0][:, 0],
    #           iter_vel_euler[0][:, 1],
    #           scale=s, angles='xy', scale_units='xy', color="k", transform=PROJ_UTM)

    # predict v_plate from both blocks
    iter_v_plate_vec = [np.zeros((fault.joint_num_patches, 3))]
    iter_v_plate_vec[0][closest_is_honshu, :] = vel_from_rotvec(
        iter_new_rotvec_hon[0],
        joint_centroids_xyz[closest_is_honshu, :],
        joint_centroids_lla[closest_is_honshu, :],
    )
    iter_v_plate_vec[0][closest_is_hokkaido, :] = vel_from_rotvec(
        iter_new_rotvec_hok[0],
        joint_centroids_xyz[closest_is_hokkaido, :],
        joint_centroids_lla[closest_is_hokkaido, :],
    )
    # loop over infinite patch indices
    for i in ix_inf:
        # find closest non-infinite patch
        j = np.argmin(
            np.linalg.norm(
                fault.joint_centroids[ix_notinf, :]
                - fault.joint_centroids[i, :][None, :],
                axis=1,
            )
        )
        iter_v_plate_vec[0][i, :] = iter_v_plate_vec[0][ix_notinf, :][j, :]

    # start loop
    n_iters = 10
    for iloop in range(1, n_iters):
        # recreate objects
        temp_fault_dict = deepcopy(fault_dict)
        # precomputed velocities
        temp_fault_dict["farfield_motion_type"] = 4
        # need to switch reference frames
        temp_fault_dict["farfield_motion"] = -iter_v_plate_vec[iloop - 1] / SEC_PER_YEAR
        iter_faults.append(
            Fault3D.from_cfg_and_files(
                temp_fault_dict,
                K_inner_asperities_file=K_INNER_ASP_FILE,
                K_inner_inner_file=K_INNER_INNER_FILE,
                cross_sections_file=CROSS_SECTION_FILE,
                write_new_files=False,
                verbose=True,
            )
        )
        iter_sims.append(
            SubductionSimulation3D.from_cfg_objs_and_files(
                sim_dict,
                rheo,
                iter_faults[-1],
                G_surf_file=G_SURF_FILE,
                write_new_files=False,
                verbose=True,
            )
        )
        # new v_star
        K_inner_red = fault.K_inner_inner[:, :2, :, :2].reshape(
            n_patches * 2, n_patches * 2
        )
        K_asp_red = fault.K_inner_asperities[:, :2, :, :2].reshape(n_patches * 2, -1)
        iter_v_star.append(
            np.linalg.lstsq(
                K_inner_red,
                K_inner_red
                @ iter_sims[-1].v_plate_ddcs_proj_eff[fault.s_inner, :].ravel()
                + K_asp_red
                @ iter_sims[-1].v_plate_ddcs_proj_eff[fault.s_asperities, :].ravel(),
                rcond=None,
            )[0].reshape(n_patches, 2)
        )
        # new vel_inter
        iter_vel_inter.append(
            (
                get_surface_displacements(
                    (
                        iter_v_star[-1]
                        - iter_sims[-1].v_plate_ddcs_proj_eff[fault.s_inner, :]
                    )[:, :, None],
                    iter_sims[-1].G_surf[:, :, fault.s_inner, :],
                )
                - get_surface_displacements(
                    iter_sims[-1].v_plate_ddcs_proj_eff[fault.s_asperities, :][
                        :, :, None
                    ],
                    iter_sims[-1].G_surf[:, :, fault.s_asperities, :],
                )
            ).squeeze()
            * SEC_PER_YEAR
        )
        # new vel_euler
        iter_vel_euler.append(vel_obs_pac - iter_vel_inter[-1])
        # new rotation vectors
        iter_new_rotvec_hon.append(
            np.rad2deg(
                estimate_euler_pole(
                    obs_loc_lla[sta_in_honshu, :2],
                    iter_vel_euler[-1][sta_in_honshu, :2],
                )[0]
            )
        )
        # iter_new_rotvec_hok.append(np.rad2deg(
        #     estimate_euler_pole(obs_loc_lla[sta_in_hokkaido, :2],
        #                         iter_vel_euler[-1][sta_in_hokkaido, :2])[0]))
        iter_new_rotvec_hok.append(iter_new_rotvec_hon[-1].copy())
        # new priors
        iter_vel_prior_hon.append(
            vel_from_rotvec(
                iter_new_rotvec_hon[-1],
                obs_loc_xyz[sta_in_honshu, :],
                obs_loc_lla[sta_in_honshu, :],
            )
        )
        iter_vel_prior_hok.append(
            vel_from_rotvec(
                iter_new_rotvec_hok[-1],
                obs_loc_xyz[sta_in_hokkaido, :],
                obs_loc_lla[sta_in_hokkaido, :],
            )
        )
        iter_vel_prior_out_hon.append(
            vel_from_rotvec(
                iter_new_rotvec_hon[-1],
                obs_loc_xyz[sta_outside, :],
                obs_loc_lla[sta_outside, :],
            )
        )
        iter_vel_prior_hok_hon.append(
            vel_from_rotvec(
                iter_new_rotvec_hon[-1],
                obs_loc_xyz[sta_in_hokkaido, :],
                obs_loc_lla[sta_in_hokkaido, :],
            )
        )
        # new v_plate
        iter_v_plate_vec.append(np.zeros((fault.joint_num_patches, 3)))
        iter_v_plate_vec[-1][closest_is_honshu, :] = vel_from_rotvec(
            iter_new_rotvec_hon[-1],
            joint_centroids_xyz[closest_is_honshu, :],
            joint_centroids_lla[closest_is_honshu, :],
        )
        iter_v_plate_vec[-1][closest_is_hokkaido, :] = vel_from_rotvec(
            iter_new_rotvec_hok[-1],
            joint_centroids_xyz[closest_is_hokkaido, :],
            joint_centroids_lla[closest_is_hokkaido, :],
        )
        # loop over infinite patch indices
        for i in ix_inf:
            # find closest non-infinite patch
            j = np.argmin(
                np.linalg.norm(
                    fault.joint_centroids[ix_notinf, :]
                    - fault.joint_centroids[i, :][None, :],
                    axis=1,
                )
            )
            iter_v_plate_vec[-1][i, :] = iter_v_plate_vec[-1][ix_notinf, :][j, :]

    # plot progression of v_inter and vel_prior

    # plt.quiver(fault.inner_centroids[:, 0], fault.inner_centroids[:, 1],
    #            iter_v_star[0][:, 0], iter_v_star[0][:, 1], color="C0")
    # plt.quiver(fault.inner_centroids[:, 0], fault.inner_centroids[:, 1],
    #            iter_sims[0].v_plate_ddcs_proj_eff[fault.s_inner, 0],
    #            iter_sims[0].v_plate_ddcs_proj_eff[fault.s_inner, 1], color="C1")

    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_obs[:, 0], vel_obs[:, 1], color="C0")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            iter_vel_inter[0][:, 0], iter_vel_inter[0][:, 1], color="C1")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            iter_vel_euler[0][:, 0], iter_vel_euler[0][:, 1], color="C2")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_euler2[:, 0], vel_euler2[:, 1], color="C3")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_euler3[:, 0], vel_euler3[:, 1], color="C4")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_prior[:, 0], vel_prior[:, 1], color="k")

    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            (iter_vel_euler[0] - vel_prior)[:, 0],
    #            (iter_vel_euler[0] - vel_prior)[:, 1],
    #            color="k")

    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            (vel_prior2 + iter_vel_inter[0])[:, 0],
    #            (vel_prior2 + iter_vel_inter[0])[:, 1], color="C1")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            (vel_prior3 + vel_inter2)[:, 0], (vel_prior3 + vel_inter2)[:, 1], color="C2")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            (vel_prior4 + vel_inter3)[:, 0], (vel_prior4 + vel_inter3)[:, 1], color="C3")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_obs[:, 0], vel_obs[:, 1], color="k")

    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            iter_vel_inter[0][:, 0], iter_vel_inter[0][:, 1], color="C0")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_inter2[:, 0], vel_inter2[:, 1], color="C1")
    # plt.quiver(obs_loc.iloc[:, 0], obs_loc.iloc[:, 1],
    #            vel_inter3[:, 0], vel_inter3[:, 1], color="C2")

    # L&M 2010 lon/lat/rate deg/Ma - relative to nominally stable eurasia
    # Pacific 	104.78° ± 0.70° 	−51.14° ± 0.27° 	0.78° ± 0.00°
    # NE Honshu 	324.70° ± 1.28° 	−40.77° ± 0.83° 	1.28° ± 0.26°
    # Central Hokkaido 320.72° ± 0.35° -43.16° ± 0.24° 2.46° ± 0.42°
    lm_pacific_euler = np.deg2rad([104.78, -51.14, 0.78])
    lm_nehonshu_euler = np.deg2rad([324.7, -40.77, 1.28])
    lm_hokkaido_euler = np.deg2rad([320.72, -43.16, 2.46])
    lm_pacific_rotvec = eulerpole2rotvec(lm_pacific_euler)
    lm_nehonshu_rotvec = eulerpole2rotvec(lm_nehonshu_euler)
    lm_hokkaido_rotvec = eulerpole2rotvec(lm_hokkaido_euler)
    # relative
    lm_pac2honshu_rotvec = lm_pacific_rotvec - lm_nehonshu_rotvec
    lm_pac2hokkaido_rotvec = lm_pacific_rotvec - lm_hokkaido_rotvec
    """
    >>> np.rad2deg(lm_pac2honshu_rotvec)
    array([-0.91600355,  1.03336493,  0.22849945])
    >>> np.rad2deg(lm_pac2hokkaido_rotvec)
    array([-1.51385169,  1.60927262,  1.07536208])
    """

    # save final Euler rotation vectors to file
    summary = {
        "rotvec_honshu": (-iter_new_rotvec_hon[-1] * 1e6).tolist(),
        "rotvec_hokkaido": (-iter_new_rotvec_hok[-1] * 1e6).tolist(),
    }
    print(summary)
    """
    {
    "rotvec_honshu": [-0.8595244894853925, 1.0793456606904304, 0.069938345018291],
    "rotvec_hokkaido": [-0.8595244894853925, 1.0793456606904304, 0.069938345018291]
    }
    """
    with open(CORRECTIONS_FILE, mode="wt") as f:
        json.dump(summary, f, indent="")
    # save final v_plate_vec
    final_v_plate_vec = -iter_v_plate_vec[-1] / SEC_PER_YEAR
    np.save(CORRECTED_V_PLATE_FILE, final_v_plate_vec)

    # stack progression of velocities
    stack_vel_inter = np.stack(iter_vel_inter)
    stack_vel_euler = np.stack(iter_vel_euler)
    stack_vel_euler_fit_hon = np.stack(iter_vel_prior_hon)
    stack_vel_euler_fit_hok = np.stack(iter_vel_prior_hok)
    stack_vel_euler_fit_out = np.stack(iter_vel_prior_out_hon)
    stack_v_plate_vec = np.stack(iter_v_plate_vec)
    stack_vel_mod = stack_vel_inter.copy()
    stack_vel_mod[:, sta_in_honshu, :] += stack_vel_euler_fit_hon
    stack_vel_mod[:, sta_in_hokkaido, :] += stack_vel_euler_fit_hok
    stack_vel_mod[:, sta_outside, :] += stack_vel_euler_fit_out

    # velocity honshu to pacific
    my_vel_hon_to_pac = vel_from_rotvec(
        iter_new_rotvec_hon[-1], obs_loc_xyz, obs_loc_lla
    )
    # velocity hokkaido to pacific
    my_vel_hok_to_pac = vel_from_rotvec(
        iter_new_rotvec_hok[-1], obs_loc_xyz, obs_loc_lla
    )

    # combine the different predicted plate velocities for all stations into one array
    combined_vel_jap_to_pac = my_vel_hon_to_pac.copy()
    combined_vel_jap_to_pac[sta_in_hokkaido, :] = my_vel_hok_to_pac[sta_in_hokkaido, :]

    # compute timeseries
    combined_ts_jap_to_pac = (
        combined_vel_jap_to_pac[:, :, None]
        * (t_obs[None, None, :] - t_obs[0]).astype(float)
        / 1e9
        / SEC_PER_YEAR
    )
    # compute velocities
    vel_obs_hon = vel_obs_pac - my_vel_hon_to_pac
    # remove from PAC timeseries to get JAP timeseries
    ts_obs_jap = ts_obs_pac - combined_ts_jap_to_pac
    # save
    np.save(DATA_OUT_BASENAME + "_jap_ok.npy", ts_obs_jap)
    shutil.copy2(f"{DATA_IN_BASENAME}.json", DATA_OUT_BASENAME + "_jap_ok.json")

    # save data for just honshu
    obs_loc_honshu = obs_loc.iloc[sta_in_honshu, :]
    obs_loc_honshu.to_csv(f"{OBS_LOC_BASENAME}_honshu_ok.csv")
    np.save(DATA_OUT_BASENAME + "_jap_honshu_ok.npy", ts_obs_jap[sta_in_honshu, :, :])
    with (
        open(f"{DATA_IN_BASENAME}.json", mode="rt") as fin,
        open(DATA_OUT_BASENAME + "_jap_honshu_ok.json", mode="wt") as fout,
    ):
        temp_info = json.load(fin)
        temp_info["ts_shape"][0] = int(sta_in_honshu.sum())
        json.dump(temp_info, fout)

    # load interseismic velocities from Loveless & Meade (2016)
    # https://doi.org/10.1016/j.epsl.2015.12.033
    v_mdl_LM16_eu = pd.read_csv(LM16_FNAME, index_col=0, delimiter=r"\s+", comment="#")
    inside_njapan_lm16 = NJAPAN_PATH.contains_points(
        v_mdl_LM16_eu[["latitude", "longitude"]].values
    )
    v_mdl_LM16_eu = v_mdl_LM16_eu.iloc[inside_njapan_lm16, :]

    # plot interseismic observed velocities split into components & fits
    s1 = 2e-1
    fig1 = plt.figure(num="Fit progression of surface velocities", figsize=(6, 6))
    ax1 = fig1.add_subplot(projection=PROJ_PC)
    for i in range(n_iters):
        ax1.quiver(
            obs_loc.iloc[:, 0],
            obs_loc.iloc[:, 1],
            stack_vel_inter[i, :, 0],
            stack_vel_inter[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.vik(0.4 - i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=i,
            transform=PROJ_UTM,
        )
        ax1.quiver(
            obs_loc.iloc[:, 0],
            obs_loc.iloc[:, 1],
            stack_vel_euler[i, :, 0],
            stack_vel_euler[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.vik(0.6 + i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=n_iters + i,
            transform=PROJ_UTM,
        )
        ax1.quiver(
            obs_loc.iloc[sta_in_honshu, 0],
            obs_loc.iloc[sta_in_honshu, 1],
            stack_vel_euler_fit_hon[i, :, 0],
            stack_vel_euler_fit_hon[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.bam(0.6 + i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=2 * n_iters + i,
            transform=PROJ_UTM,
        )
        ax1.quiver(
            obs_loc.iloc[sta_in_hokkaido, 0],
            obs_loc.iloc[sta_in_hokkaido, 1],
            stack_vel_euler_fit_hok[i, :, 0],
            stack_vel_euler_fit_hok[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.bam(0.6 + i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=2 * n_iters + i,
            transform=PROJ_UTM,
        )
        ax1.quiver(
            obs_loc.iloc[sta_outside, 0],
            obs_loc.iloc[sta_outside, 1],
            stack_vel_euler_fit_out[i, :, 0],
            stack_vel_euler_fit_out[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.bam(0.6 + i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=2 * n_iters + i,
            transform=PROJ_UTM,
        )
        ax1.quiver(
            obs_loc.iloc[:, 0],
            obs_loc.iloc[:, 1],
            stack_vel_mod[i, :, 0],
            stack_vel_mod[i, :, 1],
            scale=s1,
            angles="xy",
            scale_units="xy",
            color=cm.bam(0.4 - i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=3 * n_iters + i,
            transform=PROJ_UTM,
        )
    q = ax1.quiver(
        obs_loc.iloc[:, 0],
        obs_loc.iloc[:, 1],
        vel_obs_pac[:, 0],
        vel_obs_pac[:, 1],
        scale=s1,
        angles="xy",
        scale_units="xy",
        color="k",
        zorder=4 * n_iters,
        transform=PROJ_UTM,
    )
    ax1.coastlines(lw=0.5)
    ax1.quiverkey(q, 0.8, 0.25, 0.1, label="100 mm/a", color="k")
    ax1.legend(
        [
            Line2D([0], [0], color="k", lw=4),
            Line2D([0], [0], color=cm.bam(0.15), lw=4),
            Line2D([0], [0], color=cm.vik(0.15), lw=4),
            Line2D([0], [0], color=cm.vik(0.85), lw=4),
            Line2D([0], [0], color=cm.bam(0.85), lw=4),
        ],
        [
            "Observed",
            "Best Fit to Observed",
            "Locking Effects",
            "Plate Motion",
            "Best Euler Pole Fit",
        ],
        loc="lower right",
    )
    fig1.suptitle("Fit progression of observed surface velocities in PAC")
    fig1.savefig("corrected_fit_surface_ok.pdf", dpi=300)

    # plot progression of fault slip rates
    s2 = 1e-1
    fig2 = plt.figure(num="Fit progression of fault slip rates", figsize=(6, 6))
    ax2 = fig2.add_subplot(projection=PROJ_PC)
    for i in range(n_iters):
        ax2.quiver(
            fault.inner_centroids[:, 0],
            fault.inner_centroids[:, 1],
            stack_v_plate_vec[i, fault.s_inner, 0],
            stack_v_plate_vec[i, fault.s_inner, 1],
            scale=s2,
            angles="xy",
            scale_units="xy",
            color=cm.vik(0.4 - i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=n_iters + i,
            transform=PROJ_UTM,
        )
        ax2.quiver(
            fault.asperities_centroids[:, 0],
            fault.asperities_centroids[:, 1],
            stack_v_plate_vec[i, fault.s_asperities, 0],
            stack_v_plate_vec[i, fault.s_asperities, 1],
            scale=s2,
            angles="xy",
            scale_units="xy",
            color=cm.vik(0.6 + i / n_iters / 4),
            width=0.005,
            headwidth=3,
            headlength=5,
            headaxislength=5,
            zorder=i,
            transform=PROJ_UTM,
        )
    ax2.quiver(
        fault.inner_centroids[:, 0],
        fault.inner_centroids[:, 1],
        -fault.v_plate_vec[fault.s_inner, 0] * SEC_PER_YEAR,
        -fault.v_plate_vec[fault.s_inner, 1] * SEC_PER_YEAR,
        scale=s2,
        angles="xy",
        scale_units="xy",
        color="k",
        width=0.003,
        headwidth=3,
        headlength=5,
        headaxislength=5,
        zorder=2 * n_iters,
        transform=PROJ_UTM,
    )
    q = ax2.quiver(
        fault.asperities_centroids[:, 0],
        fault.asperities_centroids[:, 1],
        -fault.v_plate_vec[fault.s_asperities, 0] * SEC_PER_YEAR,
        -fault.v_plate_vec[fault.s_asperities, 1] * SEC_PER_YEAR,
        scale=s2,
        angles="xy",
        scale_units="xy",
        color="k",
        width=0.003,
        headwidth=3,
        headlength=5,
        headaxislength=5,
        zorder=2 * n_iters,
        transform=PROJ_UTM,
    )
    ax2.coastlines(lw=0.5)
    ax2.quiverkey(q, 0.8, 0.25, 0.1, label="100 mm/a", color="k")
    ax2.legend(
        [
            Line2D([0], [0], color="k", lw=4),
            Line2D([0], [0], color=cm.vik(0.15), lw=4),
            Line2D([0], [0], color=cm.vik(0.85), lw=4),
        ],
        ["Initial (single EP)", "Fitted (Creeping)", "Fitted (Locked)"],
        loc="lower right",
    )
    fig2.suptitle("Fit progression of plate interface fault slip rate")
    fig2.savefig("corrected_fit_plate_ok.pdf", dpi=300)

    # plot observations in JAP reference frame
    s3 = 5e-2
    fig3 = plt.figure(num="Observations in JAP", figsize=(6, 6))
    ax3 = fig3.add_subplot(projection=PROJ_PC)
    _ = ax3.quiver(
        v_mdl_LM16_eu["longitude"],
        v_mdl_LM16_eu["latitude"],
        v_mdl_LM16_eu["east velocity (mm/yr)"] / 1000,
        v_mdl_LM16_eu["north velocity (mm/yr)"] / 1000,
        scale=s3,
        angles="xy",
        scale_units="xy",
        color="C1",
        transform=PROJ_PC,
        label="L&M 2016",
    )
    q = ax3.quiver(
        obs_loc.iloc[:, 0],
        obs_loc.iloc[:, 1],
        vel_obs_hon[:, 0],
        vel_obs_hon[:, 1],
        scale=s3,
        angles="xy",
        scale_units="xy",
        color="C0",
        transform=PROJ_UTM,
        label="DISSTANS re-oriented",
    )
    ax3.coastlines(lw=0.5)
    ax3.quiverkey(q, 0.8, 0.25, 0.1, label="100 mm/a", color="k")
    ax3.legend(loc="lower right")
    fig3.suptitle("Observations in Overriding Plate Reference Frames")
    fig3.savefig("corrected_final_observations_ok.pdf", dpi=300)

    # # show
    # plt.show()
