"""
Simple functions to interact with the Matlab files provided by the SRCMOD database.
"""

import numpy as np
from scipy.io import loadmat


# function copied from
# https://github.com/thearagon/srcmod2cmt/blob/f1d701f0927c2a5d923c2a36eb5dd8c56cc85281/cmtsol.py
def fh(lon, lat, z, strike, distance):
    """

    Given a start point (lon lat), bearing (degrees), and distance (m),
    calculates the destination point (lon lat)
    """

    theta = strike
    delta = distance / 6371000.

    theta = theta * np.pi / 180.
    lat1 = lat * np.pi / 180.
    lon1 = lon * np.pi / 180.

    lat2 = np.arcsin(np.sin(lat1) * np.cos(delta) +
                     np.cos(lat1) * np.sin(delta) * np.cos(theta))

    lon2 = lon1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(lat1),
                             np.cos(delta) - np.sin(lat1) * np.sin(lat2))

    return (lon2 * 180.0 / np.pi, lat2 * 180.0 / np.pi, z)


# function copied from
# https://github.com/thearagon/srcmod2cmt/blob/f1d701f0927c2a5d923c2a36eb5dd8c56cc85281/cmtsol.py
def fd(lon, lat, z, strike, dip, ddip):
    """

    Given a start point (lon lat z), strike, dip of the fault (degrees),
    and distance along dip (m), calculates the destination point (lon lat z)
    """

    theta = strike + 90

    z2 = z + ddip * np.sin(dip * np.pi / 180.)
    D = np.cos(dip * np.pi / 180.) * ddip

    return fh(lon, lat, z2, theta, D)


def load_file(fname: str) -> dict:
    """
    Load a SRCMOD-formatted .mat file with all original fields preserved.

    Parameters
    ----------
    fname
        Path to file.

    Returns
    -------
        Dictionary of items included in the Matlab file.
    """
    struct = loadmat(fname, struct_as_record=False, squeeze_me=True)
    event_id = [k for k in struct.keys() if k[:2] != "__"]
    if len(event_id) > 1:
        raise RuntimeError("Unrecognized file format: found multiple events in Matlab struct ("
                           f"{event_id}).")
    event_id = event_id[0]
    return {field: getattr(struct[event_id], field) for field in struct[event_id]._fieldnames}


class Segment():
    """ Container class for fault segment data """

    def __init__(self,
                 strike: float,
                 dip: float,
                 Nz: int,
                 width: float,
                 lon: np.ndarray,
                 lat: np.ndarray,
                 depth: np.ndarray,
                 slip: np.ndarray
                 ) -> None:

        # save data
        self.strike = float(strike)
        """ Strike angle [°] """
        self.dip = float(dip)
        """ Dip angle [°] """
        self.Nz = int(Nz)
        """ Number of patches in the depth direction [-] """
        self.width = float(width)
        """ Fault width [km] """
        self.lon = np.atleast_2d(lon)
        """ Longitude of each patch [°] """
        self.lat = np.atleast_2d(lat)
        """ Latitude of each patch [°] """
        self.depth = np.atleast_2d(depth)
        """ Depth of each patch [km] """
        self.slip = np.atleast_2d(slip)
        """ Slip at each patch [m] """

        # test data
        assert self.lon.shape == self.lat.shape == self.depth.shape == self.slip.shape
        assert self.slip.shape[0] == self.Nz


def is_single_segment(file_dict: dict) -> bool:
    """
    Test a file dictionary whether it has one or multiple segments.

    Parameters
    ----------
    file_dict
        Input data as output by `load_file`.

    Returns
    -------
        Whether the file has a single segment (`True`) or multiple ones (`False`).

    Raises
    ------
    ValueError
        If the structure is not recognized to be either single or multiple segments.
    """
    SINGLE_SEG_VARS = ["srcAStke", "srcDipAn", "invNzNx", "srcDimWL",
                       "geoLON", "geoLAT", "geoZ", "slipSPL"]
    MULTI_SEG_VARS = ["AStke", "DipAn", "DimWL", "geoLON", "geoLAT", "geoZ", "SLIP"]
    try:
        if all([k in file_dict.keys() for k in SINGLE_SEG_VARS]) \
           and (file_dict["invSEGM"] == 1):
            is_single_segment = True
        elif all([f"seg{i}{k}" for k in MULTI_SEG_VARS
                  for i in range(1, int(file_dict["invSEGM"]) + 1)]):
            is_single_segment = False
        else:
            raise ValueError
    except BaseException:
        raise ValueError(f"Unrecognized file structure with keys {sorted(file_dict.keys())}.")
    return is_single_segment


def extract_slip(file_dict: dict, slip_factor: float = 0.01) -> list:
    """
    Load all segments and extract the location and amount of the slip.

    Parameters
    ----------
    file_dict
        Input data as output by `load_file`.
    slip_factor
        Factor to multiply the slip with to get units of meters.

    Returns
    -------
        List of segments containing the location and slip amount for each slip.
    """

    # extract the data of the patch(es)
    if is_single_segment(file_dict):
        segs_raw = [Segment(strike=file_dict["srcAStke"],
                            dip=file_dict["srcDipAn"],
                            Nz=file_dict["invNzNx"][0],
                            width=file_dict["srcDimWL"][0],
                            lon=file_dict["geoLON"],
                            lat=file_dict["geoLAT"],
                            depth=file_dict["geoZ"],
                            slip=file_dict["slipSPL"])]
    else:
        segs_raw = [Segment(strike=file_dict[f"seg{i}AStke"],
                            dip=file_dict[f"seg{i}DipAn"],
                            Nz=np.atleast_2d(file_dict[f"seg{i}geoZ"]).shape[0],
                            width=file_dict[f"seg{i}DimWL"][0],
                            lon=file_dict[f"seg{i}geoLON"],
                            lat=file_dict[f"seg{i}geoLAT"],
                            depth=np.atleast_2d(file_dict[f"seg{i}geoZ"]),
                            slip=file_dict[f"seg{i}SLIP"])
                    for i in range(1, int(file_dict["invSEGM"]) + 1)]

    # convert nominal locations to centroids and slip to meters
    segs = []
    for s in segs_raw:
        new_lon, new_lat, new_depth = fd(s.lon, s.lat, s.depth,
                                         s.strike, s.dip, s.width / s.Nz / 2)
        segs.append(Segment(strike=s.strike,
                            dip=s.dip,
                            Nz=s.Nz,
                            width=s.width,
                            lat=new_lat,
                            lon=new_lon,
                            depth=new_depth,
                            slip=s.slip * slip_factor))

    # done
    return segs
