import math
import numpy as np
import pandas as pd
import monetio as mio
import xarray as xr
import dask


class bdy_value:
    def __init__(self, df=None, nc=None, mode=None):
        """
        Initialize the BdyValue class.
        
        Args:
            df (str): Path to the CSV file containing latitude, longitude, and height data.
            nc (str): Path to the netCDF file.
            mode (str): Simulation mode ('wrf' or 'cmaq').
        """
        self.nc = nc
        self.mode = mode
        if self.mode not in ["wrf", "cmaq"]:
            raise ValueError("Invalid mode option. Choose either 'wrf' or 'cmaq'.")

        self.df = pd.read_csv(
            df,
            names=["latitude", "longitude", "height"],
            skiprows=1,
            header=None,
            sep=";",
        )

    def _get_mode(self, varname=None, time=0):
        """
        Get simulation mode-specific data from the netCDF file.

        Args:
            varname (str): Variable name.
            time (int): Time index.

        Returns:
            lat (xarray.DataArray): Latitude values.
            lon (xarray.DataArray): Longitude values.
            height (xarray.DataArray): Height values.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            v_var (xarray.DataArray): Variable values.
        """
        if self.mode == "wrf":
            nc_file = xr.open_dataset(self.nc)
            if varname == "U":
                v_var = _dstag(nc_file[varname][time], 2)
            elif varname == "V":
                v_var = _dstag(nc_file[varname][time], 1)
            elif varname == "W":
                v_var = _dstag(nc_file[varname][time], 0)
            # To calculate WRF height:
            # The height of model levels above ground can be obtained as follow
            # [(PH+PHB)/9.8 - terrain height].
            dx = nc_file.DX
            dy = nc_file.DY
            ph = _dstag(nc_file["PH"][time], 0)
            ph_b = _dstag(nc_file["PHB"][time], 0)
            hgt = nc_file["HGT"][0]
            height = (ph + ph_b) / 9.8 - hgt
            lat = nc_file["XLAT"][time]
            lon = nc_file["XLONG"][time]

        elif self.mode == "cmaq":
            nc_file = mio.cmaq.open_mfdataset(self.nc, engine="netcdf4")
            m3d = xr.open_dataset("METCRO3D.nc")
            v_var = nc_file[varname][time]
            dx = nc_file.XCELL
            dy = nc_file.YCELL
            m3d = m3d.rename_dims(dims_dict={"LAY": "z", "ROW": "y", "COL": "x"})
            height = m3d["ZH"][time]
            lat = nc_file.latitude
            lon = nc_file.longitude

        return lat, lon, height, dx, dy, v_var

    def _get_crop(self, lat, lon, height, dx, dy, v_var):
        """
        Crop the variable data based on latitudinal and longitudinal boundaries.

        Args:
            lat (xarray.DataArray): Latitude values.
            lon (xarray.DataArray): Longitude values.
            height (xarray.DataArray): Height values.
            dx (float): Grid spacing in the x-direction.
            dy (float): Grid spacing in the y-direction.
            v_var (xarray.DataArray): Variable values.

        Returns:
            crop_value (xarray.DataArray): Cropped variable data.
        """
        crop_value = v_var.where(height <= self.df.height.max(), drop=True)
        latmin, latmax = self.df.latitude.min(), self.df.latitude.max()
        lonmin, lonmax = self.df.longitude.min(), self.df.longitude.max()
        lat_values = find_nearest(lat, (latmin + latmax) / 2, coord="lat")
        lon_values = find_nearest(lon, (lonmin + lonmax) / 2, coord="lon")

        with xr.set_options(keep_attrs=True):
            if latmin != latmax and lonmin != lonmax:
                crop_value = crop_value.where(
                    (lat >= latmin) & (lat <= latmax), drop=True
                )
                crop_value = crop_value.where(
                    (lon >= lonmin) & (lon <= lonmax), drop=True
                )
            elif (lonmin == lonmax and latmin == latmax) or (
                distance(latmin, latmax, val="lat") < dx
                and distance(lonmin, lonmax, val="lon") < dy
            ):
                if self.mode == "wrf":
                    crop_value = crop_value.isel(
                        south_north=slice(lon_values, lon_values + 1),
                        west_east=slice(lat_values, lat_values + 1),
                        drop=True,
                    )
                elif self.mode == "cmaq":
                    crop_value = crop_value.isel(
                        x=slice(lon_values, lon_values + 1),
                        y=slice(lat_values, lat_values + 1),
                        drop=True,
                    )
                # print(crop_value.shape)
                # crop_value = crop_value.swap_dims({'x': 'longitude', 'y': 'latitude'})
            elif latmin == latmax or distance(latmin, latmax, val="lat") < dx:
                if self.mode == "wrf":
                    crop_value = crop_value[:, lat_values, :]
                elif self.mode == "cmaq":
                    crop_value = crop_value.isel(y=lat_values)
                crop_value = crop_value.where(
                    (lon >= lonmin) & (lon <= lonmax), drop=True
                )
            else:
                if self.mode == "wrf":
                    crop_value = crop_value[:, :, lon_values]
                elif self.mode == "cmaq":
                    crop_value = crop_value.isel(y=lon_values)
                crop_value = crop_value.where(
                    (lat >= latmin) & (lat <= latmax), drop=True
                )
        return crop_value

    def crop_data(self, varname="NO", time=0):
        """
        Crop the netCDF file based on latitude and longitude boundaries.

        Args:
            varname (str): Variable name.
            time (int): Time index.

        Returns:
            crop_variable (xarray.DataArray): Cropped variable data.
            crop_height (xarray.DataArray): Cropped height data.
        """
        lat, lon, height, dx, dy, v_var = self._get_mode(varname=varname, time=time)
        crop_variable = self._get_crop(lat, lon, height, dx, dy, v_var)
        crop_height = self._get_crop(lat, lon, height, dx, dy, height)
        return crop_variable, crop_height

    def _make_dict(self, varname="NO", time=0):
        """
        Create a dictionary with coordinates and height as keys and variable values.

        Args:
            varname (str): Variable name.
            time (int): Time index.

        Returns:
            dict_n (dict): Dictionary with coordinates and variable values.
        """

        var_, var_height = self.crop_data(varname=varname, time=time)
        if self.mode == "wrf":
            return {
                (float(v.XLAT), float(v.XLONG), float(v.values)): float(var_[i])
                for i, v in enumerate(var_height)
                if v.XLAT == var_[i].XLAT and v.XLONG == var_[i].XLONG
            }
        elif self.mode == "cmaq":
            return {
                (float(v.y), float(v.x), float(v.values)): float(var_[i])
                for i, v in enumerate(var_height)
                if v.y == var_[i].y and v.x == var_[i].x
            }

    def _get_ll(self, varname=None, time=None):
        """
        Calculate latitude and longitude corresponding to boundary coordinates.

        Args:
            varname (str): Variable name.
            time (int): Time index.

        Returns:
            lat (float): Latitude value.
            lon (float): Longitude value.
        """
        lat, lon, _, _, _, _ = self._get_mode(varname=varname, time=time)
        y = find_nearest(lat, self.df.latitude, coord="lat")
        x = find_nearest(lon, self.df.longitude, coord="lon")
        return lat[y, 0], lon[0, x]

    def var_dict(self, varname="NO", time=0):
        """
        Create a dictionary of boundary variable values.

        Args:
            varname (str): Variable name.
            time (int): Time index.

        Returns:
            new_val (list): List of boundary variable values.
        """

       wrf_var = self._make_dict(varname=varname, time=time)
        lat, lon = self._get_ll(varname=varname, time=time)
        df_copy = self.df.copy()
        df_copy["lat_wrf"] = lat
        df_copy["long_wrf"] = lon

        d_ = {}
        new_val = []
        for linha in df_copy.values:
            if tuple(linha) not in d_:
                d_[tuple(linha)] = search_tuple(wrf_var, linha)
            new_val.append(d_[tuple(linha)])
        return new_val


def _dstag(varName, stagger_dim):
    """
    Perform staggered interpolation for a variable.

    Args:
        varName (xarray.DataArray): Variable to be staggered.
        stagger_dim (int): Staggered dimension.

    Returns:
        varName_staggered (xarray.DataArray): Staggered variable.
    """
    var_shape = varName.shape
    num_dims = varName.ndim
    stagger_dim_size = var_shape[stagger_dim]

    # Dynamically building the range slices to create the appropriate
    # number of ':'s in the array accessor lists.
    # For example, for a 3D array, the calculation would be
    # result = .5 * (var[:,:,0:stagger_dim_size-2]
    #                    + var[:,:,1:stagger_dim_size-1])
    # for stagger_dim=2.  So, full slices would be used for dims 0 and 1, but
    # dim 2 needs the special slice.
    full_slice = slice(None)
    slice1 = slice(0, stagger_dim_size - 1, 1)
    slice2 = slice(1, stagger_dim_size, 1)

    # default to full slices
    dim_ranges_1 = [full_slice] * num_dims
    dim_ranges_2 = [full_slice] * num_dims

    # for the stagger dim, insert the appropriate slice range
    dim_ranges_1[stagger_dim] = slice1
    dim_ranges_2[stagger_dim] = slice2

    if varName.dims == ("bottom_top", "south_north", "west_east_stag"):
        varName = varName.rename({"west_east_stag": "west_east"})
    elif varName.dims == ("bottom_top", "south_north_stag", "west_east"):
        varName = varName.rename({"south_north_stag": "south_north"})
    elif varName.dims == ("bottom_top_stag", "south_north", "west_east"):
        varName = varName.rename({"bottom_top_stag": "bottom_top"})

    return 0.5 * (varName[tuple(dim_ranges_1)] + varName[tuple(dim_ranges_2)])


def find_nearest(array, values, coord=None):
    """
    Find the index of the nearest latitude or longitude value.

    Args:
        array (xarray.DataArray): Array of latitude or longitude values.
        values (float or list): Target latitude or longitude values.
        coord (str): Coordinate dimension ("lat" or "lon").

    Returns:
        nearest_indices (int or list): Index or list of indices.
    """
    if isinstance(values, (float, int)):
        if coord == "lon":
            return np.argmin(np.abs(array[0, :].values - values))
        if coord == "lat":
            return np.argmin(np.abs(array[:, 0].values - values))
    else:
        nearest = []
        for value in values:
            if coord == "lon":
                nearest.append(np.argmin(np.abs(array[0, :].values - value)))
            if coord == "lat":
                nearest.append(np.argmin(np.abs(array[:, 0].values - value)))
        return nearest


# def find_nearest(array, values, coord=None):
#     """
#     Get the index of nearest latitude or longitude value
#     """
#     if type(values) in [np.float64, np.int]:
#         if coord == "lon":
#             return np.argmin(np.abs(array[0, :].values - values))
#         if coord == "lat":
#             return np.argmin(np.abs(array[:, 0].values - values))
#     else:
#         nearest = []
#         for value in values:
#             if coord == "lon":
#                 nearest.append(np.argmin(np.abs(array[0, :].values - value)))
#             if coord == "lat":
#                 nearest.append(np.argmin(np.abs(array[:, 0].values - value)))
#         return nearest


def distance(val1, val2, val=None):
    """
    Calculate the great-circle distance between two points.

    Args:
        val1 (float): First value (latitude or longitude).
        val2 (float): Second value (latitude or longitude).
        val (str): Coordinate type ("lat" or "lon").

    Returns:
        distance (float): Great-circle distance.
    """
    R = 6371  # Earth's radius in kilometers
    if val == "lat":
        lat1, lat2 = val1, val2
        lon1, lon2 = 0, 0
    elif val == "lon":
        lat1, lat2 = 0, 0
        lon1, lon2 = val1, val2
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) ** 2 + math.sin(dLon / 2) ** 2 * math.cos(lat1) * math.cos(
        lat2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def search_tuple(dict_item, new_key):
    """
    Search for the nearest key in a dictionary based on Euclidean distance.

    Args:
        dict_item (dict): Dictionary to search.
        new_key (tuple): Target key.

    Returns:
        value (float): Value corresponding to the nearest key.
    """
    closest_key = None
    closest_distance = None
    for key in dict_item.keys():
        # calculate the Euclidean distance between the two tuples
        distance = sum((a - b) ** 2 for a, b in zip(key, new_key)) ** 0.5
        # update the closest key and shortest distance
        if closest_distance is None or distance < closest_distance:
            closest_key = key
            closest_distance = distance
    return dict_item.get(closest_key)


# def search_tuple(dict_item, new_key):
#     closest_key = None
#     closest_distance = None
#     for key in dict_item.keys():
#         # search the nearest value to key
#         distance = sum(abs(a - b) for a, b in zip(key, new_key))
#         #  update the nearest key and shortest distance
#         if closest_distance is None or distance < closest_distance:
#             closest_key = key
#             closest_distance = distance
#     return dict_item.get(closest_key)
