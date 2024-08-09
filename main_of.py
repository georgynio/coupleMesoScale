import math
import numpy as np
import pandas as pd
import monetio as mio
import xarray as xr
import dask

from collections import defaultdict, Counter, namedtuple

Coordinate = namedtuple('Coordinate', ['latitude', 'longitude', 'height'])
earth_radius = 6370000

class bdy_value:
    def __init__(self, df=None, nc=None, mode=None, metcro3d=None, units=None):
        """Initialize values
        Args:
        - nc: path to netcdf file
        - df: path to dataframe file
        - mode: either 'wrf' or 'cmaq'
        - units: default ppmV, option ug/m3
        """
        assert mode in ["wrf", "cmaq"], "Invalid mode option. Choose either 'wrf' or 'cmaq'."

        self.nc = nc
        self.metcro3d = metcro3d
        self.mode = mode
        self.df = pd.read_csv(
            df,
            names=["latitude", "longitude", "height"],
            skiprows=1,
            header=None,
            sep=";",
        )
        self.coordinate_dict = self._make_coordinate_dict()
        self.units = units
    
    def _get_mode(self, varname=None, time=0):
        nc_file = xr.open_dataset(self.nc, chunks={"time": 1})
        if self.mode == "wrf":
            # Asegúrate de que todas las variables necesarias sean extraídas aquí
            if varname in ["U", "V", "W"]:
                v_var = _dstag(nc_file[varname][time], self._get_stagger_dim(varname))
            else:
                v_var = nc_file[varname][time]
            ph = _dstag(nc_file["PH"][time], 0)
            ph_b = _dstag(nc_file["PHB"][time], 0)
            hgt = nc_file["HGT"][0] # topografia
            height = (ph + ph_b) / 9.8 - hgt
            lat = nc_file["XLAT"][time]
            lon = nc_file["XLONG"][time]
            dx = nc_file.DX  # Asegúrate de que DX está definido en tu archivo nc
            dy = nc_file.DY  # Asegúrate de que DY está definido en tu archivo nc
        elif self.mode == "cmaq":
            grid = grid_from_dataset(nc_file, earth_radius=earth_radius)
            area_def = get_ioapi_pyresample_area_def(nc_file, grid)
            nc_file = nc_file.assign_attrs({"proj4_srs": grid})
            for i in nc_file.variables:
                nc_file[i] = nc_file[i].assign_attrs({"proj4_srs": grid})
                for j in nc_file[i].attrs:
                    nc_file[i].attrs[j] = nc_file[i].attrs[j].strip()
            nc_file = _get_latlon(nc_file, area_def)
            m3d = xr.open_dataset(self.metcro3d)
            if self.units == 'ug/m3':
                press = m3d['PRES'][time]
                temp = m3d['TA'][time]
                v_var = nc_file[varname][time] * press / (temp*8.314)
            else:
                v_var = nc_file[varname][time]
            height = m3d["ZH"][time]
            lat = nc_file.latitude
            lon = nc_file.longitude
            dx = nc_file.XCELL
            dy = nc_file.YCELL

        return lat, lon, height, dx, dy, v_var

    def _get_stagger_dim(self, varname):
        """
        Determine the stagger dimension for WRF variables based on their standard staggering.

        Args:
            varname (str): Variable name to check the staggering dimension.

        Returns:
            int: Stagger dimension index for the variable.
        """
        if varname in ["U", "V", "W"]:
            if varname == "U":
                return 2  # U is typically staggered in the west_east direction
            elif varname == "V":
                return 1  # V is typically staggered in the south_north direction
            elif varname == "W":
                return 0  # W is typically staggered in the bottom_top direction
        return None  # Return None if no staggering dimension is applicable
        
    def _make_coordinate_dict(self):
        return {(row.latitude, row.longitude): Coordinate(row.latitude, row.longitude, row.height) 
                for row in self.df.itertuples()}

    def count_coordinates(self):
        # Utilizar Counter para contar la ocurrencia de coordenadas
        coordinate_counts = Counter((row.latitude, row.longitude) for row in self.df.itertuples())
        return coordinate_counts

    def get_coordinates(self):
        # Utilizar namedtuple para devolver las coordenadas como una estructura de datos simple
        return [Coordinate(row.latitude, row.longitude, row.height) for row in self.df.itertuples()]

    def _get_crop(self, lat, lon, height, dx, dy, v_var):
        """
        Make a real crop of data from netcdf file
        """
        with xr.set_options(keep_attrs=True):
            crop_value = v_var.where(height.compute() <= self.df.height.max(), drop=True)
            latmin, latmax = self.df.latitude.min(), self.df.latitude.max()
            lonmin, lonmax = self.df.longitude.min(), self.df.longitude.max()
            lat_values = find_nearest(lat.compute(), [latmin, latmax], coord="lat")
            lon_values = find_nearest(lon.compute(), [lonmin, lonmax], coord="lon")
            
            if (distance(latmin, latmax, val="lat") > dy
                and distance(lonmin, lonmax, val="lon") > dx):
                eqs = 'difs'
                crop_value = crop_value.where(
                    (lat >= latmin) & (lat <= latmax) & (lon >= lonmin) & (lon <= lonmax),
                    drop=True
                )
            elif (distance(latmin, latmax, val="lat") < dy
                  and distance(lonmin, lonmax, val="lon") < dx):
                eqs = 'lats_lons'
                if self.mode == "wrf":
                    crop_value = crop_value[:, lat_values[0], lon_values[0]]
                elif self.mode == "cmaq":
                    crop_value = crop_value[:-1, lat_values[0], lon_values[0]]
            elif (distance(latmin, latmax, val="lat") < dy
                  and distance(lonmin, lonmax, val="lon") > dx):
                eqs = 'lats_eqs'
                crop_value = crop_value.where(
                    (lon.compute() >= lonmin) & (lon.compute() <= lonmax),
                    drop=True)
                if self.mode == "wrf":
                    crop_value = crop_value[:, lat_values[0], :]
                elif self.mode == "cmaq":
                    crop_value = crop_value[:-1, lat_values[0], :]
            elif (distance(latmin, latmax, val="lat") > dy
                  and distance(lonmin, lonmax, val="lon") < dx):
                eqs = 'lons_eqs'
                crop_value = crop_value.where(
                    (lat.compute() > latmin) & (lat.compute() < latmax),
                    drop=True)
                if self.mode == "wrf":
                    crop_value = crop_value.loc[:-1, :, lon_values[0]]
                elif self.mode == "cmaq":
                    crop_value = crop_value.loc[:-1, 1:, lon_values[0]]
        return crop_value, eqs

    def crop_data(self, varname="NO", time=0):
        """
        Crop the netcdf file from lat and lon boundaries to any time.
        Compare the maximum and minimum values into WRF file.

            time:    simulation time
            varname: default NO (nitrogen oxide)
            mode:    'wrf' or 'cmaq'

        Return:
            xarray: croped variable values
            xarray: croped hight values
        """
        lat, lon, height, dx, dy, v_var = self._get_mode(varname=varname, time=time)
        crop_variable, eqs = self._get_crop(lat, lon, height, dx, dy, v_var)
        crop_height, eqs = self._get_crop(lat, lon, height, dx, dy, height)
        return crop_variable, crop_height, eqs
    
    def _make_dict(self, varname="NO", time=0):
        """
        Make a dictionary with coordinates and height as key and the WRF variable value.
        
        Args:
            varname (str): WRF variable key to look up the highest height. Default is "NO".
            time (int): Time index. Default is 0.
        
        Returns:
            dict_n (dict): Dictionary with coordinates and height as key and the WRF variable value.
                        Example: {(-20.264686584472656, -40.2711181640625, 271.5928649902344): -0.4926649332046509}
        """
        
        varn_, var_height, _ = self.crop_data(varname=varname, time=time)
        
        coordinate_dict = defaultdict(list)
        for height, value in zip(var_height, varn_):
            for h, val in zip(height, value):
                # Determine the key based on the mode
                if self.mode == "wrf":
                    coordinate = (float(h.XLAT), float(h.XLONG), float(h.values))
                else:
                    coordinate = (float(h.latitude), float(h.longitude), float(h.values))
                coordinate_dict[coordinate].append(float(val))
        
        return coordinate_dict

    def _are_coordinates_equal(self, v1, v2):
        """
        Check if the coordinates of two points are equal.
        """
        if self.mode == "wrf":
            return v1.XLAT == v2.XLAT and v1.XLONG == v2.XLONG
        elif self.mode == "cmaq":
            return v1.y == v2.y and v1.x == v2.x

    def _get_ll(self, varname=None, time=None):
        """
        Calculate the lat and long from latitude and longitude bound_file
        latitude: netcdf latitude
        longitude: netcdf lontigute
        Return
        A list with the WRF lat and long corresponding to bound_file
        """
        lat, lon, _, _, _, _ = self._get_mode(varname=varname, time=time)
        y = find_nearest(lat.compute(), self.df.latitude, coord="lat")
        x = find_nearest(lon.compute(), self.df.longitude, coord="lon")
        return lat[y, 0], lon[0, x]

    def var_dict(self, varname="NO", time=0):
        """
        Make a pair (boundary, variable) dictionary
        Inputs:
        - varname: variable name
        - time: time index

        Return:
            Dictionary of each boundary variable value
        """
        _var = self._make_dict(varname=varname, time=time)
        lat, lon = self._get_ll(varname=varname, time=time)
        # Assuming self.df is a DataFrame and 'lat_wrf', 'long_wrf' are existing columns
        df_copy = self.df.copy()
        df_copy = df_copy.drop(['latitude', 'longitude'], axis=1)
        self.df['lat_wrf'] = lat
        self.df['long_wrf'] = lon

        d_ = {}
        new_val = []
        # search in each line of dataframe
        for linha in self.df.values:
            if tuple(linha) not in d_:
                d_[tuple(linha)] = search_tuple(_var, linha)
            new_val.append(d_[tuple(linha)])
        return new_val

def _dstag(varName, stagger_dim):
    var_shape = varName.shape
    num_dims = varName.ndim
    stagger_dim_size = var_shape[stagger_dim]

    # Dinámicamente construyendo los rangos de slices para crear el número adecuado de ':' en las listas de acceso de array.
    full_slice = slice(None)
    slice1 = slice(0, stagger_dim_size - 1, 1)
    slice2 = slice(1, stagger_dim_size, 1)

    # Por defecto a full slices
    dim_ranges_1 = [full_slice] * num_dims
    dim_ranges_2 = [full_slice] * num_dims

    # Para la dimensión stagger, inserta el rango de slice adecuado
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
    Obtener el índice del valor de latitud o longitud más cercano
    """
    if isinstance(values, (float, int)):
        if coord == "lon":
            return np.argmin(np.abs(array[0, :].values - values)) + 1
        if coord == "lat":
            return np.argmin(np.abs(array[:, 0].values - values)) + 1
    else:
        nearest = []
        for value in values:
            if coord == "lon":
                nearest.append(np.argmin(np.abs(array[0, :].values - value)) + 1)
            if coord == "lat":
                nearest.append(np.argmin(np.abs(array[:, 0].values - value)) + 1)
        return nearest

def distance(val1, val2, val=None):
    R = 6371  # Radio de la Tierra en kilómetros
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

    a = math.sin(dLat / 2) ** 2 + math.sin(dLon / 2) ** 2 * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c * 1000  # en metros

def search_tuple(dict_item, new_key):
    closest_key = None
    closest_distance = None
    for key in dict_item.keys():
        # calcula la distancia euclidiana entre las dos tuplas
        distance = sum((a - b) ** 2 for a, b in zip(key, new_key)) ** 0.5
        # actualiza la clave más cercana y la distancia más corta
        if closest_distance is None or distance < closest_distance:
            closest_key = key
            closest_distance = distance
    return dict_item.get(closest_key)

def _ioapi_grid_from_dataset(ds, earth_radius=6370000):
    """Obtiene la proyección IOAPI del archivo en proj4.

    Parameters
    ----------
    ds : type
        Descripción del parámetro `ds`.
    earth_radius : type
        Descripción del parámetro `earth_radius`.

    Returns
    -------
    type
        Descripción del objeto devuelto.
    """

    pargs = dict()
    pargs["lat_1"] = ds.P_ALP
    pargs["lat_2"] = ds.P_BET
    pargs["lat_0"] = ds.YCENT
    pargs["lon_0"] = ds.P_GAM
    pargs["center_lon"] = ds.XCENT
    pargs["x0"] = ds.XORIG
    pargs["y0"] = ds.YORIG
    pargs["r"] = earth_radius
    proj_id = ds.GDTYP
    if proj_id == 2:
        # Lambert
        p4 = (
            "+proj=lcc +lat_1={lat_1} +lat_2={lat_2} "
            "+lat_0={lat_0} +lon_0={lon_0} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +a={r} +b={r}"
        )
        p4 = p4.format(**pargs)
    elif proj_id == 4:
        # Polar stereo
        p4 = "+proj=stere +lat_ts={lat_1} +lon_0={lon_0} +lat_0=90.0" "+x_0=0 +y_0=0 +a={r} +b={r}"
        p4 = p4.format(**pargs)
    elif proj_id == 3:
        # Mercator
        p4 = (
            "+proj=merc +lat_ts={lat_1} " "+lon_0={center_lon} " "+x_0={x0} +y_0={y0} +a={r} +b={r}"
        )
        p4 = p4.format(**pargs)
    else:
        raise NotImplementedError("IOAPI proj not implemented yet: " "{}".format(proj_id))
    return p4

def grid_from_dataset(ds, earth_radius=6370000):
    """Resumen corto.

    Parameters
    ----------
    ds : type
        Descripción del parámetro `ds`.
    earth_radius : type
        Descripción del parámetro `earth_radius`.

    Returns
    -------
    type
        Descripción del objeto devuelto.
    """
    # quizás es un archivo IOAPI
    if hasattr(ds, "IOAPI_VERSION") or hasattr(ds, "P_ALP"):
        return _ioapi_grid_from_dataset(ds, earth_radius=earth_radius)

def get_ioapi_pyresample_area_def(ds, proj4_srs):
    from pyresample import geometry, utils

    y_size = ds.NROWS
    x_size = ds.NCOLS
    projection = utils.proj4_str_to_dict(proj4_srs)
    proj_id = "IOAPI_Dataset"
    description = "IOAPI area_def for pyresample"
    area_id = "Object_Grid"
    x_ll, y_ll = ds.XORIG + ds.XCELL * 0.5, ds.YORIG + ds.YCELL * 0.5
    x_ur, y_ur = (
        ds.XORIG + (ds.NCOLS * ds.XCELL) + 0.5 * ds.XCELL,
        ds.YORIG + (ds.YCELL * ds.NROWS) + 0.5 * ds.YCELL,
    )
    area_extent = (x_ll, y_ll, x_ur, y_ur)
    area_def = geometry.AreaDefinition(
        area_id, description, proj_id, projection, x_size, y_size, area_extent
    )
    return area_def

def _get_latlon(dset, area):
    """Calcula las latitudes y longitudes del pyreample.geometry.AreaDefinition

    Parameters
    ----------
    dset : xarray.Dataset
        Datos del modelo CMAQ

    Returns
    -------
    xarray.Dataset
        Datos del modelo CMAQ incluyendo la latitud y longitud en formato estándar.
    """
    lon, lat = area.get_lonlats()
    dset["longitude"] = xr.DataArray(lon[::-1, :], dims=["ROW", "COL"])
    dset["latitude"] = xr.DataArray(lat[::-1, :], dims=["ROW", "COL"])
    dset = dset.assign_coords(longitude=dset.longitude, latitude=dset.latitude)
    return dset

