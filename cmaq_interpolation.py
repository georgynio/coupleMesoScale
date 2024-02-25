import os
import pandas as pd
import numpy as np
from main_of import bdy_value

def process_boundary_file(file_name, nc, time, path, initial_time, metcro3d=None):
    print(f"Working at ************ {file_name} *********")
    f_bound = os.path.join(path, file_name)
    bdy = pd.read_csv(f_bound, skiprows=1, names=["latitude", "longitude", "altura"], sep=";")

    c_path = "constant/boundaryData/"
    val_wrf = bdy_value(nc=nc, mode="cmaq", df=f_bound, metcro3d=metcro3d)
    val_no = np.asarray(val_wrf.var_dict(varname="NO", time=time))/1.23   # NO  -> ppb/1.23 = ug/m3
    val_no2 = np.asarray(val_wrf.var_dict(varname="NO2", time=time))/1.88 # NO2 -> ppb/1.88 = ug/m3

    tempo =  time - initial_time
    # save each boundary file at /constant/boundaryData/bound_file
    output_folder = os.path.join(c_path, f"{file_name[6:]}/{tempo*3600}")
    
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, "tracer"), "w", encoding="utf-8") as f_nox:
        f_nox.write(f"{bdy.shape[0]} \n")
        f_nox.write("(\n")
        for index, row in enumerate(val_no):
            nox = row + val_no2[index]
            # the position depends on the building main axis
            f_nox.write(f"{float(nox)}\n")
        f_nox.write(")")



#path = os.getcwd()
path = "/home/yossimar/Documentos/of-3.5m/"
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Lista de nombres a procesar
names_to_process = ['bound_north', 'bound_south', 'bound_east', 'bound_west']


#print(files)
ncfls = [a for a in files if "CCTM_ACONC" in a]
nc = os.path.join(path, ncfls[0])

initial = 10*24
end = 31*24
# Procesar cada nombre de archivo
for time in range(initial, end, 1):
    for name in names_to_process:
        if name in files:
            process_boundary_file(name, nc, time, path, initial, metcro3d=f'{path}/METCRO3D_d03.nc')

    print(f"*-*-*-*- End {time} -*-*-*-*")
    print()
