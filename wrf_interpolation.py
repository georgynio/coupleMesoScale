import os
import pandas as pd
import numpy as np

from main_of import bdy_value


def process_boundary_file(file_name, nc, time, initial_time, path):
    print(f"Working at ************ {file_name} *********")
    f_bound = os.path.join(path, file_name)
    bdy = pd.read_csv(f_bound, skiprows=1, names=["latitude", "longitude", "altura"], sep=";")

    c_path = "constant/boundaryData/"
    val_wrf = bdy_value(nc=nc, mode="wrf", df=f_bound)
    val_u = val_wrf.var_dict(varname="U", time=time)
    val_v = val_wrf.var_dict(varname="V", time=time)
    val_w = val_wrf.var_dict(varname="W", time=time)
    #print(val_u)
    tempo = time - initial_time
    # save each boundary file at /constant/boundaryData/boud_file
    output_dir = f"{c_path+file_name[6:]}/{tempo*3600}"
    os.makedirs(output_dir, exist_ok=True)

    val_u = [float(val[0]) for val in val_u]
    val_v = [float(val[0]) for val in val_v]
    val_w = [float(val[0]) for val in val_w]

    with open(os.path.join(output_dir, "U"), "w", encoding="utf-8") as f_u:
        f_u.write(f"{bdy.shape[0]}\n(\n")
        for i, v in enumerate(zip(val_u, val_v, val_w)):
            f_u.write(f'({float(v[0])} {float(v[1])} {float(v[2])})\n')            
        f_u.write(")")


# Directorio actual
path = os.getcwd()

# Lista de archivos en el directorio
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# Lista de nombres a procesar
#names_to_process = ["bound_Av_Armando_Duarte_Rabelo", "bound_Av_Carlos_Martins", "bound_rodovia_norte_sul_1", "bound_Av_Dante_Michelini_1", "bound_rodovia_norte_sul_2", "bound_Av_Dante_Michelini_2", "bound_Rua_Carlos_Martins", "bound_Av_Italina_Pereira_Mota", "bound_Rua_Filogonio_Mota", "bound_Av_Jose_Celso_Claudio_1", "bound_Ruas_secundarias", "bound_Av_Jose_Celso_Claudio_2", "bound_bairro", "bound_ground"]
names_to_process = ['bound_north', 'bound_south', 'bound_east', 'bound_west']

# Lista de archivos que contienen "wrfout"
ncfls = list(filter(lambda a: "wrfout" in a, files))
nc = os.path.join(path, ncfls[0])

initial = 0*24
end = 33*24
# Procesar cada nombre de archivo
for time in range(initial, end, 1):
    for name in names_to_process:
        if name in files:
            process_boundary_file(name, nc, time, initial, path)

    print(f"*-*-*-*- End {time} -*-*-*-*")
    print()
