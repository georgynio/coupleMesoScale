import os
import numpy as np
import pandas as pd
import geopandas as gpd

from main_of import bdy_value

path = os.getcwd()
files = os.listdir()

time = 0

altura_max = 350


ncfls = list(filter(lambda a: "CCTM_ACONC" in a, files))
nc = os.path.join(path, ncfls[0])

cabecalho_NO = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:                                        |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      NO;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

cabecalho_NO2 = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:                                        |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      NO2;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""


cabecalho_NOX = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:                                        |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      NOX;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""

bounds = []
for tin, time in enumerate(range(0, 5 * 24 * 3600, 3600)):
    for f in files:
        if "ground" in f or "top" in f:
            pass
        elif "bound" in f:
            print(f'\n Work in boundary --> {f}')
            f_bound = os.path.join(path, f)
            # read point file from boundaryData
            bdy = pd.read_csv(
                f_bound, skiprows=1, names=["latitude", "longitude", "altura"], sep=";"
            )
            # bounds.append(bdy)
            c_path = "constant/boundaryData/"
            val_cmaq = bdy_value(nc=nc, mode='cmaq', df=f_bound)
            val_no = val_cmaq.var_dict(varname='NO', time=tin)
            val_no2 = val_cmaq.var_dict(varname='NO2', time=tin)
            # convert ppmv to ug/m3
            val_nox = val_no + val_no2

            # save each boundary file at /constant/boundaryData/boud_file
            os.makedirs(f"{c_path+f[6:]}/{time}", exist_ok=True)
            with open(f"{c_path+f[6:]}/{time}/tracer", "w", encoding="utf-8") as f_nox:
                f_nox.write(f"{bdy.shape[0]} \n")
                f_nox.write("(\n")
                for index, row in enumerate(val_no):
                    #print(row)
                    nox = row + val_no2[index]
                    # the position depends on the building main axis
                    f_nox.write(f"{nox}\n")
                f_nox.write(")")

            # # save each initial condition
            # if time==0:
            #     with open('0.org/include/initialConditions', 'w', encoding='utf-8') as f_0:
            #         f_0.write(cabecalho_2)
            #         f_0.write(f'flowVelocity        ({np.mean(val_u)} {np.mean(val_v)} {np.mean(val_w)});\n')
            #         f_0.write('pressure             0;\n')
            #         f_0.write('turbulentKE          1.3;\n')
            #         f_0.write('turbulentEpsilon     0.01;\n')
            #         f_0.write('// ************************************************************************* //')
    print(f"\n*-*-*-*- End {time} -*-*-*-*\n")
    print()