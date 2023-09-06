import os
import pandas as pd
import numpy as np

from main_of import bdy_value


# Obtém o diretório atual e a lista de arquivos nele
path = os.getcwd()
files = os.listdir()

# Define o tempo inicial
time = 0

# Define a altura máxima para filtrar os limites
altura_max = 350

# Filtra os arquivos contendo "wrfout" na lista de arquivos para obter os arquivos netcdf
ncfls = list(filter(lambda a: "wrfout" in a, files))
nc = os.path.join(path, ncfls[0])

# Lista de vias (nomes de ruas e outros) para excluir dos arquivos de limites
vias = [ "Av_Armando_Duarte_Rabelo",
"Av_Carlos_Martins",
"Av_Dante_Michelini_1",
"Av_Dante_Michelini_2",
"Av_Italina_Pereira_Mota",
"Av_Jose_Celso_Claudio_1",
"Av_Jose_Celso_Claudio_2",
"bairro",
"rodovia_norte_sul_1",
"rodovia_norte_sul_2",
"Rua_Carlos_Martins",
"Rua_Filogonio_Mota",
"Ruas_secundarias",
"ground",
"top"
]


# Cabeçalho do arquivo boundaryData
cabecalho = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    class       vectorField;
    object      points;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"""


cabecalho_2 = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\    /   O peration     | Version:                                        |
|   \\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
\n\n"""

# Lista para armazenar os dados de limites
bounds = []
for tin, time in enumerate(range(0, 5 * 24 * 3600, 3600)):
    # Itera sobre os arquivos no diretório
    for f in files:
        # Verifica se alguma via está presente no nome do arquivo
        if any(via in f for via in vias):
            pass
        # Caso contrário, verifica se o arquivo é um arquivo para ser processado
        elif "bound" in f:
            print(f"Working at ************ {f} *********")
            f_bound = os.path.join(path, f)

            # Lê o arquivo CSV contendo os dados de limite
            bdy = pd.read_csv(
                f_bound, skiprows=1, names=["latitude", "longitude", "altura"], sep=";"
            )
            # bounds.append(bdy)

            # Obtém os valores das variáveis U, V e W a partir do arquivo netcdf
            val_wrf = bdy_value(nc=nc, mode="wrf", df=f_bound)
            val_u = val_wrf.var_dict(varname="U", time=tin + 45)
            val_v = val_wrf.var_dict(varname="V", time=tin + 45)
            val_w = val_wrf.var_dict(varname="W", time=tin + 45)

            # Cria o diretório para armazenar os arquivos de limites
            c_path = "constant/boundaryData/"
            os.makedirs(f"{c_path+f[6:]}/{time}", exist_ok=True)

            # Escreve os valores das variáveis nos arquivos de processamento
            with open(f"{c_path+f[6:]}/{time}/U", "w", encoding="utf-8") as f_u:
                f_u.write(f"{bdy.shape[0]} \n")
                f_u.write("(\n")
                for row in zip(val_u, val_v, val_w):
                    f_u.write(f"({row[1]}  {row[0]}  {row[2]})\n")
                f_u.write(")")

            # Salva as condições iniciais para o tempo inicial
            if time == 0:
                with open(
                    "0.org/include/initialConditions", "w", encoding="utf-8"
                ) as f_0:
                    f_0.write(cabecalho_2)
                    f_0.write(
                        f"flowVelocity        ({np.mean(val_u)} {np.mean(val_v)} {np.mean(val_w)});\n"
                    )
                    f_0.write("pressure             0;\n")
                    f_0.write("turbulentKE          1.3;\n")
                    f_0.write("turbulentEpsilon     0.01;\n")
                    f_0.write(
                        "// ************************************************************************* //"
                    )
    print(f"*-*-*-*- End {time} -*-*-*-*")
    print()
