/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2012 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    ofWRF

Description
    Write the boundary conditions to any time.
    To use this feature, we shouldn't forget this was created for linear interpolation.
    We prefer to use this feature after the blockMesh because if we have complex
    geometry the process it takes too long.

\*------------i---------------------------------------------------------------*/

#include <fvCFD.H>
#include <typeinfo>
#include <vector>
#include <string.h>
#include <sys/stat.h> // Librería para verificar la existencia de directorios

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //


bool directoryExists(const char *path)
{
    struct stat info;
    if (stat(path, &info) != 0)
        return false;
    else if (info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}

void createDirectory(const char *path)
{
#ifdef _WIN32
    _mkdir(path);
#else
    mkdir(path, 0777);
#endif
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    // read the namelist.inp file
    dictionary dict(IFstream("namelist.inp")());

    // geometrical position
    scalar X_min(readScalar(dict.lookup("Xmin")));
    scalar X_max(readScalar(dict.lookup("Xmax")));
    scalar Y_min(readScalar(dict.lookup("Ymin")));
    scalar Y_max(readScalar(dict.lookup("Ymax")));
    scalar Z_min(readScalar(dict.lookup("Zmin")));
    scalar Z_max(readScalar(dict.lookup("Zmax")));
    
    // offset
    scalar offset_z(readScalar(dict.lookup("offsetZ")));
    
    // geographical position
    scalar Lat_min(readScalar(dict.lookup("Latmin")));
    scalar Lat_max(readScalar(dict.lookup("Latmax")));
    scalar Lon_min(readScalar(dict.lookup("Lonmin")));
    scalar Lon_max(readScalar(dict.lookup("Lonmax")));

    // Info << "The mesh has " << mesh.C().size() << "cells and " << mesh.Cf().size()
    //     << "internal faces in it..." << nl << nl << endl;

    Info << "Read and write the latitude, longitude and heigth of centres " << endl;
    Info << endl;
    std::ofstream outfile;
    forAll (mesh.boundaryMesh(), patchI){
        string name=mesh.boundary()[patchI].name();
        // Info << "Boundary " << mesh.boundaryMesh()[patchI].name() << endl;
        outfile.open("bound_" + name);
        outfile<<mesh.boundary()[patchI].Cf().size()<<"\n";
        Info << "Processing boundary: " << name << endl;
        for (label cellI = 0; cellI < mesh.boundary()[patchI].Cf().size(); cellI++)
            {
            scalar latitude = (mesh.boundary()[patchI].Cf()[cellI][1] - Y_min)*(Lat_max - Lat_min)/(Y_max - Y_min) + Lat_min;
            scalar longitude = (mesh.boundary()[patchI].Cf()[cellI][0] - X_min)*(Lon_max - Lon_min)/(X_max - X_min) + Lon_min;
            scalar height = mesh.boundary()[patchI].Cf()[cellI][2] + offset_z;
            outfile<<latitude<<";"<<longitude<<";"<<height<<"\n";
            }
        outfile.close();
        }

    Info << "Write the latitude, longitude and heigth of centres on" << endl;
    Info << "constant/boundaryData/<bound>/points" << endl;
    Info << endl;
    forAll (mesh.boundaryMesh(), patchI){
        string name=mesh.boundary()[patchI].name();
        // Info << "Boundary " << mesh.boundaryMesh()[patchI].name() << endl;
        string directoryPath = "constant/boundaryData/" + name;
        if (!directoryExists(directoryPath.c_str())) {
            createDirectory(directoryPath.c_str());
        }
        outfile.open("constant/boundaryData/" + name+ "/points");
        outfile<<mesh.boundary()[patchI].Cf().size()<<"\n";
        Info << "Processing boundary: " << name << endl;
        outfile << "(\n";
        for (label cellI = 0; cellI < mesh.boundary()[patchI].Cf().size(); cellI++)
            {
            scalar x_pos = mesh.boundary()[patchI].Cf()[cellI][0];
            scalar y_pos = mesh.boundary()[patchI].Cf()[cellI][1];
            scalar z_pos = mesh.boundary()[patchI].Cf()[cellI][2];
            outfile<<"("<<x_pos<<"\t"<<y_pos<<"\t"<<z_pos<<")"<<"\n";
            }
        outfile << ")";
        outfile.close();
        }


//    Info << "Start the Python processing...";
//    system("python3 wrf_interpolation.py");
    return 0;
}
