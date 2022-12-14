# ======================================================================
#  AUTHOR:  Ben Rafferty, Purdue University
#  Copyright (c) 2010  Purdue Research Foundation
#
#  See the file "LICENSE.txt" for information on usage and
#  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
# ======================================================================

"""
contactmaps is a Python module for creating contact maps and distance maps
for protein molecules.  Two classes are defined: ContactMap and DistanceMap.
Each class's initializer requires a Bio.PDB.Model object as input, in
addition to other optional parameters.  See Biopython for more information
on the Bio.PDB.Model object.  The ability to generate two dimensional data
maps for proetin molecules is extensible, as both classes are derived from
the GenericMap class.  New types of data maps can be easily generated by 
creating a subclass of GenericMap, and overriding the calc_matrix method.

Python docstrings are present where appropriate, so please make use of 
python's builtin help function (ex: help(contactmaps.DistanceMap))

Two scripts are also provided: cmap.py and dmap.py.  These scripts are 
provided for convenience, and make use of the functionality given by the
python module. They allow contact maps and distance maps, respectively, to
be generated from a command line environment without needing to write any
Python code.

Biopython and matplotlib are required.

Usage examples:

    Creating distance map from command line:
        dmap.py 1UBQ.pdb -o distancemap.png

    Creating contact map from Python and display interactively:
        import contactmaps
        structure = contactmaps.get_structure("1UBQ.pdb")
        model = structure[0]
        map = contactmaps.ContactMap(model)
        map.show()

    Creating sequence of distance maps for all models in pdb file:
        import contactmaps
        structure = contactmaps.get_structure("trajectory.pdb")
        for i, model in enumerate(structure):
            m = i + 1 # Model numbers begin at 1
            map = contactmaps.DistanceMap(model)
            # Get maximum distance from first map and apply this limit to
            # all others.
            if i==0:
                lim = map.get_maximum()
            else:
                if lim:
                    map.saturate(lim)
            map.print_figure("distancemap%d.png" % m)

Copyright (c) 2010, Purdue Research Foundation
All rights reserved.

Developed by:  Ben Rafferty, Zachary Flohr
               Network for Computational Nanotechnology
               Purdue University, West Lafayette, Indiana

See the file LICENSE.txt for licensing information.

"""
#
from .utils import *
from .ContactMap import ContactMap
from .DistanceMap import DistanceMap
