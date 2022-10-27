# ======================================================================
#  AUTHOR:  Ben Rafferty, Purdue University
#  Copyright (c) 2010  Purdue Research Foundation
#
#  See the file "LICENSE.txt" for information on usage and
#  redistribution of this file, and for a DISCLAIMER OF ALL WARRANTIES.
# ======================================================================

import Bio.PDB
from .utils import remove_hetero

class GenericMap():
    """
    Generic class for 2D protein data maps.  To define a specific map, create
    a child class and override calc_matrix(model).  This class cannot be
    instantiated directly, you must use a child class.  Any additional keyword
    arguments given to the constructor will be passed to calc_matrix.
    """
    def __init__(self, model, title="", xlabel="", ylabel="", colorbar=True, \
                 colorbarlabel="", contour=False, interpolation="nearest", \
                 cmap="jet", **kwargs):
        if not isinstance(model, Bio.PDB.Model.Model):
            raise TypeError("Input argument is not of class Bio.PDB.Model")
        if len(model) < 1:
            raise ValueError("No chains found in model.")
        remove_hetero(model)
        self.matrix = self.calc_matrix(model, **kwargs)
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.colorbar = colorbar
        self.colorbarlabel = colorbarlabel
        self.contour = contour
        self.interpolation = interpolation

    def calc_matrix(model, **kwargs):
        """
        Override this method in any child classes.  Must perform some
        calculation on the provided model, and return a matrix representing
        the data for the map.
        """
        raise NotImplementedError( \
              "GenericMap cannot be instantiated directly.")

    def get_data(self):
        """
        Returns the matrix containing the map data.
        """
        return self.matrix

    def __repr__(self):
        return repr(self.matrix)
