"Module for creating diagrams illustrating variables shared between submodels."
from itertools import combinations
from numpy import sin, cos, pi


# pylint: disable=too-many-locals
def linking_diagram(topmodel, subsystems, filename):
    """
    Method to create a latex diagram illustrating how variables
    are linked between a parent model and its submodels

    Inputs
    ------
    topmodel - a model object, the parent model of all submodels
    susbystems - a list of model objects, each a submodel of
    the topmodel
    filename - a string which is the name of the file latex output
    will be written to

    note: the following packages must be used in the latex file
    \\usepackage{tikz}
    \\usetikzlibrary{backgrounds}
    """
    keychain = {}
    for vk in topmodel.varkeys:
        model = vk.descr.get("models")
        if model == [topmodel.name] and vk not in topmodel.substitutions:
            v = vk.str_without(["models"])
            keychain[v] = []
            for ss in subsystems:
                if v in ss.varkeys:
                    keychain[v].append(type(ss).__name__)
            if len(keychain[v]) <= 1:
                del keychain[v]

    # Get all combinations of sub-models
    modellist = [type(ss).__name__ for ss in subsystems]
    modellistcombo = []
    for n in range(2, len(subsystems) + 1):
        modellistcombo += combinations(modellist, n)

    # Create a dictionary that has each combination (tuple) as a key
    zidane = {}
    for modelgroup in modellistcombo:
        zidane[modelgroup] = []

    # Fill this dictionary in with all the varkeys for which each combo applies
    for key in keychain:
        zidane[tuple(keychain[key])] += [key]

    # Get rid of any empty entries
    viera = {k: v for k, v in zidane.items() if v}

    with open(filename, "w") as outfile:

        outfile.write("\\begin{center}\n" +
                      "\\begin{tikzpicture}[thick]\n")

        # Create a circular ring of nodes, one for each model
        nodepos = {}
        i = 0
        I = len(modellist)
        R = 6
        for model in modellist:
            nodepos[model] = (R*sin(2*pi*i/I), R*cos(2*pi*i/I))
            outfile.write("\\node at {0}".format(str(nodepos[model])) +
                          "[circle, minimum size=3cm, fill=blue!20]" +
                          "({0}){{{0}}};\n".format(model))
            i = i + 1

        j = 0
        colours = ["red", "blue", "cyan ", "magenta", "brown", "gray",
                   "olive", "orange"]
        for key in viera:
            # Create a node for every group of variables
            vargroup = viera[key]
            varnodeposx = 0
            varnodeposy = 0
            for k in key:
                varnodeposx += nodepos[k][0]
                varnodeposy += nodepos[k][1]
            varnodeposx = varnodeposx/len(key)
            varnodeposy = varnodeposy/len(key)
            outfile.write("\\node at (%.2f,%.2f)" % (varnodeposx, varnodeposy) +
                          "[draw,rectangle, color=%s, align=left, fill=white]({"
                          % colours[j] +
                          " ".join(vargroup).replace("\\", "") + "}){$" +
                          "$\\\\$".join(vargroup) + "$};\n")
            # Create edges from vargroups to models they appear in
            for k in key:
                outfile.write("\\begin{pgfonlayer}{background}\n" +
                              "\\draw[color=%s] ({" % colours[j] +
                              " ".join(vargroup).replace("\\", "") +
                              "})--("+k+");\n" +
                              "\\end{pgfonlayer}\n")

            j += 1

        outfile.write("\\end{tikzpicture}\n" +
                      "\\end{center}")
