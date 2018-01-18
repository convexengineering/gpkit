"""Rendering methods"""

"""Save a set of variables to a .csm, for usage
with the Engineering Sketch Pad rendering software.

Arguments:

- filepath is a string with the full filepath to the .csm,
for example "design.csm" or "renders/version1.csm"

- sol is a GPkit Solution instance

- paramdict is a dictionary comprised of GPkit Variables
indexed by their design parameter name in ESP (String).
"""

def saveVariablesToCSM(filepath,sol,paramdict):
        contents = ""
        if filepath != "":
            with open(filepath, 'r') as file:
                contents=file.read()
        lines = contents.splitlines()
        for i,line in enumerate(lines):
            if line[0:7] == 'despmtr':
                key = line[10:25].strip()
                if key in paramdict:
                    valStr = str(float(sol(paramdict[key])))
                    line = line[:len(line)-len(valStr)] + valStr 
                    lines[i] = line
        output = ""
        for line in lines:
            output = output + line + '\n'
        contents = output
        with open(filepath, "w") as file:
            file.write(contents)
