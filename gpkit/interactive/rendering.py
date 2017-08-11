"""Rendering methods"""


def saveDesignParametersToCSM(filepath,paramdict):
        contents = ""
        if filepath != "":
            with open(filepath, 'r') as file:
                contents=file.read()
        lines = contents.splitlines()
        for i,line in enumerate(lines):
            if line[0:7] == 'despmtr':
                key = line[10:25].strip()
                if key in paramdict:
                    line = line[:25] + str(paramdict[key])
                    lines[i] = line
        output = ""
        for line in lines:
            output = output + line + '\n'
        contents = output
        with open(filepath, "w") as file:
            file.write(contents)
            print 'successfully updated ESP file'
