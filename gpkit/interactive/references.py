"Code to make variable references plots"

import os
import shutil
import webbrowser
from collections import defaultdict


# pylint:disable=too-many-locals
def referencesplot(model, *, openimmediately=True):
    """Makes a references plot.

    1) Creates the JSON file for a d3 references plot
    2) Places it and the corresponding HTML file in the working directory
    3) (optionally) opens that HTML file up immediately in a web browser

    """
    imports = {}
    totalv_ss = defaultdict(dict)
    for constraint in model.flat():
        for varkey in constraint.vks:
            vlineage = varkey.lineagestr()
            clineage = constraint.lineagestr()
            if not vlineage:
                vlineage = "%s [%s]" % (varkey, varkey.unitstr())
            for lin in (clineage, vlineage):
                if lin not in imports:
                    imports[lin] = set()
            if vlineage != clineage:
                imports[clineage].add(vlineage)
                if constraint.v_ss:
                    totalv_ss[clineage] += constraint.v_ss

    def clean_lineage(lineage, clusterdepth=2):
        prelineage = ".".join(lineage.split(".")[:clusterdepth])
        last = "0".join(lineage.split(".")[clusterdepth:])
        return "model."+prelineage + "." + last

    lines = ['jsondata = [']
    for lineage, limports in imports.items():
        name, short = clean_lineage(lineage), lineage.split(".")[-1]
        limports = map(clean_lineage, limports)
        lines.append(
            '  {"name":"%s","fullname":"%s","shortname":"%s","imports":%s},'
            % (name, lineage, short, repr(list(limports)).replace("'", '"')))
    lines[-1] = lines[-1][:-1]
    lines.append("]")

    if totalv_ss:
        def get_total_senss(clineage, vlineage, normalize=False):
            v_ss = totalv_ss[clineage]
            num = sum(abs(ss) for vk, ss in v_ss.items()
                      if vk.lineagestr() == vlineage)
            if not normalize:
                return num
            return num/sum(abs(ss) for ss in v_ss.values())
        lines.append("globalsenss = {")
        for clineage, limports in imports.items():
            if not limports:
                continue
            limports = {vl: get_total_senss(clineage, vl) for vl in limports}
            lines.append('  "%s": %s,' %
                         (clineage, repr(limports).replace("'", '"')))
        lines[-1] = lines[-1][:-1]
        lines.append("}")
        lines.append("normalizedsenss = {")
        for clineage, limports in imports.items():
            if not limports:
                continue
            limports = {vl: get_total_senss(clineage, vl, normalize=True)
                        for vl in limports}
            lines.append('  "%s": %s,' %
                         (clineage, repr(limports).replace("'", '"')))
        lines[-1] = lines[-1][:-1]
        lines.append("}")

    with open("referencesplot.json", "w") as f:
        f.write("\n".join(lines))

    htmlfile = "referencesplot.html"
    if not os.path.isfile(htmlfile):
        shutil.copy(os.path.join(os.path.dirname(__file__), htmlfile), htmlfile)

    if openimmediately:
        webbrowser.open("file://" + os.path.join(os.getcwd(), htmlfile),
                        autoraise=True)
