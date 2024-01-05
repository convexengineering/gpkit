"Finds solvers, sets gpkit settings, and builds gpkit"
import os
import sys
import shutil
import subprocess

LOGSTR = ""
settings = {}


def log(*args):
    "Print a line and append it to the log string."
    global LOGSTR  # pylint: disable=global-statement
    print(*args)
    LOGSTR += " ".join(args) + "\n"


def pathjoin(*args):
    "Join paths, collating multiple arguments."
    return os.sep.join(args)


def isfile(path):
    "Returns true if there's a file at $path. Logs."
    if os.path.isfile(path):
        log(f"#     Found {path}")
        return True
    log(f"#     Could not find {path}")
    return False


def replacedir(path):
    "Replaces directory at $path. Logs."
    log("#     Replacing directory", path)
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def call(cmd):
    "Calls subprocess. Logs."
    log(f"#     Calling '{cmd}'")
    log("##")
    log("### CALL BEGINS")
    retcode = subprocess.call(cmd, shell=True)
    log("### CALL ENDS")
    log("##")
    return retcode


def diff(filename, diff_dict):
    "Applies a simple diff to a file. Logs."
    with open(filename, "r", encoding="UTF-8") as a:
        with open(filename+".new", "w", encoding="UTF-8") as b:
            for line_number, line in enumerate(a):
                if line[:-1].strip() in diff_dict:
                    newline = diff_dict[line[:-1].strip()]+"\n"
                    log(f"#\n#     Change in {filename}"
                        f" on line {line_number + 1}")
                    log("#     --", line[:-1][:70])
                    log("#     ++", newline[:70])
                    b.write(newline)
                else:
                    b.write(line)
    shutil.move(filename+".new", filename)


class SolverBackend:
    "Inheritable class for finding solvers. Logs."
    name = look = None

    def __init__(self):
        log(f"\n# Looking for `{self.name}`")
        found_in = self.look()  # pylint: disable=not-callable
        if found_in:
            log(f"\nFound {self.name} {found_in}")
            self.installed = True
        else:
            log("# Did not find\n#", self.name)
            self.installed = False


class MosekCLI(SolverBackend):
    "MOSEK command line interface finder."
    name = "mosek_cli"

    def look(self):  # pylint: disable=too-many-return-statements
        "Looks in default install locations for a mosek before version 9."
        log("#   (A \"success\" is if mskexpopt complains that")
        log("#    we haven't specified a file for it to open.)")
        already_in_path = self.run()
        if already_in_path:
            return already_in_path

        log("# Looks like `mskexpopt` was not found in the default PATH,")
        log("#  so let's try locating that binary ourselves.")

        if sys.platform[:3] == "win":
            rootdir = "C:\\Program Files\\Mosek"
            mosek_platform = "win64x86"
        elif sys.platform[:6] == "darwin":
            rootdir = pathjoin(os.path.expanduser("~"), "mosek")
            mosek_platform = "osx64x86"
        elif sys.platform[:5] == "linux":
            rootdir = pathjoin(os.path.expanduser("~"), "mosek")
            mosek_platform = "linux64x86"
        else:
            return log(f"# Platform unsupported: {sys.platform}")

        if "MSKHOME" in os.environ:  # allow specification of root dir
            rootdir = os.environ["MSKHOME"]
            log(f"# Using MSKHOME environment variable (value {rootdir})"
                " instead of OS-default MOSEK home directory")
        if not os.path.isdir(rootdir):
            return log("# expected MOSEK directory not found: {rootdir}")

        possible_versions = [f for f in os.listdir(rootdir)
                             if len(f) == 1 and f < "9"]
        if not possible_versions:
            return log("# no version folders (e.g. '7', '8') found"
                       f" in mosek directory \"{rootdir}\"")
        version = sorted(possible_versions)[-1]
        tools_dir = pathjoin(rootdir, version, "tools")
        lib_dir = pathjoin(tools_dir, "platform", mosek_platform)
        bin_dir = pathjoin(lib_dir, "bin")
        settings["mosek_bin_dir"] = bin_dir
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + bin_dir
        log(f"#   Adding {bin_dir} to the PATH")

        return self.run("in " + bin_dir)

    def run(self, where="in the default PATH"):
        "Attempts to run mskexpopt."
        try:
            if call("mskexpopt") in (1052, 28):  # 28 for MacOSX
                return where
        except:   # pylint: disable=bare-except
            pass  # exception type varies by operating system
        return None


class CVXopt(SolverBackend):
    "CVXopt finder."
    name = "cvxopt"

    def look(self):
        "Attempts to import cvxopt."
        try:
            log("#   Trying to import cvxopt...")
            import cvxopt  # pylint: disable=unused-import
            return "in the default PYTHONPATH"
        except ImportError:
            pass


class MosekConif(SolverBackend):
    "MOSEK exponential cone solver finder."
    name = "mosek_conif"

    def look(self):
        "Attempts to import a mosek supporting exponential cones."
        try:
            log("#   Trying to import mosek...")
            import mosek
            if hasattr(mosek.conetype, "pexp"):
                return "in the default PYTHONPATH"
            return None
        except ImportError:
            pass

def build():
    "Builds GPkit"
    import gpkit
    log(f"# Building GPkit version {gpkit.__version__}")
    log("# Moving to the directory from which GPkit was imported.")
    start_dir = os.getcwd()
    os.chdir(gpkit.__path__[0])

    log("\nAttempting to find and build solvers:")
    solvers = [MosekCLI(), MosekConif(), CVXopt()]
    installed_solvers = [solver.name for solver in solvers if solver.installed]
    if not installed_solvers:
        log("Can't find any solvers!\n")
    if "GPKITSOLVERS" in os.environ:
        log(f"Replaced found solvers ({installed_solvers}) with environment "
            f"var GPKITSOLVERS ({os.environ['GPKITSOLVERS']})")
        settings["installed_solvers"] = os.environ["GPKITSOLVERS"]
    else:
        settings["installed_solvers"] = ", ".join(installed_solvers)
    log("\nFound the following solvers: " + settings["installed_solvers"])

    # Write settings
    envpath = "env"
    replacedir(envpath)
    with open(pathjoin(envpath, "settings"), "w", encoding="UTF-8") as f:
        for setting, value in sorted(settings.items()):
            f.write(f"{setting} : {value}\n")
    with open(pathjoin(envpath, "build.log"), "w", encoding="UTF-8") as f:
        f.write(LOGSTR)

    os.chdir(start_dir)

if __name__ == "__main__":
    build()
