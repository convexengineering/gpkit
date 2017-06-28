"Finds solvers, sets gpkit settings, and builds gpkit"
from __future__ import print_function

import os
import sys
import shutil
import subprocess
import glob

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
        log("#     Found %s" % path)
        return True
    else:
        log("#     Could not find %s" % path)
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
    log("#     Calling '%s'" % cmd)
    log("##")
    log("### CALL BEGINS")
    retcode = subprocess.call(cmd, shell=True)
    log("### CALL ENDS")
    log("##")
    return retcode


def diff(filename, diff_dict):
    "Applies a simple diff to a file. Logs."
    with open(filename, "r") as a:
        with open(filename+".new", "w") as b:
            for line_number, line in enumerate(a):
                if line[:-1].strip() in diff_dict:
                    newline = diff_dict[line[:-1].strip()]+"\n"
                    log("#\n#     Change in %s"
                        "on line %i" % (filename, line_number + 1))
                    log("#     --", line[:-1][:70])
                    log("#     ++", newline[:70])
                    b.write(newline)
                else:
                    b.write(line)
    shutil.move(filename+".new", filename)


class SolverBackend(object):
    "Inheritable class for finding solvers. Logs."
    installed = False
    name = None
    look = None
    build = None

    def __init__(self):
        log("# Looking for", self.name)
        location = self.look()  # pylint: disable=not-callable
        if location is not None:
            log("# Found %s %s" % (self.name, location))
            if not self.build:
                self.installed = True
            else:
                log("#\n# Building %s..." % self.name)
                self.installed = self.build()  # pylint: disable=not-callable
                status = "Done" if self.installed else "Failed"
                log("# %s building %s" % (status, self.name))
        else:
            log("# Did not find", self.name)


class MosekCLI(SolverBackend):
    "MOSEK command line interface finder."
    name = "mosek_cli"

    def look(self):
        "Attempts to run mskexpopt."
        try:
            log("#   Trying to run mskexpopt...")
            if call("mskexpopt") in (1052, 28):  # 28 for MacOSX
                return "in system path"
        except:  # pylint: disable=bare-except
            # exception type varies by operating system
            return


class CVXopt(SolverBackend):
    "CVXopt finder."
    name = "cvxopt"

    def look(self):
        "Attempts to import cvxopt."
        try:
            log("#   Trying to import cvxopt...")
            # Testing the import, so the variable is intentionally not used
            import cvxopt  # pylint: disable=unused-variable
            return "in Python path"
        except ImportError:
            return


class Mosek(SolverBackend):
    "MOSEK finder and builder."
    name = "mosek"

    # Some of the expopt code leaks log(statements onto stdout,)
    # instead of handing them to the task's stream message system.
    patches = {
        'dgopt.c': {
            # line 683:
            'printf("Number of Hessian non-zeros: %d\\n",nlh[0]->numhesnz);':
            'MSK_echotask(task,MSK_STREAM_MSG,"Number of Hessian non-zeros: %d\\n",nlh[0]->numhesnz);',  # pylint: disable=line-too-long
        },
        'expopt.c': {
            # line 1115:
            'printf ("solsta = %d, prosta = %d\\n", (int)*solsta,(int)*prosta);':  # pylint: disable=line-too-long
            'MSK_echotask(expopttask,MSK_STREAM_MSG, "solsta = %d, prosta = %d\\n", (int)*solsta,(int)*prosta);',  # pylint: disable=line-too-long
            """printf("Warning: The variable with index '%d' has only positive coefficients akj.\\n The problem is possibly ill-posed.\\n.\\n",i);""":  # pylint: disable=line-too-long
            """MSK_echotask(expopttask,MSK_STREAM_MSG, "Warning: The variable with index '%d' has only positive coefficients akj.\\n The problem is possibly ill-posed.\\n.\\n",i);""",  # pylint: disable=line-too-long
            """printf("Warning: The variable with index '%d' has only negative coefficients akj.\\n The problem is possibly ill-posed.\\n",i);""":  # pylint: disable=line-too-long
            """MSK_echotask(expopttask,MSK_STREAM_MSG, "Warning: The variable with index '%d' has only negative coefficients akj.\\n The problem is possibly ill-posed.\\n",i);""",  # pylint: disable=line-too-long
        }
    }

    def __init__(self):
        super(Mosek, self).__init__()
        self.expopt_files = None
        self.bin_dir = None
        self.flags = None
        self.lib_path = None

    def look(self):
        "Looks in default install locations for latest mosek version."
        if sys.platform == "win32":
            rootdir = "C:\\Program Files\\Mosek"
            mosek_platform = "win64x86"
            libpattern = "mosek64_?_?.dll"
            self.flags = "-Wl,--export-all-symbols,-R"
        elif sys.platform == "darwin":
            rootdir = pathjoin(os.path.expanduser("~"), "mosek")
            mosek_platform = "osx64x86"
            libpattern = "libmosek64.?.?.dylib"
            self.flags = "-Wl,-rpath"

        elif sys.platform == "linux2":
            rootdir = pathjoin(os.path.expanduser("~"), "mosek")
            mosek_platform = "linux64x86"
            libpattern = "libmosek64.so"
            self.flags = "-Wl,--export-dynamic,-R"

        else:
            log("# Build script does not support"
                " your platform (%s)" % sys.platform)
            return

        if not os.path.isdir(rootdir):
            return

        possible_versions = [f for f in os.listdir(rootdir) if len(f) == 1]
        version = sorted(possible_versions)[-1]
        tools_dir = pathjoin(rootdir, version, "tools")
        lib_dir = pathjoin(tools_dir, "platform", mosek_platform)
        h_path = pathjoin(lib_dir, "h", "mosek.h")
        self.bin_dir = pathjoin(lib_dir, "bin")
        try:
            self.lib_path = glob.glob(self.bin_dir+os.sep+libpattern)[0]
        except IndexError:  # mosek folder found, but mosek not installed
            return

        if not isfile(h_path) or not isfile(self.lib_path):
            return

        expopt_dir = pathjoin(tools_dir, "examples", "c")
        expopt_filenames = ["scopt-ext.c", "expopt.c", "dgopt.c",
                            "scopt-ext.h", "expopt.h", "dgopt.h"]
        self.expopt_files = [pathjoin(expopt_dir, fname)
                             for fname in expopt_filenames]
        self.expopt_files += [h_path]
        for expopt_file in self.expopt_files:
            if not isfile(expopt_file):
                return

        global settings  # pylint: disable=global-variable-not-assigned
        settings["mosek_bin_dir"] = self.bin_dir
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + self.bin_dir

        return "version %s, installed to %s" % (version, rootdir)

    def build(self):
        "Builds a dynamic library to GPKITBUILD or $HOME/.gpkit"
        try:
            # Testing the import, so the variable is intentionally not used
            import ctypesgencore  # pylint: disable=unused-variable
        except ImportError:
            log("## SKIPPING MOSEK INSTALL: CTYPESGENCORE WAS NOT FOUND")
            return

        lib_dir = replacedir(pathjoin("gpkit", "_mosek", "lib"))
        if "GPKITBUILD" in os.environ:
            solib_dir = replacedir(os.environ["GPKITBUILD"])
        else:
            solib_dir = replacedir(pathjoin(os.path.expanduser("~"), ".gpkit"))
        f = open(pathjoin(lib_dir, "__init__.py"), 'w')
        f.close()

        build_dir = replacedir(pathjoin("gpkit", "_mosek", "build"))
        log("#\n#   Copying expopt library files to", build_dir)
        expopt_build_files = []
        for old_location in self.expopt_files:
            new_location = pathjoin(build_dir, os.path.basename(old_location))
            log("#     Copying %s" % old_location)
            shutil.copyfile(old_location, new_location)
            if new_location[-2:] == ".c":
                expopt_build_files.append(new_location)

        log("#\n#   Applying expopt patches...")
        for filename, patch in self.patches.items():
            diff(pathjoin(build_dir, filename), patch)

        log("#\n#   Building expopt library...")
        built_expopt_lib = call("gcc -fpic -shared" +
                                ' %s "%s"' % (self.flags, self.bin_dir) +
                                "    " + " ".join(expopt_build_files) +
                                '   "' + self.lib_path + '"' +
                                " -o " + pathjoin(solib_dir, "expopt.so"))
        if sys.platform == "darwin":
            link_library = call("install_name_tool -change @loader_path/libmosek64.7.1.dylib "  # pylint: disable=line-too-long
                                + self.lib_path + " "
                                + pathjoin(solib_dir, "expopt.so"))
            if link_library != 0:
                return False

        if built_expopt_lib != 0:
            return False

        log("#\n#   Building Python bindings for expopt and Mosek...")
        # mosek_h_path = pathjoin(lib_dir, "mosek_h.py")
        built_expopt_h = call("python gpkit/modified_ctypesgen.py -a" +
                              " -l " + pathjoin(solib_dir, "expopt.so").replace("\\", "/") +   # pylint: disable=line-too-long
                              ' -l "' + self.lib_path.replace("\\", "/") + '"' +
                              # ' -o "' + mosek_h_path.replace("\\", "/") + '"'+
                              " -o "+pathjoin(lib_dir, "expopt_h.py") +
                              "    "+pathjoin(build_dir, "expopt.h"))

        if built_expopt_h != 0:
            return False

        return True

def rebuild():
    "Changes to the installed gpkit directory and runs build_gpkit()"
    import gpkit
    log("# Moving to the directory from which GPkit was imported.")
    os.chdir(gpkit.__path__[0])
    build_gpkit()

def build_gpkit():
    "Builds GPkit"
    if isfile("__init__.py"):
        #call("ls")
        log("#     Don't want to be in a folder with __init__.py, going up!")
        os.chdir("..")

    log("Started building gpkit...\n")

    log("Attempting to find and build solvers:\n")
    solvers = [Mosek(), MosekCLI(), CVXopt()]
    installed_solvers = [solver.name
                         for solver in solvers
                         if solver.installed]
    if not installed_solvers:
        log("Can't find any solvers!\n")

    log("...finished building gpkit.")

    if "GPKITSOLVERS" in os.environ:
        log("Replaced found solvers (%s) with environment var GPKITSOLVERS"
            " (%s)" % (installed_solvers, os.environ["GPKITSOLVERS"]))
        settings["installed_solvers"] = os.environ["GPKITSOLVERS"]
    else:
        settings["installed_solvers"] = ", ".join(installed_solvers)

    # Choose default solver
    log("\nFound the following solvers: " + settings["installed_solvers"])

    # Write settings
    envpath = pathjoin("gpkit", "env")
    replacedir(envpath)
    log("Replaced the directory gpkit/env\n")
    settingspath = envpath + os.sep + "settings"
    with open(settingspath, "w") as f:
        for setting, value in settings.items():
            f.write("%s : %s\n" % (setting, value))
        f.write("\n")

    with open("gpkit/build.log", "w") as f:
        f.write(LOGSTR)
