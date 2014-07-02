import os
import sys
import shutil
import subprocess


def pathjoin(*args):
    return os.sep.join(args)


def isfile(path):
    if os.path.isfile(path):
        print "#     Found %s" % path
        return True
    else:
        print "#     Could not find %s" % path
        return False


def replacedir(path):
    print "#     Replacing directory", path
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def call(cmd):
    print "#     Calling '%s'" % cmd
    print "##"
    print "### CALL BEGINS"
    retcode = subprocess.call(cmd, shell=True)
    print "### CALL ENDS"
    print "##"
    return retcode


def diff(filename, diff_dict):
    with open(filename, "r") as a:
        with open(filename+".new", "w") as b:
            for line_number, line in enumerate(a):
                if line[:-1] in diff_dict:
                    newline = diff_dict[line[:-1]]+"\n"
                    print ("#\n#     Change in"
                           " %s on line %i" % (filename, line_number + 1))
                    print "#     --", line[:-1][:70]
                    print "#     ++", newline[:70]
                    b.write(newline)
                else:
                    b.write(line)
    shutil.move(filename+".new", filename)


class SolverBackend(object):
    installed = False

    def __init__(self):
        print "# Looking for", self.name
        location = self.look()
        if location is not None:
            print "# Found %s %s" % (self.name, location)
            if not hasattr(self, 'build'):
                self.installed = True
            else:
                print "#\n# Building %s..." % self.name
                self.installed = self.build()
                status = "Done" if self.installed else "Failed"
                print "# %s building %s" % (status, self.name)
        else:
            print "# Did not find", self.name
        print


class Mosek_CLI(SolverBackend):
    name = "mosek_cli"

    def look(self):
        try:
            print "#   Trying to run mskexpopt..."
            if call("mskexpopt") == 1052:
                return "in system path"
        except Exception: pass
        return None


class CVXopt(SolverBackend):
    name = "cvxopt"

    def look(self):
        try:
            print "#   Trying to import cvxopt..."
            import cvxopt
            return "in Python path"
        except ImportError:
            return None


class Mosek(SolverBackend):
    name = "mosek"

    # Some of the expopt code leaks print statements onto stdout,
    # instead of handing them to the task's stream message system.
    # If you add a new leak, make sure to escape any backslashes!
    #   (that is, replace '\' with '\\')
    patches = {
        'dgopt.c': {
            # line 683:
            '          printf("Number of Hessian non-zeros: %d\\n",nlh[0]->numhesnz);':
            '          MSK_echotask(task,MSK_STREAM_MSG,"Number of Hessian non-zeros: %d\\n",nlh[0]->numhesnz);',
        },
        'expopt.c': {
            # line 1115:
            '    printf ("solsta = %d, prosta = %d\\n", (int)*solsta,(int)*prosta);':
            '    MSK_echotask(expopttask,MSK_STREAM_MSG, "solsta = %d, prosta = %d\\n", (int)*solsta,(int)*prosta);',
        }
    }

    def look(self):
        if sys.platform == "win32":
            try:
                self.dir = "C:\\Program Files\\Mosek"
                self.platform = "win64x86"
                self.libname = "mosek64_7_0.dll"
                self.version = sorted(os.listdir(self.dir))[-1]
            except WindowsError:
                try:
                    self.dir = "C:\\Program Files (x86)\\Mosek"
                    self.platform = "win32x86"
                    self.libname = "mosek7_0.dll"
                    self.version = sorted(os.listdir(self.dir))[-1]
                except WindowsError:
                    return None
        elif sys.platform == "darwin":
            try:
                self.dir = "/usr/local/mosek"
                self.platform = "osx64x86"
                self.libname = "libmosek64.7.0.dylib"
                self.version = sorted(os.listdir(self.dir))[-2]
            except OSError:
                return None
        else:
            print ("# Build script does not support"
                   " your platform (%s)" % sys.platform)
            return None

        self.tools_dir = pathjoin(self.dir, self.version, "tools")
        self.lib_dir = pathjoin(self.tools_dir, "platform", self.platform)
        self.h_path = pathjoin(self.lib_dir, "h", "mosek.h")
        self.lib_path = pathjoin(self.lib_dir, "bin", self.libname)
        if not isfile(self.h_path): return None
        if not isfile(self.lib_path): return None

        self.expopt_dir = pathjoin(self.tools_dir, "examples", "c")
        expopt_filenames = ["scopt-ext.c", "expopt.c", "dgopt.c",
                            "scopt-ext.h", "expopt.h", "dgopt.h"]
        self.expopt_files = [pathjoin(self.expopt_dir, fname)
                             for fname in expopt_filenames]
        self.expopt_files += [self.h_path]
        for expopt_file in self.expopt_files:
            if not isfile(expopt_file): return None

        return "version %s, installed to %s" % (self.version, self.dir)

    def build(self):
        lib_dir = replacedir(pathjoin("gpkit", "_mosek", "lib"))
        f = open(pathjoin(lib_dir, "__init__.py"), 'w')
        f.close()

        build_dir = replacedir(pathjoin("gpkit", "_mosek", "build"))
        print "#\n#   Copying expopt library files to", build_dir
        expopt_build_files = []
        for old_location in self.expopt_files:
            new_location = pathjoin(build_dir, os.path.basename(old_location))
            print "#     Copying %s" % old_location
            shutil.copyfile(old_location, new_location)
            if new_location[-2:] == ".c":
                expopt_build_files.append(new_location)

        print "#\n#   Applying expopt patches..."
        for filename, patch in self.patches.iteritems():
            diff(pathjoin(build_dir, filename), patch)

        print "#\n#   Building expopt library..."
        built_expopt_lib = call("gcc -fpic -shared" +
                                "    " + " ".join(expopt_build_files) +
                                '   "' + self.lib_path + '"' +
                                " -o " + pathjoin(lib_dir, "expopt.so"))
        if built_expopt_lib != 0: return False

        print "#\n#   Building Python bindings for expopt and Mosek..."
        mosek_h_path = pathjoin(lib_dir, "mosek_h.py")
        built_expopt_h = call("python ctypesgen.py -a" +
                              " -l "+pathjoin(lib_dir, "expopt.so") +
                              ' -l "' + self.lib_path.replace("\\", "/") + '"' +
                              ' -o "' + mosek_h_path.replace("\\", "/") + '"' +
                              " -o "+pathjoin(lib_dir, "expopt_h.py") +
                              "    "+pathjoin(build_dir, "expopt.h"))

        if built_expopt_h != 0: return False

        return True

solvers = [CVXopt(), Mosek_CLI(), Mosek()]
installed_solvers = [solver.name
                     for solver in solvers
                     if solver.installed]
