"Implements tests for all external repositories."
import os
import sys
import subprocess
from time import sleep
from collections import defaultdict


def test_repo(repo=".", xmloutput=False):
    """Test repository.

    If no repo name given, runs in current directory.
    Otherwise, assumes is in directory above the repo
    with a shared gplibrary repository.
    """
    os.chdir(repo)
    settings = get_settings()
    print
    print "SETTINGS"
    print settings
    print

    if repo == "." and not os.path.isdir("gpkitmodels"):
        git_clone("gplibrary")
        pip_install("gplibrary", local=True)

    # install dependencies other than gplibrary
    if settings["pip install"]:
        for package in settings["pip install"].split(","):
            package = package.strip()
            pip_install(package)
    if os.path.isfile("setup.py"):
        pip_install(".")

    skipsolvers = None
    if "skipsolvers" in settings:
        skipsolvers = [s.strip() for s in settings["skipsolvers"].split(",")]

    testpy = ("from gpkit.tests.from_paths import run;"
              "run(xmloutput=%s, skipsolvers=%s)"
              % (xmloutput, skipsolvers))
    subprocess.call(["python", "-c", testpy])
    if repo != ".":
        os.chdir("..")


def test_repos(repos=None, xmloutput=False, ingpkitmodels=False):
    """Get the list of external repos to test, and test.

    Arguments
    ---------
    xmloutput : bool
        True if the tests should produce xml reports

    ingpkitmodels : bool
        False if you're in the gpkitmodels directory that should be considered
        as the default. (overriden by repo-specific branch specifications)
    """
    if not ingpkitmodels:
        git_clone("gplibrary")
        repos_list_filename = "gplibrary"+os.sep+"EXTERNALTESTS"
        pip_install("gplibrary", local=True)
    else:
        print "USING LOCAL DIRECTORY AS GPKITMODELS DIRECTORY"
        repos_list_filename = "EXTERNALTESTS"
        pip_install(".", local=True)
    repos = [line.strip() for line in open(repos_list_filename, "r")]
    for repo in repos:
        git_clone(repo)
        test_repo(repo, xmloutput)


def get_settings():
    "Gets settings from a TESTCONFIG file"
    settings = defaultdict(str)
    if os.path.isfile("TESTCONFIG"):
        with open("TESTCONFIG", "r") as f:
            for line in f:
                if len(line.strip().split(" : ")) > 1:
                    key, value = line.strip().split(" : ")
                    settings[key] = value
    return settings


def git_clone(repo, branch="master"):
    "Tries several times to clone a given repository"
    cmd = ["git", "clone", "--depth", "1"]
    cmd += ["-b", branch]
    cmd += ["https://github.com/convexengineering/%s.git" % repo]
    call_and_retry(cmd)


def pip_install(package, local=False):
    "Tries several times to install a pip package"
    if sys.platform == "win32":
        cmd = ["pip"]
    else:
        cmd = ["python", os.environ["PIP"]]
    cmd += ["install"]
    if local:
        cmd += ["--no-cache-dir", "--no-deps", "-e"]
    cmd += [package]
    call_and_retry(cmd)


def call_and_retry(cmd, max_iterations=5, delay=5):
    "Tries max_iterations times (waiting d each time) to run a command"
    iterations = 0
    return_code = None
    print "calling", cmd
    while return_code != 0 and iterations < max_iterations:
        iterations += 1
        print "  attempt", iterations
        return_code = subprocess.call(cmd)
        sleep(delay)
    return return_code
