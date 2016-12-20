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
    with a shared gpkit-models repository.
    """
    if os.path.isfile(repo+os.sep+"setup.py"):
        pip_install(repo, local=True)
    os.chdir(repo)
    settings = get_settings()
    print
    print "SETTINGS"
    print settings
    print

    # install gpkit-models
    if "gpkit-models branch" in settings:
        branch = settings["gpkit-models branch"]
        if repo == ".":
            git_clone("gpkit-models", branch=branch)
            pip_install("gpkit-models", local=True)
        else:
            os.chdir("..")
            os.chdir("gpkit-models")
            call_and_retry(["git", "fetch", "--depth", "1", "origin",
                            branch])
            subprocess.call(["git", "checkout", "FETCH_HEAD"])
            os.chdir("..")
            pip_install("gpkit-models", local=True)
            os.chdir(repo)

    # install other dependencies
    if settings["pip install"]:
        for package in settings["pip install"].split(","):
            package = package.strip()
            pip_install(package)

    skipsolvers = None
    if "skipsolvers" in settings:
        skipsolvers = [s.strip() for s in settings["skipsolvers"].split(",")]

    testpy = ("from gpkit.tests.from_paths import run;"
              "run(xmloutput=%s, skipsolvers=%s)"
              % (xmloutput, skipsolvers))
    subprocess.call(["python", "-c", testpy])
    if repo != ".":
        os.chdir("..")


def test_repos(repos=None, xmloutput=False):
    "Get the list of external repos to test, and test."
    git_clone("gpkit-models")
    repos_list_filename = "gpkit-models"+os.sep+"EXTERNALTESTS"
    repos = [line.strip() for line in open(repos_list_filename, "r")]
    for repo in repos:
        git_clone(repo)
        test_repo(repo, xmloutput=xmloutput)


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
    cmd += ["https://github.com/hoburg/%s.git" % repo]
    call_and_retry(cmd)


def pip_install(package, local=False):
    "Tries several times to install a pip package"
    if sys.platform == "win32":
        cmd = ["pip"]
    else:
        cmd = ["python", os.environ["PIP"]]
    if local:  # remove any other local packages of the same name...
        subprocess.call(cmd + ["uninstall", package])
    cmd += ["install"]
    if local:
        cmd += ["--no-cache-dir", "--no-deps", "-e"]
    else:
        cmd += ["--upgrade"]
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
