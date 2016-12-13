import os
import sys
import subprocess
from time import sleep
from collections import defaultdict

from gpkit.tests.from_paths import run


def test_repos(repos=["d8", "gas_solar_trade"]):
    gpkit_models = False
    for repo in repos:
        git_clone(repo, branch="test_prep")
        if os.path.isfile("repo"+os.sep+"setup.py"):
            pip_install(repo, local=True)
        os.chdir(repo)
        settings = get_settings()

        # install gpkit-models
        if "gpkit-models branch" in settings:
            branch = settings["gpkit-models branch"]
            os.chdir("..")
            if not os.path.isdir("gpkit-models"):
                git_clone("gpkit-models", branch=branch)
                pip_install("gpkit-models", local=True)
            else:
                os.chdir("gpkit-models")
                call_and_retry(["git", "fetch", "--depth", "1", "origin",
                                branch])
                subprocess.call(["git", "checkout", "FETCH_HEAD"])
                os.chdir("..")
            os.chdir(repo)

        # install other dependencies
        if settings["pip install"]:
            for package in settings["pip install"].split(","):
                package = package.strip()
                pip_install(package)

        # TODO: xmloutput=True
        testpy = ("from gpkit.tests.from_paths import run;"
                  "run(skipsolvers=%s)" % repr(settings["skipsolvers"]))
        subprocess.call(["python", "-c", testpy])
        os.chdir("..")


def get_settings():
    settings = defaultdict(str)
    if os.path.isfile("TESTCONFIG"):
        with open("TESTCONFIG", "r") as f:
            for line in f:
                if len(line.strip().split(" : ")) > 1:
                    key, value = line.strip().split(" : ")
                    settings[key] = value
    return settings


def git_clone(repo, branch):
    cmd = ["git", "clone", "--depth", "1"]
    cmd += ["-b", branch]
    cmd += ["https://github.com/hoburg/%s.git" % repo]
    call_and_retry(cmd)


def pip_install(package, local=False):
    if sys.platform == "win32":
        cmd = ["pip"]
    else:
        cmd = ["python", os.environ["PIP"]]
    subprocess.call(cmd + ["uninstall", package])
    cmd += ["install"]
    if local:
        cmd += ["-e"]
    else:
        cmd += ["--upgrade"]
    cmd += [package]
    call_and_retry(cmd)


def call_and_retry(cmd, max_iterations=5, delay=5):
    iterations = 0
    return_code = None
    print "calling", cmd
    while return_code != 0 and iterations < max_iterations:
        iterations += 1
        print "  attempt", iterations
        return_code = subprocess.call(cmd)
        sleep(delay)
    return return_code
