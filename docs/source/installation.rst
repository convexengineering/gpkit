.. _installation:

Installation
************

1. If you are on Mac or Windows, we recommend installing `Anaconda <https://www.anaconda.com>`_. Alternatively, `install pip and create a virtual environment <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_.
2. (optional) Install the MOSEK 9 solver with ``pip install Mosek``, then a license as described below
3. (optional) Install the MOSEK 8 solver as described below
4. Run ``pip install gpkit`` in the appropriate terminal or command prompt.
5. Open a Python prompt and run ``import gpkit`` to finish installation and run unit tests.

If you encounter any bugs please email ``gpkit@mit.edu``
or `raise a GitHub issue <http://github.com/convexengineering/gpkit/issues/new>`_.


Installing MOSEK 8
==================
GPkit interfaces with two off the shelf solvers: cvxopt, and MOSEK (versions 8 and 9).
Cvxopt is open source and installed by default; MOSEK requires a commercial licence or (free)
academic license. In MOSEK version 8 GPkit uses the command-line interface ``mskexpopt`` solver, while
in MOSEK 9 it uses the more active exponential-cone interface (and hence supports :ref:`migp`).

Mac OS X
  - If ``which gcc`` does not return anything, install the `Apple Command Line Tools <https://developer.apple.com/downloads/index.action?=command%20line%20tools>`_.
  - Download `MOSEK 8 <https://www.mosek.com/downloads/>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Mac <http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html>`_.
  - If applicable, request an `academic license file <https://www.mosek.com/products/academic-licenses/>`_ and put it in ``~/mosek/``

Linux
  - Download `MOSEK 8 <https://www.mosek.com/downloads/>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Linux <http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html>`_.
  - If applicable, request an `academic license file <https://www.mosek.com/products/academic-licenses/>`_ and put it in ``~/mosek/``

Windows
    - Make sure ``gcc`` is on your system path.
        - To do this, type ``gcc`` into a command prompt.
        - If you get ``executable not found``, then install the 64-bit version (x86_64 installer architecture dropdown option) with GCC version 6.4.0 or older of `mingw <http://sourceforge.net/projects/mingw-w64/>`_.
        - In an Anaconda command prompt (or equivalent), run ``cd C:\Program Files\mingw-w64\x86_64-6.4.0-posix-seh-rt_v5-rev0\`` (or whatever corresponds to the correct installation directory; note that if mingw is in ``Program Files (x86)`` instead of ``Program Files`` you've installed the 32-bit version by mistake)
        - Run ``mingw-w64`` to add it to your executable path. For step 3 of the install process you'll need to run ``pip install gpkit`` from this prompt.
    - Download `MOSEK 8 <https://www.mosek.com/downloads/>`_, then:
        - Follow `these steps for Windows <http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html>`_.
    - If applicable, request an `academic license file <https://www.mosek.com/products/academic-licenses/>`_ and put it in ``C:\Users\(your_username)\mosek\``

Debugging your installation
===========================

You may need to rebuild GPkit if any of the following occur:
  - You install MOSEK after installing GPkit
  - You see ``Could not load settings file.`` when importing GPkit, or
  - ``Could not load MOSEK library: ImportError('expopt.so not found.')``

To rebuild GPkit run ``python -c "from gpkit.build import rebuild; rebuild()"``.

If that doesn't solve your issue then try the following:
  - ``pip uninstall gpkit``
  - ``pip install --no-cache-dir --no-deps gpkit``
  - ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If any tests fail, please email ``gpkit@mit.edu`` or `raise a GitHub issue <http://github.com/convexengineering/gpkit/issues/new>`_.


Bleeding-edge installations
===========================

Active developers may wish to install the `latest GPkit <http://github.com/convexengineering/gpkit>`_ directly from Github. To do so,

  1. ``pip uninstall gpkit`` to uninstall your existing GPkit.
  2. ``git clone https://github.com/convexengineering/gpkit.git``
  3. ``pip install -e gpkit`` to install that directory as your environment-wide GPkit.
  4. ``cd ..; python -c "import gpkit.tests; gpkit.tests.run()"`` to test your installation from a non-local directory.
