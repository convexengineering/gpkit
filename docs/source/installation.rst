Installation Instructions
*************************

If you encounter any bugs during installation, please email ``gpkit@mit.edu``,
or raise a `new issue <http://github.com/hoburg/gpkit/issues/new>`_.

Install dependencies
====================
GPkit's dependencies are as follows.
 - ``pip``,
 - ``numpy`` version 1.8 or newer,
 - ``scipy``,
 - ``pint``,

plus at least one solver (which we will install in a later step).

There are many ways to install these dependencies.
Below is one suggestion for how to do so.

Get ``pip``
-----------

**Mac OS X**

Run ``easy_install pip`` at a terminal window.

**Linux**

Use your package manager to install ``pip`,
e.g. ``sudo apt-get install python-pip`` for Ubuntu.

**Windows**

Do nothing at this step.

Get python packages
-------------------

**Mac OS X**

Run the following commands.
 - ``pip install pip --upgrade``
 - ``pip install numpy``
 - ``pip install scipy``
 - ``pip install pint``

**Linux**

 - Use your package manager to install ``numpy`` and ``scipy`,
e.g. ``sudo apt-get install python-numpy`` for Ubuntu.
 - Run ``pip install pint``

**Windows**

Do nothing at this step.

Install a GP solver
===================
GPkit interfaces with two off the shelf solvers: cvxopt, and mosek.
Cvxopt is open source; mosek requires a commercial licence or (free)
academic license. The solvers tend to have comparable performance on small
problems, but mosek appears faster on most larger problems we have tested.

At least one solver is required.

Unfortunately, on Windows, due to 32-bit vs 64 bit issues, we do not
currently know of a way to install both cvxopt and mosek simultaneously.
If you are a Windows user, you should pick one solver or the other.
For Windows 10, cvxopt does not appear to be an option.

Installing cvxopt
-----------------

**Mac OSX**

Run ``pip install cvxopt``.

**Linux**

Run ``pip install cvxopt``.

**Windows**

If you are using Windows 10, stop. Go to Installing mosek.

  - Install the Python 2.7 version of `Python (x,y) <https://python-xy.github.io/downloads.html>`_. (Note: Python (x,y) is 32-bit.)
      - Installing CVXOPT with Anaconda or another Python distribution can be difficult, which is why we reccomend Python (x,y).
      - Python (x,y) recommends removing any previous installations of Python before installation.
      - Be sure to click the cvxopt and pint check boxes under "Choose components" during installation.

Installing mosek
----------------

Note: if you do not have a paid license,
you will need an academic or trial license to proceed.

**Mac OS X**

  - If ``which gcc`` does not return anything, install ``XCode`` and the `Apple Command Line Tools <https://developer.apple.com/downloads/index.action?=command%20line%20tools>`_.
  - Install cytypesgen via ``pip install ctypesgen --pre`` (gpkit uses ctypesgen to interface with the MOSEK C bindings).
  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Mac <http://docs.mosek.com/7.0/toolsinstall/Mac_OS_X_installation.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``

**Linux**

  - Install cytypesgen via ``pip install ctypesgen --pre`` (gpkit uses ctypesgen to interface with the MOSEK C bindings).
  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Move the ``mosek`` folder to your home directory
      - Follow `these steps for Linux <http://docs.mosek.com/7.0/toolsinstall/Linux_UNIX_installation_instructions.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``~/mosek/``

**Windows**

  - If you have a 32-bit version of Windows, stop. Go to cvxopt instructions.
  - Install the 64-bit version of `Anaconda<http://www.continuum.io/downloads#_windows>`_.
  - Install cytypesgen via ``pip install ctypesgen --pre`` (gpkit uses ctypesgen to interface with the MOSEK C bindings).
  - Download `MOSEK <http://mosek.com/resources/downloads>`_, then:
      - Follow `these steps for Windows <http://docs.mosek.com/7.0/toolsinstall/Windows_installation.html>`_.
      - Request an `academic license file <http://license.mosek.com/academic>`_ and put it in ``C:\Users\(your_username)\mosek\``
      - Make sure ``gcc`` is on your system path.
        To do this, type ``gcc`` into a command prompt.
        If you get ``executable not found``, then install the 
        64-bit version of `mingw<http://sourceforge.net/projects/mingw-w64/>`_.     - Make sure the ``mingw`` bin directory is on your system path (you may have to add it manually).


Install GPkit
=============
  - Run ``pip install gpkit`` at the command line.
  - Run ``pip install ipywidgets`` for interactive control of models (recommended)
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - *Optional:* to install gpkit into an isolated python environment, install virtualenv, run ``virtualenv $DESTINATION_DIR`` then activate it with ``source $DESTINATION_DIR/bin/activate``.

Debugging installation
======================

You may need to rebuild GPkit if any of the following occur:
  - You install a new solver (mosek or cvxopt) after installing GPkit
  - You delete the ``.gpkit`` folder from your home directory
  - You see a ``Could not load settings file.`` message
  - You see a ``Could not load MOSEK library: ImportError('$HOME/.gpkit/expopt.so not found.`` error.
To rebuild GPkit, do the following:
  - Run ``pip uninstall gpkit``
  - Run ``pip install --no-cache-dir --no-deps gpkit``
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"``
  - If any tests fail, email ``gpkit@mit.edu``.


Updating GPkit between releases
===============================

Active developers may wish to install the `latest GPkit <http://github.com/hoburg/gpkit>`_ directly from the source code on Github. To do so,

  - Run ``pip uninstall gpkit`` to uninstall your existing GPkit.
  - Run ``git clone https://github.com/hoburg/gpkit.git`` to clone the GPkit repository, or ``cd gpkit; git pull origin master; cd ..`` to update your existing repository.
  - Run ``pip install -e gpkit`` to reinstall GPkit.
  - Run ``python -c "import gpkit.tests; gpkit.tests.run()"`` to test your installation.
