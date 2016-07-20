crstal
===============================

The Cuda Risk Simulation & Trading Analytic Library is a quantitative framework for Counterparty risk and xVA's using nvidia's cuda framework. An interactive widget set is also provided as a jupyter notebook extension.

Installation
------------

To install use pip:

    $ pip install crstal
    $ jupyter nbextension enable --py --sys-prefix crstal


For a development installation (requires npm),

    $ git clone https://github.com/sylam/crstal.git
    $ cd crstal
    $ pip install -e .
    $ jupyter nbextension install --py --symlink --user crstal
    $ jupyter nbextension enable --py --user crstal
