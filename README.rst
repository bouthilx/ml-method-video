Videos for the paper `Accounting for Variance in Machine Learning Benchmarks` @ MLSys2021
-----------------------------------------------------------------------------------------

This repository contains the code to create the short and oral videos of our paper at MLSys2021.

The animation are all generated with matplotlib.

It does not generate the audio-track unfortunately, this part was recorded. :)

Install
-------

For installation, simply install the requirements with ``pip``.


::

   $ pip install -r requirements.txt


The data used for some of the visualizations is available on dropbox at
``https://www.dropbox.com/sh/cwcjd91gw9wu6wv/AABg8h4Iq8ZQUuhoIkpDNNW-a?dl=0``.

The default path used by the script is ``~/Dropbox/Olympus-Data``. When running the script, 
make sure to pass ``--data-folder`` with correct path if you save the data elsewhere.


Execution
---------

For the short video

::

   $ python procedure.py --fps 60 --dpi 100 --parallel --concat

For the oral video

::

   $ python mlsys2021.py --fps 60 --dpi 100 --parallel --concat
