.. vim: set fileencoding=utf-8 :

.. _house_prices_pred_userguide:

===========
 User Guide
===========

This guide explains how to use this package and obtain results published in our
paper.  Results can be re-generated automatically by executing the following
command:
.. code-block:: sh

   (house_prices_env) $ python toolchain_all_params.py #to get the comparaison of the prediction power of the two algorithms(RF and Decision Tree) used
   (house_prices_env) $ python toolchain_all_params.py #to get results about the analysis on the relevant params
   
Here are the plotting of the results of the executions above:
-------------------------------------------------------------
.. image:: ../results/all_params_results.png