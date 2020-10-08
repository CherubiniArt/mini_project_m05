.. vim: set fileencoding=utf-8 :

.. _house_prices_pred_hypotheses:

===========
 Hypotheses
===========

One of the most important steps of scientific method is to define which are the working
hypotheses of a specific project. In our case we have two hypotheses, the first one which
is kind of general while the second one is more precise:

    If the intrinsic value of a property is defined by different parameters such as the
    number of rooms or the total surface for example, then it must be possible to have
    an algorithm which can predict with good accuracy the price of a property.

    If some parameters are more meaningful than others to explain the price of a house,
    then by taking into account only these parameters the price of a house should be still
    predictable.

To verify the first hypothesis, what we will do is to give at the input of the machine learning
algorithms the 80 variables we have and analysis the results to see whether the hypothesis has
to be rejected or not. To verify the second hypothesis, we will first determine which are the
variables having the highest correlation to the property price and we will use only those variables
to predict the property price.

.. include:: links.rst