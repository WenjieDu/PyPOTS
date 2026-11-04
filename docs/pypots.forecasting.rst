pypots.forecasting
==========================

pypots.forecasting.lifts
------------------------------------

This task API provides an experimental free-running multi-step rollout of the
learned latent transition. ``n_steps`` is the history length and
``n_pred_steps`` is the future horizon. Prediction reads only the history.
For the original in-sample one-step forecasting definition, use
``pypots.imputation.LIFTS.forecast_in_sample()`` on the jointly fitted model.

Training follows the original history-only joint objective and training-loss
checkpoint policy. ``X_pred`` is checked for this task API's shape compatibility
but is not added to the training objective; the original method does not train
a multi-step rollout objective. The default latent-dimension rule, batch size,
and 100 prediction samples match the imputation task API.

Multi-step rollout accuracy must be evaluated separately. In the Case 1 MCAR
reference comparison (100 epochs, ``lambda_enc=0``, 100 prediction samples),
the 12-step rollout RMSE was 1.185 versus 0.577 for a Kalman oracle knowing the
true generative parameters. This extension should not be interpreted as a
reproduction of an original multi-step forecasting benchmark.

.. automodule:: pypots.forecasting.lifts
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


pypots.forecasting.transformer
------------------------------

.. automodule:: pypots.forecasting.transformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.mixlinear
------------------------------

.. automodule:: pypots.forecasting.mixlinear
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.timellm
------------------------------

.. automodule:: pypots.forecasting.timellm
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.gpt4ts
------------------------------

.. automodule:: pypots.forecasting.gpt4ts
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.moment
------------------------------

.. automodule:: pypots.forecasting.moment
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.tsicl
------------------------------

.. automodule:: pypots.forecasting.tsicl
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.tefn
------------------------------

.. automodule:: pypots.forecasting.tefn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.micn
------------------------------

.. automodule:: pypots.forecasting.micn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.dlinear
------------------------------

.. automodule:: pypots.forecasting.dlinear
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.segrnn
------------------------------

.. automodule:: pypots.forecasting.segrnn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.moderntcn
------------------------------

.. automodule:: pypots.forecasting.moderntcn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.fits
------------------------------

.. automodule:: pypots.forecasting.fits
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.film
------------------------------

.. automodule:: pypots.forecasting.film
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.timesnet
------------------------------

.. automodule:: pypots.forecasting.timesnet
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.timemixerpp
------------------------------

.. automodule:: pypots.forecasting.timemixerpp
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.timemixer
------------------------------

.. automodule:: pypots.forecasting.timemixer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.csdi
------------------------------

.. automodule:: pypots.forecasting.csdi
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.forecasting.bttf
------------------------------

.. automodule:: pypots.forecasting.bttf
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
