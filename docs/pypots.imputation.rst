pypots.imputation
=========================

pypots.imputation.lifts
------------------------------------

LIFTS trains imputation and in-sample one-step forecasting jointly. Inputs use
``[samples, steps, features]``, with NaN for missing entries, and must be
standardized using training-set statistics. ``predict()`` returns imputation
means and samples; ``forecast_in_sample()`` reuses the same fitted weights and
returns predictions at the input time positions, conditioning only on earlier
observations (the first position uses the learned initial distribution).

Both prediction methods generate 100 samples by default. The default batch
size is 64. If ``d_latent`` is omitted, it is ``n_features`` up to 50 features
and ``max(50, n_features // 2)`` above that threshold.

The supported reference training path has ``lambda_enc=0`` and runs all epochs.
It selects the minimum mean training joint loss from zero-based epoch
``epochs // 4`` onward: epochs 26 through 100 when training for 100 epochs.
Validation data do not select checkpoints or trigger early stopping.

Two boundary corrections are intentional differences from the pinned reference:
the filter always embeds the observation-mask channel, avoiding a branch on
batch-wide completeness; a one-dimensional latent transition bypasses its final
LayerNorm so it retains history dependence. Neither boundary occurs in the
partially observed, five-dimensional Case 1 reference comparison.

.. automodule:: pypots.imputation.lifts
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:


pypots.imputation.saits
------------------------------

.. automodule:: pypots.imputation.saits
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.helix
------------------------------

.. automodule:: pypots.imputation.helix
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.transformer
------------------------------------

.. automodule:: pypots.imputation.transformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tefn
------------------------------------

.. automodule:: pypots.imputation.tefn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.csai
------------------------------------

.. automodule:: pypots.imputation.csai
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.moment
------------------------------------

.. automodule:: pypots.imputation.moment
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tsicl
------------------------------------

.. automodule:: pypots.imputation.tsicl
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tslanet
------------------------------------

.. automodule:: pypots.imputation.tslanet
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.timellm
------------------------------------

.. automodule:: pypots.imputation.timellm
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.gpt4ts
------------------------------------

.. automodule:: pypots.imputation.gpt4ts
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.segrnn
------------------------------------

.. automodule:: pypots.imputation.segrnn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.fits
------------------------------------

.. automodule:: pypots.imputation.fits
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.timemixerpp
------------------------------------

.. automodule:: pypots.imputation.timemixerpp
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.timemixer
------------------------------------

.. automodule:: pypots.imputation.timemixer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.moderntcn
------------------------------------

.. automodule:: pypots.imputation.moderntcn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.imputeformer
------------------------------------

.. automodule:: pypots.imputation.imputeformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.totem
------------------------------------

.. automodule:: pypots.imputation.totem
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tkan
------------------------------------

.. automodule:: pypots.imputation.tkan
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.itransformer
------------------------------------

.. automodule:: pypots.imputation.itransformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.koopa
------------------------------------

.. automodule:: pypots.imputation.koopa
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.frets
------------------------------

.. automodule:: pypots.imputation.frets
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.crossformer
------------------------------

.. automodule:: pypots.imputation.crossformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.timesnet
------------------------------

.. automodule:: pypots.imputation.timesnet
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.patchtst
------------------------------

.. automodule:: pypots.imputation.patchtst
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.etsformer
------------------------------

.. automodule:: pypots.imputation.etsformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.micn
------------------------------
.. automodule:: pypots.imputation.micn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.dlinear
------------------------------

.. automodule:: pypots.imputation.dlinear
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tide
------------------------------

.. automodule:: pypots.imputation.tide
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.scinet
------------------------------

.. automodule:: pypots.imputation.scinet
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.nonstationary_transformer
---------------------------------------------

.. automodule:: pypots.imputation.nonstationary_transformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.film
------------------------------

.. automodule:: pypots.imputation.film
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.revin_scinet
------------------------------

.. automodule:: pypots.imputation.film
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.pyraformer
------------------------------

.. automodule:: pypots.imputation.pyraformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.fedformer
------------------------------

.. automodule:: pypots.imputation.fedformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.informer
------------------------------

.. automodule:: pypots.imputation.informer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.autoformer
------------------------------

.. automodule:: pypots.imputation.autoformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.csdi
------------------------------

.. automodule:: pypots.imputation.csdi
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.usgan
------------------------------

.. automodule:: pypots.imputation.usgan
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.stemgnn
------------------------------

.. automodule:: pypots.imputation.stemgnn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.reformer
------------------------------

.. automodule:: pypots.imputation.reformer
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.gpvae
------------------------------

.. automodule:: pypots.imputation.gpvae
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.brits
------------------------------

.. automodule:: pypots.imputation.brits
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.mrnn
------------------------------

.. automodule:: pypots.imputation.mrnn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.grud
------------------------------

.. automodule:: pypots.imputation.grud
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.tcn
------------------------------

.. automodule:: pypots.imputation.tcn
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.trmf
------------------------------

.. automodule:: pypots.imputation.trmf
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.lerp
-----------------------------

.. automodule:: pypots.imputation.lerp
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.locf
-----------------------------

.. automodule:: pypots.imputation.locf
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.median
-----------------------------

.. automodule:: pypots.imputation.median
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

pypots.imputation.mean
-----------------------------

.. automodule:: pypots.imputation.mean
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
