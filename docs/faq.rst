Frequently Asked Questions
==========================

Inclusion Criteria
^^^^^^^^^^^^^^^^^^
Similarly to `scikit-learn <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_,
PyPOTS mainly considers well-established models/algorithms for inclusion. A rule of thumb is the paper should be
published for at least 1 year, have 20+ citations, and the usefulness to our users can be claimed.

**However**, we encourage the authors of proposed new models to share and add your implementations into PyPOTS
to help boost research accessibility and reproducibility in the field of POTS modeling.
Note this exception only applies if you commit to the maintenance of your model for at least two years.


Join PyPOTS
^^^^^^^^^^^
.. _becoming-a-volunteer:

Becoming a Volunteer Developer
""""""""""""""""""""""""""""""
To become a member of PyPOTS volunteer development team, you should

1. love open-source science and be active on GitHub;
2. be familiar with the PyPOTS codebase and have made at least one pull request merged into branch ``main`` of PyPOTS,
   which is not for fixing typos or improving the docs;
3. watch PyPOTS repository to receive the latest news from it;
4. join the `PyPOTS community on Slack <https://join.slack.com/t/pypots-org/shared_invite/zt-1gq6ufwsi-p0OZdW~e9UW_IA4_f1OfxA>`_
   and become a member of the channel ``#dev-team``. ``#dev-team`` currently is a public channel, and you don't need an invitation to join it;
5. commit to constantly maintain PyPOTS project and obey our development principles;

Once you obtain the role, you'll be listed as a member on the ``About Us`` pages of
`PyPOTS main site <https://pypots.com/about/>`_
and
`PyPOTS docs site <https://docs.pypots.com/en/latest/about_us.html>`_.

Becoming a Lead
"""""""""""""""
To become a lead at PyPOTS, surely you have to already be a volunteer developer first, i.e. you've met all requirements in the section :ref:`becoming-a-volunteer`.
Your research should be highly related to data mining/machine learning on POTS data, and
you need to prove that you're capable of proposing a research plan solely and conducting it.
You're willing to take developing PyPOTS as your responsibility and commit to constantly and regularly
contribute you time and ideas to PyPOTS things (including community culture construction,
code maintenance, current research implementation, new research planning).
The lead is a permanent role unless your research is no longer related to the field of modeling POTS or
you no longer want to get involved with affairs at PyPOTS.

If you believe you want to do this, you can drop an email with anything you want to tell and your CV attachment to
`team@pypots.com <mailto:team@pypots.com>`_. We will schedule a meeting for you and all other members at PyPOTS for further discussion.
This is absolutely not a so-called interview, please don't take it formal and we just would like to listen to your thoughts about the field of POTS ;-)


Our Development Principles
^^^^^^^^^^^^^^^^^^^^^^^^^^
1. `Do one thing and do it well (Unix philosophy) <https://en.wikipedia.org/wiki/Unix_philosophy#Do_One_Thing_and_Do_It_Well>`_.
   We're PyPOTS: we don't build everything related to time series and we don't have to, but only things related to partially-observed time series.
   And when we build something in PyPOTS, we're responsible and trying our best to do it well;
2. `Eat our own dog food <https://en.wikipedia.org/wiki/Eating_your_own_dog_food>`_.
   We develop PyPOTS and we should try the best to use it in any scenarios related to POTS data.
   Only in this way, we can figure out how it tastes like, if it is a good toolset for users, and what other features and models should be included into PyPOTS;
3. `No silver bullet <https://en.wikipedia.org/wiki/No_Silver_Bullet>`_ and `No free launch <https://en.wikipedia.org/wiki/No_free_lunch_theorem>`_.
   There is no one solution to all problems in the universe. In PyPOTS, we keep things modular, so one can easily try and replace parts of the pipeline
   in search for the optimal combination for the particular task;
4. Keep things easy to use and familiar. We try to keep PyPOTS intuitive without compromising flexibility and without forcing users to learn a completely new technology.
   We do this by keeping the toolkit close to APIs in scikit-learn and pytorch that people know and love;
