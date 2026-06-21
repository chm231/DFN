# Profile Likelihood

Profile likelihood curves are printed to stdout during each run
via the [DIAGNOSTIC] tag in `p32_estimator.estimate_size_model()`.

To generate profile_ll_setX.png, pass the `size_estimate_results` dict
from the dfnrec pipeline directly to `diagnostic_report.plot_profile_ll()`.
This stub will be promoted to a full plot when the dfnrec pipeline
pipes profile data through to this module.
