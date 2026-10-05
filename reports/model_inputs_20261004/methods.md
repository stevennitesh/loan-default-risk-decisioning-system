# Current assessed model inputs

All five frozen current history models; 1,000 uniformly sampled assessment applicants per fold, seed 20261004, declared before computation. Sampling never reads the outcome. Saved fitting-only preprocessing and native LightGBM TreeSHAP are reused without fitting or selection.

Absolute encoded-column contributions are summed by original field (including all one-hot categories), then by source group; each sample is averaged and five fold magnitudes receive equal weight. Quantities are cumulative mean absolute raw-margin contributions in natural-log-odds units, not signed net effects, probability points or shares of predictive performance. Larger groups can accumulate more magnitude; correlated and derived inputs share information.

Intercept-inclusive signed contributions reproduce each sampled raw margin within 1e-8; residuals and source/model identities are in provenance.json. Exact IDs, individual scores and SHAP arrays stay local and are not exported. This describes fitted-model behavior, not causality, predictive-value ablation or adverse-action reasons. The renderer reads these anonymous tables without data/models.
