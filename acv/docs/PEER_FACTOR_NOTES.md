# Exploratory peer comparison

This is development work across six training cases, not a new independent validation result. Test data were not used in these comparisons, and the original baseline is unchanged. Later plot commands accept a separate workbook for display only; this does not authorize test-based tuning.

Peer gap is the subject car's indoor reading minus the median of at least three other usable cars at the same recorded timestamp. The subject is excluded from its reference. Three is an exploratory minimum, not an official requirement or tuned parameter. Usability and provisional rich-layout mappings follow the existing baseline. Missing comparisons remain unavailable.

The step 10 table averages max(control gap, 0) and max(peer gap, 0) over the same rows for each car. Negative values contribute zero; zero values remain in the denominator. An empty selection has unavailable averages. The matched control average can differ from the baseline, which uses all individually usable rows. Different cars can still have different accepted timestamps. No combined weighting or persistence factor has been selected.

Both factors share the indoor measurement, so they are correlated evidence. Peer warmth can reflect different control settings rather than a cooling fault. Case 04 still lacks an established validity flag and the necessary readings for Cars 05-08.
