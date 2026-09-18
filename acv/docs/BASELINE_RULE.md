# Fixed exploratory baseline

Recorded before reading case 04's label or computing rankings for cases 02-06.

- Find every car identifier in the case's own column headers.
- Standard layout: indoor average minus cooling control; require Information Valid = Valid and Running Mode = Automatic Cooling.
- Rich layout (provisional interpretation): passenger cabin detected value minus target value; require Running Mode = Full Cooling or Half Cooling. No equivalent information-valid flag is established.
- Require both numbers to be finite. Missing/non-finite readings are excluded; numeric zero is not automatically excluded.
- Score each car by the sample average of max(difference, 0), including zeros in the denominator. This does not measure consecutive duration and is not a time-weighted average.
- Rank scored cars descending by full-precision score; exact ties use ascending car ID.
- Cars without usable readings have an unavailable score, never a score of zero. Place them after scored cars in ascending car ID. This is an arbitrary completion rule, not evidence that they are healthy.
- Unrecognised layouts raise an error rather than silently guessing a mapping.

These are modelling recommendations, not official competition rules. Temperature units and control semantics remain unconfirmed. Case 04's missing temperature pairs and mode values for Cars 05-08, plus its field semantics, warrant organiser clarification. No trained parameters or labels are used by this rule. Cases 01 and 04 informed development; evaluating them is not an untouched holdout assessment. The rule was selected before test download; no test-based tuning is permitted.
