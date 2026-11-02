# 3. Results

## 3.1 Reference-capacity distributions

Across 55 XJTU cells, the first-comparable Qref was below the 2.000 Ah nominal reference in every cell. The median difference was -0.123 Ah (-6.150%), with a cell-level range from -0.213 to -0.058 Ah (-10.650% to -2.900%). The early-median-5 comparable Qref was also below nominal in all 55 cells, with a median difference of -0.077 Ah (-3.850%) and a range from -0.195 to -0.038 Ah. The XJTU-only initial-test reference was available for 47 cells and showed a mixed direction: 38 values were below and 9 were above nominal; its median difference was -0.012 Ah (-0.600%).

Across 77 HUST cells, both measured comparable definitions were above the 1.100 Ah nominal reference in every cell. The first-comparable difference had a median of +0.082 Ah (+7.429%) and ranged from +0.059 to +0.132 Ah. The early-median-5 difference had a median of +0.082 Ah (+7.424%) and ranged from +0.058 to +0.130 Ah. HUST had no separate standardized initial-test Qref, so these results represent mechanism-level replication with common comparable definitions rather than an exact nominal-versus-initial-test replication.

## 3.2 Qref-induced EOL and censoring changes at 80%

Under the predefined 80% labeling rule, XJTU nominal Qref produced 20 exact, 32 interval-censored, and 3 right-censored EOL states. First-comparable Qref produced 0, 1, and 54, respectively, while early-median-5 produced 0, 3, and 52. Relative to nominal, EOL status changed in 51 of 55 cells for first-comparable and 49 of 55 for early-median-5. The median absolute 80% threshold shifts were -0.0984 and -0.0616 Ah, respectively. No exact EOL-cycle delta was created for these censored pairs.

In HUST, nominal Qref produced 77 right-censored and no exact EOL states at 80%, whereas both measured comparable definitions produced 77 exact states and no right-censored states. Thus, EOL status changed in all 77 cells for each measured definition. The measured references shifted the median 80% analytical threshold by +0.0654 Ah for first-comparable and +0.0653 Ah for early-median-5. Because the nominal endpoints were right-censored, these were not exact paired EOL differences; the minimum observed earlier-by-cycle bound had a median of 188 cycles for both comparisons. These results must be interpreted in the HUST released observation-window and stopping-rule context near the nominal 80% region.

## 3.3 RUL point-label availability

At 80%, point-RUL labels were available for 20 of 55 XJTU cells (36.4%) under nominal Qref and for 0 of 55 cells under either measured comparable definition. The corresponding availability change involved 20 cells for each measured comparison. In HUST, availability changed from 0 of 77 cells under nominal Qref to 77 of 77 cells under both measured definitions. Censored EOL states retained missing point-RUL labels; no last-observed cycle was substituted as EOL.

## 3.4 Threshold robustness at 75%, 80%, and 85%

The Qref-related labeling effect was not unique to the 80% threshold. At 75%, XJTU nominal Qref yielded exact/interval/right counts of 0/1/54, compared with 0/0/55 for first-comparable and 0/0/55 for early-median-5; this level was mostly observation-window limited. At 85%, the corresponding XJTU counts were 23/32/0, 5/31/19, and 10/32/13. Among exact pairs at 85%, the median EOL shift was 29.0 cycles for first-comparable (n=5) and 26.5 cycles for early-median-5 (n=10).

In HUST at 75%, nominal Qref remained right-censored in all 77 cells, while first-comparable produced 61 exact and 16 right-censored states and early-median-5 produced 57 exact and 20 right-censored states. The minimum observed earlier-by-cycle bound had medians of 22 and 23 cycles, respectively. At 85%, all three HUST Qrefs produced 77 exact EOL states; measured-versus-nominal exact EOL shifts remained observable, with medians of -222 cycles for first-comparable and -222 cycles for early-median-5. The HUST 75% and 80% counts remain dependent on the finite released observation window, whereas the 85% threshold is encountered earlier in the recorded trajectories.

## 3.5 SOH benchmark metric sensitivity

The primary A9 benchmark used 55 XJTU cells with 10,427 observations and 77 HUST cells with 146,121 feature-complete observations. In physical-capacity space, XJTU median held-out-cell MAE_Q ranges across Qrefs were 0.0243-0.0248 Ah for Random Forest, 0.0265-0.0269 Ah for Gradient Boosting, and 0.0332-0.0381 Ah for Ridge. Their SOH-space MAE ranges were 1.240-1.284, 1.327-1.433, and 1.705-2.023 percentage points, respectively. In HUST, physical-space ranges were 0.0203-0.0210, 0.0215-0.0228, and 0.0319-0.0322 Ah, whereas SOH-space ranges were 1.708-1.913, 1.800-2.073, and 2.709-2.903 percentage points. Thus, Qref definition was associated with changes in reported error magnitudes, and physical-capacity-space reporting reduced ambiguity attributable solely to SOH label scaling without eliminating model or dataset effects.

## 3.6 Model-ranking robustness

Using median held-out-cell MAE_Q_Ah as the predefined ranking metric, the primary A9 order was Random Forest, Gradient Boosting, then Ridge for every XJTU and HUST common Qref. The same order was retained in the B10 A9-plus-cycle-index and A9 post-cycle-5 analyses, and in HUST after removing CV duration. In the XJTU A8 analysis, the observed order was Random Forest, Gradient Boosting, then Ridge under nominal Qref but Gradient Boosting, Random Forest, then Ridge under both measured definitions. The corrected RF-versus-GB difference of medians was 0.000744 Ah under nominal Qref, -0.002557 Ah under first-comparable, and -0.002062 Ah under early-median-5. All corrected 95% bootstrap intervals included zero: [-0.004461, 0.003582], [-0.003454, 0.005258], and [-0.008512, 0.002603] Ah. The same-draw cell bootstrap reproduced the nominal-RF-lower and measured-GB-lower pattern in 31.04% of replicates for first-comparable and 52.52% for early-median-5. This was a descriptive rank reversal, not a bootstrap-robust model-ranking effect.
