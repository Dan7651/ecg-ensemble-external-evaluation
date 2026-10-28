# Technical clinical-label mapping audit

The map is research harmonization, not clinician-adjudicated equivalence to SCP
superclasses. Its source is the original SPH dictionary and standardized ECG
terminology (Mason et al., DOI 10.1016/j.jacc.2007.01.025).

- NORM requires explicit code 1; sinus rhythm alone does not imply normal ECG.
- Codes 82 (prolonged PR) and 147 (T-wave abnormality) have form-statement
  cross-references; 173 CD and 2,106 STTC positive records depend solely on them.
- Code 148 (prolonged QT) is outside the frozen map despite its catalogue link
  to STTC. It was not added after viewing results.
- All 44 source primary statements and 15 modifiers were retained in the
  dictionary audit; modifiers do not add targets. Counts of occurrences, ECGs
  and unique patients were not treated as interchangeable.
- Withdrawing code 147 changes blend AUROC from 0.94493 to about 0.93157.
  This sensitivity is larger than the original fusion gain; it is not a new
  clinically certified ground truth.
- The 4,587 mapped-zero recordings are annotation negatives, not certified
  healthy controls. Their inclusion is reported separately without rescuing
  eligibility or thresholds.

Qualified clinical review remains necessary to adjudicate ontology equivalence.
No expert signoff, new normal definition, or revised deployment map is asserted.
