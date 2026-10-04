# Overview: where the capstone stands

Updated at the end of every week. Last updated: 2026-10-04 (end of week 28_9).

We have finished replicating GLIMMER and moved into uncertainty. GLIMMER's danger-weighted loss catches more dangerous readings but makes overall accuracy worse. Wrapping error bands around the plain model gives better alarms, and of the bands tested, volatility methods adapt their width to the errors. The open question is whether uncertainty can be personalized: a model fine-tuned on a new patient's first few days should grow more confident about that patient. Early tests say no, but they used five patients and one seed. Next week the direction gets decided with Prof. Watson, and a learning-curve test starts that checks which kind of uncertainty shrinks with more data.
