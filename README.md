# transfer-pretraining-Ch-Eng

This repository hosts an English-Cherokee Neural Machine Translation (NMT) model, leveraging advanced knowledge transfer techniques from a pre-trained Russian-English model. Both the encoder and decoder components of our model have been enriched with linguistic features and structural insights gained from the Russian-English language pair, aiming to enhance translation quality and fluency for the low-resource Cherokee language. This approach not only addresses the challenge of limited Cherokee-English parallel corpora but also sets a new precedent for leveraging cross-lingual transfer learning in NMT systems.


You can save russian model in /hugging-face/model folder, for that you need to uncomment several lines at the beginning of '/hugging-face/main.py'
Base Russian-English model has BLEU score of 61.1

BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.2 = 61.1 83.0/67.4/57.1/48.8 
The overall BLEU score is 61.1. The numbers following it represent the precision of 1-gram/2-gram/3-gram/4-gram matches between the translated text and the reference. These precisions decrease as the n-gram size increases, which is typical because finding exact matches becomes harder with longer sequences.

(BP = 0.972 ratio = 0.973)
BP stands for Brevity Penalty, which penalizes shorter translations. A BP close to 1 (like 0.972 here) indicates the translation length is close to the reference length, hence minimal penalty. The ratio of the hypothesis length to the reference length is 0.973, supporting the same conclusion.

hyp_len = 36215 ref_len = 37235
The total length of the hypothesis (translated text) and the reference text, respectively

chrF2+case.mixed+numchars.6+numrefs.1+space.False+version.1.4.2 = 0.736



Data processing:
(train-dev-test) sizes (80%-10%-10%) or (11,334 - 1400 - 1400) lines