#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: Jennifer Hu
# Date: 2019-10-27
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Usage: python vocab_hist.py
# Function: Generates histogram of word frequencies in specified vocabulary.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load data with pickle.
corpus = "ptb"
parent = "/om/group/cpl/language-models/syntaxgym/data/%s" % corpus
with open("%s/train_vocab.pkl" % parent, "rb") as f:
    vocab = pickle.load(f)

# Sort vocabulary by frequency to see most frequent words.
sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
print("== Top 10 words in vocabulary ==")
print("\n".join(str(tup) for tup in sorted_vocab[:10]))

# Get frequencies for plotting and print descriptive stats.
freqs = [freq for word, freq in vocab.items()]
print("== Descriptive statistics of frequencies ==")
print("Max: %d" % max(freqs))
print("Min: %d" % min(freqs))
print("Mean: %f" % np.mean(freqs))
print("Std: %f" % np.std(freqs))

# Only plot frequencies on lower end.
threshold = 5 # np.quantile(freqs, 0.5)
low_freqs = [freq for freq in freqs if freq <= threshold]
prop_low = len(low_freqs) / len(freqs) * 100.0
print("== Plotting frequencies <= %.2f (%.2f%% of vocabulary) ==" % (threshold, prop_low))

# Generate plot.
plt.hist(low_freqs, bins=5)
plt.xlabel("# of occurrences in %s training data" % corpus.upper())
plt.ylabel("# of tokens")
plt.title("Token frequencies")
plt.savefig("%s/train_vocab_hist.png" % parent, dpi=300, bbox_inches="tight")
