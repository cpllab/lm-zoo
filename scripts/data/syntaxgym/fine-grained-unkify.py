import argparse
import pickle

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper constants and functions
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SUFFIXES = ['ed', 'ing', 'ion', 'er', 'est', 'ly', 'ity', 'y', 'al']

def get_dict(lines, outpath=None, n=1):
    """
    Expects lines to contain tokens separated by whitespace.
    """
    # Populate dictionary with frequency counts of each token.
    freq_dict = {}
    for line in lines:
        tokens = line.split()
        for token in tokens:
            if token not in freq_dict:
                freq_dict[token] = 1
            else:
                freq_dict[token] += 1
    # If outpath specified, then save dictionary to file.
    if outpath is not None:
        with open(outpath, 'wb') as f:
            pickle.dump(freq_dict, f)
    # Only return words that appear more than n times in the data.
    vocab = [word for word, freq in freq_dict.items() if freq > n]
    return vocab 

def unkify(tokens, vocab, brackets=False):
    final_tokens = []
    for token in tokens:
        # Only unkify the words that are not in vocab.
        if token.rstrip() == '':
            final_tokens.append('UNK')
        # If brackets=True, then replace parentheses with LRB/RRB tokens.
        elif brackets and token.rstrip() == '(':
            final_tokens.append('-LRB-')
        elif brackets and token.rstrip() == ')':
            final_tokens.append('-RRB-')
        # Else, perform fine-grained unkification process.
        elif not(token.rstrip() in vocab):
            numCaps = 0
            hasDigit = False
            hasDash = False
            hasLower = False
            for char in token.rstrip():
                if char.isdigit():
                    hasDigit = True
                elif char == '-':
                    hasDash = True
                elif char.isalpha():
                    if char.islower():
                        hasLower = True
                    elif char.isupper():
                        numCaps += 1
            unked = 'UNK'
            lower = token.rstrip().lower()
            ch0 = token.rstrip()[0]
            if ch0.isupper():
                if numCaps == 1:
                    unked = unked + '-INITC'    
                    if lower in vocab:
                        unked = unked + '-KNOWNLC'
                else:
                    unked = unked + '-CAPS'
            elif not(ch0.isalpha()) and numCaps > 0:
                unked += '-CAPS'
            elif hasLower:
                unked += '-LC'
            if hasDigit:
                unked += '-NUM'
            if hasDash:
                unked += '-DASH' 
            if lower[-1] == 's' and len(lower) >= 3:
                ch2 = lower[-2]
                if not(ch2 == 's') and not(ch2 == 'i') and not(ch2 == 'u'):
                    unked += '-s'
            elif len(lower) >= 5 and not(hasDash) and not(hasDigit and numCaps > 0):
                for suffix in SUFFIXES:
                    if lower[-len(suffix):] == suffix:
                        unked += '-' + suffix
                        break
            final_tokens.append(unked)
        else:
            # If token is in the vocabulary, then no need to unk.
            final_tokens.append(token.rstrip())
    return final_tokens 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main function
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main(args):
    """
    Maps *tokenized* file (one sentence per line) to fine-grained unk tokens.
    Expects tokens to be separated by whitespace.
    Prints unkified content to stdout.
    To unkify the train file, pass the same file as args.train and args.dev.
    """
    with open(args.train, "r") as train_file:
        train_lines = train_file.readlines()
    with open(args.dev, "r") as dev_file:
        dev_lines = dev_file.readlines()
    vocab = get_dict(train_lines, outpath=args.save_dict, n=args.n) 

    for line in dev_lines:
        tokens = line.strip().split()
        unkified = unkify(tokens, vocab, brackets=args.brackets)
        print(' '.join(unkified))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Unkify tokenized file")
    p.add_argument("train", type=str)
    p.add_argument("dev", type=str)
    p.add_argument("--save_dict", "-save_dict", type=str, default=None,
                   help="path to save pickled vocabulary (token-to-freq)")
    p.add_argument("--brackets", "-brackets", default=False, action="store_true",
                   help="map parentheses to LRB/RRB tokens")
    p.add_argument("--n", "-n", type=int, default=1, 
                   help="only include that appear > n times in training data")
    args = p.parse_args()
    main(args)