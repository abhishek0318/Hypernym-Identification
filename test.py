"""Check if given two words form hypernym hyponym relationship.

Usage: python3 test.py word1 word2
Returns if word1 is hypernym of word2.
"""

import os.path
import sys

from sklearn.externals import joblib

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: {} word1 word2', sys.argv[0])
        sys.exit()
        
    model = joblib.load(os.path.join('data', 'trained_model.pkl'))
    
    if model.predict([(sys.argv[1], sys.argv[2])]) == 1:
        print("{} is a hypernym of {}.".format(sys.argv[1], sys.argv[2]))
    else:
        print("{} is not a hypernym of {}.".format(sys.argv[1], sys.argv[2]))
