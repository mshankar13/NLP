import argparse
from helpers import *


class Training:
    """ This application takes as input a file containing POS tagged sentences and generates
        three output files which contains various probabilities necessary for implementing a frequency
        based tagger and a HMM tagger"""

    if __name__ == "__main__":

        """ This method parses arguments from the command line and calls several functions to produce 3 files:
            transitions.txt, emissions.txt and laplace-tag-unigrams.txt. Handles calling all helper functions to
            process training data and output results.

            Args:
                param1 (str): filename containing the training data consisting of one tagged sentence per line
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('filename')
        parser.add_argument('transitions')
        parser.add_argument('emissions')
        parser.add_argument('laplace')
        results = vars(parser.parse_args())

        # dict of filenames by type vars(results)
        sentences = read_file(results['filename'])
        mle_dict1, mle_dict2, mle_dict3, laplace_dict, unigram_tag_dict = transitions_emissions_evaluator(sentences)
        write_transitions(mle_dict1, str(results['transitions']))
        write_emissions(mle_dict2, mle_dict3, laplace_dict, str(results['emissions']))
        write_tag_unigrams(unigram_tag_dict, str(results['laplace']))