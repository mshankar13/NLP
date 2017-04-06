import argparse
from helpers import *


class HMMTagger:
    """ This application takes as input a file containing POS tagged sentences and generates
        three output files which contains various probabilities necessary for implementing a frequency
        based tagger and a HMM tagger"""

    if __name__ == "__main__":

        """ This method parses arguments from the command line and calls several functions to produce 3 files:
            transitions.txt, emissions.txt and laplace-tag-unigrams.txt. It also invokes several functions which run
            the application. See helpers for all functions and explanations.

            Args:
                param1 (str): filename containing the training data consisting of one tagged sentence per line
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('filename')
        parser.add_argument('transitions')
        parser.add_argument('emissions')
        parser.add_argument('tag')
        parser.add_argument('outfile')
        results = vars(parser.parse_args())

        sentences = read_file(results['filename'])
        filtered_sentences, sentences_tags = filter_sentences(sentences)

        emissions = read_file(results['emissions'])
        emissions_dict, all_tags = reformat_emissions(emissions, results['tag'])

        transitions = read_file(results['transitions'])
        transitions_dict = reformat_transitions(transitions)

        new_tagged_sentences, best_probs = evaluate_sentences_hmm(filtered_sentences, all_tags, emissions_dict, transitions_dict)

        write_hmm_tagger(best_probs, new_tagged_sentences, sentences_tags, results['outfile'])