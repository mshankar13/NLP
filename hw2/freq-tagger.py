import argparse
from helpers import *


class FreqTagger:
    """ """

    if __name__ == "__main__":

        """ This method parses arguments from the command line and calls several functions to produce 3 files:
            transitions.txt, emissions.txt and laplace-tag-unigrams.txt. Applies the training results on the test data
            to try and predict the tags.

            Args:
                param1 (str): filename containing the training data consisting of one tagged sentence per line
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('filename')
        parser.add_argument('emissions')
        parser.add_argument('test-fbtagger')
        results = vars(parser.parse_args())

        sentences = read_file(results['filename'])
        filtered_sentences, sentence_tags = filter_sentences(sentences)
        emissions = read_file(results['emissions'])
        emissions_dict = parse_emissions_txt(emissions)
        tagged_sentences = evaluate_test_sentences(emissions_dict, filtered_sentences)
        write_fb_tagger(tagged_sentences, results['test-fbtagger'])