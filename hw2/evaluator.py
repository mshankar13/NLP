import argparse
from helpers import *


class Evaluator:
    """ This application takes as input a file containing POS tagged sentences and generates
        three output files which contains various probabilities necessary for implementing a frequency
        based tagger and a HMM tagger"""

    if __name__ == "__main__":

        """ This method reads the test predictions from frequency-based tagger and hmm tagger for mle and laplace.
            It outputs the accuracy, precision, recall and f1 for each POS tag.
        """

        parser = argparse.ArgumentParser()
        parser.add_argument('mle')
        parser.add_argument('laplace')
        results = vars(parser.parse_args())

        hmm_mle_test = read_file(results['mle'])
        predicted_mle, actual_mle = evaluate_result(hmm_mle_test)

        hmm_laplace_test = read_file(results['laplace'])
        predicted_laplace, actual_laplace = evaluate_result(hmm_laplace_test)

        mle_accuracy, mle_mistagged, mle_words = calculate_accuracy(actual_mle, predicted_mle)
        laplace_accuracy, laplace_mistagged, laplace_words = calculate_accuracy(actual_laplace, predicted_laplace)
        print('HMM MLE Accuracy:\t', mle_accuracy)
        print('HMM Laplace Accuracy:\t', laplace_accuracy)

        get_top_ten(laplace_mistagged, laplace_words)
        with open('evaluator_mle.txt', 'w') as fileObj:
            for tag in list(set(actual_mle)):
                p = calculate_precision(tag, actual_mle, predicted_mle)
                r = calculate_recall(tag, actual_mle, predicted_mle)
                f1 = calculate_f1(p, r)

                fileObj.write('{}\t{}\t{}\t{}\n'.format(tag, p, r, f1))

        with open('evaluator_laplace.txt', 'w') as file:
            for tag in list(set(actual_laplace)):
                p = calculate_precision(tag, actual_laplace, predicted_laplace)
                r = calculate_recall(tag, actual_laplace, predicted_laplace)
                f1 = calculate_f1(p, r)

                file.write('{}\t{}\t{}\t{}\n'.format(tag, p, r, f1))