import sys
import operator


# Fill a buffer with the file
def get_buffer(unigramfile, buffer, count_buffer):
    with open(unigramfile, 'r') as fileObj:
        # Loop through and read the file line by line
        while True:
            line = fileObj.readline()
            if not line:
                break
            splitline = line.split(", ")
            buffer.append(splitline[0])
            count_buffer.append(splitline[1].replace("\n", ""))


# Recalculate mle for the bigram using the first word
def recalculate_mle(word1, unigramfile):

    buffer = []
    count_buffer = []
    get_buffer(unigramfile, buffer, count_buffer)

    try:
        neu = count_buffer[buffer.index(word1)]
        total_tokens = 0
        for count in count_buffer:
            total_tokens += int(count)

    except ValueError:
        print(word1, "does not exist in the training data")
        sys.exit()
    else:
        return int(neu) / total_tokens


# Recalculate the laplace for the bigram using the first word
def recalculate_laplace(word1, unigramfile):

    buffer = []
    count_buffer = []
    get_buffer(unigramfile, buffer, count_buffer)

    try:
        den = int(count_buffer[buffer.index(word1)]) + len(buffer) + 1
    except ValueError:
        den = len(buffer) + 1
    return 1 / den


# Recalculate the interpolation for the bigram
def recalculate_interpolation(training_lambda, word1, word2, unigramfile):

    mle = recalculate_mle(word1, unigramfile)
    laplace = recalculate_laplace(word2, unigramfile)

    return (training_lambda * mle) + ((1 - training_lambda) * laplace)


# Recalculate the Katz Backoff for unseen bigrams
def recalculate_katz(word1, word2, unigramfile, other_words, other_ads):
    alpha = 0
    for ad in other_ads:
        alpha += float(ad)
    alpha = 1 - alpha

    buffer = []
    count_buffer = []
    get_buffer(unigramfile, buffer, count_buffer)

    # Calculate Total Tokens
    total_tokens = 0
    for count in count_buffer:
        total_tokens += int(count)

    # Calculate Laplace Probability
    try:
        beta = (count_buffer[buffer.index(word2)] + 1) / (total_tokens + len(count_buffer) + 1)
    except ValueError:
        beta = 1 / (total_tokens + len(count_buffer) + 1)

    return alpha * beta


# Get the top ten words to follow the first word in the query
def get_top_ten(other_words, other_ads):
    words_dict = {}
    for x in other_ads:
        words_dict[other_words[other_ads.index(x)]] = float(x)
    sorted_dict = sorted(words_dict.items(), key=operator.itemgetter(1), reverse=True)
    # other_ads.sort(key = lambda i: i, reverse = True)
    counter = 10
    reverse = 1
    print("Top Ten Words")
    for x in sorted_dict:
        if counter == 0:
            break
        print('{}{}\t{}\t{}'.format(str(reverse), ": ", x[1], x[0]))
        counter -= 1
        reverse += 1


# Remove unnecessary new lines and tabs
def process_line(line):
    if "\n" in line:
        line.replace("\n", "")
        line2 = line.replace("\t", "")
    return line2


# Process the input from command line
def process_input(self):

    # Get the inputs using sys
    bigramfile = self[1]
    unigramfile = self[2]
    word1 = str.lower(self[3])
    word2 = str.lower(self[4])
    smoothing = self[5]

    # Test inputs
    # bigramfile = "../outputs/bigram.lm"
    # unigramfile = "../outputs/unigram.lm"
    # word1 = "of"
    # word2 = "the"
    # smoothing = "I"

    # Check which smoothing probabilities to get where s = index in the line
    if smoothing is "M":
        s = 3
    elif smoothing is "L":
        s = 4
    elif smoothing is "I":
        s = 5
    elif smoothing is "K":
        s = 6
    else:
        print("Invalid Smoothing\nUSAGE:\tbigram.lm unigram.lm word1 word1 smoothing< M, L, I, K>")
        sys.exit()

    try:    # Try to open the file and make the necessary calculations or queries

        result = ['0', '0', '0', '0', '0', '0', '0']
        other_words = []
        other_ads = []
        first_line = -1
        training_lambda = 0

        # Open the bigram file
        with open(bigramfile, 'r') as fileObj:
            # Loop through and read the file line by line
            counter = 0
            while True:

                line = fileObj.readline()

                if not line:
                    break

                # If the line exists process it
                if first_line == -1:
                    first_line = 0
                    # Save the training lambda from the first line in the file
                    training_lambda = float(line.split("LAMBDA ")[1].replace("\n", ""))
                else:
                    split_line = process_line(line).split(", ")
                    if word1 == split_line[0] and word2 == split_line[1]:
                        result = split_line
                        counter += 1
                    elif word1 == split_line[0]:
                        other_words.append(split_line[1])
                        other_ads.append(split_line[s])

        # Check the see if the first word in the user bigram is in the training data
        if not other_words:
            print("Training data does not include ", word1)
            sys.exit()

        if smoothing is "M":    # MLE smoothing
            if result == ['0', '0', '0', '0', '0', '0', '0']:
                print(recalculate_mle(word1, unigramfile))
            else:
                print(result[3])
                # get_top_ten(other_words, other_ads)

        elif smoothing is "L":  # Laplace smoothing
            if result == ['0', '0', '0', '0', '0', '0', '0']:
                print(recalculate_laplace(word1, unigramfile))
            else:
                print(result[4])
                # get_top_ten(other_words, other_ads)

        elif smoothing is "I":  # Interpolation smoothing
            if result == ['0', '0', '0', '0', '0', '0', '0']:
                print(recalculate_interpolation(training_lambda, word1, word2, unigramfile))
            else:
                print(result[5])
                # get_top_ten(other_words, other_ads)
        elif smoothing is "K":  # Katz Backoff Smoothing
            if result == ['0', '0', '0', '0', '0', '0', '0']:
                print(recalculate_katz(word1, word2, unigramfile, other_words, other_ads))
            else:
                print(result[6].replace("\n", ""))
                # get_top_ten(other_words, other_ads)

    except FileNotFoundError:
        print("Bigram OR Unigram LM file does not exist")
        sys.exit()


class BQ:
    if __name__ == "__main__":

        # Verify that the user input the correct arguments
        if len(sys.argv) < 6:
            print("USAGE:\tbigram.lm unigram.lm word1 word1 smoothing< M, L, I, K>")
            sys.exit()

        process_input(sys.argv)
