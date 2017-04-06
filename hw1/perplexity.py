import sys
import math


# Fill a buffer with the file
def get_buffer(filename, buffer, count_buffer):
    with open(filename, 'r') as fileObj:
        # Loop through and read the file line by line
        while True:
            line = fileObj.readline()
            if not line:
                break
            splitline = line.split(", ")
            buffer.append(splitline[0])
            count_buffer.append(splitline[1].replace("\n", ""))


# Calculate the mle for a bigram
def calculate_mle(bi, i, bigrams_count, unigrams_count, unigrams):
    if bi == ['</s>', '<s>']:
        return 1
    if i != -1:
        num = bigrams_count[i]
        den = unigrams_count[unigrams.index(bi[0])]
        return num / den
    else:
        return 0


# Calculate the laplace for a bigram
def calculate_laplace(bi, v, i, bigrams_count, unigrams_count, unigrams):
    neu = bigrams_count[i] + 1
    den = unigrams_count[unigrams.index(bi[0])] + v + 1
    return neu / den


# Calculate the interpolation for a bigram
def calculate_inter(l, bi, mle_prob, v, all_unigrams, unigrams, unigrams_count):
    try:
        py = (unigrams_count[unigrams.index(bi[1])] + 1) / (len(all_unigrams) + v + 1)
        result = (l * mle_prob) + ((1 - l) * py)
    except ValueError:
        py = (0 + 1) / (len(all_unigrams) + v + 1)
        result = (l * mle_prob) + ((1 - l) * py)
    return result


# Fill words and word_counts with the unigrams and unigram counts
def get_unigram_prob(unigramfile, words, word_counts):
    try:
        with open(unigramfile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break
                split_line = line.split(", ")
                # Process the lines into lists
                words.insert(len(words), split_line[0])
                word_counts.insert(len(word_counts), int(split_line[1]))

    except IOError:
        print(unigramfile, " not found")
        sys.exit()


# Fill the training_bigrams, training_bigrams_count, laplace, and interpolated lists with appropriate values from the lm
def get_bigram_prob(bigramfile, training_bigrams, training_bigrams_count, laplace, interpolated):
    training_lambda = 0
    try:
        # Open the file
        with open(bigramfile, 'r') as fileObj:
            # Loop through and read the file line by line
            first_line = -1
            while True:
                line = fileObj.readline()
                if not line:
                    break
                if first_line == -1:
                    first_line = 0
                    # Get the training lambda for the interpolation calculations later on (Stored at the first line)
                    training_lambda = float(line.split("LAMBDA ")[1].replace("\n", ""))
                else:
                    split_line = line.split(", ")
                    # Process the lines into lists
                    bigram = []
                    bigram.append(split_line[0])
                    bigram.append(split_line[1])

                    training_bigrams.append(bigram)
                    training_bigrams_count.append(int(split_line[2]))
                    laplace.append(float(split_line[4]))
                    interpolated.append(float(split_line[5]))
        return training_lambda
    except IOError:
        print(bigramfile, " not found")
        sys.exit()


# Calculate the unigram laplace
def unigram_laplace(v, total_tokens, x):
    return (x + 1) / (total_tokens + v + 1)


# Recalculate interpolation for a bigram if it could not be retrieved
def recalculate_interpolated_bigram(b, bigrams, bigrams_count, unigrams_count, unigrams, total_tokens, l, v):
    try:
        m = calculate_mle(b, bigrams.index(tuple(b)), bigrams_count, unigrams_count, unigrams)
    except ValueError:
        if b == ['</s>', '<s>']:
            m = 1
        else:
            m = 0

    # Sum all interpolation probabilities
    try:
        u = unigram_laplace(v, total_tokens, unigrams_count[unigrams.index(b[1])])
    except ValueError:
        u = (0 + 1) / (total_tokens + v + 1)

    return (l * m) + ((1 - l) * u)


# Recalculate the laplace bigram probability if it could not be found
def recalculate_laplace_bigram(unigrams, unigrams_counts, bigrams, bigram_counts, bigram, v):
    try:
        neu = (bigram_counts[bigrams.index(bigram)] + 1)
        try:  # Try to get the counts of the first word
            den = unigrams_counts[unigrams.index(bigram[0])] + v + 1
        except ValueError:
            den = v + 1
        return neu / den
    except ValueError:
        neu = 1
        try:  # Try to get the counts of the first word
            den = unigrams_counts[unigrams.index(bigram[0])] + v + 1
        except ValueError:
            den = v + 1
        return neu / den


# Calculate the perplexities for laplace unigram, laplace bigram and interpolated bigram
def calculate_perplexities(all_words, all_bigrams, bigramfile, unigramfile):

    # Holds the training data from the lm
    training_bigrams = []
    training_laplace = []
    training_interpolated = []
    training_bigrams_count = []

    # Holds the unigram data from the lm
    training_unigrams = []
    training_unigrams_count = []

    # Keep track of the probability sums
    laplace_sum = 0
    laplace_sum_u = 0
    inter_sum = 0

    # Get the lambda and fill the training data bigram lists
    training_lambda = get_bigram_prob(bigramfile, training_bigrams, training_bigrams_count, training_laplace,
                                      training_interpolated)

    # Fill the training data unigram lists
    get_unigram_prob(unigramfile, training_unigrams, training_unigrams_count)

    # Calculate the total number of tokens
    total_tokens = 0
    for count in training_unigrams_count:
        total_tokens += int(count)

    # Run calculations for get the probabilities for each bigram
    for bigram in all_bigrams:

        try:    # Try to get the bigram data if it exists
            i = training_bigrams.index(list(bigram))
            lap = training_laplace[i]
            inter = training_interpolated[i]

            try:    # Try to get the unigram data if it exists
                lapu = unigram_laplace(total_tokens, len(training_unigrams),
                                           training_unigrams_count[training_unigrams.index(bigram[0])])
            except ValueError:
                lapu = (0 + 1) / (total_tokens + len(training_unigrams) + 1)

        except ValueError:

            # Recalculate laplace bigram probability
            lap = recalculate_laplace_bigram(training_unigrams, training_unigrams_count, training_bigrams,
                                             training_bigrams_count, bigram, len(training_unigrams))

            # Recalculate laplace unigram probability
            try:
                lapu = unigram_laplace(total_tokens, len(training_unigrams),
                                           training_unigrams_count[training_unigrams.index(bigram[0])])
            except ValueError:
                lapu = (0 + 1) / (total_tokens + len(training_unigrams) + 1)

            # Recalculate interpolated bigram probability
            inter = recalculate_interpolated_bigram(bigram, training_bigrams, training_bigrams_count,
                                                    training_unigrams_count, training_unigrams, total_tokens,
                                                    training_lambda, len(training_unigrams))

        # Sum all log 2 probabilities
        laplace_sum += math.log2(lap)
        inter_sum += math.log2(inter)
        laplace_sum_u += math.log2(lapu)

    # Apply the rest of the PP formula; N = total words in the test sequence
    laplace_pp = math.pow(2, (-1 / len(all_words)) * laplace_sum)
    inter_pp = math.pow(2, (-1 / len(all_words)) * inter_sum)
    laplace_pp_u = math.pow(2, (-1 / len(all_words)) * laplace_sum_u)

    print("Laplace Bigram PP: ", laplace_pp)
    print("Laplace Unigram PP: ", laplace_pp_u)
    print("Interpolated Bigram PP: ", inter_pp)


# Constructs the bigram ml to be written to a file
def get_bigrams(all_words):

    # Get the bigrams from all the words
    all_bigrams = []
    bigrams = []

    # Organize all bigrams into a list for counting

    sentence_bigrams = [all_words[y: y + 2] for y in range(len(all_words) - 1)]
    for y in sentence_bigrams:
        all_bigrams.insert(len(all_bigrams), y)

    # Delete extra stop starts at the end of the bigrams list
    del all_bigrams[len(all_bigrams) - 1]
    del all_bigrams[len(all_bigrams) - 1]

    # Get rid of duplicate bigrams
    bigrams_set = set(map(tuple, all_bigrams))
    for b in bigrams_set:
        bigrams.append(b)
    return all_bigrams


# Constructs the unigram ml to be written to a file
def sequence_tokens(words, all_words):
    # Organize all unigrams into a list for counting
    for x in words:
        for y in x:
            all_words.insert(len(all_words), y)


# Incorporates the pre-processing steps for each line in the inputFile
def process_input(buffer, words):
    # Split the sentences by period
    buffer.replace(". ", ".")
    sentences = buffer.split(".")

    # Loop through and split each word into word arrays
    for i in sentences:
        split_sentence = i.split(" ")
        split_sentence.insert(0, "<s>")
        split_sentence.append("</s>")
        # Remove any empty words
        words.append([str.lower(x) for x in split_sentence if x])


# Take out unnecessary newlines and tabs from line
def process_line(self):
    line = self.replace("\n", " ")
    line2 = line.replace("\t", "")
    return line2


# Fill the buffer and words with information from the file
def fill_buffer(inputfile, buffer, words):
    # Try opening the file
    try:
        with open(inputfile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break
                # Process the lines into a buffer
                buffer += process_line(line)
        process_input(buffer, words)
    except IOError:
        print(inputfile, " not found")


class PP:
    if __name__ == "__main__":
        buffer = ""
        words = []
        all_words = []

        # Validate correct number of arguments
        if len(sys.argv) < 3:
            print("Insufficient arguments")
            print("USAGE:\ttest.txt bigram.lm unigram.lm")
            sys.exit()

        # Test input
        # inputFile = "../inputs/test.txt"
        # bigramFile = "../outputs/bigram.lm"
        # unigramFile = "../outputs/unigram.lm"

        bigramfile = sys.argv[1]
        unigramfile = sys.argv[2]
        inputfile = sys.argv[3]

        fill_buffer(inputfile, buffer, words)
        sequence_tokens(words, all_words)
        all_bigrams = get_bigrams(all_words)  # Get all bigrams

        calculate_perplexities(all_words, all_bigrams, bigramfile, unigramfile)
