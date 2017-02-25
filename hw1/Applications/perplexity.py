import sys
import math


def get_unigram_prob(unigramFile, words, word_counts):

    unigram_laplace = []
    try:
        fileObj = open(unigramFile, 'r')
        fileObj.close()
    except IOError:
        print(unigramFile, " not found")
        sys.exit()
    else:
        with open(unigramFile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break
                split_line = line.split(", ")
                # Process the lines into lists
                words.insert(len(words), split_line[0])
                word_counts.insert(len(word_counts), int(split_line[1]))
        # Close the file
        fileObj.close()

        # Calculate unigram laplace
        total = 0   # Keep track of unique v
        for count in word_counts:
                total += count

        for count in word_counts:
            unigram_laplace.insert(len(unigram_laplace), (count + 1) / (total + len(word_counts) + 1))
    return unigram_laplace


def get_bigram_prob(bigramFile, training_bigrams, laplace, interpolated):


    try:
        fileObj = open(bigramFile, 'r')
        fileObj.close()
    except IOError:
        print(bigramFile, " not found")
        sys.exit()
    else:
        with open(bigramFile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break
                split_line = line.split(", ")
                # Process the lines into lists
                bigram = []
                bigram.append(split_line[0])
                bigram.append(split_line[1])

                training_bigrams.insert(len(training_bigrams), bigram)
                laplace.insert(len(laplace), float(split_line[4]))
                interpolated.insert(len(interpolated), float(split_line[5]))
        # Close the file
        fileObj.close()
    return


def calculate_perplexities(all_words, all_bigrams, bigramFile, unigramFile):

    training_word1 = []
    training_word2 = []
    training_bigram = []
    laplace = []
    interpolated = []

    words = []
    word_counts = []

    laplace_sum = 0
    laplace_sum_u = 0
    inter_sum = 0

    get_bigram_prob(bigramFile, training_bigram, laplace, interpolated)
    # Sum each bigram log 2 probability
    for bigram in all_bigrams:
        # Laplace Bigram
        # Interpolated Bigram
        try:
            i = training_bigram.index(list(bigram))
            lap = laplace[i]
            inter = interpolated[i]

            laplace_sum += math.log(lap, 2)
            inter_sum += math.log(inter, 2)
        except ValueError:
            i = -1

    x = math.pow(2, -1/len(all_bigrams))
    laplace_pp = x * laplace_sum
    inter_pp = x * inter_sum

    laplace_u = get_unigram_prob(unigramFile, words, word_counts)
    for word in all_words:
        try:
            i = words.index(word)
            lap = laplace_u[i]
            laplace_sum_u *= lap

        except ValueError:
            i = -1
    laplace_pp_u = math.pow(laplace_sum_u, len(all_words))

    print("Laplace Bigram PP: ", laplace_pp)
    print("Laplace Unigram PP: ", laplace_pp_u)
    print("Interpolated Bigram PP: ", inter_pp)
    return


# Constructs the bigram ml to be written to a file
def get_bigrams(all_words):

    all_bigrams = []
    bigrams = []

    # Organize all bigrams into a list for counting

    sentence_bigrams = [all_words[y: y + 2] for y in range(len(all_words) - 1)]
    for y in sentence_bigrams:
        all_bigrams.insert(len(all_bigrams), y)

    # Get rid of duplicate bigrams
    bigrams_set = set(map(tuple, all_bigrams))
    for b in bigrams_set:
        bigrams.append(b)

    return bigrams


# Constructs the unigram ml to be written to a file
def sequence_tokens(words, all_words):
    # Organize all unigrams into a list for counting
    for x in words:
        for y in x:
            all_words.insert(len(all_words), y)

    return


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

    return


def process_line(self):
    line = self.replace("\n", " ")
    line2 = line.replace("\t", "")
    return line2


def fill_buffer(inputFile, buffer, words):
    # Try opening the file
    try:
        fileObj = open(inputFile, 'r')
        fileObj.close()
    except IOError:
        print(inputFile, " not found")
    else:
        with open(inputFile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break
                # Process the lines into a buffer
                buffer += process_line(line)
        # Close the file
        fileObj.close()
        process_input(buffer, words)
    return


class PP:
    if __name__ == "__main__":
        buffer = ""
        words = []
        all_words = []

        inputFile = "../inputs/test.txt"
        bigramFile = "../outputs/bigram.lm"
        unigramFile = "../outputs/unigram.lm"

        fill_buffer(inputFile, buffer, words)
        sequence_tokens(words, all_words)
        all_bigrams = get_bigrams(all_words)    # Get all bigrams

        calculate_perplexities(all_words, all_bigrams, bigramFile, unigramFile)