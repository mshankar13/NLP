import sys
import itertools
import collections
import math


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


# Calculate the unique number of words for V
def calculate_v(unigrams_count):
    return len(unigrams_count)


def calculate_mle(bigram, i, bigrams_count, unigrams_count, unigrams):
    num = 0
    if i != -1:
        num = bigrams_count[i]
        den = unigrams_count[unigrams.index(bigram[1])]
        return num / den
    else:
        return 0


def calculate_laplace(bigram, v, i, bigrams_count, unigrams_count, unigrams):
    neu = bigrams_count[i] + 1
    den = unigrams_count[unigrams.index(bigram[0])] + v + 1
    return neu / den


def calculate_inter(l, bigram, mle_prob, v, all_unigrams, unigrams, unigrams_count):
    py = (unigrams_count[unigrams.index(bigram[1])] + 1) / (len(all_unigrams) + v + 1)
    result = (l * mle_prob) + ((1 - l) * py)
    return result


# For training Katz = AD
def calculate_ad(bigram, i, bigrams_count, unigrams_count, unigrams):
    # D = 0.5
    return (bigrams_count[i] - 0.5) / unigrams_count[unigrams.index(bigram[0])]


def write_top_bigrams(bigrams, probs):
    file = open("../outputs/top-bigrams.txt", 'w')
    index = 0
    for b in bigrams:
        file.write('{}\n'.format(b))


def write_bigram(bigrams, bigrams_count, mle, laplace, inter, katz):
    file = open("../outputs/bigram.lm", 'w')
    index = 0
    for b in bigrams:
        w1, w2, bc = b[0], b[1], bigrams_count[index]
        m, l, i, k = mle[index], laplace[index], inter[index], katz[index]
        file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(w1, w2, bc, m, l, i, k))
        index += 1
    return


def write_unigram(unigrams, unigrams_count):
    file = open("../outputs/unigram.lm", 'w')
    index = 0
    for u in unigrams:
        file.write('{}, {}\n'.format(u, unigrams_count[index]))
        index += 1
    return


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


def calculate_lambda(bigram, mle_prob, v, all_unigrams, unigrams, unigrams_count):
    try:
        py = (unigrams_count[unigrams.index(bigram[1])] + 1) / (len(all_unigrams) + v + 1)
    except ValueError:
        py = 1 / (len(all_unigrams) + v + 1)

    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    lower = 0
    first = -1

    inter = []
    for l in lambdas:
        value = (l * mle_prob) + ((1 - l) * py)
        inter.append(value)

    # Calculate perplexity
    index = 0
    opt_inter = 0
    for i in inter:
        pp = math.pow(2, -1) * math.log(i, 2)
        if first == -1:
            lower = pp
            index = inter.index(i)
        elif pp < lower:
            lower = pp
            opt_inter = i
            index = inter.index(i)

    return lambdas[index]


def get_lambda(v, unigrams_count, all_unigrams, unigrams, bigrams, bigrams_count):
    buffer = ""
    words = []
    all_dev_bigrams = []
    dev_bigrams = []

    l = 0

    inputFile = "../inputs/dev.txt"
    fill_buffer(inputFile, buffer, words)

    # Organize all dev bigrams into a list for counting
    for x in words:
        sentence_bigrams = [x[y: y + 2] for y in range(len(x) - 1)]
        for y in sentence_bigrams:
            all_dev_bigrams.insert(len(all_dev_bigrams), y)

    # Get rid of duplicate dev bigrams
    bigrams_set = set(map(tuple, all_dev_bigrams))
    for b in bigrams_set:
        dev_bigrams.append(b)

    lambdas = []
    # Make necessary calculations for each bigram
    i = 0
    for b in dev_bigrams:
        try:
            bigrams.index(b)
        except ValueError:
            i = -1
        m = calculate_mle(b, i, bigrams_count, unigrams_count, unigrams)
        l = calculate_lambda(b, m, v, all_unigrams, unigrams, unigrams_count)
        lambdas.append(l)
        i += 1

    return l


def top_bigrams(laplace, v, all_unigrams, bigrams):
    joint_probs = {}
    joint_prob = []
    index = 0
    for b in bigrams:
        pl = (all_unigrams.count(b[0]) + 1) / (len(all_unigrams) + v + 1)
        k = pl * laplace[index]
        joint_prob.insert(index, k)
        joint_probs.setdefault(k, []).append(index)
        index += 1

    ordered_probs = dict(collections.OrderedDict(sorted(joint_probs.items(), reverse=True)))
    reverse_order = list(reversed(sorted(ordered_probs.keys())))
    count = 0
    joint20_prob = []
    joint20_bigrams = []
    for r in reverse_order:
        if count == 20:
            break
        lst = ordered_probs.get(r)
        for l in lst:
            w1 = bigrams[l][0]
            w2 = bigrams[l][1]
            if '<s>' not in w1 and '</s>' not in w1 and '<s>' not in w2 and '</s>' not in w2:
                joint20_prob.insert(count, joint_prob[l])
                joint20_bigrams.insert(count, bigrams[l])
                count += 1

            if count == 20:
                break
    write_top_bigrams(joint20_bigrams, joint20_prob)
    return


# Constructs the bigram ml to be written to a file
def bigram(words, v, all_unigrams, all_bigrams, bigrams_count, bigrams, unigrams_count, unigrams):
    mle = []  # Keep track of mle calculations
    laplace = []
    inter = []
    katz = []

    # Organize all bigrams into a list for counting
    for x in words:
        sentence_bigrams = [x[y: y + 2] for y in range(len(x) - 1)]
        for y in sentence_bigrams:
            all_bigrams.insert(len(all_bigrams), y)

    # Get rid of duplicate bigrams
    bigrams_set = set(map(tuple, all_bigrams))
    for b in bigrams_set:
        bigrams.append(b)

    i = 0
    for b in bigrams:
        bigrams_count.insert(i, all_bigrams.count(list(b)))  # Calculate the frequency of each bigram

    l = get_lambda(v, unigrams_count, all_unigrams, unigrams, bigrams, bigrams_count)
    # Make necessary calculations for each bigram
    i = 0
    for b in bigrams:

        m = calculate_mle(b, i, bigrams_count, unigrams_count, unigrams)
        mle.insert(i, m)  # Calculate the MLE Probability
        laplace.insert(i, calculate_laplace(b, v, i, bigrams_count, unigrams_count,
                                            unigrams))  # Calculate the Laplace Probability
        # Calculate the Interpolated Probability
        inter.insert(i, calculate_inter(l, b, m, v, all_unigrams, unigrams, unigrams_count))
        katz.insert(i, calculate_ad(b, i, bigrams_count, unigrams_count, unigrams))  # Calculate the AD probability

        i += 1

    write_bigram(bigrams, bigrams_count, mle, laplace, inter, katz)
    return laplace


# Constructs the unigram ml to be written to a file
def unigram(words, unigrams_count, all_unigrams, unigrams):
    # Organize all unigrams into a list for counting
    for x in words:
        for y in x:
            all_unigrams.insert(len(all_unigrams), y)
    # Get rid of duplicate unigrams
    unigram_set = set(map(str, all_unigrams))
    for u in unigram_set:
        unigrams.append(u)

    for u in unigrams:
        unigrams_count.append(all_unigrams.count(u))

    write_unigram(unigrams, unigrams_count)
    return calculate_v(unigrams_count)


class LMB:
    if __name__ == "__main__":
        buffer = ""
        words = []

        all_unigrams = []
        unigrams_count = []
        all_bigrams = []  # Keep track of all bigrams
        bigrams_count = []  # Keep track of the frequency of each bigram
        bigrams = []  # Keep track of unique bigrams
        unigrams = []

        # if len(sys.argv) < 2:
        #    print("Please specify file to convert")
        #    sys.exit()

        # Get the filename
        # inputFile = sys.argv[1]
        inputFile = "../inputs/train.txt"

        fill_buffer(inputFile, buffer, words)
        v = unigram(words, unigrams_count, all_unigrams, unigrams)
        laplace = bigram(words, v, all_unigrams, all_bigrams, bigrams_count, bigrams, unigrams_count, unigrams)
        top_bigrams(laplace, v, all_unigrams, bigrams)
