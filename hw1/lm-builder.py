import sys
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


# Removes newlines and tabs from each line read from the file
def process_line(self):
    line = self.replace("\n", " ")
    line2 = line.replace("\t", "")
    return line2


# Calculate the unique number of words for V
def calculate_v(unigrams_count):
    return len(unigrams_count)


# Calculates mle for a bigram
def calculate_mle(bi, i, bigrams_count, unigrams_count, unigrams):
    if bi == ['</s>', '<s>']:
        return 1
    if i != -1:
        num = bigrams_count[i]
        den = unigrams_count[unigrams.index(bi[0])]
        return num / den
    else:
        return 0


# Calculates the Laplace for a bigram
def calculate_laplace(bi, v, i, bigrams_count, unigrams_count, unigrams):
    neu = bigrams_count[i] + 1
    den = unigrams_count[unigrams.index(bi[0])] + v + 1
    return neu / den


# Calculates the interpolation for a bigram
def calculate_inter(l, bi, mle_prob, v, all_unigrams, unigrams, unigrams_count):
    try:
        py = (unigrams_count[unigrams.index(bi[1])] + 1) / (len(all_unigrams) + v + 1)
        result = (l * mle_prob) + ((1 - l) * py)
    except ValueError:
        py = (0 + 1) / (len(all_unigrams) + v + 1)
        result = (l * mle_prob) + ((1 - l) * py)
    return result


# For training Katz = AD
def calculate_ad(bi, i, bigrams_count, unigrams_count, unigrams):
    # D = 0.5
    return (bigrams_count[i] - 0.5) / unigrams_count[unigrams.index(bi[0])]


# Writes the top bigrams to a file, excluding start and stop tags
def write_top_bigrams(bigrams, probs):
    with open("top-bigrams.txt", 'w') as file:
        for b in bigrams:
            file.write('{}, {}\n'.format(b, probs[bigrams.index(b)]))


def write_bigram(l, bigrams, bigrams_count, mle, laplace, inter, katz):
    with open("bigram.lm", 'w') as file:
        index = 0
        file.write('LAMBDA {}\n'.format(l))
        for b in bigrams:
            w1, w2, bc = b[0], b[1], bigrams_count[index]
            m, l, i, k = mle[index], laplace[index], inter[index], katz[index]
            file.write('{}, {}, {}, {}, {}, {}, {}\n'.format(w1, w2, bc, m, l, i, k))
            index += 1


def write_unigram(unigrams, unigrams_count):
    with open("unigram.lm", 'w') as file:
        index = 0
        for u in unigrams:
            file.write('{}, {}\n'.format(u, unigrams_count[index]))
            index += 1


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


def process_dev(words, all_dev_unigrams, dev_unigrams, all_dev_bigrams, dev_bigrams):

    # Organize all unigrams into a list for counting
    for x in words:
        for y in x:
            all_dev_unigrams.insert(len(all_dev_unigrams), y)
    # Get rid of duplicate unigrams
    unigram_set = set(all_dev_unigrams)
    for u in unigram_set:
        dev_unigrams.append(u)

    # Organize all dev bigrams into a list for counting
    for x in words:
        sentence_bigrams = [x[y: y + 2] for y in range(len(x) - 1)]
        for y in sentence_bigrams:
            all_dev_bigrams.insert(len(all_dev_bigrams), y)
            if y[1] == '</s>':
                all_dev_bigrams.insert(len(all_dev_bigrams), ['</s>', '<s>'])
    del all_dev_bigrams[len(all_dev_bigrams) - 1]
    del all_dev_bigrams[len(all_dev_bigrams) - 1]
    del all_dev_bigrams[len(all_dev_bigrams) - 1]

    # Get rid of duplicate dev bigrams
    bigrams_set = set(map(tuple, all_dev_bigrams))
    for b in bigrams_set:
        dev_bigrams.append(b)


def unigram_laplace(v, total_tokens, x):
    return (x + 1) / (total_tokens + v + 1)


def get_lambda(v, unigrams_count, all_unigrams, unigrams, bigrams, bigrams_count):
    buffer = ""
    words = []
    all_dev_bigrams = []
    dev_bigrams = []
    all_dev_unigrams = []
    dev_unigrams = []

    inputfile = "dev.txt"
    fill_buffer(inputfile, buffer, words)

    process_dev(words, all_dev_unigrams, dev_unigrams, all_dev_bigrams, dev_bigrams)

    # Make necessary calculations for each bigram
    # Sum all interpolated probabilities together with different lambdas
    lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]
    inters = []
    for l in lambdas:
        val = 0
        for b in all_dev_bigrams:
            try:
                m = calculate_mle(b, bigrams.index(tuple(b)), bigrams_count, unigrams_count, unigrams)
            except ValueError:
                if b == ['</s>', '<s>']:
                    m = 1
                else:
                    m = 0
            # Sum all interpolation probabilities
            try:
                u = unigram_laplace(v, len(all_unigrams), unigrams_count[unigrams.index(b[1])])
            except ValueError:
                u = (0 + 1) / (len(all_unigrams) + v + 1)

            inter = (l * m) + ((1 - l) * u)
            if inter != 0:
                val += math.log2(inter)

        inters.append(val)

    # Multiple each value by -1/N and make exp for 2
    lower = 0
    lower_val = 0
    first = -1
    index = 0
    for inte in inters:
        inte *= (-1 / len(all_dev_unigrams))
        inte = math.pow(2, inte)
        if first == -1:
            lower = index
            lower_val = inte
            first = 0
        elif inte < lower_val:
            lower_val = inte
            lower = index
        index += 1
    return lambdas[lower]


def top_bigrams(laplace, v, all_unigrams, bigrams):
    joint_probs = {}
    joint_prob = []
    index = 0
    for b in bigrams:
        pl = (all_unigrams.count(b[0]) + 1) / (len(all_unigrams) + v + 1)
        k = pl * laplace[bigrams.index(b)]
        joint_prob.insert(bigrams.index(b), k)
        joint_probs.setdefault(k, []).append(bigrams.index(b))
        index += 1

    # Sort the bigrams by probabilities from greatest to least
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
            if y[1] == '</s>':
                all_bigrams.insert(len(all_bigrams), ['</s>', '<s>'])

    del all_bigrams[len(all_bigrams) - 1]
    del all_bigrams[len(all_bigrams) - 1]
    del all_bigrams[len(all_bigrams) - 1]

    # Get rid of duplicate bigrams
    bigrams_set = set(map(tuple, all_bigrams))
    for b in bigrams_set:
        bigrams.append(b)

    # Add each bigram to the end of the list
    for b in bigrams:
        bigrams_count.append(all_bigrams.count(list(b)))  # Calculate the frequency of each bigram

    l = get_lambda(v, unigrams_count, all_unigrams, unigrams, bigrams, bigrams_count)
    # Make necessary calculations for each bigram
    for b in bigrams:

        # Calculate the MLE Probability
        m = calculate_mle(b, bigrams.index(b), bigrams_count, unigrams_count, unigrams)
        mle.append(m)

        # Calculate the Laplace Probability
        laplace.append(calculate_laplace(b, v, bigrams.index(b), bigrams_count, unigrams_count, unigrams))

        # Calculate the Interpolated Probability
        inter.append(calculate_inter(l, b, m, v, all_unigrams, unigrams, unigrams_count))

        # Calculate the AD probability
        katz.append(calculate_ad(b, bigrams.index(b), bigrams_count, unigrams_count, unigrams))

    write_bigram(l, bigrams, bigrams_count, mle, laplace, inter, katz)
    return laplace


# Constructs the unigram ml to be written to a file
def unigram(words, unigrams_count, all_unigrams, unigrams):
    # Organize all unigrams into a list for counting
    for x in words:
        for y in x:
            all_unigrams.insert(len(all_unigrams), y)
    # Get rid of duplicate unigrams
    unigram_set = set(all_unigrams)
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

        if len(sys.argv) < 2:
          print("Please specify file to convert")
          sys.exit()

        # Get the filename
        inputFile = sys.argv[1]
        # inputFile = "../inputs/train.txt"

        fill_buffer(inputFile, buffer, words)
        v = unigram(words, unigrams_count, all_unigrams, unigrams)
        laplace = bigram(words, v, all_unigrams, all_bigrams, bigrams_count, bigrams, unigrams_count, unigrams)
        top_bigrams(laplace, v, all_unigrams, bigrams)
