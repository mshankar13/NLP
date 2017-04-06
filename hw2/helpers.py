import sys
import math
import operator


def write_fb_tagger(tagged_sentences, filename):
    """Writes the results for the frequency based tagger containing the tagged sentences
        Args:
            :param tagged_sentences the tagged sentences from the resulting frequency based analysis
            :param filename name of the file to write the data to"""
    with open(filename, 'w') as file:
        index = 0
        for tagged_sentence in tagged_sentences:
            file.write('{} '.format(index))
            index = index + 1
            for word in range(len(tagged_sentence) - 1):
                file.write('{} '.format(tagged_sentence[word]))
            file.write('{}\n'.format(tagged_sentence[len(tagged_sentence) - 1]))
    return


def write_tag_unigrams(unigram_tag_dict, filename):
    """Writes the unigram probabilities for each tag seen in training
        Args:
            :param unigram_tag_dict the dictionary containing the tag and probability represented as a key value pair
            :param filename name of the file to write the data to"""
    with open(filename, 'w') as file:
        for unigram in unigram_tag_dict:
            file.write('{}\t{}\n'.format(unigram, unigram_tag_dict[unigram]))
    return


def write_emissions(mle_dict1, mle_dict2, laplace_dict, filename):
    """Writes the mle and laplace emissions probabilities for a word given a tag and mle for a tag given a word
        Args:
            :param mle_dict1 the mle probabilities for a word given a tag
            :param mle_dict2 the mle probabilities for a tag given a word
            :param laplace_dict laplace probabilities for a word given a tag
            :param filename name of the file to write the data to"""
    with open(filename, 'w') as file:
        for mle in mle_dict1:
            tag_pair = list(mle)
            file.write('{}\t{}\t{}\t{}\t{}\n'.format(tag_pair[1], tag_pair[0], mle_dict1[mle], mle_dict2[mle],
                                                     laplace_dict[mle]))
    return


def write_transitions(mle_dict, filename):
    """Writes the mle transitions probabilities for a tag given another tag
        Args:
            :param mle_dict the mle probabilities for a tag given another tag
            :param filename name of the file to write the data to"""
    with open(filename, 'w') as file:
        for mle in mle_dict:
            tag_pair = list(mle)
            file.write('{}\t{}\t{}\n'.format(tag_pair[0], tag_pair[1], mle_dict[mle]))
    return


def write_hmm_tagger(best_probs, tagged_sentences, sentences_tags, filename):
    """Writes the tagged sentences from the applied HMM Tagger Viterbi Algorithm
        Args:
            :param tagged_sentences sentences tagged based on the results of the algorithm
            :param filename name of the file to write the data to"""
    with open(filename, 'w') as file:
        j = 0
        for sentence in tagged_sentences:
            i = 0
            file.write('{} '.format(best_probs[j]))
            for word in sentence:
                if i != len(sentence) - 1:
                    file.write('{}/{} '.format(word, sentences_tags[j][i]))
                else:
                    file.write('{}/{}\n'.format(sentence[i], sentences_tags[j][i]))
                i = i + 1
            j = j + 1
    return


def calculate_mle(specific_element, total_elements):
    """Calculates the mle probability for a given a specific element and the total count of all the elements
        Args:
            :param specific_element specific element count
            :param total_elements total number of all elements in the list

        :return the mle probability of the specific element"""
    return specific_element / total_elements


def calculate_mle_helper1(tag_frequencies, tags_pairs_frequencies):
    """ Helper to calculate mle probabilities by taking in the bigrams and frequencies of the w - 1 argument
        Intended for calculating the mle of a word given a tag
        Args:
            :param tag_frequencies the count of each tag in training sorted into a dictionary
            :param tags_pairs_frequencies the count of each pair of tags seen in training sorted into a dictionary

        :return mle_dict the mles for the tag bigrams returned as a dictionary"""
    mle_dict = {}
    for tag_pair in tags_pairs_frequencies:
        mle_dict[tag_pair] = calculate_mle(tags_pairs_frequencies[tag_pair], tag_frequencies[tag_pair[1]])
    return mle_dict


def calculate_mle_helper2(mle_dict1, tag_frequencies, all_tags_size):
    """ Helper to calculate mle probabilities by taking in the bigrams and frequencies of the w - 1 argument
            Intended for calculating the mle of a tag given a word
            Args:
                :param mle_dict1 the mles for each tag seen in training
                :param tag_frequencies the count of each tag in training sorted into a dictionary
                :param tags_pairs_frequencies the count of each pair of tags seen in training sorted into a dictionary

            :return mle_dict the mles for the tag bigrams returned as a dictionary"""
    mle_dict2 = {}
    for mle in mle_dict1:
        tag_pair = list(mle)
        mle_dict2[mle] = mle_dict1[mle] * (tag_frequencies[tag_pair[1]] / all_tags_size)
    return mle_dict2


def calculate_unigram_laplace(tag_frequencies, total_tokens, V):
    """Calculates the unigram laplace probabilities for tags seen in training
        Args:
            :param tag_frequencies the count of each tag in training sorted into a dictionary
            :param total_tokens the total count of all tags seen in training
            :param V the number of unique tags seen in training

        :return laplace_dict a dictionary of the laplace probabilities hashed by the tag"""
    laplace_dict = {}
    for tag in tag_frequencies:
        laplace_dict[tag] = (tag_frequencies[tag] + 1) / (total_tokens + V + 1)
    return laplace_dict


def calculate_laplace(tag_pair_count, prev_tag_count, V):
    """Calculates the laplace probability for a word given a tag
        Args:
            :param tag_pair_count the count of a pair of tags seen in training
            :param prev_tag_count the count of the previous tag
            :param V the number of unique tags seen in training

        :return laplace probability of a specific tag pair"""
    return (tag_pair_count + 1) / (prev_tag_count + V + 1)


def calculate_laplace_helper(tag_frequencies, tags_pairs_frequencies, V):
    """Helper to calculate the laplace probabilities by taking in the bigrams and frequencies of the t-1 argument
        Intended for calculating the laplace of a word given a tag
        Args:
            :param tag_frequencies the count of each tag in training sorted into a dictionary
            :param tags_pairs_frequencies the count of each tag pair seen in training sorted into a dictionary
            :param V the number of unique tags seen in training"""
    laplace_dict = {}
    for tag_pair in tags_pairs_frequencies:
        laplace_dict[tag_pair] = calculate_laplace(tags_pairs_frequencies[tag_pair],
                                                   tag_frequencies[tag_pair[1]], V)
    return laplace_dict


def hash_unique_tags_pairs(all_tag_pairs):
    """Processes the list of all tag combinations and returns a list containing the unique pairs
        Args:
            :param all_tag_pairs seen in training"""
    unique_tag_pairs_dict = {}
    unique_tag_pairs = set(map(tuple, all_tag_pairs))
    for unique_tag_pair in unique_tag_pairs:
        unique_tag_pairs_dict[unique_tag_pair] = all_tag_pairs.count(list(unique_tag_pair))
    return unique_tag_pairs_dict


def get_tags_pairs(sentences_tags):
    """Processes the training sentence tags and gets the tag pairs for each sentence
        Args:
            :param sentences_tags the sentence tags

        :return the tag pairs for each sentence seen in training"""
    sentences_tag_pairs = []
    for sentence_tags in sentences_tags:
        sentences_tag_pairs.append([sentence_tags[y: y + 2] for y in range(len(sentence_tags) - 1)])
    return sentences_tag_pairs


def get_all_tags_pairs(sentences_tags):
    """ Processes each sentences' tags and returns all the tag pairs in one list for analysis
        Args:
            :param sentences_tags the sentences tags seen in training

        :return every tag pair including repeats seen in training"""
    sentences_tag_pairs = get_tags_pairs(sentences_tags)
    all_tags_pairs = []

    for sentences_tag_pair in sentences_tag_pairs:
        for tag_pair in sentences_tag_pair:
            all_tags_pairs.append(tag_pair)
    return all_tags_pairs


def get_unique_tags(all_elements):
    """Gets the unique list of elements
        Args:
            :param all_tags

        :return the unique elements from the list"""
    return set(all_elements)


def get_all_tags(all_sentences_tags):
    """Gets all the tags from each sentence and stores them in a list including repeats
        Args:
            :param all_sentences_tags all the sentence tags

        :return all_tags the list of all sentence tags"""
    all_tags = []
    for sentence_tags in all_sentences_tags:
        for tag in sentence_tags:
            all_tags.append(tag)
    return all_tags


def get_frequencies(all_elements):
    """Gets the frequencies of elements in the list
        Args:
            :param all_elements to count the frequency of

        :return all_tag_frequencies all the frequencies of each unique elements returned as a dictionary"""
    all_tag_frequencies = {}
    unique_tags = get_unique_tags(all_elements)
    for tag in unique_tags:
        all_tag_frequencies[tag] = all_elements.count(tag)
    return all_tag_frequencies


def get_all_frequencies(sentences_tags):
    """Helper to get the frequencies of all elements in a list
        Args:
            :param sentences_tags list to get all the frequencies of the unique elements

        :return frequencies as a dictionary where the key is the element"""
    all_tags = get_all_tags(sentences_tags)
    return get_frequencies(all_tags), len(all_tags)


def get_sentences_tags(sentences):
    """Gets the sentence tags from a list of sentences with words tagged
        Args:
            :param sentences to parse for tags

        :return sentence_tags the tags for each sentence managed in a list
        :return word_tag_pairs the words with their respective tags for each sentence
        :return all_words all the words seen in the text"""
    sentences_tags = []
    word_tag_pairs = []
    all_words = []
    for sentence in sentences:
        split_sentence = sentence.split(" ")
        first_word = split_sentence[0].split("\t")
        del split_sentence[0]
        split_sentence.insert(0, first_word[1])
        tags = ["<s>"]
        for word in split_sentence:
            word_tag_pair = word.rsplit("/", 1)
            all_words.append(word_tag_pair[0])
            tags.append(word_tag_pair[1])
            if '\n' in word_tag_pair[1]:
                remove_newline = word_tag_pair[1].split('\n')
                del word_tag_pair[1]
                word_tag_pair.append(remove_newline[0])
            word_tag_pairs.append(word_tag_pair)
        del tags[len(tags) - 1]
        tags.append('.')
        tags.append('</s>')
        sentences_tags.append(tags)
    return sentences_tags, word_tag_pairs, all_words


def transitions_helper(sentences_tags):
    """Helper to invoke the transitions calculations
        Args:
            :param sentences_tags the sentence tags to be analyzed

        :return mle_dict the dictionary of mle probabilities for each pair of sentence tags"""
    all_tag_pairs = get_all_tags_pairs(sentences_tags)
    tags_pairs_frequencies = hash_unique_tags_pairs(all_tag_pairs)
    tag_frequencies, all_tags_len = get_all_frequencies(sentences_tags)

    mle_dict = calculate_mle_helper1(tag_frequencies, tags_pairs_frequencies)
    return mle_dict


def get_most_frequent_tag(tag_frequencies):
    """Gets the tag seen most frequently in training
        Args:
            :param tag_frequencies the list of tags with respective frequencies to analyze"""
    max = 0
    max_tag = ''
    for tag in tag_frequencies:
        if tag_frequencies[tag] > max:
            max_tag = tag
            max = tag_frequencies[tag]
    print(max_tag)
    return


def emissions_helper(tag_word_pairs, sentences_tags, all_words):
    """Helper function to invoke the emissions calculations on the training set
        Args:
            :param tag_word_pairs the tag word pairs seen in training
            :param sentences_tags all the sentence tags seen in training
            :param all_words all the words seen in training including repeats

        :return mle_dict1 the mle probabilities for a word given a tag
        :return mle_dict2 the mle probabilities for a tag given a word
        :return laplace_dict the laplace probabilities for a word given a tag
        :return unigram_laplace the unigram laplace probabilities for each tag"""
    word_tag_frequencies = hash_unique_tags_pairs(tag_word_pairs)
    tag_frequencies, all_tags_len = get_all_frequencies(sentences_tags)
    get_most_frequent_tag(tag_frequencies)

    mle_dict1 = calculate_mle_helper1(tag_frequencies, word_tag_frequencies)
    mle_dict2 = calculate_mle_helper2(mle_dict1, tag_frequencies, all_tags_len)
    laplace_dict = calculate_laplace_helper(tag_frequencies, word_tag_frequencies, len(all_words))
    unigram_laplace = calculate_unigram_laplace(tag_frequencies, all_tags_len, len(all_words))
    return mle_dict1, mle_dict2, laplace_dict, unigram_laplace


def transitions_emissions_evaluator(sentences):
    """ Processes each line from the input training file and computes the mle probabilities
        for each unique set of tags. Inserts the start and stop tags for sentences.
        Splits each sentence by spaces and splits by '/' to get the second element, containing the tag.
        Calls the functions to carry out transitions and emissions methodology

        Args:
            param1 (str): Line from the input file to analyze

        :return mle_dict1 the mle probabilities for the transitions
        :return mle_dict2 the mle probabilities for the emissions word given a tag
        :return mle_dict3 the mle probabilities for the emissions tag given a word
        :return unigram_laplace the unigram laplace probabilities for each tag
    """
    sentences_tags, word_tag_pairs, all_words = get_sentences_tags(sentences)
    mle_dict1 = transitions_helper(sentences_tags)
    mle_dict2, mle_dict3, laplace_dict, unigram_laplace = emissions_helper(word_tag_pairs, sentences_tags, all_words)
    return mle_dict1, mle_dict2, mle_dict3, laplace_dict, unigram_laplace


def filter_sentences(sentences):
    """Function to filter each sentence and clean each word into a list fo words and the corresponding tags in another list
        Args:
            :param sentences the raw sentences to be cleaned

        :return new_sentence_list the new sentence broken into a list of a list of words
        :return sentence_tags_list the tags broken into a list of a list of tags corresponding to the sentence list"""
    new_sentence_list = []
    sentence_tags_list = []
    for sentence in sentences:
        sentence = sentence.replace('\n', '')
        split_sentence = sentence.split(' ')
        first_word = split_sentence[0].split("\t")
        del split_sentence[0]
        if len(first_word) != 1:
            split_sentence.insert(0, first_word[1])
        cleaned_sentence = []
        sentence_tags = []
        for word in split_sentence:
            tag_word_pair = word.rsplit("/", 1)
            cleaned_sentence.append(tag_word_pair[0])
            sentence_tags.append(tag_word_pair[1])
        new_sentence_list.append(cleaned_sentence)
        sentence_tags_list.append(sentence_tags)
    return new_sentence_list, sentence_tags_list


def evaluate_result(sentences):
    predicted = []
    actual = []
    for sentence in sentences:
        sentence = sentence.replace('\n', '')
        split_sentence = sentence.split(' ')
        first_word = split_sentence[0].split("\t")
        del split_sentence[0]
        predicted_tags = []
        actual_tags = []
        for word in split_sentence:
            part = word.split('/')
            predicted.append(part[1])
            actual.append(part[2])
        #predicted.append(predicted_tags)
        #actual.append(actual_tags)
    return predicted, actual


def evaluate_test_sentences(emissions_dict, sentences):
    """ Looks for the most probable tag given a word. If the word is unseen in training the MLE is recalculated
        Args:
            :param emissions_dict the emissions data parsed from the emissions file
            :param sentences all the sentences to analyze using emissions data

        :return tagged_sentences the tagged sentences based on the emissions data applied on the test set"""
    tagged_sentences = []
    for sentence in sentences:
        tagged_words = []
        for word in sentence:
            word_tag = "NN"
            list = [key for key in emissions_dict.keys() if word in key[1]]
            max = 0
            for key in list:
                if emissions_dict[key][1] > max:
                    max = emissions_dict[key][1]
                    word_tag = key[0]
                elif emissions_dict[key] == max:
                    word_tag = "NN"
            tagged_words.append(word + "/" + word_tag)
        tagged_sentences.append(tagged_words)
    return tagged_sentences


def reformat_emissions(emissions, prob):
    """Turns the list of emissions strings into a dict where i = tag and j = word
        Args:
            :param emissions list to be converted

        :return a 2D array of the emissions data"""

    # emissions_matrix = [[0 for i in range(len(emissions))] for i in range(len(emissions))]
    emissions_dict = {}
    all_tags = []
    index = 4
    if prob == 'M':
        index = 2

    for emission in emissions:
        emission = emission.split('\t')
        emission[len(emission) - 1].replace('\n', '')
        emissions_dict[emission[0], emission[1]] = float(emission[index])
        all_tags.append(emission[0])
    return emissions_dict, all_tags


def reformat_transitions(transitions):
    """Hashes the transitions probabilities into an easy access dictionary
        Args:
            :param transitions raw transitions info

        :return transitions_dict easy to manage form"""
    transitions_dict = {}

    for transition in transitions:
        transition = transition.split('\t')
        transition[len(transition) - 1].replace('\n', '')
        transitions_dict[transition[0], transition[1]] = float(transition[2])
    return transitions_dict


def check_emissions(emissions, first_word, tag):
    """Checks to see if the emission probability exists
        Args:
            :param emissions probabilities from training
            :param first_word to search for
            :param tag to search for

        :return b the emission probability"""
    if (tag, first_word) in emissions.keys():
        b = emissions[tag, first_word]
    else:
        b = 0.000000000000000000000000000001
    return b


def get_value(transitions, emissions, tag, index, first_word):
    """Tries to get the word tag pair given the transitions and emissions probabilities
        Args:
            :param transitions probabilities from training
            :param emissions probabilities from training
            :param tag to be analyzed
            :param index
            :param first_word to be analyzed

        :return the product of the two probabilities searched"""
    if ('<s>', tag) in transitions.keys():
        transition = transitions['<s>', tag]
        emission = check_emissions(emissions, first_word, tag)
    else:
        transition = 0.000000000000000000000000000001
        emission = check_emissions(emissions, first_word, tag)
    return transition * emission


def get_max(sentence_tags, i, j, best_path_scores, transitions, emissions, word):
    """Gets the maximum index and value for a given tag and word. Applies the emissions and transitions probabilities
        to calculate the best tag.
        Args:
            :param sentence_tags the sentence tags to iterate over to check for the best one
            :param i index of the word
            :param j index of the tag
            :param best_path_scores scores of the best path
            :param transitions probabilities from training
            :param emissions probabilities from training
            :param word to be analyzed

        :return max_value the max value
        :return max_k the max value index
        """
    k = 0
    max_value = 0
    max_k = 0

    transition = 0.000000000000000000000000000001
    emission = 0.000000000000000000000000000001
    for tag in sentence_tags:
        if (tag, sentence_tags[j]) in transitions.keys():
            transition = transitions[tag, sentence_tags[j]]
            if (sentence_tags[j], word) in emissions.keys():
                emission = emissions[sentence_tags[j], word]
        else:
            if (sentence_tags[j], word) in emissions.keys():
                emission = emissions[sentence_tags[j], word]

        temp = best_path_scores[i - 1][k] + math.log2(transition) + math.log2(emission)
        if max_value == 0:
            max_value = temp
            max_k = k
        if temp > max_value:
            max_value = temp
            max_k = k
        k = k + 1
    return max_value, max_k


def viterbi(sentence, sentence_tags, emissions, transitions):
    """Evaluates a given sentence and returns the best tags for each word in the sentence using the viterbi algorithm
        Args:
            :param sentence to evaluate
            :param unique_tags
            :param emissions probabilities from the training set
            :param transitions probabilities from the training set

        :return best_tags the best tags for each word in the sentence"""
    best_path_scores = [[0 for i in range(len(sentence_tags))] for i in range(len(sentence))]
    back_pointers = [[0 for i in range(len(sentence_tags))] for i in range(len(sentence))]
    n = len(sentence)

    # Initialization
    index = 0
    for tag in sentence_tags:
        best_path_scores[0][index] = get_value(transitions, emissions, tag, index, sentence[0])
        back_pointers[0][index] = 0
        index = index + 1

    # Forward pass
    i = 1
    for word in sentence[1:]:
        j = 0
        for tag in sentence_tags:
            max_value, max_index = get_max(sentence_tags, i, j, best_path_scores, transitions, emissions, word)
            best_path_scores[i][j] = max_value
            back_pointers[i][j] = max_index
            j = j + 1
        i = i + 1

    best_tag_indicies = n * [0]
    best_prob = 0
    k = 0
    temp = 0
    for tag in sentence_tags:
        if temp == 0:
            temp = best_path_scores[n - 1][k]
            best_tag_indicies[n - 1] = k
        if temp < best_path_scores[n - 1][k]:
            temp = best_path_scores[n - 1][k]
            best_tag_indicies[n - 1] = k
        k = k + 1
    best_prob += temp
    k = n - 2
    while k > -1:
        best_tag_indicies[k] = back_pointers[k + 1][best_tag_indicies[k + 1]]
        best_prob += best_path_scores[k + 1][best_tag_indicies[k + 1]]
        k = k - 1

    best_tags = []
    for index in best_tag_indicies:
        best_tags.append(sentence_tags[index])

    return best_tags, best_prob


def evaluate_sentences_hmm(filtered_sentences, sentences_tags, emissions, transitions):
    """Evaluates each sentence in the test set and calls the viterbi function to predict the best tags for each word
        in each sentence
        Args:
            :param filtered_sentences the cleaned sentences from the test set
            :param sentences_tags the sentence tags used to predict which tag belongs to the given word
            :param emissions probabilities depending on user input for MLE or Laplace
            :param transitions probabilities

        :return new_tagged_sentences based on the viterbi algorithm"""
    best_probs = []
    index = 0
    new_tagged_sentences = []
    unique_tags = list(get_unique_tags(sentences_tags))
    for sentence in filtered_sentences:
        best_tags, best_prob = viterbi(sentence, unique_tags, emissions, transitions)
        best_probs.append(best_prob)
        i = 0
        tagged_sentence = []
        for word in sentence:
            tagged_sentence.append(word + '/' + best_tags[i])
            i = i + 1
        new_tagged_sentences.append(tagged_sentence)
        index = index + 1
    return new_tagged_sentences, best_probs


def parse_emissions_txt(emissions_list):
    """Parses the emissions.txt file and returns a dictionary of the entries for easy access
        Args:
            :param emissions_list the list of emissions entries to parse

        :return emissions_dict"""
    emissions_dict = {}
    for emission in emissions_list:
        emission = emission.replace('\n', '')
        split_emission = emission.split('\t')
        emissions_dict[tuple([split_emission[0], split_emission[1]])] = \
            tuple([float(split_emission[2]), float(split_emission[3]), float(split_emission[4])])
    return emissions_dict


def get_top_three(words_dict_list):
    top_three_list = {}
    for word in set(words_dict_list):
        top_three_list[word] = 0
    for word in words_dict_list:
        top_three_list[word] = top_three_list[word] + 1

    sorted_tags = list(sorted(top_three_list.items(), key=operator.itemgetter(1)))
    sorted_tags.reverse()

    top_three = [sorted_tags[0], sorted_tags[1], sorted_tags[2]]
    return top_three


def get_top_ten(tag_dict, words_dict):
    sorted_tags = list(sorted(tag_dict.items(), key=operator.itemgetter(1)))
    sorted_tags.reverse()
    i = 0
    print('Top 10 mismatched tags\n')
    for i in range(10):
        tup = list(sorted_tags[i])
        print(tup[0], ' Mismatched with: ')
        j = 0
        top_three_list = get_top_three(words_dict[tup[0]])
        print(top_three_list)
    return


def calculate_accuracy(actual_tags, predicted_tags):
    correct_count = 0
    index = 0
    unique_tags = list(get_unique_tags(predicted_tags))
    tag_dict = {}
    words = {}
    for tag in actual_tags:
        tag_dict[tag] = 0
        words[tag] = []

    for tag in actual_tags:
        if tag == predicted_tags[index]:
            correct_count = correct_count + 1
        else:
            tag_dict[tag] = tag_dict[tag] + 1
            words[tag].append(predicted_tags[index])
        index = index + 1
    return correct_count / len(actual_tags) , tag_dict, words


def calculate_precision(pos_tag, actual_tags, predicted_tags):
    correct_count = 0
    index = 0
    for tag in actual_tags:
        if (tag == predicted_tags[index]) and (pos_tag == tag):
            correct_count = correct_count + 1
        index = index + 1
    if predicted_tags.count(pos_tag) == 0:
        return 0

    return (correct_count / predicted_tags.count(pos_tag))


def calculate_recall(pos_tag, actual_tags, predicted_tags):
    correct_count = 0
    index = 0
    for tag in actual_tags:
        if (tag == predicted_tags[index]) and (pos_tag == tag):
            correct_count = correct_count + 1
        index = index + 1
    if predicted_tags.count(pos_tag) == 0:
        return 0

    return (correct_count / actual_tags.count(pos_tag))


def calculate_f1(p, r):
    if p  == 0 and r == 0:
        return 0
    else:
        return (2 * p * r) / (p + r)


def read_file(inputfile):
    """Reads the lines for the file to be processed
        Args:
            :param inputfile the filename of the file to read

        :return sentences the list of lines read from the file"""
    sentences = []
    try:
        with open(inputfile) as fileObj:
            sentences = fileObj.readlines()
        return sentences
    except IOError:
        print(inputfile, ' not found')
        sys.exit(-1)
