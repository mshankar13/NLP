import sys


def recalculate_katz(word1, word2, unigramFile, bigramFile):

    return


def get_top_ten(other_words, index):
    other_words.sort(key = lambda i: i[index], reverse = True)
    counter = 10
    print("Top Ten Words")
    for x in other_words:
        print('{}{}\t{}\t{}'.format(str(counter), ": ", x[1], x[index].replace("\n", "")))
        counter -= 1
        if counter == 0:
            break
    return


def process_line(line):
    if "\n" in line:
        line.replace("\n", "")
    return line


def process_input(self, top_ten_words, top_ten_prob):
    bigramFile = self[1]
    unigramFile = self[2]
    word1 = self[3]
    word2 = self[4]
    smoothing = self[5]

    try:
        fileObj = open(bigramFile, 'r')
        fileObj2 = open(unigramFile, 'r')
        fileObj.close()
        fileObj2.close()
    except FileNotFoundError:
        print("Bigram OR Unigram LM file does not exist")
        sys.exit()
    else:
        result = ['0', '0', '0', '0', '0', '0', '0']
        other_words = []
        with open(bigramFile, 'r') as fileObj:
            # Loop through and read the file line by line
            while True:
                line = fileObj.readline()
                if not line:
                    break

                split_line = process_line(line).split(", ")
                if word1 == split_line[0] and word2 == split_line[1]:
                    result.append(split_line)
                elif word1 == split_line[0]:
                    other_words.append(split_line)
        fileObj.close()

        if smoothing is "M":
            print(result[3])
            get_top_ten(other_words, 3)
        elif smoothing is "L":
            print(result[4])
            get_top_ten(other_words, 4)
        elif smoothing is "I":
            print(result[5])
            get_top_ten(other_words, 5)
        elif smoothing is "K":
            if result[6] == 0:
                recalculate_katz(word1, word2, unigramFile, bigramFile)
            else:
                print(result[6].replace("\n", ""))
            get_top_ten(other_words, 6)

    return


class BQ:
    if __name__ == "__main__":
        top_ten_words = []
        top_ten_prob = []

        if len(sys.argv) < 6:
           print("Insufficient Arguments")
           sys.exit()

        process_input(sys.argv, top_ten_words, top_ten_prob)
