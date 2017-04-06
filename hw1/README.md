Simple Language Model Builder
-----------------------------

How to run:  
    
    LM-Builder (Runtime ~10 seconds): python3 lm-builder.py <path/to/train.txt>
    Bigram-Query: python3 bigram-query.py <path/to/bigram.lm> <path/to/unigram.lm> <word1> <word2> <smoothing: M, L, I, K>
    Perplexity: python3 perplexity.py <path/to/bigram.lm> <path/to/unigram.lm> <path/to/test.txt>

Methodology:

    <s> and </s> were added in the training and dev data for Part 1
    Performed dumb tokenization specified in the homework pdf, splitting on the spaces and periods
    Calculated probabilities as per the Estimation methods