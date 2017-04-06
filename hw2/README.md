Parts of Speech Tagger
----------------------

How to run:
    
    Training: python3 train-tagger.py train.txt <transitions.txt> <emissions.txt> <laplace-tag-unigrams.txt>
    FBTagger: python3 freq-tagger.py test.txt emissions.txt <test-fbtagger.txt>
    HMMTagger: python3 hmm-tagger.py test.txt transitions.txt emissions.txt <Smoothing Method 'M' or 'L'> <Outfile>
    
    Evaluator(please keep hmmtagger): python3 evaluator.py test-hmm-mle-tagger.txt test-hmm-laplace-tagger.txt
    **Evaluator shows the accuracy for the two files AND top 10 mismatched with the top three confusion for each tag.
    
    
Analysis:
    
        Top 10 mismatched tags with their top three mismatches

            1. NNP  Mismatched with: UH, POS, NN
            2. NN  Mismatched with: UH, POS, PDT
            3. JJ  Mismatched with: UH, PDT, POS
            4. NNS  Mismatched with: POS, UH, NN
            5. VB  Mismatched with: UH, NN, IN
            6. VBN  Mismatched with: UH, VBD, PRP$
            7. -NONE-  Mismatched with: UH, POS, VBN
            8. VBG  Mismatched with: UH, ., POS
            9. VBD  Mismatched with: VBN, UH, IN
            10. IN  Mismatched with: DT, RP, WDT
            
        Please see evaluator_laplace.txt and evaluator_mle.txt for calculated Precision, Recall and F1 for each POS tag
        in the format per line:
                        tag     precision     recall   f1

Assumptions:
    
        
        Training: emissions.txt:
            tag     word    Pr(w|t) for MLE     Pr(t|w) for MLE     Pr(w|t) Laplace
            
            NOTE: When calculating Pr(t|w), Naive Bayes formula was applied so Pr(t|w) = Pr(t) * Pr(w|t)
        
        FBTagger: Ties and unseen words were tagged with the most frequent tag 'NN' which was calculated during training
        
        HMMTagger: Any unseen tags in the test set were defaulted to a very small decimal
            value for the emissions and transitions probability lookup for MLE. 
            
            *****NOTE: The probability of each tag sequence is the log sum probabilities of each tag from the Viterbi algorithm 
            best paths for that tag. For zero probabilities found a default very small number was used to replace the zero value.
            
            