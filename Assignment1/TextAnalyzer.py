import sys
import argparse
import numpy as np
import math
from pyspark import SparkContext

'''Ilkay YILDIZ
    EECE 5698
    HW1'''


def toLowerCase(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()


def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text Analysis through TFIDF computation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', help='Mode of operation', choices=['TF', 'IDF', 'TFIDF', 'SIM', 'TOP'])
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master', default="local[20]", help="Spark Master")
    parser.add_argument('--idfvalues', type=str, default="idf",
                        help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other', type=str, help='Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()

    sc = SparkContext(args.master, 'Text Analysis')

    if args.mode == 'TF':
        # READS TEXTS
        # Read without unicode
        F = sc.textFile(args.input, 20).map(lambda x: x.encode('ascii', 'ignore'))
        # Separate each word and turn to lower case
        sep_word = F.flatMap(lambda line: line.lower().split())
        # Keep alphabetic characters
        alpha_word = sep_word.map(stripNonAlpha)
        # Eliminate empty strings
        alpha_word = alpha_word.filter(lambda x: x != '')

        # Append 1s and count the number of occurrences of the same key
        al_tup = alpha_word.map(lambda x: (x, 1))
        dist = al_tup.reduceByKey(lambda x, y: x + y)
        # Store result in file args.output
        dist.saveAsTextFile(args.output)

    if args.mode == 'TOP':
        # READS WORD-TF PAIRS
        # Read all text files at args.input, (filename, (term,val)) as a pairRDD
        # Read without unicode
        rdd = sc.wholeTextFiles(args.input, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt = rdd.reduce(lambda x, y: x + y)
        # Get rid of escape characters
        F = sc.parallelize([txt]).map(lambda x: x.replace('\n', '')).map(lambda x: x.replace('\t', ''))
        # Separate tuples
        pairs = F.flatMap(lambda x: x.split(')')).filter(lambda x: x != '').map(lambda x: eval(x + ')'))

        # Sort in descending order with respect to values and take top 20
        sorted20 = pairs.takeOrdered(20, lambda pair: -pair[1])
        # Return
        sc.parallelize(sorted20, 1).saveAsTextFile(args.output)

    if args.mode == 'IDF':
        # READS TEXTS
        # Read all text files at args.input
        # Read without unicode
        rdd = sc.wholeTextFiles(args.input, 20).mapValues(lambda x: x.encode('ascii', 'ignore'))
        # Get number of files in corpus
        k = rdd.count()
        # Get rid of escape characters
        F = rdd.mapValues(
            lambda x: x.replace('\n', ' ').replace('\t', ' ').replace('.', ' ').replace('-', ' ').replace(',', ' '))
        # Create source-word pairs and turn to lower case
        sep_word = F.flatMapValues(lambda text: text.lower().split())
        # Keep alphabetic characters
        alpha_word = sep_word.mapValues(stripNonAlpha)

        # Eliminate empty strings and find distinct (word,source) pairs
        all_words = alpha_word.filter(lambda x: x[1] != '').distinct().map(lambda pair: (pair[1], pair[0]))
        # Map second element to count and sum all counts
        word_counts = all_words.mapValues(lambda val: 1).reduceByKey(lambda count1, count2: count1 + count2)
        # Calculate IDF
        word_idf = word_counts.mapValues(lambda count: math.log(1.0 * k / count))

        # Return
        word_idf.saveAsTextFile(args.output)

    if args.mode == 'TFIDF':
        # READS WORD-TF and WORD-IDF PAIRS
        # Read  TF scores from file args.input, (filename, (term,val)) as a pairRDD
        # Read without unicode
        rdd = sc.wholeTextFiles(args.input, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt = rdd.reduce(lambda x, y: x + y)
        # Get rid of escape characters
        F = sc.parallelize([txt]).map(lambda x: x.replace('\n', '')).map(lambda x: x.replace('\t', ''))
        # Separate tuples
        tf_pairs = F.flatMap(lambda x: x.split(')')).filter(lambda x: x != '').map(lambda x: eval(x + ')'))

        # Read IDF scores from file args.idfvalues
        rdd_idf = sc.wholeTextFiles(args.idfvalues, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt_idf = rdd_idf.reduce(lambda x, y: x + y)
        # Get rid of escape characters
        F_idf = sc.parallelize([txt_idf]).map(lambda x: x.replace('\n', '')).map(lambda x: x.replace('\t', ''))
        # Separate tuples
        idf_pairs = F_idf.flatMap(lambda x: x.split(')')).filter(lambda x: x != '').map(lambda x: eval(x + ')'))

        # Compute TDIDF for each word
        join_scores = tf_pairs.join(idf_pairs)  # 1st score is tf, 2nd score is idf
        tdidf_pairs = join_scores.mapValues(lambda scores: (scores[0]) * (scores[1]))
        # Sort wrt TDIDF
        tdidf_pairs = tdidf_pairs.sortBy(lambda pair: -pair[1])

        # Return
        tdidf_pairs.saveAsTextFile(args.output)

    if args.mode == 'SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase, letter-only string and VAL is a numeric value.

        # READS WORD-TFIDF
        # Read TFIDF scores from file args.input, (filename, (term,val)) as a pairRDD
        # Read without unicode
        rdd = sc.wholeTextFiles(args.input, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt = rdd.reduce(lambda x, y: x + y)
        # Get rid of escape characters
        F = sc.parallelize([txt]).flatMap(lambda x: x.split('\n')).filter(lambda x: x != '').map(lambda x: eval(x))

        # Read TFIDF scores from file args.other
        # Read without unicode
        rdd2 = sc.wholeTextFiles(args.other, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt2 = rdd2.reduce(lambda x, y: x + y)
        # Get rid of escape characters
        F2 = sc.parallelize([txt2]).flatMap(lambda x: x.split('\n')).filter(lambda x: x != '').map(lambda x: eval(x))

        # Compute cosine similarity
        join_scores = F.join(F2)  # 1st score is tf for F, 2nd score is tfidf for F'
        # Products of scores of matching words in F and F2
        prods = join_scores.mapValues(lambda scores: (scores[0]) * (scores[1]))
        numerator = prods.values().reduce(lambda score1, score2: score1 + score2)
        # Square of scores in F and F2
        Fsq = F.mapValues(lambda score: score ** 2)
        F2sq = F2.mapValues(lambda score: score ** 2)
        # Sum of Square of scores in F and F2
        FsumSq = Fsq.values().reduce(lambda score1, score2: score1 + score2)
        F2sumSq = F2sq.values().reduce(lambda score1, score2: score1 + score2)
        # Multiply sum of squares and take sqrt
        denominator = math.sqrt(FsumSq * F2sumSq)
        cosSim = numerator / denominator

        # Return
        with open(args.output, 'w') as f:
            f.write(str(cosSim))
        f.close()




