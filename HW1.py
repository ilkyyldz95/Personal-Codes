import sys
import argparse
import numpy as np
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
        # Read without unicode
        rdd = sc.wholeTextFiles(args.input, 20).values().map(lambda x: x.encode('ascii', 'ignore'))
        # Concatenate all texts
        txt = rdd.reduce(lambda x, y: x + y)
        # Get rid of escape characters but do not lose spaces
        F = sc.parallelize([txt]).map(lambda x: x.replace('\n', ' ')).map(lambda x: x.replace('\t', ' '))

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
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        pass

    if args.mode == 'TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value.
        pass

    if args.mode == 'SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase, letter-only string and VAL is a numeric value.
        pass





