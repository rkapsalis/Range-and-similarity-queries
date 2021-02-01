from __future__ import annotations
from os.path import dirname
from pandas import DataFrame

import bPlusTree
import lsh
import pandas
import matplotlib.pyplot as plt

# The maximum number of keys each node can hold (branching factor)
order = 4
# number of hash functions
hash_num = 150
# number of buckets (in a row of buckets array)
k = 30
# k for k-shingles
s_k = 5

MYDIR = dirname(__file__)  # gives back your directory path

def main():
    query_type = ""
    dictionary = []
    # to keep track of which text we are searching into
    point = 0

    print("\nPreprocessing...")
    # get directory
    # path_of_docs = MYDIR + '/Datasets/corpus20090418/'
    path_of_docs = MYDIR + '/sample/'

    documents = bPlusTree.docs_to_search(path_of_docs)
    # for every document in directory
    for doc in documents[1]:
        # open file
        file = open(doc, "r", encoding="UTF-8", errors='ignore')
        # read file
        words = file.read()
        # text to list
        raw_dictionary = list(words.split())
        # words preprocessing
        doc_dictionary = [bPlusTree.Preprocessing(raw_dictionary), documents[0][point]]
        point += 1
        dictionary.append(doc_dictionary)
    # create b+ tree
    tree = bPlusTree.bplustree(dictionary, order)

    # queries
    while query_type != "r" and query_type != "e" and query_type != "q":
        print("\nType 'r' for range query")
        print("Type 'e' for exact query")
        print("Type 'q' for exit")
        query_type = input("Please select the type of the query:")

        # sim_docs list contains:
        # - in the preprocessing phase: only the documents names in which the query found
        # - at the beginning of LSH (on each row): · the name of one documentsin which the query found
        #                                          · all the text of the document as one string variable
        sim_docs = []

        # if user select "range query"
        if query_type == "r":
            l_bound = input("Please type the lower bound: ").lower()
            u_bound = input("Please type the upper bound: ").lower()
            # if not given in the correct order, then change the order
            if l_bound > u_bound:
                temp = u_bound
                u_bound = l_bound
                l_bound = temp

            # search_results: list with all the documents for words in thw given range
            # contains duplicates
            search_results = tree.retrieve(l_bound, u_bound, query_type)
            # if non empty
            if search_results:
                # for each row of search_results
                for word in range(len(search_results)):
                    # for each document in document list of this row
                    for doc_list in range(len(search_results[word][1])):
                        # check for duplicates
                        if search_results[word][1][doc_list] not in sim_docs:
                            # append only unique documents in list
                            sim_docs.append(search_results[word][1][doc_list])
            print("sim_docs", sim_docs)
            query_type = ""
        # if user select "exact query"
        elif query_type == 'e':
            l_bound = input("Please type the word you are looking for: ").lower()
            u_bound = l_bound
            search_results = tree.retrieve(l_bound, u_bound, query_type)
            # if non empty
            if search_results:
                # for each row (document name) of search_results
                for i, word in enumerate(search_results):
                    # check for duplicates
                    if word not in sim_docs:
                        # append only unique documents in list
                        sim_docs.append(search_results[i])
            query_type = ""
        elif query_type == 'q':
            return

        # if keywords found only in one document
        if len(sim_docs) == 1:
            print("Only 1 document found. We cannot perform LSH algorithm for similarities.")
            # skip this while loop and print menu
            continue

        if sim_docs:
            for i, text in enumerate(sim_docs):
                for d_row in dictionary:
                    if text == d_row[1]:
                        joined_row = ' '.join(d_row[0])
                        sim_docs[i] = [sim_docs[i], joined_row]

            # read documents, preprocessing, hashed shingles creation
            hashed_shingles, docs = lsh.create_shingles(sim_docs, s_k)
            # create input matrix with 0/1 for hashed shingles and documents
            inp_mtrx = lsh.create_input_matrix(hashed_shingles, docs)
            # create signature matrix given the input matrix
            sign_mtrx = lsh.minHash(inp_mtrx, docs, hash_num)
            # for r in range(len(sign_mtrx)):
            #     print("sign_mtrx:", r, "-->", sign_mtrx[r])

            # create bands given the signature matrix
            bands = lsh.create_bands(sign_mtrx)
            # print("bands: ")
            # for r in range(len(bands)):
            #     print(r, "-->", end='')
            #     for s_r in range(len(bands[r][1])):
            #         if s_r == 0:
            #             print(bands[r][1][s_r])
            #         else:
            #             print("\t", bands[r][1][s_r])

            # create buckets given the bands
            buckets = lsh.create_hash_table(bands, k)
            # for r in range(len(buckets)):
            #     print("buckets:", r, "-->", buckets[r])

            # find candidate pairs
            cand_pairs = lsh.candidate_column_pairs(buckets)
            # for r in range(len(cand_pairs)):
            #     print("cand_pairs:", r, "-->", cand_pairs[r])

            # calculate pairs similarities (Jaccard similarity)
            similarities = lsh.document_similarities(cand_pairs, sign_mtrx, docs)

            # DataFrame for similarities
            similarities_df = DataFrame(similarities, columns=['First_document', 'Second_document', 'Similarity_with_signatures'])
            # option to print all rows and columns of DataFrame
            pandas.set_option('display.max_rows', None)
            pandas.set_option('display.max_columns', None)
            pandas.set_option('display.width', None)

            # uncomment the above to print DataFrame with one more column (Similarity_with_words) for experiments

            # word_js = []
            # for r in range(len(similarities)):
            #     for d_r, d_i in enumerate(dictionary):
            #         if dictionary[d_r][1] == similarities[r][0]:
            #             l1_list = dictionary[d_r][0]
            #         if dictionary[d_r][1] == similarities[r][1]:
            #             l2_list = dictionary[d_r][0]
            #     word_js.append(lsh.jaccard_similarity(l1_list, l2_list))
            # similarities_df.insert(3, 'Similarity_with_words', word_js, True)


            print(similarities_df)


if __name__ == "__main__":
    main()
