from __future__ import annotations
from os.path import dirname

import bPlusTree
import lsh

MYDIR = dirname(__file__)  # gives back your directory path


def main():
    query_type = ""
    dictionary = []
    point = 0  # to keep track of which text I am searching into

    # build tree
    print("Preprocessing...")
    # path_of_docs = MYDIR + '/Datasets/corpus20090418'  # + input_theme  # execute path+files or dataset +file of documents
    path_of_docs = MYDIR + '/sample/'  # + input_theme  # execute path+files or dataset +file of documents
    documents = bPlusTree.docs_to_search(path_of_docs)
    for doc in documents[1]:
        file = open(doc, "r", encoding="UTF-8", errors='ignore')  # open file
        words = file.read()  # read file
        raw_dictionary = list(words.split())  # text to list
        doc_dictionary = [bPlusTree.Preprocessing(raw_dictionary), documents[0][point]]  # words preprocessing
        point += 1
        dictionary.append(doc_dictionary)

    tree = bPlusTree.bplustree(dictionary)

    # queries
    while query_type != "r" and query_type != "e" and query_type != "q":
        print("\nType 'r' for range query")
        print("Type 'e' for exact query")
        print("Type 'q' for exit")
        query_type = input("Please select the type of the query:")

        if query_type == "r":
            l_bound = input("Please type the lower bound: ")
            u_bound = input("Please type the upper bound: ")

            if l_bound > u_bound:  # if not given in the correct order, then change the order
                temp = u_bound
                u_bound = l_bound
                l_bound = temp

            sim_docs = []
            search_results = tree.retrieve(l_bound, u_bound, query_type)
            for word in range(len(search_results)):
                for doc_list in range(len(search_results[word][1])):
                    # print(doc_list, search_results[word][1][doc_list])
                    if search_results[word][1][doc_list] not in sim_docs:
                        sim_docs.append(search_results[word][1][doc_list])
            print("sim_docs", sim_docs)
            query_type = ""
        elif query_type == 'e':
            l_bound = input("Please type word you are looking for: ")
            u_bound = l_bound
            sim_docs = tree.retrieve(l_bound, u_bound, query_type)
            query_type = ""
        elif query_type == 'q':
            return

        hash_num = 10  # number of hash functions
        k = 30  # number of buckets (in a row of bucket buckets array)

        # read documents, preprocessing, hashed shingles creation
        hashed_shingles, docs = lsh.create_shingles(MYDIR, sim_docs)
        # create input matrix with 0/1 for hashed shingles and documents
        inp_mtrx = lsh.create_input_matrix(hashed_shingles, docs)
        # create signature matrix given the input matrix
        sign_mtrx = lsh.minHash(inp_mtrx, docs, hash_num)
        for r in range(len(sign_mtrx)):
            print("sign_mtrx:", r, "-->", sign_mtrx[r])
        # create bands given the signature matrix
        bands = lsh.create_bands(sign_mtrx)
        print("bands: ")
        for r in range(len(bands)):
            print(r, "-->", end='')
            for s_r in range(len(bands[r][1])):
                if s_r == 0:
                    print(bands[r][1][s_r])
                else:
                    print("\t", bands[r][1][s_r])
        # create buckets given the bands
        buckets = lsh.create_hash_table(bands, k)
        for r in range(len(buckets)):
            print("buckets:", r, "-->", buckets[r])
        # find candidate pairs
        cand_pairs = lsh.candidate_column_pairs(buckets)
        for r in range(len(cand_pairs)):
            print("cand_pairs:", r, "-->", cand_pairs[r])
        # calculate pairs similarities (Jaccard similarity)
        similarities = lsh.document_similarities(cand_pairs, sign_mtrx, docs)
        # remove dublicates
        similarities = [ii for n, ii in enumerate(similarities) if ii not in similarities[:n]]
        # sort results
        similarities = sorted(similarities, key=lambda s: s[2], reverse=True)
        for r in range(len(similarities)):
            print("similarities:", r, "-->", similarities[r])

if __name__ == "__main__":
    main()
