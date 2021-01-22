import math
import binascii
import random
import re
from os.path import dirname
import os
from nltk.corpus import stopwords

# get documents directory


# read documents from directory,
# preprocessing on text (words to lowercase, removing punctuation marks, split, remove stopwords)
# create shingles
# return:
# 1. zip_list: list of which each row contains hashed shingle and the names of the documents in which it appears
# 2. docs: the names of all documents
def create_shingles(MYDIR, sim_docs):
    k = 5
    docs = [""]
    zip_list = []
    shingles = []
    hashed_shingles = []
    path_of_docs = MYDIR + '/sample/'
    print("Preprocessing...")

    # for every file
    for name in sim_docs:
        print("name:", name)
        doc = os.path.join(path_of_docs, name)
        # read file
        file = open(doc, "r", encoding="UTF-8", errors='ignore')
        # put all words in one line
        file = file.read().replace('\n', ' ').replace('\r', ' ')
        # remove punctuation
        new_s = re.sub(r'[^\w\s]', ' ', file.lower())

        set(stopwords.words("english"))
        # remove stopwords
        filteredContents = [word for word in new_s.split() if word not in stopwords.words('english')]
        filteredContents = ' '.join(filteredContents)
        # store file names in array
        docs.append(name)
        for i in range(len(filteredContents) - k + 1):
            # create singles
            shingle = filteredContents[i:i + k]
            # create hashed shingle
            hashed = binascii.crc32(shingle.encode('utf8')) & 0xffffffff  # 32bit
            if hashed not in hashed_shingles:
                # store unique hashed shingles
                hashed_shingles.append(hashed)
            # if the shingle is not in the list
            if not any((x[0] == hashed) for x in zip_list):
                # store shingle in array
                shingles.append(shingle)
                # store shingle and document name in array
                zip_list.append([hashed, [name]])
            else:
                # store name of document in the corresponding position
                position = [(i, el.index(hashed)) for i, el in enumerate(zip_list) if hashed == el[0]]
                if name not in zip_list[position[0][0]][1]:
                    zip_list[position[0][0]][1].append(name)

    for r in range(len(zip_list)):
        print("zip_list: ", zip_list[r])
    return zip_list, docs


# Given the zip_list and docs list it creates the input matrix (sparse matrix).
# input_matrix: rows: hashed shingles
#               columns: documents names
#               cells: 1 --> hashed shingles of row r is a member of the document of column c
#                      0 --> hashed shingles of row r is not a member of the document of column c
def create_input_matrix(zip_list, docs):
    input_matrix = []
    # store document names in first row
    input_matrix.append(docs)

    for zip_row in zip_list:
        row = []
        # store shingle name in first column of current row
        row.append(zip_row[0])
        for doc_order_in_matrix in range(len(docs) - 1):
            # if shingle is in document
            if input_matrix[0][doc_order_in_matrix + 1] in zip_row[1]:
                row.append(1)
            else:
                row.append(0)
        # store row in input matrix
        input_matrix.append(row)

    for r in range(len(input_matrix)):
        print(input_matrix[r])
    return input_matrix


# find next prime number of a number
def next_prime(value):
    for num in range(value, value*value):  # for num in range(value,2**33): ν παίρνεις την επόμενη δύναμη
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            return num


def randomCoefficients(functions_no):
    randomNumbers = []
    functions_counter = 0
    max_shingle_value = 2 ** 32 - 1
    while functions_counter < functions_no:
        # generate random number in range (0, max_shingle_value)
        random_number = random.randint(0, max_shingle_value)
        # We want the number to be unique.
        while random_number in randomNumbers:
            # create new random number
            random_number = random.randint(0, max_shingle_value)
        # Add the unique random number to the list
        randomNumbers.append(random_number)
        # Increment the counter.
        functions_counter = functions_counter + 1
    return randomNumbers


# Creates the signature matrix given the input matrix and the number of hash functions
def minHash(input_matrix, docs, hash_num):
    # max shingle value
    max_shingle_value = 2 ** 32 - 1
    p = next_prime(max_shingle_value + 1)  # p = next_prime(max_shingle_value)
    a_coef = randomCoefficients(hash_num)
    b_coef = randomCoefficients(hash_num)
    signature_matrix = []
    # initialize signature matrix
    for j in range(hash_num):
       signature_matrix.append([p + 1] * (len(docs) - 1))
    print("signature_matrix: ", signature_matrix)

    # for every row of the input matrix (every hashed single)
    for row in range(1, len(input_matrix)):
        # create hash functions for this row
        hash_func = []
        for i in range(hash_num):
            hashed_function = (a_coef[i] * row + b_coef[i]) % p
            hash_func.append(hashed_function)
        print("hash_func: ", hash_func)

        # for every column of the input matrix
        for column in range(len(docs) - 1):
            # if column c in input matrix has 1 in row r
            # print("input_matrix[row][column]: ", input_matrix[row][column+1])
            if input_matrix[row][column + 1] == 1:
                # for each i= 1, 2, ..., n
                for hf in range(hash_num):
                    # if hash_function(hf) is smaller than sig_matrix(row,column)
                    if hash_func[hf] < signature_matrix[hf][column]:
                        # set sig_matrix(row,column) to hash_function(hf)
                        signature_matrix[hf][column] = hash_func[hf]
    return signature_matrix


# Finds the set of table size divisors and
# assigns appropriate values to the variables r and b for for the number of bands and rows
def get_b_r(length_of_sign_matrix):
    factors = []
    for i in range(1, length_of_sign_matrix + 1):
        if length_of_sign_matrix % i == 0:
            factors.append(i)
    if len(factors) > 3:
        r = factors[int(len(factors) / 2) - 1]
    elif len(factors) == 3:
        r = factors[int(len(factors) / 2)]
    else:
        print("Give number of functions diferent from", length_of_sign_matrix)
        return
    b = int(length_of_sign_matrix / r)
    return b, r


# Creates bands given the signature matrix.
# Separates the signature_matrix rows (n in tolat)
# to b bands each of which consists of r rows (b*r=n)
def create_bands(signature_matrix):
    bands = []
    b, r = get_b_r(len(signature_matrix))
    print("\n#bands = ", b, "#rows = ", r)

    row_to_add = 0
    for band in range(b):
        rows = []
        for row in range(r):
            # print("band,row", band, row)
            rows.append(signature_matrix[row_to_add])
            row_to_add = row_to_add+1
        bands.append([band, rows])
    return bands


# Create buckets given the bands.
# Hashes each column of bands to some large number of buckets.
# The hash function gives the position in which the column
# in which it will be placed in the buckets array.
# We use a separate bucket array for each band, so columns with the same vector
# in different bands will not hash to the same bucket.
def create_hash_table(bands, k):
    buckets = []
    for i in range(len(bands)):
        row = []
        for j in range(k):
            row.append([])
        buckets.append(row)

    a = random.randint(1, 99)  # hash function parameters
    b = random.randint(1, 99)
    for band in range(len(bands)):
        for column in range(len(bands[band][1][0])):
            sum = 0
            for band_len in range(len(bands[band][1])):
                # print(bands[band][1][band_len][column], end=', ')
                sum = sum + bands[band][1][band_len][column]
            position_in_hashed = ((a * sum + b) % 99) % k
            buckets[band][position_in_hashed].append(column)
        #     print("\tposition -->", position_in_hashed)
        # print("end band\n")
    return buckets


# The buckets array is a sparse array, so candidate_column_pairs function
# returns only the candidate pairs of documents (zip array of buckets)
def candidate_column_pairs(buckets):
    candidate_pairs_list = []
    for row in range(len(buckets)):
        row_candidates_matches = []
        for column in range(len(buckets[row])):
            if len(buckets[row][column]) > 1:
                column_candidates_matches = []
                for matches in range(len(buckets[row][column])):
                    column_candidates_matches.append(buckets[row][column][matches])
                row_candidates_matches.append(column_candidates_matches)
        candidate_pairs_list.append(row_candidates_matches)
    return candidate_pairs_list


# jaccard_similarity function calculates the Jaccard similarity of 2 lists
def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


# document_similarities function caclulates the Jaccard similarity of all pairs
def document_similarities(cand_pairs, sign_mtrx, docs):
    similarities = []
    for bucket in range(len(cand_pairs)):
        for row in range(len(cand_pairs[bucket])):
            for l in range(len(cand_pairs[bucket][row])):
                for s_l in range(l + 1, len(cand_pairs[bucket][row])):
                    first = cand_pairs[bucket][row][l]
                    second = cand_pairs[bucket][row][s_l]
                    list_1 = [i[first] for i in sign_mtrx]
                    list_2 = [i[second] for i in sign_mtrx]
                    js = jaccard_similarity(list_1, list_2)
                    similarities.append([docs[first + 1], docs[second + 1], js])
    return similarities
