import math
import binascii
import random
import re
from os.path import isfile, join, dirname
import os
import nltk
from nltk.corpus import stopwords

# get documents directory


# read documents from directory,
# preprocessing on text (words to lowercase, removing punctuation marks, split, remove stopwords)
# create shingles
# return:
# 1. zip_list: list of which each row contains hashed shingle and the names of the documents in which it appears
# 2. docs: the names of all documents
def create_shingles(MYDIR):
    k = 5
    docs = [""]
    zip_list = []
    shingles = []
    hashed_shingles = []
    path_of_docs = MYDIR + '/sample/'
    print("Preprocessing...")
    for root, dirs, files in os.walk(path_of_docs, topdown=False):
        # for every file
        for name in files:
            print("name:", name)
            doc = os.path.join(root, name)
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


# creates bands given the signature matrix
def create_bands(signature_matrix):
    bands = []
    b, r = get_b_r(len(signature_matrix))
    print("\n#bands = ", b, "#rows = ", r)

    for band in range(b):
        rows = []
        for row in range(r):
            rows.append(signature_matrix[band + row])
        bands.append([band, rows])
    return bands


def main():
    MYDIR = dirname(__file__)  # gives back your directory path
    hash_num = 10

    hashed_shingles, docs = create_shingles(MYDIR)
    inp_mtrx = create_input_matrix(hashed_shingles, docs)
    sign_mtrx = minHash(inp_mtrx, docs, hash_num)
    for r in range(len(sign_mtrx)):
        print("minHash:", r, "-->", sign_mtrx[r])
    bands = create_bands(sign_mtrx)
    print("bands: ")
    for r in range(len(bands)):
        print(r, "-->", end='')
        for s_r in range(len(bands[r][1])):
            if s_r == 0:
                print(bands[r][1][s_r])
            else:
                print("\t", bands[r][1][s_r])


    #print(sm)

if __name__ == "__main__":
    main()
