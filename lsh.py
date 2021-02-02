import math
import binascii
import random


def create_shingles(sim_docs, k):
    # read documents from directory,
    # preprocessing on text (words to lowercase, removing punctuation marks, split, remove stopwords)
    # create shingles
    # return:
    # 1. zip_list: list of which each row contains hashed shingle and the names of the documents in which it appears
    # 2. docs: the names of all documents

    docs = [""]
    zip_list = []
    shingles = []
    hashed_shingles = []

    print("\nCalculations...")

    # for every file
    for name in sim_docs:
        doc_name = name[0]
        text = name[1].lower()

        # store file names in array
        docs.append(doc_name)
        for i in range(len(text) - k + 1):
            # create singles
            shingle = text[i:i + k]
            # create hashed shingle
            hashed = binascii.crc32(shingle.encode('utf8')) & 0xffffffff  # 32bit
            if hashed not in hashed_shingles:
                # store unique hashed shingles
                hashed_shingles.append(hashed)
            # if the shingle is not in the list
            if not any((x[0] == hashed) for x in zip_list):
                # store shingle in array
                shingles.append(shingle)
                # store hashed shingle and document name in array
                zip_list.append([hashed, [doc_name]])
            else:
                # store name of document in the corresponding position
                position = [(i, el.index(hashed)) for i, el in enumerate(zip_list) if hashed == el[0]]
                # check if document name, already exists in list
                if doc_name not in zip_list[position[0][0]][1]:
                    zip_list[position[0][0]][1].append(doc_name)

    # for r in range(len(zip_list)):
    #     print("zip_list: ", zip_list[r])
    return zip_list, docs


def create_input_matrix(zip_list, docs):
    # Given the zip_list and docs list it creates the input matrix (sparse matrix).
    # input_matrix: rows: hashed shingles
    #               columns: documents names
    #               cells: 1 --> hashed shingles of row r is a member of the document of column c
    #                      0 --> hashed shingles of row r is not a member of the document of column c

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

    # for r in range(len(input_matrix)):
    #     print(input_matrix[r])
    return input_matrix


def next_prime(value):
    # Returns next prime number after a given number

    for num in range(value, value * value):
        # square root method to find if a number is prime or not
        if all(num % i != 0 for i in range(2, int(math.sqrt(num)) + 1)):
            return num


def randomCoefficients(funct_no):
    # Generates random numbers a,b for every hash function

    randomNumbers = []
    functions_counter = 0
    max_shingle_value = 2 ** 32 - 1
    # for every hash function
    while functions_counter < funct_no:
        # generate random number in range (0, max_shingle_value)
        # random.seed(0)
        random_number = random.randint(0, max_shingle_value)
        # we want the number to be unique.
        # random.seed(0)
        while random_number in randomNumbers:
            # if the number is not unique, create new random number
            random_number = random.randint(0, max_shingle_value)
        # add the unique random number to the list
        randomNumbers.append(random_number)
        # increment the counter.
        functions_counter = functions_counter + 1
    return randomNumbers


def minHash(input_matrix, docs, hash_num):
    # Implements minhash technique.
    # Creates the signature matrix,
    # given the input matrix and the number of hash functions

    # max shingle value
    max_shingle_value = 2 ** 32 - 1
    # next prime
    p = next_prime(max_shingle_value + 1)
    # get random coefficients
    a_coef = randomCoefficients(hash_num)
    b_coef = randomCoefficients(hash_num)
    # initialize signature matrix
    signature_matrix = []
    # for every hash function
    for j in range(hash_num):
        signature_matrix.append([p + 1] * (len(docs) - 1))

    # for every row of the input matrix (every hashed single)
    for row in range(1, len(input_matrix)):
        # create hash functions for this row
        hash_func = []
        for i in range(hash_num):
            # universal hashing
            hashed_function = (a_coef[i] * row + b_coef[i]) % p
            hash_func.append(hashed_function)
        # print("hash_func: ", hash_func)

        # for every column of the input matrix
        for column in range(len(docs) - 1):
            # if column c in input matrix has 1 in row r
            if input_matrix[row][column + 1] == 1:
                # for each hash_function = 1, 2, ..., n
                for hf in range(hash_num):
                    # if hash_function(hf) is less than sig_matrix(row,column)
                    if hash_func[hf] < signature_matrix[hf][column]:
                        # set sig_matrix(row,column) to hash_function(hf)
                        signature_matrix[hf][column] = hash_func[hf]
    return signature_matrix


def get_b_r(length_of_sign_matrix):
    # Finds the set of table size divisors and
    # assigns appropriate values to the variables
    # r and b for for the number of bands and rows

    factors = []
    for i in range(1, length_of_sign_matrix + 1):
        # if i exactly divides the length_of_sign_matrix
        if length_of_sign_matrix % i == 0:
            factors.append(i)

    # if the number of divisors is greater than 3
    if len(factors) > 3:
        # number of rows (r) takes the first lower than the mid value
        r = factors[int(len(factors) / 2) - 1]
    # if the number of divisors is 3
    elif len(factors) == 3:
        # number of rows (r) takes the the mid value
        r = factors[int(len(factors) / 2)]
    else:
        # if the number of divisors is less than 3 (inappropriate number for bands)
        print("Give number of functions different from", length_of_sign_matrix)
        return
    # must be length_of_sign_matrix = b*r
    b = int(length_of_sign_matrix / r)
    return b, r


def create_bands(signature_matrix):
    # Creates bands given the signature matrix.
    # Separates the signature_matrix rows (n in total)
    # to b bands each of which consists of r rows (b*r=n)

    bands = []
    # get the number of bands and the number of rows
    b, r = get_b_r(len(signature_matrix))
    row_to_add = 0

    # for each one of b bands
    for band in range(b):
        rows = []
        for row in range(r):
            # get r rows from signature_matrix
            rows.append(signature_matrix[row_to_add])
            row_to_add = row_to_add + 1
        # add r rows to this band
        bands.append([band, rows])
    return bands


def create_hash_table(bands, k):
    # Creates buckets given the bands.
    # Hashes each column of bands to some large number of buckets.
    # The hash function gives the position in which the column
    # will be placed in the buckets array.
    # We use a separate bucket array for each band, so columns  vector
    # with the same in different bands will not hash to the same bucket.

    buckets = []
    for i in range(len(bands)):
        row = []
        for j in range(k):
            row.append([])
        buckets.append(row)

    # hash function parameters
    # random.seed(0)
    a = random.randint(1, 99)
    # random.seed(0)
    b = random.randint(1, 99)
    for band in range(len(bands)):
        for column in range(len(bands[band][1][0])):
            sum = 0
            for band_len in range(len(bands[band][1])):
                sum = sum + bands[band][1][band_len][column]
            position_in_hashed = ((a * sum + b) % 99) % k
            buckets[band][position_in_hashed].append(column)
        #     print("\tposition -->", position_in_hashed)
        # print("end band\n")
    return buckets


def candidate_column_pairs(buckets):
    # The buckets array is a sparse array, so candidate_column_pairs function
    # returns only the candidate pairs of documents (zip array of buckets)

    candidate_pairs_list = []
    # for each bucket array (for row of buckets)
    for row in range(len(buckets)):
        row_candidates_matches = []
        # for each bucket in bucket array
        for column in range(len(buckets[row])):
            # if bucket contains 2 or more documents
            if len(buckets[row][column]) > 1:
                # add bucket to new zip bucket array
                row_candidates_matches.append(buckets[row][column])
        # add zip bucket array to candidate_pairs_list
        candidate_pairs_list.append(row_candidates_matches)
    return candidate_pairs_list


def jaccard_similarity(list1, list2):
    # calculates the Jaccard similarity of 2 lists

    s1 = set(list1)
    s2 = set(list2)
    return float(str(round(len(s1.intersection(s2)) / len(s1.union(s2)), 3)))


def document_similarities(cand_pairs, sign_mtrx, docs):
    # calculates the Jaccard similarity of all pairs

    similarities = []
    # for each bucket array in candidate pairs list
    for bucket_array in range(len(cand_pairs)):
        # for each bucket in bucket array
        for bucket in range(len(cand_pairs[bucket_array])):
            # find all pairs from a list (if list size > 2)
            # A,B,C,D --> AB,AC,AD - BC,BD - CD
            for doc_1 in range(len(cand_pairs[bucket_array][bucket])):
                for doc_2 in range(doc_1 + 1, len(cand_pairs[bucket_array][bucket])):
                    first = cand_pairs[bucket_array][bucket][doc_1]
                    second = cand_pairs[bucket_array][bucket][doc_2]
                    # if documents are different from each other
                    if first != second:
                        # get signatures of documents from signature matrix
                        list_1 = [s[first] for s in sign_mtrx]
                        list_2 = [s[second] for s in sign_mtrx]
                        # calculate jaccard similarity
                        js = jaccard_similarity(list_1, list_2)
                        # if record / pair does not exist in similarities add it
                        if ([docs[first + 1], docs[second + 1], js] or [docs[second + 1], docs[
                        first + 1], js]) not in similarities:
                            similarities.append([docs[first + 1], docs[second + 1], js])

    # sort results
    similarities = sorted(similarities, key=lambda s: s[2], reverse=True)

    return similarities
