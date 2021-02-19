# Range and similarity queries
In this repository we have implemented a B+ Tree to execute range and exact match queries and a Locality Sensitive Hashing (LSH) algorithm, using the MinHash method for finding similar documents as measured by Jaccard similarity.

## Implementation
### B+ Tree

### LSH
The documents returned from B+ tree are then processed by the LSH algorithm. The LSH algorithm consists of these steps:
1. **Shingling of Documents**: <p align="justify"> we convert each document into a set of characters of length k (also known as k-shingles). Then we convert every k-shingle into a 32-bit number via a hash function. </p>
2. **Input-matrix creation**: <p align="justify">the columns of the input-matrix correspond to the documents, and the rows correspond to the hashed shingles. There is an 1 in position (r,c), where r is a row and c a column of the matrix, if the document of column c contains the hashed shingle of row r. Otherwise the value in position (r, c) is 0.</p>
3. **Minhashing**: <p align="justify"> in this step we create a new matrix, the signature matrix. Initially, this matrix consists of very large values. For every row of the input-matrix, we construct 100 universal hash functions ( h<sub>a,b</sub>(x) = (ax + b) mod p ). For every row r of the input matrix, we iterate every column. If column c has 1 in row r, then for each hash function, if hash_function(hf) is less than signature_matrix(r, c), set siganture_matrix(r, c) to the smaller of the current value of siganture_matrix(r, c) and hf(r). </p>
4. **Locality-sensitive hashing**
   * **Band partition**: we separate the signature_matrix rows to b bands each of which consists of r rows (b*r=n). For each band, we hash its portion of each column, using the same universal hash function, to a hash table with k buckets.
   * **Find candidate pairs**: we consider any pair that hashed to the same bucket for any of the hashings to be a candidate pair.
### Jaccard Similarity
In this final step, we calculate the [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index) of all candidate pairs.
## Installation
Execute:
* `git clone https://github.com/rkapsalis/Range-and-similarity-queries.git` to clone the repository
* `cd Range-and-similarity-queries` to access the project directory
* `pip install -r requirements.txt` to install all dependencies
* `python main.py`

## License
