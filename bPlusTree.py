from __future__ import annotations
from os import listdir
from os.path import isfile, join, dirname
import re
from math import floor

MYDIR = dirname(__file__)  # gives back your directory path


class Node:
    """
    Node object
    Attributes:
        order (int): The maximum number of keys each node can hold (branching factor).
    """
    uidCounter = 0

    def __init__(self, order):
        self.order = order
        self.parent: Node = None
        self.keys = []
        self.values = []

        #  This is for Debugging purposes only!
        Node.uidCounter += 1
        self.uid = self.uidCounter

    def split(self) -> Node:  # Split a full Node to two new ones.
        left = Node(self.order)
        right = Node(self.order)
        mid = int(self.order // 2)

        left.parent = right.parent = self

        left.keys = self.keys[:mid]
        left.values = self.values[:mid + 1]

        right.keys = self.keys[mid + 1:]
        right.values = self.values[mid + 1:]

        self.values = [left, right]  # Setup the pointers to child nodes.

        self.keys = [self.keys[mid]]  # Hold the first element from the right subtree.

        # Setup correct parent for each child node.
        for child in left.values:
            if isinstance(child, Node):
                child.parent = left

        for child in right.values:
            if isinstance(child, Node):
                child.parent = right

        return self  # Return the 'top node'

    def isEmpty(self) -> bool:
        return len(self.keys) == 0

    def isRoot(self) -> bool:
        return self.parent is None


class LeafNode(Node):
    def __init__(self, order):
        super().__init__(order)

        self.prevLeaf: LeafNode = None
        self.nextLeaf: LeafNode = None

    def add(self, key, value):
        if not self.keys:  # Insert key if it doesn't exist
            self.keys.append(key)
            self.values.append([value])
            return

        for i, item in enumerate(self.keys):  # Otherwise, search key and append value.
            if key == item:  # Key found => Append Value
                self.values[i].append(value)  # Remember, this is a list of data. Not nodes!
                break

            elif key < item:  # Key not found && key < item => Add key before item.
                self.keys = self.keys[:i] + [key] + self.keys[i:]
                self.values = self.values[:i] + [[value]] + self.values[i:]
                break

            elif i + 1 == len(self.keys):  # Key not found here. Append it after.
                self.keys.append(key)
                self.values.append([value])
                break

    def split(self) -> Node:  # Split a full leaf node. (Different method used than before!)
        top = Node(self.order)
        right = LeafNode(self.order)
        mid = int(self.order // 2)

        self.parent = right.parent = top

        right.keys = self.keys[mid:]
        right.values = self.values[mid:]
        right.prevLeaf = self
        right.nextLeaf = self.nextLeaf

        top.keys = [right.keys[0]]
        top.values = [self, right]  # Setup the pointers to child nodes.

        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        self.nextLeaf = right  # Setup pointer to next leaf

        return top  # Return the 'top node'


class BPlusTree(object):
    def __init__(self, order=4):
        self.root: Node = LeafNode(order)  # First node must be leaf (to store data).
        self.order: int = order

    @staticmethod
    def _find(node: Node, key):
        for i, item in enumerate(node.keys):
            if key < item:
                return node.values[i], i
            elif i + 1 == len(node.keys):
                return node.values[i + 1], i + 1  # return right-most node/pointer.

    @staticmethod
    def _mergeUp(parent: Node, child: Node, index):
        parent.values.pop(index)
        pivot = child.keys[0]

        for c in child.values:
            if isinstance(c, Node):
                c.parent = parent

        for i, item in enumerate(parent.keys):
            if pivot < item:
                parent.keys = parent.keys[:i] + [pivot] + parent.keys[i:]
                parent.values = parent.values[:i] + child.values + parent.values[i:]
                break

            elif i + 1 == len(parent.keys):
                parent.keys += [pivot]
                parent.values += child.values
                break

    def insert(self, key, value):
        node = self.root

        while not isinstance(node, LeafNode):  # While we are in internal nodes... search for leafs.
            node, index = self._find(node, key)

        # Node is now guaranteed a LeafNode!
        node.add(key, value)

        while len(node.keys) == node.order:  # 1 over full
            if not node.isRoot():
                parent = node.parent
                node = node.split()  # Split & Set node as the 'top' node.
                jnk, index = self._find(parent, node.keys[0])
                self._mergeUp(parent, node, index)
                node = parent
            else:
                node = node.split()  # Split & Set node as the 'top' node.
                self.root = node  # Re-assign (first split must change the root!)

    def retrieve(self, key, max, query):
        keys_values = []
        node = self.root
        flag = True
        while not isinstance(node, LeafNode):
            node, index = self._find(node, key)

        if query == 'r':  # range query
            while flag and node:
                for i, node_data in enumerate(node.keys):
                    print(node_data, node.values[i])
                    if node_data >= max:
                        flag = False
                        break
                    keys_values.append([node_data, node.values[i]])
                node = node.nextLeaf
            return keys_values

        else:  # exact query
            for i, item in enumerate(node.keys):
                if key == item:
                    print("found: keys:", node.keys[i], " found: value ", node.values[i])
                    return node.values[i]

        print("Word not found")
        return None

    def printTree(self):
        if self.root.isEmpty():
            print('The bpt+ Tree is empty!')
            return
        queue = [self.root, 0]

        while len(queue) > 0:
            node = queue.pop(0)
            height = queue.pop(0)

            if not isinstance(node, LeafNode):
                queue += self.intersperse(node.values, height + 1)
            print('Level ' + str(height), '|'.join(map(str, node.keys)), ' -->\t current -> ', node.uid,
                  '\t parent -> ', node.parent.uid if node.parent else None)

    def getLeftmostLeaf(self):
        if not self.root:
            return None

        node = self.root
        while not isinstance(node, LeafNode):
            node = node.values[0]

        return node

    def getRightmostLeaf(self):
        if not self.root:
            return None

        node = self.root
        while not isinstance(node, LeafNode):
            node = node.values[-1]

    def showAllData(self):
        node = self.getLeftmostLeaf()
        if not node:
            return None

        while node:
            for node_data in node.values:
                print('[{}]'.format(', '.join(map(str, node_data))), end=' -> ')

            node = node.nextLeaf
        print('Last node')

    def showAllDataReverse(self):
        node = self.getRightmostLeaf()
        if not node:
            return None

        while node:
            for node_data in reversed(node.values):
                print('[{}]'.format(', '.join(map(str, node_data))), end=' <- ')

            node = node.prevLeaf
        print()

    @staticmethod
    def intersperse(lst, item):
        result = [item] * (len(lst) * 2)
        result[0::2] = lst
        return result


def bplustree(dictionary):
    bplustree = BPlusTree(order=4)

    for doc in dictionary:  # for each document
        for word in doc[0]:  # for each word in document
            bplustree.insert(word, doc[1])
            print(word, doc[1])

    bplustree.printTree()
    bplustree.showAllData()
    return bplustree


def docs_to_search(path_of_docs):
    file_info = []
    try:
        list_files = [file for file in listdir(path_of_docs) if isfile(join(path_of_docs, file))]
        list_paths = [join(path_of_docs, file) for file in listdir(path_of_docs) if isfile(join(path_of_docs, file))]
        file_info.append(list_files)
        file_info.append(list_paths)
    except FileNotFoundError:
        print("Not a folder, stop the programm and rerun with valid folder name!")
        return file_info
    return file_info


def Preprocessing(contentsRaw):
    # convert to lowercase
    contentsRaw = [term.lower() for term in contentsRaw]

    # remove punctuation
    sth = []
    for word in contentsRaw:
        new_s = re.sub(r'[^\w\s]', '', word)
        sth.append(new_s)

    filteredContents = sth

    # remove stopwords
    from nltk.corpus import stopwords
    set(stopwords.words("english"))
    filteredContents = [word for word in filteredContents if word not in stopwords.words('english')]

    return filteredContents


def main():
    query_type = ""
    dictionary = []
    point = 0  # to keep track of which text I am searching into

    # build tree
    path_of_docs = MYDIR + '/Datasets/corpus20090418'  # + input_theme  # execute path+files or dataset +file of documents
    documents = docs_to_search(path_of_docs)
    for doc in documents[1]:
        file = open(doc, "r", encoding="UTF-8", errors='ignore')  # open file
        words = file.read()  # read file
        raw_dictionary = list(words.split())  # text to list
        doc_dictionary = [Preprocessing(raw_dictionary), documents[0][point]]  # words preprocessing
        point += 1
        dictionary.append(doc_dictionary)

    tree = bplustree(dictionary)

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

            tree.retrieve(l_bound, u_bound, query_type)
            query_type = ""
        elif query_type == 'e':
            l_bound = input("Please type word you are looking for: ")
            u_bound = l_bound
            tree.retrieve(l_bound, u_bound, query_type)
            query_type = ""
        elif query_type == 'q':
            return


if __name__ == '__main__':
    main()
