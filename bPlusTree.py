from __future__ import annotations
from os import listdir
from os.path import isfile, join, dirname
import re

# directory path
MYDIR = dirname(__file__)


class Node:
    # represents Node of tree

    uidCounter = 0

    def __init__(self, order):
        # The max number of keys each node can hold
        self.order = order
        self.parent: Node = None
        self.keys = []
        self.values = []

        Node.uidCounter += 1
        self.uid = self.uidCounter

    def split(self) -> Node:
        # Split a full Node to two new ones.
        left = Node(self.order)
        right = Node(self.order)
        # Break the node at m/2th position.
        mid = int(self.order // 2)

        # Set parent for left and right node
        left.parent = right.parent = self

        # Left node takes keys till mid index
        left.keys = self.keys[:mid]
        # Left node takes values till mid+1 index
        left.values = self.values[:mid + 1]

        # Right node takes keys from mid+1 index till end
        right.keys = self.keys[mid + 1:]
        # Right node takes values from mid+1 index till end
        right.values = self.values[mid + 1:]

        # Setup the pointers(values) to child nodes.
        self.values = [left, right]
        # Hold the first element from the right subtree.
        self.keys = [self.keys[mid]]

        # Setup parent for each child node.
        for child in left.values:
            if isinstance(child, Node):
                child.parent = left

        for child in right.values:
            if isinstance(child, Node):
                child.parent = right
        # Return temp parent node
        return self

    def isEmpty(self) -> bool:
        # Check if node is empty
        return len(self.keys) == 0

    def isRoot(self) -> bool:
        # Check if node is root
        return self.parent is None


class LeafNode(Node):
    # represents leaf Node of tree

    def __init__(self, order):
        super().__init__(order)
        # Pointers to previous and next leaves
        self.prevLeaf: LeafNode = None
        self.nextLeaf: LeafNode = None

    def add(self, key, value):
        # Insert key and value if keys list is empty
        if not self.keys:
            self.keys.append(key)
            self.values.append([value])
            return
        # Otherwise, search for every key in keys list.
        for i, item in enumerate(self.keys):
            # Key found
            if key == item:
                # Append value in a list of data
                self.values[i].append(value)
                break
            # Key not found && key < item
            elif key < item:
                # Append key and value before item
                self.keys = self.keys[:i] + [key] + self.keys[i:]
                self.values = self.values[:i] + [[value]] + self.values[i:]
                break
            # If we have reached last iteration.
            elif i + 1 == len(self.keys):
                # Append it in last position.
                self.keys.append(key)
                self.values.append([value])
                break

    def split(self) -> Node:
        # Splits a full leaf node
        # New parent node
        top = Node(self.order)
        # New right node
        right = LeafNode(self.order)
        # Break the node at m/2th position.
        mid = int(self.order // 2)

        # Set parent for left and right leaves
        self.parent = right.parent = top

        # Right leaf takes keys from mid index till end
        right.keys = self.keys[mid:]
        # Right leaf takes values from mid index till end
        right.values = self.values[mid:]
        # Set right leafs' prevLeaf (the splitted leaf)
        right.prevLeaf = self
        # Set right leafs' nextLeaf
        right.nextLeaf = self.nextLeaf

        # Parent takes as keys the first key of right leaf
        top.keys = [right.keys[0]]
        # Setup the pointers to child nodes (the 2 leaves)
        top.values = [self, right]

        # Left leaf takes keys till mid index
        self.keys = self.keys[:mid]
        # Left leaf takes values till mid index
        self.values = self.values[:mid]
        # Setup pointer to next leaf (right leaf)
        self.nextLeaf = right

        # Return the temp parent
        return top


class BPlusTree(object):
    # represents a b+ tree
    def __init__(self, order):
        # First node must be leaf
        self.root: Node = LeafNode(order)
        self.order: int = order

    @staticmethod
    def _find(node: Node, key):
        # Given a node and the search key, _find() returns the value of the node
        # that corresponds or leads to the node containing the key

        # for every key of the given node
        for i, item in enumerate(node.keys):
            # if search key is lexically less than node key
            if key < item:
                return node.values[i], i
            # if search key is lexically greater than node key (end of node)
            elif i + 1 == len(node.keys):
                # return right-most node/pointer.
                return node.values[i + 1], i + 1

    @staticmethod
    def _mergeUp(parent: Node, child: Node, index):
        # After the split in the node-child, this function corrects the pointers to and from the parent node
        # and the intermediate value, that resulted after the split, is inserted in the parent.

        # remove parent pointer to node-child
        parent.values.pop(index)
        # store to pivot the key of node-child
        pivot = child.keys[0]

        # set as the parent of the nodes-children the node parent
        for c in child.values:
            if isinstance(c, Node):
                c.parent = parent

        # Search in the parent node for the appropriate location
        # to add the key and the 2 values of node-child
        for i, item in enumerate(parent.keys):
            #  if key of the child is lexically smaller than key of the parent
            if pivot < item:
                # add pivot and child values to parent in position i
                parent.keys = parent.keys[:i] + [pivot] + parent.keys[i:]
                parent.values = parent.values[:i] + child.values + parent.values[i:]
                break
            # if key of the child is lexically less than the key of the parent (end of node)
            elif i + 1 == len(parent.keys):
                # append pivot and child values to parent in the end of parent node
                parent.keys += [pivot]
                parent.values += child.values
                break

    def insert(self, key, value):
        # inserts a new key,value pair into tree

        node = self.root

        # While we are in internal nodes... search for leaves.
        while not isinstance(node, LeafNode):
            node, index = self._find(node, key)

        # Node is a LeafNode
        node.add(key, value)

        # node overflow
        while len(node.keys) == node.order:
            if not node.isRoot():
                parent = node.parent
                # split & set node as temp parent node.
                node = node.split()
                # search for node index in parent
                jnk, index = self._find(parent, node.keys[0])
                # merge parent and node
                self._mergeUp(parent, node, index)
                # set parent as the current node
                node = parent
            # node is root
            else:
                # split and set node as the temp parent node.
                node = node.split()
                # set node as tree root
                self.root = node

    def retrieve(self, key, max_value, query):
        # returns the documents that contain the requested key

        keys_values = []
        node = self.root
        flag = True
        while not isinstance(node, LeafNode):
            node, index = self._find(node, key)

        # range query
        if query == 'r':
            while flag and node:
                for i, node_data in enumerate(node.keys):
                    # if we have reached the upper bound
                    if node_data >= max_value:
                        flag = False
                        break
                    keys_values.append([node_data, node.values[i]])
                # get next leaf
                node = node.nextLeaf
            return keys_values
        # exact query
        else:
            for i, item in enumerate(node.keys):
                if key == item:
                    print("found word", node.keys[i], " in documents:", node.values[i])
                    return node.values[i]

        print("Word not found. Please, try again!")
        return None

    def printTree(self):

        # check if tree is empty
        if self.root.isEmpty():
            print('The b+ tree is empty!')
            # empty tree --> stop program
            return

        # queue: list of nodes to print and their height from the root
        # set as first node the root with height=0
        queue = [self.root, 0]

        # while queue is not empty
        while len(queue) > 0:
            # take the first node and its height and remove them from the queue
            node = queue.pop(0)
            height = queue.pop(0)

            # if node is not leaf
            if not isinstance(node, LeafNode):
                # add to queue the children of the node
                queue += self.intersperse(node.values, height + 1)
            # print node keys and id info
            print('Level ' + str(height), '|'.join(map(str, node.keys)), ' -->\t current -> ', node.uid,
                  '\t parent -> ', node.parent.uid if node.parent else None)

    def getLeftmostLeaf(self):

        # check if tree is empty
        if not self.root:
            return None

        node = self.root
        # while node is not leaf
        while not isinstance(node, LeafNode):
            node = node.values[0]
        # return leftmost leaf
        return node

    def showAllData(self):
        node = self.getLeftmostLeaf()
        if not node:
            return None

        while node:
            for node_data in node.values:
                print('[{}]'.format(', '.join(map(str, node_data))), end=' -> ')

            node = node.nextLeaf
        print('Last node')

    @staticmethod
    def intersperse(lst, item):
        # new list that in even positions contains lst elements
        # and in the odd positions contains the item value (integer number)
        result = [item] * (len(lst) * 2)
        result[0::2] = lst
        return result


def bplustree(dictionary, ord):
    # Create a new B+ tree with order ord
    bplustree = BPlusTree(order=ord)

    # for each document
    for doc in dictionary:
        # for each word in document
        for word in doc[0]:
            bplustree.insert(word, doc[1])

    # bplustree.printTree()
    # bplustree.showAllData()
    return bplustree


def docs_to_search(path_of_docs):
    file_info = []
    try:
        # add to list_files the names of the files in the directory
        list_files = [file for file in listdir(path_of_docs) if isfile(join(path_of_docs, file))]
        # add to list_paths the paths of the files in the directory
        list_paths = [join(path_of_docs, file) for file in listdir(path_of_docs) if isfile(join(path_of_docs, file))]
        # append lists to the file_info list
        file_info.append(list_files)
        file_info.append(list_paths)
    except FileNotFoundError:
        # directory not found
        print("Not a directory, stop the program and rerun with valid folder name!")
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