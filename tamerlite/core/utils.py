
class TrieNode:
    def __init__(self):
        self.children = {}  # Each child is a key-value pair where the key is a macro
        self.is_end = False  # True if the node represents the end of a sequence that expands a state


class PrefixTree:
    def __init__(self):
        self.root = TrieNode()
        self.counter_skip = 0

    def insert(self, word: tuple):
        """
        Insert a word (tuple of strings) into the prefix tree if it is not already in it and return True, otherwise return False.
        """
        inserted = False
        node = self.root
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode()
                inserted = True
            node = node.children[letter]
        if not node.is_end:
            inserted = True
        node.is_end = True
        return inserted
        
    
    def print_structure(self, node=None, prefix=''):
        """
        Prints the structure of the Trie in a readable format.
        """
        if node is None:
            node = self.root  # Start from the root node

        for letter, child_node in node.children.items():
            # Mark the end of a word
            end_marker = " (end of word)" if child_node.is_end else ""
            print(f"{prefix}└── {letter}{end_marker}")
            # Recursively print the child nodes
            self.print_structure(child_node, prefix=prefix + "    ")
