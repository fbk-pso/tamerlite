import re
import json
import time
import ast
from scripts.utils import extract_lifted_macros_from_json, generate_ground_macros, select_best_lifted_macros

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



def read_macros_from_csv(file_path):
    macros_list = []

    with open(file_path, 'r') as file:
        # Skip the header line
        next(file)

        # Read the rest of the lines
        for line in file:
            # Split the line by commas and strip any extra whitespace
            parts = line.split(',')
            macro = parts[0].strip()
            macros_list.append(re.findall(r"'(.*?)'", macro))

    return macros_list

def read_macros_from_json(file_path, problem, macros_usage):
    macros_list = []

    with open(file_path, 'r') as file:
        database_read = json.load(file)

    best_lifted_macros = extract_lifted_macros_from_json(database_read, problem)

    if 'best' not in file_path:
        start_time = time.time()
        best_lifted_macros = select_best_lifted_macros(best_lifted_macros, problem, macros_usage)
        print(f"Time_for_selection: {(time.time() - start_time)}")
    
    # best_lifted_macros = best_lifted_macros[:-1]
    for i, ma in enumerate(best_lifted_macros):
        print(f"{i})  {ma}")

    for lifted in best_lifted_macros:
        for ground in generate_ground_macros(lifted, problem):
            assert len(ground) == len(lifted)
            macros_list.append(ast.literal_eval(str(ground)))

    return macros_list


def read_macros(macros_path, macros_usage, problem):
    if '.csv' in macros_path:
        macros = read_macros_from_csv(macros_path)
    elif '.json' in macros_path:
        macros = read_macros_from_json(macros_path, problem, macros_usage)
    else:
        raise ValueError("Unknown file format for the macros file.")
    return macros