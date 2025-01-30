class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True

    def search_fuzzy(self, query, max_distance):
        """ Finds words in the Trie within the given edit distance threshold. """
        def dfs(node, word, prev_row):
            word_len = len(query)
            curr_row = [prev_row[0] + 1]

            for j in range(1, word_len + 1):
                cost = 0 if query[j - 1] == word[-1] else 1
                curr_row.append(min(
                    prev_row[j] + 1,      # Deletion
                    curr_row[j - 1] + 1,  # Insertion
                    prev_row[j - 1] + cost  # Substitution
                ))

            if curr_row[-1] <= max_distance and node.is_end_of_word:
                results.append((word, curr_row[-1]))

            if min(curr_row) <= max_distance:
                for char, next_node in node.children.items():
                    dfs(next_node, word + [char], curr_row)

        results = []
        start_row = list(range(len(query) + 1))
        for char, next_node in self.root.children.items():
            dfs(next_node, [char], start_row)

        return sorted(results, key=lambda x: x[1])

if __name__ == "__main__":
    corpus = ["apple", "apricot", "banana", "grape", "grapefruit"]
    query = "applying"
    max_distance = 1  # Allow up to 2 edits

    trie = Trie()
    for word in corpus:
        trie.insert(word)

    closest_matches = trie.search_fuzzy(query, max_distance)
    print("Closest Matches:", closest_matches)
