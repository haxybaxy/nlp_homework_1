import re
from collections import defaultdict
import random
import nltk
from nltk.corpus import gutenberg

# Download required NLTK data
try:
    nltk.data.find('corpora/gutenberg')
except LookupError:
    nltk.download('gutenberg')

class ShakespeareTextGenerator:
    def __init__(self):
        # Initialize with nested defaultdict(int) for counts
        self.from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
        self.from_bigram_to_next_token_probs = defaultdict(lambda: defaultdict(float))

        self.from_trigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
        self.from_trigram_to_next_token_probs = defaultdict(lambda: defaultdict(float))

        self.from_quadgram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
        self.from_quadgram_to_next_token_probs = defaultdict(lambda: defaultdict(float))

    def preprocess_text(self, text):
        """Preprocess the text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def build_ngram_counts(self, tokens, n):
        """Build n-gram counts dictionary."""
        if n == 2:
            counts_dict = self.from_bigram_to_next_token_counts
        elif n == 3:
            counts_dict = self.from_trigram_to_next_token_counts
        else:  # n == 4
            counts_dict = self.from_quadgram_to_next_token_counts

        for i in range(len(tokens) - n + 1):
            if n == 2:
                prefix = (tokens[i],)  # Single token for bigrams
            elif n == 3:
                prefix = (tokens[i], tokens[i+1])  # Two tokens for trigrams
            else:  # n == 4
                prefix = (tokens[i], tokens[i+1], tokens[i+2])  # Three tokens for quadgrams

            next_token = tokens[i + n - 1]
            counts_dict[prefix][next_token] += 1

    def calculate_probabilities(self, n):
        """Calculate probabilities from counts."""
        if n == 2:
            counts_dict = self.from_bigram_to_next_token_counts
            probs_dict = self.from_bigram_to_next_token_probs
        elif n == 3:
            counts_dict = self.from_trigram_to_next_token_counts
            probs_dict = self.from_trigram_to_next_token_probs
        else:  # n == 4
            counts_dict = self.from_quadgram_to_next_token_counts
            probs_dict = self.from_quadgram_to_next_token_probs

        # Calculate probabilities for each n-gram
        for ngram, counts in counts_dict.items():
            total = sum(counts.values())
            if total > 0:
                for token, count in counts.items():
                    probs_dict[ngram][token] = float(count) / total

    def sample_next_token(self, current_ngram, n):
        """Sample the next token based on n-gram probabilities."""
        probs_dict = self.from_bigram_to_next_token_probs
        if n == 3:
            probs_dict = self.from_trigram_to_next_token_probs
        elif n == 4:
            probs_dict = self.from_quadgram_to_next_token_probs

        if current_ngram not in probs_dict:
            return random.choice(['the', 'and', 'to', 'of', 'a'])  # fallback

        tokens = list(probs_dict[current_ngram].keys())
        probabilities = list(probs_dict[current_ngram].values())
        return random.choices(tokens, weights=probabilities, k=1)[0]

    def generate_text(self, initial_ngram, num_words, n):
        """Generate text using n-grams."""
        if not isinstance(initial_ngram, tuple):
            raise ValueError("initial_ngram must be a tuple")

        generated = list(initial_ngram)

        for _ in range(num_words - len(initial_ngram)):
            current_ngram = tuple(generated[-(n-1):])
            next_token = self.sample_next_token(current_ngram, n)
            generated.append(next_token)

        return ' '.join(generated)

    def train(self):
        """Train the model using Shakespeare's texts from NLTK Gutenberg corpus."""
        # Get all Shakespeare texts from Gutenberg
        shakespeare_files = [file for file in gutenberg.fileids() if 'shakespeare' in file.lower()]
        print(f"\nFound {len(shakespeare_files)} Shakespeare texts: {shakespeare_files}")

        all_text = []
        total_words = 0

        for file in shakespeare_files:
            words = gutenberg.words(file)
            text = ' '.join(words)
            all_text.append(text)
            total_words += len(words)
            print(f"Processed {file}: {len(words)} words")

        # Combine all texts
        combined_text = ' '.join(all_text)
        tokens = self.preprocess_text(combined_text)
        print(f"\nTotal words before preprocessing: {total_words}")
        print(f"Total tokens after preprocessing: {len(tokens)}")

        # Build n-gram counts
        print("\nBuilding n-gram models...")
        self.build_ngram_counts(tokens, 2)  # bigrams
        self.build_ngram_counts(tokens, 3)  # trigrams
        self.build_ngram_counts(tokens, 4)  # quadgrams

        print(f"Number of unique bigrams: {len(self.from_bigram_to_next_token_counts)}")
        print(f"Number of unique trigrams: {len(self.from_trigram_to_next_token_counts)}")
        print(f"Number of unique quadgrams: {len(self.from_quadgram_to_next_token_counts)}")

        # Calculate probabilities
        print("\nCalculating probabilities...")
        self.calculate_probabilities(2)
        self.calculate_probabilities(3)
        self.calculate_probabilities(4)

        # Print some example probabilities
        print("\nExample probability distributions:")
        common_bigram = ('to', 'be')
        print(f"\nTop 5 words following {common_bigram}:")
        probs = self.from_bigram_to_next_token_probs[common_bigram]
        for word, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {word}: {prob:.4f}")

def main():
    print("Initializing Shakespeare Text Generator...")
    generator = ShakespeareTextGenerator()

    print("\nTraining the model...")
    generator.train()

    print("\nGenerating sample texts...")

    # Generate text using different n-grams
    print("\nBigram generation:")
    bigram_text = generator.generate_text(('to', 'be'), 50, 2)
    print(bigram_text)

    print("\nTrigram generation:")
    trigram_text = generator.generate_text(('to', 'be', 'or'), 50, 3)
    print(trigram_text)

    print("\nQuadgram generation:")
    quadgram_text = generator.generate_text(('to', 'be', 'or', 'not'), 50, 4)
    print(quadgram_text)

    print("\nGeneration complete!")

if __name__ == "__main__":
    main()
