import unittest
from homework import ShakespeareTextGenerator

class TestShakespeareGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ShakespeareTextGenerator()

    def test_preprocess_text(self):
        """Test text preprocessing"""
        test_text = "To be, or not to be: that is the question!"
        processed = self.generator.preprocess_text(test_text)
        self.assertEqual(processed, ['to', 'be', 'or', 'not', 'to', 'be', 'that', 'is', 'the', 'question'])

    def test_build_ngram_counts(self):
        """Test n-gram counting"""
        # Simple test case with repeated sequence
        tokens = ['a', 'b', 'c', 'a', 'b', 'd']
        self.generator.build_ngram_counts(tokens, 2)

        # Print the counts for debugging
        print("\nBigram counts:")
        print(dict(self.generator.from_bigram_to_next_token_counts))

        # Test all expected bigram transitions
        expected_counts = {
            ('a',): {'b': 2},  # 'a' is followed by 'b' twice
            ('b',): {'c': 1, 'd': 1},  # 'b' is followed by 'c' once and 'd' once
            ('c',): {'a': 1},  # 'c' is followed by 'a' once
        }

        for prefix, expected_next_tokens in expected_counts.items():
            for next_token, expected_count in expected_next_tokens.items():
                self.assertEqual(
                    self.generator.from_bigram_to_next_token_counts[prefix][next_token],
                    expected_count,
                    f"Expected {expected_count} occurrence(s) of '{next_token}' following '{prefix}'"
                )

    def test_generate_text(self):
        """Test text generation"""
        self.generator.train()  # First train the model

        # Test generation with different n-grams
        bigram_text = self.generator.generate_text(('to', 'be'), 10, 2)
        self.assertEqual(len(bigram_text.split()), 10)

        trigram_text = self.generator.generate_text(('to', 'be', 'or'), 10, 3)
        self.assertEqual(len(trigram_text.split()), 10)

    def test_invalid_input(self):
        """Test error handling"""
        with self.assertRaises(ValueError):
            self.generator.generate_text(['not', 'a', 'tuple'], 10, 2)

    def test_fallback_mechanism(self):
        """Test fallback for unknown n-grams"""
        next_token = self.generator.sample_next_token(('unknown', 'ngram'), 2)
        self.assertIn(next_token, ['the', 'and', 'to', 'of', 'a'])

if __name__ == '__main__':
    unittest.main()
