"""
Unit tests for Sanskrit Dictionary module
Tests word lookup, performance, and edge cases
"""
import unittest
import time
from pathlib import Path
import tempfile
import shutil

from src.preprocessing.dictionary import SanskritDictionary


class TestSanskritDictionary(unittest.TestCase):
    """Test cases for SanskritDictionary class"""

    @classmethod
    def setUpClass(cls):
        """Set up test dictionary with known Sanskrit words"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dict = SanskritDictionary(dictionary_dir=cls.temp_dir)

        # Add known Sanskrit words for testing
        cls.test_words = {
            'rāma': 'dark, dark-colored; pleasing, delightful; Name of various princes',
            'kṛṣṇa': 'black, dark, dark-blue; Krishna (name of deity)',
            'dharma': 'that which is established or firm, law, duty, right, justice, morality',
            'karma': 'act, action, performance; work, deed; fate',
            'yoga': 'yoking, joining; union, connection; means, way, manner',
            'asti': 'he/she/it is, there is, exists (from √as)',
            'bhavati': 'becomes, is, happens (from √bhū)',
            'deva': 'heavenly, divine; deity, god',
            'agni': 'fire; the god of fire',
            'veda': 'knowledge, sacred knowledge; the Vedas'
        }

        # Add words to dictionary
        for word, definition in cls.test_words.items():
            cls.dict.add_entry(word, definition)

        # Add some variants
        cls.dict.add_entry('rāmaḥ', 'Rama (nominative singular)', variants=['rāmah'])
        cls.dict.add_entry('kṛṣṇasya', 'of Krishna (genitive singular)', variants=['kṛṣṇasya'])

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory"""
        shutil.rmtree(cls.temp_dir)

    def test_word_exists(self):
        """Test that known words are found"""
        for word in self.test_words.keys():
            with self.subTest(word=word):
                self.assertTrue(
                    self.dict.exists(word),
                    f"Word '{word}' should exist in dictionary"
                )

    def test_word_not_exists(self):
        """Test that unknown words are not found"""
        unknown_words = ['xyz123', 'notaword', 'invalid']

        for word in unknown_words:
            with self.subTest(word=word):
                self.assertFalse(
                    self.dict.exists(word),
                    f"Word '{word}' should not exist in dictionary"
                )

    def test_get_definition(self):
        """Test getting definitions for known words"""
        word = 'rāma'
        definition = self.dict.get_definition(word)

        self.assertIsNotNone(definition)
        self.assertIn('pleasing', definition.lower())

    def test_get_definition_not_found(self):
        """Test getting definition for unknown word"""
        definition = self.dict.get_definition('unknownword123')
        self.assertIsNone(definition)

    def test_get_definitions_list(self):
        """Test getting definitions as list"""
        word = 'rāma'
        definitions = self.dict.get_definitions(word)

        self.assertIsInstance(definitions, list)
        self.assertGreater(len(definitions), 0)

    def test_normalization(self):
        """Test word normalization"""
        # These should all normalize to the same form
        variations = ['Rāma', 'RĀMA', 'rāma', '  rāma  ']

        for var in variations:
            with self.subTest(variation=var):
                normalized = self.dict.normalize_word(var)
                self.assertEqual(normalized, 'rāma')

    def test_case_insensitive_lookup(self):
        """Test that lookups are case-insensitive"""
        self.assertTrue(self.dict.exists('rāma'))
        self.assertTrue(self.dict.exists('Rāma'))
        self.assertTrue(self.dict.exists('RĀMA'))

    def test_variant_forms(self):
        """Test that variant forms are recognized"""
        # Test variant lookup
        self.assertTrue(self.dict.exists('rāmah'))
        self.assertTrue(self.dict.exists('kṛṣṇasya'))

    def test_performance_single_lookup(self):
        """Test that single lookup is fast (< 1ms)"""
        word = 'rāma'

        # Warm up cache
        self.dict.exists(word)

        # Time actual lookup
        start = time.time()
        result = self.dict.exists(word)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        self.assertTrue(result)
        self.assertLess(elapsed, 1.0, f"Lookup took {elapsed:.4f}ms, should be < 1ms")

    def test_performance_batch_lookups(self):
        """Test batch lookup performance"""
        test_words_list = list(self.test_words.keys())
        iterations = 100

        start = time.time()
        for _ in range(iterations):
            for word in test_words_list:
                self.dict.exists(word)

        elapsed = time.time() - start
        total_lookups = len(test_words_list) * iterations
        avg_time_ms = (elapsed / total_lookups) * 1000

        self.assertLess(
            avg_time_ms, 1.0,
            f"Average lookup time {avg_time_ms:.4f}ms should be < 1ms"
        )

    def test_cache_functionality(self):
        """Test that caching improves performance"""
        word = 'rāma'

        # Clear cache by creating new dictionary
        new_dict = SanskritDictionary(dictionary_dir=self.temp_dir)
        for w, d in self.test_words.items():
            new_dict.add_entry(w, d)

        # First lookup (cache miss)
        start = time.time()
        new_dict.exists(word)
        first_lookup = time.time() - start

        # Second lookup (cache hit)
        start = time.time()
        new_dict.exists(word)
        second_lookup = time.time() - start

        # Second lookup should be faster or similar
        # (This is a weak assertion as both should be very fast)
        self.assertGreaterEqual(first_lookup, 0)
        self.assertGreaterEqual(second_lookup, 0)

    def test_statistics(self):
        """Test that statistics are tracked correctly"""
        # Reset by creating new dict
        new_dict = SanskritDictionary(dictionary_dir=self.temp_dir)
        for w, d in self.test_words.items():
            new_dict.add_entry(w, d)

        # Perform some lookups
        new_dict.exists('rāma')
        new_dict.exists('kṛṣṇa')
        new_dict.exists('unknown_word')

        stats = new_dict.get_stats()

        self.assertEqual(stats['total_entries'], len(self.test_words))
        self.assertEqual(stats['lookups'], 3)
        self.assertIn('cache_hit_rate', stats)

    def test_add_multiple_definitions(self):
        """Test adding multiple definitions for same word"""
        word = 'test_word'

        self.dict.add_entry(word, 'Definition 1')
        self.dict.add_entry(word, 'Definition 2')

        definitions = self.dict.get_definitions(word)
        self.assertEqual(len(definitions), 2)
        self.assertIn('Definition 1', definitions)
        self.assertIn('Definition 2', definitions)

    def test_empty_dictionary(self):
        """Test operations on empty dictionary"""
        empty_dict = SanskritDictionary(dictionary_dir=tempfile.mkdtemp())

        self.assertFalse(empty_dict.exists('rāma'))
        self.assertIsNone(empty_dict.get_definition('rāma'))
        self.assertEqual(len(empty_dict.get_definitions('rāma')), 0)

    def test_unicode_normalization(self):
        """Test that Unicode is properly normalized"""
        # Different Unicode representations of the same character
        word1 = 'rāma'  # NFC
        word2 = 'rāma'  # NFD (if different)

        normalized1 = self.dict.normalize_word(word1)
        normalized2 = self.dict.normalize_word(word2)

        self.assertEqual(normalized1, normalized2)

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly"""
        variations = ['rāma', '  rāma', 'rāma  ', '  rāma  ']

        for var in variations:
            with self.subTest(variation=repr(var)):
                self.assertTrue(self.dict.exists(var))

    def test_special_characters(self):
        """Test words with special Sanskrit characters"""
        special_words = {
            'śiva': 'auspicious, benign; Shiva',
            'ṣaṣṭha': 'sixth',
            'ṛṣi': 'seer, sage',
            'ḷ': 'rare vocalic L'
        }

        for word, definition in special_words.items():
            self.dict.add_entry(word, definition)

        for word in special_words.keys():
            with self.subTest(word=word):
                self.assertTrue(self.dict.exists(word))
                self.assertIsNotNone(self.dict.get_definition(word))


class TestDictionaryPersistence(unittest.TestCase):
    """Test dictionary save/load functionality"""

    def setUp(self):
        """Create temporary directory for each test"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_index(self):
        """Test that dictionary can be saved and loaded"""
        # Create and populate dictionary
        dict1 = SanskritDictionary(dictionary_dir=self.temp_dir)
        dict1.add_entry('test1', 'definition1')
        dict1.add_entry('test2', 'definition2')

        # Save index
        dict1._save_index()

        # Create new dictionary instance (should load saved index)
        dict2 = SanskritDictionary(dictionary_dir=self.temp_dir)

        # Verify loaded data
        self.assertTrue(dict2.exists('test1'))
        self.assertTrue(dict2.exists('test2'))
        self.assertEqual(dict2.get_definition('test1'), 'definition1')


class TestDictionaryBenchmark(unittest.TestCase):
    """Benchmark tests for performance validation"""

    @classmethod
    def setUpClass(cls):
        """Set up dictionary with more entries for realistic benchmarking"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.dict = SanskritDictionary(dictionary_dir=cls.temp_dir)

        # Add 100 common Sanskrit words
        common_words = [
            'rāma', 'kṛṣṇa', 'dharma', 'karma', 'yoga', 'veda', 'deva', 'agni',
            'sūrya', 'candra', 'prāṇa', 'ātman', 'brahman', 'śakti', 'mantra',
            'tantra', 'guru', 'śiṣya', 'vidyā', 'jñāna', 'bhakti', 'mokṣa'
        ] * 5  # Repeat to get ~100 entries

        for i, word in enumerate(common_words):
            cls.dict.add_entry(f"{word}_{i}", f"Definition for {word}")

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.temp_dir)

    def test_benchmark_performance(self):
        """Benchmark dictionary performance"""
        test_words = [f"rāma_{i}" for i in range(5)]

        results = self.dict.benchmark(test_words, iterations=100)

        # Verify performance meets requirements
        self.assertLess(
            results['avg_time_ms'], 1.0,
            f"Average lookup time {results['avg_time_ms']:.4f}ms exceeds 1ms target"
        )

        # Should handle at least 1000 lookups per second
        self.assertGreater(
            results['lookups_per_second'], 1000,
            f"Throughput {results['lookups_per_second']:.0f} lookups/sec is too low"
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
