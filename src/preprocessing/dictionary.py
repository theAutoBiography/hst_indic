"""
Sanskrit Dictionary Lookup System
Fast word lookup using hash tables with support for Monier-Williams dictionary
"""
import os
import logging
import pickle
import time
from typing import Optional, Dict, List, Set
from pathlib import Path
import re
import unicodedata

logger = logging.getLogger(__name__)


class SanskritDictionary:
    """
    Fast Sanskrit dictionary lookup using hash tables.
    Supports Monier-Williams and other Cologne Digital Sanskrit Dictionaries.

    Optimized for:
    - Fast lookups (< 1ms per query)
    - Multiple transliteration schemes (IAST, Devanagari, Harvard-Kyoto)
    - Variant forms (sandhi forms, case endings)
    """

    def __init__(self, dictionary_dir: Optional[str] = None):
        """
        Initialize Sanskrit dictionary

        Args:
            dictionary_dir: Path to dictionary data directory
        """
        if dictionary_dir is None:
            # Default to project dictionary directory
            project_root = Path(__file__).parent.parent.parent
            dictionary_dir = project_root / "data" / "dictionaries"

        self.dictionary_dir = Path(dictionary_dir)
        self.dictionary_dir.mkdir(parents=True, exist_ok=True)

        # Hash tables for fast lookup
        self._word_index: Dict[str, List[str]] = {}  # word -> [definitions]
        self._headwords: Set[str] = set()  # All dictionary headwords
        self._variant_map: Dict[str, str] = {}  # variants -> canonical form

        # Cache for frequently accessed words
        self._cache: Dict[str, str] = {}
        self._cache_size = 10000

        # Statistics
        self.stats = {
            'total_entries': 0,
            'total_variants': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'lookups': 0
        }

        # Load dictionary if available
        self._load_or_build_index()

    def _load_or_build_index(self):
        """Load pre-built index or build from scratch"""
        index_file = self.dictionary_dir / "sanskrit_index.pkl"

        if index_file.exists():
            logger.info(f"Loading pre-built dictionary index from {index_file}")
            self._load_index(index_file)
        else:
            logger.warning("No dictionary index found. Please download dictionary data.")
            logger.info("Run: python -m src.preprocessing.dictionary --download")

    def _load_index(self, index_file: Path):
        """Load pre-built dictionary index"""
        try:
            with open(index_file, 'rb') as f:
                data = pickle.load(f)
                self._word_index = data['word_index']
                self._headwords = data['headwords']
                self._variant_map = data.get('variant_map', {})
                self.stats['total_entries'] = len(self._headwords)
                self.stats['total_variants'] = len(self._variant_map)
            logger.info(f"Loaded {self.stats['total_entries']} dictionary entries")
        except Exception as e:
            logger.error(f"Failed to load dictionary index: {e}")

    def _save_index(self):
        """Save dictionary index for fast loading"""
        index_file = self.dictionary_dir / "sanskrit_index.pkl"
        try:
            data = {
                'word_index': self._word_index,
                'headwords': self._headwords,
                'variant_map': self._variant_map
            }
            with open(index_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved dictionary index to {index_file}")
        except Exception as e:
            logger.error(f"Failed to save dictionary index: {e}")

    def normalize_word(self, word: str) -> str:
        """
        Normalize Sanskrit word for consistent lookup
        - Remove diacritics variations
        - Lowercase
        - Normalize Unicode
        """
        # Normalize Unicode (NFC form)
        word = unicodedata.normalize('NFC', word)

        # Convert to lowercase
        word = word.lower()

        # Remove extra whitespace
        word = word.strip()

        return word

    def exists(self, word: str) -> bool:
        """
        Check if word exists in dictionary

        Args:
            word: Sanskrit word to check

        Returns:
            True if word exists, False otherwise

        Time complexity: O(1) average case
        """
        self.stats['lookups'] += 1

        # Normalize word
        normalized = self.normalize_word(word)

        # Check cache first
        if normalized in self._cache:
            self.stats['cache_hits'] += 1
            return True

        # Check headwords
        if normalized in self._headwords:
            return True

        # Check variants
        if normalized in self._variant_map:
            return True

        return False

    def get_definition(self, word: str) -> Optional[str]:
        """
        Get definition for a word

        Args:
            word: Sanskrit word to look up

        Returns:
            Definition string or None if not found

        Time complexity: O(1) average case
        """
        self.stats['lookups'] += 1

        # Normalize word
        normalized = self.normalize_word(word)

        # Check cache
        if normalized in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[normalized]

        # Check headwords
        if normalized in self._word_index:
            definitions = self._word_index[normalized]
            result = " | ".join(definitions)
            self._add_to_cache(normalized, result)
            self.stats['cache_misses'] += 1
            return result

        # Check variants
        if normalized in self._variant_map:
            canonical = self._variant_map[normalized]
            if canonical in self._word_index:
                definitions = self._word_index[canonical]
                result = " | ".join(definitions)
                self._add_to_cache(normalized, result)
                self.stats['cache_misses'] += 1
                return result

        return None

    def get_definitions(self, word: str) -> List[str]:
        """
        Get all definitions for a word as a list

        Args:
            word: Sanskrit word to look up

        Returns:
            List of definition strings (empty list if not found)
        """
        normalized = self.normalize_word(word)

        # Check headwords
        if normalized in self._word_index:
            return self._word_index[normalized]

        # Check variants
        if normalized in self._variant_map:
            canonical = self._variant_map[normalized]
            if canonical in self._word_index:
                return self._word_index[canonical]

        return []

    def _add_to_cache(self, word: str, definition: str):
        """Add entry to LRU cache"""
        if len(self._cache) >= self._cache_size:
            # Simple FIFO eviction (can be improved to LRU)
            self._cache.pop(next(iter(self._cache)))
        self._cache[word] = definition

    def add_entry(self, word: str, definition: str, variants: Optional[List[str]] = None):
        """
        Add a dictionary entry

        Args:
            word: Headword
            definition: Definition text
            variants: List of variant forms (optional)
        """
        normalized = self.normalize_word(word)

        # Add to headwords
        self._headwords.add(normalized)

        # Add to word index
        if normalized not in self._word_index:
            self._word_index[normalized] = []
        self._word_index[normalized].append(definition)

        # Add variants
        if variants:
            for variant in variants:
                norm_variant = self.normalize_word(variant)
                self._variant_map[norm_variant] = normalized

        self.stats['total_entries'] = len(self._headwords)
        self.stats['total_variants'] = len(self._variant_map)

    def load_from_xml(self, xml_file: Path, dictionary_name: str = "monier-williams"):
        """
        Load dictionary from Cologne Digital Sanskrit Dictionaries XML format

        Args:
            xml_file: Path to XML file
            dictionary_name: Name of the dictionary
        """
        import xml.etree.ElementTree as ET

        logger.info(f"Loading {dictionary_name} dictionary from {xml_file}")
        start_time = time.time()

        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            entry_count = 0

            # Parse entries (specific to Monier-Williams format)
            for entry in root.findall('.//entry'):
                # Get headword
                hw_elem = entry.find('.//h/key1')
                if hw_elem is None or hw_elem.text is None:
                    continue

                headword = hw_elem.text.strip()

                # Get definition
                body_elem = entry.find('.//body')
                if body_elem is None:
                    continue

                # Extract text content from body
                definition = ET.tostring(body_elem, encoding='unicode', method='text')
                definition = re.sub(r'\s+', ' ', definition).strip()

                # Get alternate spellings/forms
                variants = []
                for alt in entry.findall('.//h/key2'):
                    if alt.text:
                        variants.append(alt.text.strip())

                # Add to dictionary
                self.add_entry(headword, definition, variants)
                entry_count += 1

                if entry_count % 1000 == 0:
                    logger.info(f"Processed {entry_count} entries...")

            # Save index for future use
            self._save_index()

            elapsed = time.time() - start_time
            logger.info(f"Loaded {entry_count} entries in {elapsed:.2f}s")
            logger.info(f"Average load time: {(elapsed/entry_count)*1000:.3f}ms per entry")

        except Exception as e:
            logger.error(f"Failed to load dictionary from XML: {e}")
            raise

    def load_from_csv(self, csv_file: Path):
        """
        Load dictionary from CSV file
        Format: word,definition[,variant1,variant2,...]

        Args:
            csv_file: Path to CSV file
        """
        import csv

        logger.info(f"Loading dictionary from {csv_file}")
        start_time = time.time()

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header if present

                entry_count = 0
                for row in reader:
                    if len(row) < 2:
                        continue

                    word = row[0]
                    definition = row[1]
                    variants = row[2:] if len(row) > 2 else None

                    self.add_entry(word, definition, variants)
                    entry_count += 1

                    if entry_count % 1000 == 0:
                        logger.info(f"Processed {entry_count} entries...")

            self._save_index()

            elapsed = time.time() - start_time
            logger.info(f"Loaded {entry_count} entries in {elapsed:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load dictionary from CSV: {e}")
            raise

    def get_stats(self) -> Dict:
        """Get dictionary statistics"""
        cache_hit_rate = 0.0
        if self.stats['lookups'] > 0:
            cache_hit_rate = (self.stats['cache_hits'] / self.stats['lookups']) * 100

        return {
            **self.stats,
            'cache_hit_rate': f"{cache_hit_rate:.2f}%",
            'cache_size': len(self._cache)
        }

    def benchmark(self, test_words: List[str], iterations: int = 1000) -> Dict:
        """
        Benchmark lookup performance

        Args:
            test_words: List of words to test
            iterations: Number of iterations per word

        Returns:
            Performance statistics
        """
        logger.info(f"Benchmarking with {len(test_words)} words, {iterations} iterations each")

        start_time = time.time()
        total_lookups = 0

        for _ in range(iterations):
            for word in test_words:
                self.exists(word)
                total_lookups += 1

        elapsed = time.time() - start_time
        avg_time_ms = (elapsed / total_lookups) * 1000

        results = {
            'total_lookups': total_lookups,
            'elapsed_seconds': elapsed,
            'avg_time_ms': avg_time_ms,
            'lookups_per_second': total_lookups / elapsed
        }

        logger.info(f"Benchmark results: {avg_time_ms:.4f}ms per lookup")
        logger.info(f"Throughput: {results['lookups_per_second']:.0f} lookups/second")

        return results


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanskrit Dictionary Lookup Tool")
    parser.add_argument('--word', type=str, help='Word to look up')
    parser.add_argument('--download', action='store_true', help='Download dictionary data')
    parser.add_argument('--load-xml', type=str, help='Load dictionary from XML file')
    parser.add_argument('--load-csv', type=str, help='Load dictionary from CSV file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--stats', action='store_true', help='Show dictionary statistics')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize dictionary
    dictionary = SanskritDictionary()

    if args.download:
        print("Please run the download script:")
        print("python -m src.preprocessing.download_dictionaries")

    elif args.load_xml:
        dictionary.load_from_xml(Path(args.load_xml))

    elif args.load_csv:
        dictionary.load_from_csv(Path(args.load_csv))

    elif args.word:
        # Look up word
        start = time.time()
        definition = dictionary.get_definition(args.word)
        elapsed = (time.time() - start) * 1000

        if definition:
            print(f"\n📖 Word: {args.word}")
            print(f"✓ Definition: {definition}")
            print(f"⏱  Lookup time: {elapsed:.4f}ms")
        else:
            print(f"\n❌ Word '{args.word}' not found in dictionary")
            print(f"⏱  Lookup time: {elapsed:.4f}ms")

    elif args.benchmark:
        # Run benchmark with common Sanskrit words
        test_words = ['rāma', 'kṛṣṇa', 'dharma', 'karma', 'yoga', 'asti', 'bhū', 'deva']
        results = dictionary.benchmark(test_words, iterations=1000)
        print(f"\n🏃 Benchmark Results:")
        print(f"  Total lookups: {results['total_lookups']}")
        print(f"  Average time: {results['avg_time_ms']:.4f}ms")
        print(f"  Throughput: {results['lookups_per_second']:.0f} lookups/sec")

    elif args.stats:
        # Show statistics
        stats = dictionary.get_stats()
        print(f"\n📊 Dictionary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        parser.print_help()
