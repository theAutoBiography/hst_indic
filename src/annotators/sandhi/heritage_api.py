"""
Sanskrit Heritage Platform API integration
French team's production sandhi splitter (~75% accuracy)
API: https://sanskrit.inria.fr/
"""
import requests
import logging
from typing import List, Optional
from dataclasses import dataclass
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


@dataclass
class HeritageSplit:
    """Split result from Heritage API"""
    surface_form: str
    split: List[str]
    analysis: str
    confidence: float


class HeritageAPIClient:
    """
    Client for Sanskrit Heritage Platform API

    API endpoint: https://sanskrit.inria.fr/cgi-bin/SKT/sktlemma.cgi

    Example:
        client = HeritageAPIClient()
        result = client.split_sandhi("rāmo'sti")
        # Returns: ['rāmaḥ', 'asti']
    """

    def __init__(self, base_url: str = "https://sanskrit.inria.fr/cgi-bin/SKT/sktlemma.cgi"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Indic-Annotation-Pipeline/1.0'
        })

    def split_sandhi(
        self,
        text: str,
        encoding: str = 'SLP1',
        timeout: int = 10
    ) -> Optional[HeritageSplit]:
        """
        Split sandhi using Heritage Platform API

        Args:
            text: Sanskrit text to split
            encoding: Input encoding (SLP1, IAST, Devanagari)
            timeout: API timeout in seconds

        Returns:
            HeritageSplit object or None if API fails
        """
        try:
            # Convert to SLP1 if needed (Heritage uses SLP1)
            if encoding == 'IAST':
                text = self._iast_to_slp1(text)
            elif encoding == 'Devanagari':
                text = self._devanagari_to_slp1(text)

            # API parameters
            params = {
                'text': text,
                'topic': 'LS',  # Lexical segmentation
                'mode': 'g',    # Graphical output
                'st': 'Y',      # Show sandhi
            }

            # Make request
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()

            # Parse response (Heritage returns HTML/XML)
            return self._parse_response(response.text, text)

        except requests.Timeout:
            logger.warning(f"Heritage API timeout for: {text}")
            return None
        except Exception as e:
            logger.error(f"Heritage API error: {e}")
            return None

    def _parse_response(self, html: str, original_text: str) -> Optional[HeritageSplit]:
        """
        Parse Heritage API response

        Heritage returns HTML with segmentation results
        We need to extract the split words
        """
        # Heritage response format is complex HTML
        # For now, use a simple approach - extract from response

        # Simple heuristic: look for word boundaries in response
        # This is a simplified parser - can be improved

        try:
            # Heritage puts splits in <span> tags or similar
            # For quick integration, extract text content

            # Example response pattern:
            # <solution>rAmaH asti</solution>

            if '<solution>' in html:
                # Extract solution text
                start = html.find('<solution>') + len('<solution>')
                end = html.find('</solution>')
                solution = html[start:end].strip()

                # Split on spaces
                words = solution.split()

                return HeritageSplit(
                    surface_form=original_text,
                    split=words,
                    analysis="Heritage Platform",
                    confidence=0.75  # Heritage baseline accuracy
                )
            else:
                logger.debug(f"No solution found in Heritage response for: {original_text}")
                return None

        except Exception as e:
            logger.error(f"Error parsing Heritage response: {e}")
            return None

    def _iast_to_slp1(self, text: str) -> str:
        """Convert IAST to SLP1 encoding"""
        from indic_transliteration import sanscript
        return sanscript.transliterate(text, sanscript.IAST, sanscript.SLP1)

    def _devanagari_to_slp1(self, text: str) -> str:
        """Convert Devanagari to SLP1 encoding"""
        from indic_transliteration import sanscript
        return sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.SLP1)

    def split_batch(
        self,
        texts: List[str],
        encoding: str = 'IAST'
    ) -> List[Optional[HeritageSplit]]:
        """
        Split multiple texts (sequential to avoid overwhelming API)

        Args:
            texts: List of Sanskrit texts
            encoding: Input encoding

        Returns:
            List of HeritageSplit objects
        """
        results = []
        for text in texts:
            result = self.split_sandhi(text, encoding=encoding)
            results.append(result)

        return results


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    client = HeritageAPIClient()

    # Test cases
    tests = [
        "rāmo'sti",
        "tathā'pi",
        "namaste"
    ]

    print("Testing Heritage API:")
    print("=" * 60)

    for text in tests:
        print(f"\nInput: {text}")
        result = client.split_sandhi(text, encoding='IAST')

        if result:
            print(f"Split: {' + '.join(result.split)}")
            print(f"Confidence: {result.confidence}")
        else:
            print("❌ API call failed")
