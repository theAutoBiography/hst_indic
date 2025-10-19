"""
Download script for Cologne Digital Sanskrit Dictionaries
Downloads and processes Monier-Williams and other Sanskrit dictionaries
"""
import os
import logging
import requests
from pathlib import Path
from typing import Optional
import zipfile
import tarfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DictionaryDownloader:
    """Download and extract Sanskrit dictionaries"""

    # Dictionary sources from Cologne Digital Sanskrit Dictionaries
    DICTIONARIES = {
        'monier-williams': {
            'name': 'Monier-Williams Sanskrit-English Dictionary',
            'url': 'https://www.sanskrit-lexicon.uni-koeln.de/scans/MWScan/2020/web/webtc/download/mw.zip',
            'files': ['mw.xml'],
            'description': 'Most comprehensive Sanskrit-English dictionary (180,000+ entries)'
        },
        'apte': {
            'name': 'Apte Practical Sanskrit-English Dictionary',
            'url': 'https://www.sanskrit-lexicon.uni-koeln.de/scans/AEScan/2014/web/webtc/download/ae.zip',
            'files': ['ae.xml'],
            'description': 'Practical Sanskrit dictionary with modern usage'
        },
        'mw72': {
            'name': 'Monier-Williams 1872 Edition',
            'url': 'https://www.sanskrit-lexicon.uni-koeln.de/scans/MW72Scan/2020/web/webtc/download/mw72.zip',
            'files': ['mw72.xml'],
            'description': 'Historical edition of Monier-Williams'
        }
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize downloader

        Args:
            output_dir: Directory to save downloaded dictionaries
        """
        if output_dir is None:
            project_root = Path(__file__).parent.parent.parent
            output_dir = project_root / "data" / "dictionaries"

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dictionary download directory: {self.output_dir}")

    def download_file(self, url: str, output_path: Path, chunk_size: int = 8192) -> bool:
        """
        Download a file with progress indication

        Args:
            url: URL to download from
            output_path: Path to save file
            chunk_size: Download chunk size in bytes

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from {url}")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')

            print()  # New line after progress
            logger.info(f"Downloaded {downloaded} bytes to {output_path}")
            return True

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error during download: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """
        Extract ZIP or TAR archive

        Args:
            archive_path: Path to archive file
            extract_to: Directory to extract to

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Extracting {archive_path} to {extract_to}")

            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)

            elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)

            else:
                logger.error(f"Unsupported archive format: {archive_path.suffix}")
                return False

            logger.info("Extraction completed")
            return True

        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False

    def download_dictionary(self, dict_name: str) -> bool:
        """
        Download and extract a specific dictionary

        Args:
            dict_name: Name of the dictionary (from DICTIONARIES)

        Returns:
            True if successful, False otherwise
        """
        if dict_name not in self.DICTIONARIES:
            logger.error(f"Unknown dictionary: {dict_name}")
            logger.info(f"Available dictionaries: {', '.join(self.DICTIONARIES.keys())}")
            return False

        dict_info = self.DICTIONARIES[dict_name]
        logger.info(f"Downloading: {dict_info['name']}")
        logger.info(f"Description: {dict_info['description']}")

        # Download archive
        archive_name = dict_name + '.zip'
        archive_path = self.output_dir / archive_name

        if not self.download_file(dict_info['url'], archive_path):
            return False

        # Extract archive
        extract_dir = self.output_dir / dict_name
        extract_dir.mkdir(exist_ok=True)

        if not self.extract_archive(archive_path, extract_dir):
            return False

        # Verify expected files exist
        for expected_file in dict_info['files']:
            file_path = extract_dir / expected_file
            if file_path.exists():
                logger.info(f"✓ Found: {file_path}")
            else:
                logger.warning(f"✗ Missing expected file: {expected_file}")

        # Cleanup archive
        try:
            archive_path.unlink()
            logger.info(f"Removed archive: {archive_path}")
        except Exception as e:
            logger.warning(f"Could not remove archive: {e}")

        logger.info(f"✅ Successfully downloaded {dict_info['name']}")
        return True

    def download_all(self) -> Dict[str, bool]:
        """
        Download all available dictionaries

        Returns:
            Dictionary mapping dictionary names to success status
        """
        results = {}

        for dict_name in self.DICTIONARIES.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {dict_name}")
            logger.info(f"{'='*60}\n")

            results[dict_name] = self.download_dictionary(dict_name)

        return results

    def list_dictionaries(self):
        """Print information about available dictionaries"""
        print("\n📚 Available Sanskrit Dictionaries:\n")

        for idx, (key, info) in enumerate(self.DICTIONARIES.items(), 1):
            print(f"{idx}. {info['name']}")
            print(f"   Key: {key}")
            print(f"   Description: {info['description']}")
            print(f"   URL: {info['url']}")
            print()


def main():
    """Main function for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Cologne Digital Sanskrit Dictionaries"
    )
    parser.add_argument(
        '--dict',
        type=str,
        choices=list(DictionaryDownloader.DICTIONARIES.keys()) + ['all'],
        default='monier-williams',
        help='Dictionary to download (default: monier-williams)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for dictionaries'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available dictionaries'
    )

    args = parser.parse_args()

    downloader = DictionaryDownloader(
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    if args.list:
        downloader.list_dictionaries()
    elif args.dict == 'all':
        print("Downloading all dictionaries...")
        results = downloader.download_all()

        print("\n" + "="*60)
        print("Download Summary:")
        print("="*60)
        for dict_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"{dict_name}: {status}")
    else:
        success = downloader.download_dictionary(args.dict)
        if success:
            print(f"\n✅ Successfully downloaded {args.dict}")

            # Provide next steps
            print("\n📖 Next steps:")
            print(f"1. Load the dictionary:")
            print(f"   python -m src.preprocessing.dictionary --load-xml data/dictionaries/{args.dict}/*.xml")
            print(f"\n2. Test lookups:")
            print(f"   python -m src.preprocessing.dictionary --word rāma")
        else:
            print(f"\n❌ Failed to download {args.dict}")
            return 1

    return 0


if __name__ == "__main__":
    import sys
    from typing import Dict
    sys.exit(main())
