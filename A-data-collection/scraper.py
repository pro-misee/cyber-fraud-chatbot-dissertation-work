import requests
import json
from pathlib import Path
import pandas as pd
from site_scrapers import scrape_site, SITE_CONFIGS


class FraudGuidanceScraper:    
    def __init__(self, source_name, delay=3):
        self.source_name = source_name
        project_root = Path(__file__).parent.parent
        self.output_dir = project_root / "data_sources" / source_name / "scraped"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Academic Research Bot) AppleWebKit/537.36'
        })
    
    def scrape_site(self):
        if self.source_name not in SITE_CONFIGS:
            print(f"No configuration available for: {self.source_name}")
            return
        
        config = SITE_CONFIGS[self.source_name]
        scrape_site(
            base_url=config['base_url'],
            test_urls=config['test_urls'],
            source_name=self.source_name,
            save_callback=self._save_content,
            session=self.session,
            delay=self.delay
        )

    def _save_content(self, data, filename):
        """Save scraped content to JSON file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def compile_dataset(self):
        """Compile all scraped content into a structured dataset"""
        all_content = []
        
        for json_file in self.output_dir.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                # Handle both single objects and lists
                if isinstance(content, list):
                    all_content.extend(content)
                else:
                    all_content.append(content)
        
        if not all_content:
            print("No content found to compile")
            return pd.DataFrame()
        
        # Save compiled dataset
        df = pd.DataFrame(all_content)
        df.to_csv(self.output_dir / 'compiled_dataset.csv', index=False)
        df.to_json(self.output_dir / 'compiled_dataset.json', orient='records', indent=2)
        
        print(f"Compiled dataset with {len(all_content)} entries")
        return df


def main():
    """Main execution function"""
    print("UK Cyber Fraud Guidance Scraper - Dataset Expansion")
    print("======================================================")
    
    # Test new sources
    new_sources = ['cifas', 'ncsc', 'nca', 'ico']
    
    print(f"Available new sources: {list(SITE_CONFIGS.keys())}")
    print(f"Testing: {new_sources}")
    
    # Test one source at a time
    source_to_test = 'ncsc' 
    print(f"\n🧪 Testing {source_to_test.upper()} scraping...")
    
    scraper = FraudGuidanceScraper(source_to_test)
    scraper.scrape_site()
    
    # Compile final dataset
    dataset = scraper.compile_dataset()
    
    print(f"\nScraping completed. Dataset saved with {len(dataset)} entries.")
    print("Sample data structure:")
    if not dataset.empty:
        print(dataset.head(2).to_string())


if __name__ == "__main__":
    main()