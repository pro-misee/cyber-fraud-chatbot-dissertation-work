import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def extract_content(soup):
    """Extract meaningful content from BeautifulSoup object"""
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()
    
    # Get main content areas
    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    
    if main_content:
        return main_content.get_text(separator='\n', strip=True)
    else:
        # Fallback to body content
        return soup.get_text(separator='\n', strip=True)


def is_relevant_link(href, link_text):
    """Determine if a link is relevant for fraud guidance including social media scams"""
    fraud_keywords = [
        'fraud', 'scam', 'phishing', 'identity', 'theft', 'online', 'cyber',
        'security', 'protect', 'safe', 'privacy', 'password', 'banking',
        'shopping', 'email', 'social', 'mobile', 'wifi', 'money', 'finance',
        # Social media specific terms
        'facebook', 'instagram', 'whatsapp', 'telegram', 'tiktok', 'snapchat',
        'linkedin', 'twitter', 'dating', 'romance', 'investment', 'crypto',
        # Recent scam types
        'deepfake', 'ai', 'voice', 'clone', 'muse', 'nft', 'marketplace',
        'ticket', 'concert', 'holiday', 'booking', 'delivery', 'parcel',
        # Current trends and alerts
        '2024', '2025', 'latest', 'new', 'recent', 'alert', 'warning', 'trend'
    ]
    
    text_lower = link_text.lower()
    href_lower = href.lower()
    
    return any(keyword in text_lower or keyword in href_lower for keyword in fraud_keywords)


def scrape_linked_pages(soup, base_url, source_name, save_callback, session, delay=3, max_links=12):
    """Find and scrape relevant linked pages"""
    links = soup.find_all('a', href=True)
    relevant_links = []
    
    for link in links:
        href = link['href']
        full_url = urljoin(base_url, href)
        
        # Filter for relevant links and same domain
        if (is_relevant_link(href, link.get_text()) and 
            base_url.split('/')[2] in full_url):
            relevant_links.append((full_url, link.get_text().strip()))
    
    print(f"Found {len(relevant_links)} relevant links to scrape")
    
    for i, (url, title) in enumerate(relevant_links[:max_links]):
        try:
            time.sleep(delay)
            response = session.get(url)
            response.raise_for_status()
            
            soup_page = BeautifulSoup(response.content, 'html.parser')
            content = extract_content(soup_page)
            
            if len(content.strip()) > 200:  # Only save meaningful content
                save_callback({
                    'url': url,
                    'title': title or f"{source_name} linked page {i+1}",
                    'content': content,
                    'scraped_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f'{source_name}_linked_{i+1}.json')
                
                print(f"Scraped linked: {title[:40]}...")
            
        except Exception as e:
            print(f"Error scraping linked page {url}: {e}")


def scrape_site(base_url, test_urls, source_name, save_callback, session, delay=3):
    """
    Generic site scraper that works for most fraud prevention websites
    
    Args:
        base_url: Base URL of the site (e.g., "https://www.citizensadvice.org.uk")
        test_urls: List of URLs to try for content
        source_name: Name for file prefixes (e.g., "citizensadvice")
        save_callback: Function to save content
        session: Requests session
        delay: Delay between requests
    """
    print(f"Starting {source_name} scraping...")
    
    successful_pages = 0
    
    for i, start_url in enumerate(test_urls):
        print(f"Testing URL {i+1}: {start_url}")
        
        try:
            response = session.get(start_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            content = extract_content(soup)
            
            # Check if we got meaningful content
            if len(content.strip()) > 200 and "JavaScript" not in content:
                print(f"Found content at: {start_url}")
                
                save_callback({
                    'url': start_url,
                    'title': soup.find('title').get_text() if soup.find('title') else f'{source_name.title()} Page {i+1}',
                    'content': content,
                    'scraped_date': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f'{source_name}_page_{i+1}.json')
                
                successful_pages += 1
                
                # Scrape linked pages for first successful page only
                if successful_pages == 1:
                    scrape_linked_pages(soup, base_url, source_name, save_callback, session, delay)
                
            else:
                print(f"No meaningful content at: {start_url}")
            
            time.sleep(delay)  # Rate limiting
            
        except Exception as e:
            print(f"Error accessing {start_url}: {e}")
    
    print(f"{source_name.title()} scraping completed - {successful_pages} pages scraped")


# Original site configurations
ORIGINAL_SITE_CONFIGS = {
    'actionfraud': {
        'base_url': 'https://www.actionfraud.police.uk',
        'test_urls': [
            'https://www.actionfraud.police.uk/reporting-fraud-and-cyber-crime',
            'https://www.actionfraud.police.uk/what-to-do-if-you-are-a-victim-of-fraud',
            'https://www.actionfraud.police.uk/'
        ]
    },
    'getsafeonline': {
        'base_url': 'https://www.getsafeonline.org',
        'test_urls': [
            'https://www.getsafeonline.org/protecting-yourself/'
        ]
    },
    'citizensadvice': {
        'base_url': 'https://www.citizensadvice.org.uk',
        'test_urls': [
            'https://www.citizensadvice.org.uk/consumer/',
            'https://www.citizensadvice.org.uk/about-us/',
            'https://www.citizensadvice.org.uk/'
        ]
    },
    'fca': {
        'base_url': 'https://www.fca.org.uk',
        'test_urls': [
            'https://www.fca.org.uk/scamsmart',
            'https://www.fca.org.uk/consumers/avoid-scams-unauthorised-firms',
            'https://www.fca.org.uk/consumers/protect-yourself-scams',
            'https://www.fca.org.uk/consumers'
        ]
    },
    'ukfinance': {
        'base_url': 'https://www.ukfinance.org.uk',
        'test_urls': [
            'https://www.ukfinance.org.uk/our-expertise/economic-crime/fraud-scams',
            'https://www.ukfinance.org.uk/fraud-and-scams',
            'https://www.ukfinance.org.uk/our-expertise/personal-banking/consumer-protection',
            'https://www.ukfinance.org.uk/'
        ]
    },
    'which': {
        'base_url': 'https://www.which.co.uk',
        'test_urls': [
            'https://www.which.co.uk/consumer-rights/scams',
            'https://www.which.co.uk/consumer-rights/advice/how-to-spot-a-scam-alFiz5h8mnJ9',
            'https://www.which.co.uk/consumer-rights/scams/card-fraud',
            'https://www.which.co.uk/consumer-rights/scams/reporting-scams'
        ]
    }
}

# Active configurations - NEW SOURCES
SITE_CONFIGS = {
    # NEW sources focusing on social media fraud, recent cyber crime news, and emerging scam types
    'cifas': {
        'base_url': 'https://www.cifas.org.uk',
        'test_urls': [
            'https://www.cifas.org.uk/newsroom',
            'https://www.cifas.org.uk/insights',
            'https://www.cifas.org.uk/secure/contentPOB/type/fraud-prevention-for-individuals'
        ]
    },
    'ncsc': {
        'base_url': 'https://www.ncsc.gov.uk',
        'test_urls': [
            'https://www.ncsc.gov.uk/news',
            'https://www.ncsc.gov.uk/guidance/phishing',
            'https://www.ncsc.gov.uk/guidance/shopping-online-securely',
            'https://www.ncsc.gov.uk/guidance/social-media-how-to-use-it-safely'
        ]
    },
    'nca': {
        'base_url': 'https://www.nationalcrimeagency.gov.uk',
        'test_urls': [
            'https://www.nationalcrimeagency.gov.uk/what-we-do/crime-threats/fraud',
            'https://www.nationalcrimeagency.gov.uk/news',
            'https://www.nationalcrimeagency.gov.uk/what-we-do/crime-threats/cyber-crime'
        ]
    },
    'ico': {
        'base_url': 'https://ico.org.uk',
        'test_urls': [
            'https://ico.org.uk/about-the-ico/media-centre/news-and-blogs/',
            'https://ico.org.uk/for-the-public/online/social-networking/',
            'https://ico.org.uk/for-the-public/personal-information/'
        ]
    }
}