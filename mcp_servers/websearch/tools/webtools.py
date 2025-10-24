from dotenv import load_dotenv
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
import os

load_dotenv()


class WebTools:
    def __init__(self) -> None:
        # Initialize Groq client with API key
        self.client = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"  # or your preferred Groq model
        )
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

    def search(self, query: str):
        """Perform a web search using Firecrawl"""
        try:
            response = self.firecrawl.search(query)
            # Convert SearchData object to string format
            if hasattr(response, 'web') and response.web:
                results = []
                for item in response.web[:5]:  # Limit to top 5 results
                    # Handle SearchResultWeb objects properly
                    title = getattr(item, 'title', 'N/A')
                    url = getattr(item, 'url', 'N/A')
                    content = getattr(item, 'content', 'N/A')
                    
                    result_text = f"Title: {title}\n"
                    result_text += f"URL: {url}\n"
                    result_text += f"Content: {content[:200]}...\n"
                    results.append(result_text)
                return "\n".join(results)
            else:
                return "No search results found"
        except Exception as e:
            return f"Error performing search: {str(e)}"

    def crawl(self, url: str, maxDepth: int, limit: int):
        """Crawl a website and extract content"""
        try:
            crawl_page = self.firecrawl.crawl_url(
                url,
                params={
                    "limit": limit,
                    "maxDepth": maxDepth,
                    "scrapeOptions": {"formats": ["markdown", "html"]},
                },
                poll_interval=30,
            )
            return crawl_page
        except Exception as e:
            return f"Error crawling pages: {str(e)}"

    def extract_info(
        self, url: list[str], enableWebSearch: bool, prompt: str, showSources: bool
    ):
        """Extract specific information from URLs using AI"""
        try:
            info_extracted = self.firecrawl.extract(
                url,
                {
                    "prompt": prompt,
                    "enableWebSearch": enableWebSearch,
                    "showSources": showSources,
                    "scrapeOptions": {
                        "formats": ["markdown"],
                        "blockAds": True,
                    },
                },
            )
            return info_extracted
        except Exception as e:
            return f"Error extracting information from page {url}: {str(e)}"

    def scrape_urls(self, url: list[str]):
        """Scrape content from URLs"""
        try:
            urls_scraped = self.firecrawl.scrape_url(
                url,
                params={
                    "formats": ["markdown", "html"],
                    "actions": [
                        {"type": "screenshot"},
                    ],
                },
            )
            return urls_scraped
        except Exception as e:
            return f"Error scraping url {url}: {str(e)}"
    
    def analyze_with_groq(self, content: str, prompt: str):
        """Use Groq to analyze content"""
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that analyzes web content."},
                {"role": "user", "content": f"{prompt}\n\nContent:\n{content}"}
            ]
            response = self.client.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error analyzing with Groq: {str(e)}"