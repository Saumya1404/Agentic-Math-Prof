from dotenv import load_dotenv
from tavily import TavilyClient
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
import os
import re
from typing import Optional
    
load_dotenv()


class WebTools:
    def __init__(self) -> None:
        # Initialize Groq client with API key
        self.client = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.3-70b-versatile"  # or your preferred Groq model
        )
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) if os.getenv("TAVILY_API_KEY") else None
        self.firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY")) if os.getenv("FIRECRAWL_API_KEY") else None

    def _optimize_query(self, query: str) -> str:
        """Optimize and clean the search query using LLM to make it more search-friendly"""
        try:
            # Basic cleaning first
            query = query.strip()
            # Remove repeated words (e.g., "days days" -> "days")
            query = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', query, flags=re.IGNORECASE)
            
            # Use LLM to optimize the query for web search
            optimization_prompt = f"""Transform the following query into an optimized web search query. 
The optimized query should:
1. Be clear and specific with proper keywords
2. Fix any typos, grammatical errors, or repeated words
3. For computational/mathematical queries: include relevant terms like "formula", "calculator", "how to calculate"
4. For financial/stock queries: include relevant terms like "stock price", "calculation", "investment"
5. Maintain the original intent and meaning
6. Be concise (max 100 words)
7. Use natural search-friendly language

Original query: {query}

Return only the optimized query, nothing else:"""
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant that optimizes search queries for web search engines. You excel at fixing typos, improving clarity, and making queries more search-friendly while preserving their original meaning."},
                {"role": "user", "content": optimization_prompt}
            ]
            response = self.client.invoke(messages)
            optimized = response.content.strip()
            
            # Remove quotes if the LLM wrapped the query in them
            optimized = re.sub(r'^["\']|["\']$', '', optimized)
            
            # Fallback to cleaned original if optimization fails
            return optimized if optimized and len(optimized) > 5 else query
            
        except Exception as e:
            # If optimization fails, return cleaned original query
            return query.strip()

    def _search_with_firecrawl(self, query: str) -> Optional[str]:
        """Perform a web search using Firecrawl"""
        if not self.firecrawl:
            return None
            
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
                return "\n".join(results) if results else None
            else:
                return None
        except Exception as e:
            return None

    def _search_with_tavily(self, query: str) -> Optional[str]:
        """Perform a web search using Tavily as fallback"""
        if not self.tavily_client:
            return None
            
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=5
            )
            
            if response and hasattr(response, 'results') and response.results:
                results = []
                for item in response.results[:5]:
                    title = getattr(item, 'title', 'N/A')
                    url = getattr(item, 'url', 'N/A')
                    content = getattr(item, 'content', 'N/A') if hasattr(item, 'content') else getattr(item, 'snippet', 'N/A')
                    
                    result_text = f"Title: {title}\n"
                    result_text += f"URL: {url}\n"
                    result_text += f"Content: {content[:200]}...\n"
                    results.append(result_text)
                return "\n".join(results) if results else None
            return None
        except Exception as e:
            return None

    def search(self, query: str):
        """Perform a web search with query optimization and fallback mechanisms"""
        if not query or not query.strip():
            return "Error: Empty query provided"
        
        # Optimize the query first
        optimized_query = self._optimize_query(query)
        
        # Try Firecrawl first
        results = self._search_with_firecrawl(optimized_query)
        
        # Fallback to Tavily if Firecrawl fails or returns no results
        if not results and self.tavily_client:
            results = self._search_with_tavily(optimized_query)
        
        if results:
            return results
        
        # If both fail, try with original query as last resort
        if optimized_query != query:
            results = self._search_with_firecrawl(query)
            if not results and self.tavily_client:
                results = self._search_with_tavily(query)
        
        if results:
            return results
        
        return f"No search results found for query: {query}. Try rephrasing your query or checking if the search terms are correct."

    def crawl(self, url: str, maxDepth: int, limit: int):
        """Crawl a website and extract content"""
        if not self.firecrawl:
            return "Error: Firecrawl API key not configured"
            
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
        if not self.firecrawl:
            return "Error: Firecrawl API key not configured"
            
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

    def scrape_urls(self, url: str):
        """Scrape content from URLs"""
        if not self.firecrawl:
            return "Error: Firecrawl API key not configured"
            
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