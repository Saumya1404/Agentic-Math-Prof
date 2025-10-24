import os
from typing import List
from dataclasses import dataclass
from tools.webtools import WebTools
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("webtools")

# Initialize WebTools (loads API keys from .env)
webtools = WebTools()


@dataclass
class QueryResponse:
    response_text: str
    sources: List[str]


@mcp.tool()
async def search(query: str) -> str:
    """Performs web searches and retrieves up-to-date information from the internet.
    Args:
    - query: Specific query or topic to search for on the internet

    Returns:
    - Search results with relevant information about the requested topic
    """
    try:
        search_results = webtools.search(query)
        return search_results
    except Exception as e:
        return f"Error performing search: {str(e)}"


@mcp.tool()
async def crawl(url: str, maxDepth: int, limit: int) -> str:
    """Crawls a website starting from the specified URL and extracts content from multiple pages.
    Args:
    - url: The complete URL of the web page to start crawling from
    - maxDepth: The maximum depth level for crawling linked pages
    - limit: The maximum number of pages to crawl

    Returns:
    - Content extracted from the crawled pages in markdown and HTML format
    """
    try:
        crawl_results = webtools.crawl(url, maxDepth, limit)
        return crawl_results
    except Exception as e:
        return f"Error crawling pages: {str(e)}"


@mcp.tool()
async def extract(
    url: list[str], prompt: str, enableWebSearch: bool, showSources: bool
) -> str:
    """Extracts specific information from a web page based on a prompt.
    Args:
    - url: The complete URL(s) of the web page to extract information from
    - prompt: Instructions specifying what information to extract from the page
    - enableWebSearch: Whether to allow web searches to supplement the extraction
    - showSources: Whether to include source references in the response

    Returns:
    - Extracted information from the web page based on the prompt
    """
    try:
        info_extracted = webtools.extract_info(
            url, enableWebSearch, prompt, showSources
        )
        return info_extracted
    except Exception as e:
        return f"Error extracting information: {str(e)}"


@mcp.tool()
async def scrape(url: str) -> str:
    """Scrapes content from a web page and returns it in markdown and HTML format.
    Args:
    - url: The complete URL of the web page to scrape

    Returns:
    - Scraped content from the web page with optional screenshot
    """
    try:
        url_scraped = webtools.scrape_urls(url)
        return url_scraped
    except Exception as e:
        return f"Error scraping url {url}: {str(e)}"


@mcp.tool()
async def analyze_content(content: str, prompt: str) -> str:
    """Uses Groq AI to analyze web content based on a specific prompt.
    Args:
    - content: The web content to analyze
    - prompt: Instructions for how to analyze the content

    Returns:
    - AI-generated analysis of the content
    """
    try:
        analysis = webtools.analyze_with_groq(content, prompt)
        return analysis
    except Exception as e:
        return f"Error analyzing content: {str(e)}"


if __name__ == "__main__":
    # Initialize and run server
    mcp.run(transport="stdio")