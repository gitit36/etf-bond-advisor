# utils/etf_utils.py

import pandas as pd
from utils.openai_utils import generate_etf_recommendations
from utils.openai_utils import get_news_content
from utils.openai_utils import generate_etf_recommendations_for_keywords
import requests
from transformers import GPT2Tokenizer
from bs4 import BeautifulSoup
import yfinance as yf
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def truncate_text_transformers(text, model_name='gpt2'):
    """
    필요시 사용...
    When dynamically identifying article content from different URLs..
    Truncates article text according to API quota.
    """

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text)
    
    if len(tokens) < 10000:
        return text  # No truncation needed
    
    truncated_tokens = tokens[:10000]
    truncated_text = tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)
    
    return truncated_text


def scrape_article(url):
    """
    Scrapes the article text from the given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for HTTP errors

        # text = response.content.decode('utf-8')
        # truncated_text = truncate_text_transformers(text)
        # article = get_news_content(truncated_text)

        soup = BeautifulSoup(response.content, 'html.parser')

        article = soup.find('article', {'class': 'go_trans _article_content', 'id': 'dic_area'})

        for tag in article(['img', 'em', 'span', 'div']):
            tag.decompose()

        if not article:
            logger.error(f"Could not find the article content in URL: {url}")
            return None

        return article

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching the article from URL {url}: {e}")
        return None

def analyze_text_with_chatgpt(text):
    """
    Analyzes the given text using ChatGPT and returns the response.
    """
    chatgpt_response = generate_etf_recommendations(text)
    print("CHECK THIS OUT")
    print("==============")
    print(chatgpt_response)
    print("==============")
    return chatgpt_response



def parse_chatgpt_response(response, keywords=None):
    """
    Parses the ChatGPT response, expected to be in JSON format, and extracts ETF information.
    If the response is a string, it tries to load it as JSON.
    """
    # Handle the case where GPT response indicates no relevance
    if response == 'NO_RELEVANT_ETFS_FOUND':
        return '해당 기사와 관련된 ETF가 없습니다'
    
    response = "[" + response + "]"
    
    print("RESPONSE: ", response)

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        return '응답을 구문 분석할 수 없습니다.'

    def process_etf(etf):
        ticker = etf.get('ticker', '').upper()
        top5 = etf.get('top5', [])
        explanation = etf.get('explanation', '')
        holdings_weight = etf.get('holdings_weight', '')
        expense_ratio = etf.get('expense_ratio', '')

        if len(ticker) <= 5 and top5 and explanation:
            etf_info = {
                'ticker': ticker,
                'top5': top5,
                'explanation': explanation,
                'holdings_weight': holdings_weight,
                'expense_ratio': expense_ratio
            }
            if keywords:
                etf_info['search_keywords'] = ', '.join(keywords)
            return etf_info
        return None

    if isinstance(response_json, list):
        etf_list = [process_etf(etf) for etf in response_json if process_etf(etf)]
        return etf_list if etf_list else '응답에 유효한 ETF 정보가 없습니다.'
    
    elif isinstance(response_json, dict):
        etf_info = process_etf(response_json)
        return [etf_info] if etf_info else '응답에 유효한 ETF 정보가 없습니다.'

    return '응답에 유효한 ETF 정보가 없습니다.'



def get_etf_insights(url):
    """
    Given a URL, scrape the article, analyze it with ChatGPT, and get ETF insights.
    Returns either a string indicating no relevance or a list of ETF information.
    """
    article_text = scrape_article(url)
    if not article_text:
        return "해당 기사 내용을 가져오지 못했습니다."

    chatgpt_response = analyze_text_with_chatgpt(article_text)
    if not chatgpt_response:
        return "ETF 추천을 가져오는 데 실패했습니다."

    parsed_response = parse_chatgpt_response(chatgpt_response)
    print("PARSED_RESPONSE: ", parsed_response)
    if parsed_response == '해당 기사와 관련된 ETF가 없습니다':
        return parsed_response
    
    return parsed_response

def get_etf_insights_from_keywords(keywords):
    """
    Given a list of keywords, analyze them with ChatGPT and get ETF insights.
    Returns either a string indicating no relevance or a list of ETF information.
    """
    chatgpt_response = generate_etf_recommendations_for_keywords(keywords)
    if not chatgpt_response:
        return "ETF 추천을 가져오는 데 실패했습니다."

    parsed_response = parse_chatgpt_response(chatgpt_response, keywords)
    if parsed_response == '해당 키워드와 관련된 ETF가 없습니다':
        return parsed_response
    
    return parsed_response



