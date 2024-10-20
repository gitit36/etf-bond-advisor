# app.py

from flask import Flask, render_template, request, flash, jsonify
from utils.etf_utils import get_etf_insights, get_etf_insights_from_keywords
from utils.bond_utils import get_bond_opportunities, analyze_mezzanine_bonds, get_potential_issuers, get_similar_mezzanine_bonds, get_market_overview
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

if not app.secret_key:
    raise ValueError("SECRET_KEY is not set in environment variables.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insight_etf', methods=['GET'])
def insight_etf():
    return render_template('insight_etf.html', etfs=None)

@app.route('/insight_etf_url', methods=['POST'])
def insight_etf_url():
    article_urls = request.form.get('article_urls', '')
    if not article_urls:
        flash("Please enter at least one URL.", "error")
        return render_template('insight_etf.html', etfs=None)

    urls = [url.strip() for url in article_urls.split(',') if url.strip()]
    if not urls:
        flash("No valid URLs provided.", "error")
        return render_template('insight_etf.html', etfs=None)
    
    all_etfs = []
    for url in urls:
        insights = get_etf_insights(url)
        if isinstance(insights, str):
            all_etfs.append({'message': insights, 'url': url})
        elif isinstance(insights, list):
            for etf in insights:
                etf['url'] = url
                all_etfs.append(etf)
    
    if not all_etfs:
        flash("Failed to get ETF recommendations. Please try again later.", "error")
    
    return render_template('insight_etf.html', etfs=all_etfs)

@app.route('/insight_etf_keywords', methods=['POST'])
def insight_etf_keywords():
    keywords = request.form.get('keywords', '')
    if not keywords:
        flash("Please enter at least one keyword.", "error")
        return render_template('insight_etf.html', etfs=None)

    keywords_list = [keyword.strip() for keyword in keywords.split(',') if keyword.strip()]
    if not keywords_list:
        flash("No valid keywords provided.", "error")
        return render_template('insight_etf.html', etfs=None)
    
    insights = get_etf_insights_from_keywords(keywords_list)
    
    if isinstance(insights, str):
        flash(insights, "info")
        return render_template('insight_etf.html', etfs=None)
    
    if not insights:
        flash("Failed to get ETF recommendations. Please try again later.", "error")
    
    return render_template('insight_etf.html', etfs=insights)

@app.route('/bond_tracker', methods=['GET', 'POST'])
def bond_tracker():
    market_overview = get_market_overview()
    similar_bonds = []  # 기본값으로 빈 리스트 설정
    potential_issuers = get_potential_issuers(min_market_cap=0, max_debt_ratio=float('inf'), min_shareholder_ratio=0)

    if request.method == 'POST':
        if 'bond_form' in request.form:
            amount = float(request.form['amount'])
            interest_rate = float(request.form['interest_rate'])
            similar_bonds = get_similar_mezzanine_bonds(amount, interest_rate)
            print("SIMILAR BONDS: ", similar_bonds)
        elif 'issuer_form' in request.form:
            min_market_cap = float(request.form.get('min_market_cap', 0))
            max_debt_ratio = float(request.form.get('max_debt_ratio', 100))
            min_shareholder_ratio = float(request.form.get('min_shareholder_ratio', 0))
            months_to_maturity = int(request.form.get('months_to_maturity', 9))
            potential_issuers = get_potential_issuers(min_market_cap, max_debt_ratio, min_shareholder_ratio, months_to_maturity)

    if similar_bonds:
        max_amount = max(bond['총발행금액 (₩)'] for bond in similar_bonds)
    else:
        max_amount = 0

    debug_info = f"Max amount: {max_amount}\n"
    debug_info += "Similar bonds:\n"
    for bond in similar_bonds:
        debug_info += f"  {bond['종목명']}: {bond['총발행금액 (₩)']}\n"

    return render_template('bond_tracker.html', 
                           similar_bonds=similar_bonds, 
                           potential_issuers=potential_issuers,
                           market_overview=market_overview,
                           max_amount=max_amount,
                           debug_info=debug_info)

@app.route('/get_chart_data', methods=['POST'])
def get_chart_data():
    amount = float(request.form['amount'])
    interest_rate = float(request.form['interest_rate'])
    similar_bonds = get_similar_mezzanine_bonds(amount, interest_rate)
    
    labels = [bond['발행대상 회사'] for bond in similar_bonds]
    amounts = [bond['총발행금액 (₩)'] for bond in similar_bonds]
    interest_rates = [bond['표면이자율 (%)'] for bond in similar_bonds]
    
    return jsonify({
        'labels': labels,
        'amounts': amounts,
        'interest_rates': interest_rates
    })

if __name__ == '__main__':
    app.run(debug=True)
