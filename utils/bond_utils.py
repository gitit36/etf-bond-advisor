# utils/bond_utils.py

import pandas as pd
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data once when the module is imported
df_bonds = pd.read_csv('data/issue_decision_regular_report.csv')
df_quarterly = pd.read_csv('data/business_quarterly_report.csv')
df_market = pd.read_csv('data/market_info.csv')

def load_bond_data(filepath='data/bond_data.csv'):
    """
    Loads bond data from a CSV file.
    """
    try:
        bond_df = pd.read_csv(filepath)
        return bond_df
    except Exception as e:
        logger.error(f"Error loading bond data: {e}")
        return pd.DataFrame()

def get_bond_opportunities(industry):
    """
    Retrieves bond opportunities based on the selected industry.
    """
    bond_df = load_bond_data()
    if bond_df.empty:
        logger.error("Bond data is empty.")
        return []
    
    # Filter bonds based on industry and sort by maturity date
    print(type(industry))
    print(industry)
    filtered_bonds = bond_df[bond_df['산업'].notna() & (bond_df['산업'] == industry)]
    # filtered_bonds = bond_df[bond_df['산업'].str.contains(industry, na=False)]
    filtered_bonds = filtered_bonds.sort_values(by='만기일')
    print(filtered_bonds)
    return filtered_bonds.to_dict('records')

def analyze_mezzanine_bonds():
    df = pd.read_csv('data/issue_decision_regular_report.csv')
    
    analysis = {
        'total_issues': len(df),
        'avg_amount': df['총발행금액 (₩)'].mean(),
        'avg_interest_rate': df['표면이자율 (%)'].mean(),
        'most_common_issuer': df['발행대상 회사'].mode().iloc[0],
        'recent_issues': df.sort_values('발행일', ascending=False).head(5).to_dict('records')
    }
    
    return analysis

def get_potential_issuers(min_market_cap=0, max_debt_ratio=float('inf'), min_shareholder_ratio=0, months_to_maturity=9, max_results=10):
    try:
        today = datetime.now()
        maturity_date = today + timedelta(days=30*months_to_maturity)
        
        # Split '종목명' and take the first part
        df_bonds['종목명'] = df_bonds['종목명'].str.split().str[0]

        # 데이터 병합
        df_combined = pd.merge(df_market, df_quarterly, on='종목명', how='inner')
        # df_combined = pd.merge(df_combined, df_bonds[['종목명', '기발행사채만기일']], on='종목명', how='left')
        
        logger.info(f"Total companies before filtering: {len(df_combined)}")
        logger.debug(f"Columns in df_combined: {df_combined.columns}")
        
        # 최신 분기 데이터 선택
        latest_quarter = df_combined.filter(regex=r'\d분기 매출액').columns[-1]
        latest_profit = latest_quarter.replace('매출액', '영업이익')
        
        # 기본 필터링
        potential_issuers = df_combined[
            (df_combined['시가총액 (₩)'] >= min_market_cap) &
            (df_combined['부채비율'] <= max_debt_ratio) &
            (df_combined['대주주 지분율(%)'] >= min_shareholder_ratio)
        ]
        
        logger.info(f"Companies after all filtering: {len(potential_issuers)}")
        
        # '기발행사채만기일' 컬럼 확인 및 처리
        if '기발행사채만기일' in potential_issuers.columns:
            potential_issuers['기발행사채만기일'] = pd.to_datetime(potential_issuers['기발행사채만기일'], errors='coerce', format='%Y-%m-%d')
            potential_issuers = potential_issuers[potential_issuers['기발행사채만기일'] <= maturity_date]
            logger.info(f"Companies after maturity date filtering: {len(potential_issuers)}")
        else:
            logger.warning("'기발행사채만기일' column not found in the dataset")
        
        # 정렬 및 결과 제한
        potential_issuers = potential_issuers.sort_values('시가총액 (₩)', ascending=False).head(max_results)
        
        result_columns = ['종목명', '주가 (₩)', '시가총액 (₩)', latest_quarter, latest_profit, '부채비율', '대주주 지분율(%)']
        if '기발행사채만기일' in potential_issuers.columns:
            result_columns.append('기발행사채만기일')
        
        logger.debug(f"Final columns in result: {result_columns}")
        
        return potential_issuers[result_columns].to_dict('records')
    except Exception as e:
        logger.error(f"Error in get_potential_issuers: {str(e)}")
        return []

def get_similar_mezzanine_bonds(amount, interest_rate, num_similar=5, amount_tolerance=0.3, interest_rate_tolerance=2):
    # Calculate the lower and upper bounds for amount and interest rate
    amount_lower = amount * (1 - amount_tolerance)
    amount_upper = amount * (1 + amount_tolerance)
    interest_rate_lower = interest_rate - (interest_rate_tolerance / 100)
    interest_rate_upper = interest_rate + (interest_rate_tolerance / 100)
    
    # Filter bonds within the specified ranges
    similar_bonds = df_bonds[
        (df_bonds['총발행금액 (₩)'].between(amount_lower, amount_upper)) | 
        (df_bonds['표면이자율 (%)'].between(interest_rate_lower, interest_rate_upper))
    ]
    
    # Calculate differences
    similar_bonds.loc[:, 'amount_diff'] = abs(similar_bonds['총발행금액 (₩)'] - amount) / amount
    similar_bonds.loc[:, 'interest_rate_diff'] = abs(similar_bonds['표면이자율 (%)'] - interest_rate)
    
    # Calculate a combined similarity score (lower is more similar)
    similar_bonds.loc[:, 'similarity_score'] = similar_bonds['amount_diff'] + similar_bonds['interest_rate_diff']
    
    # Sort by similarity score and select top matches
    similar_bonds = similar_bonds.sort_values('similarity_score').head(num_similar)
    
    # 시가총액 정보 추가
    similar_bonds = pd.merge(similar_bonds, df_market[['종목명', '시가총액 (₩)']], on='종목명', how='left')
    
    print(f"Total bonds before filtering: {len(df_bonds)}")
    print(f"Bonds within amount range: {len(df_bonds[(df_bonds['총발행금액 (₩)'].between(amount_lower, amount_upper))])}")
    print(f"Bonds within interest rate range: {len(df_bonds[(df_bonds['표면이자율 (%)'].between(interest_rate_lower, interest_rate_upper))])}")
    print(f"Similar bonds found: {len(similar_bonds)}")
    
    return similar_bonds.to_dict('records')

def get_market_overview():
    total_market_cap = df_market['시가총액 (₩)'].sum()
    avg_shareholder_ratio = df_quarterly['대주주 지분율(%)'].mean()
    
    return {
        'total_market_cap': total_market_cap,
        'avg_shareholder_ratio': avg_shareholder_ratio,
        'num_companies': len(df_market)
    }
