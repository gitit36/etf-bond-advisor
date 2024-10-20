# ETF & Bond Advisor (ETF & 채권 어드바이저)

**Smart investment analysis tool for ETFs and bonds, delivering personalized recommendations and insights.

**ETF 및 채권 추적기 MVP**는 생성형 AI를 활용하여 금융 산업에서의 투자 의사결정을 지원하고, 내부 영업 활동의 효율성을 극대화하는 두 가지 핵심 서비스를 제공하는 웹 애플리케이션입니다. 이 프로젝트는 Flask를 기반으로 구축되었으며, OpenAI의 ChatGPT API와 Yahoo Finance를 통해 실시간 금융 데이터를 통합하여 사용자에게 유용한 정보를 제공합니다.

---

## 목차

- [특징](#특징)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [설치 및 실행](#설치-및-실행)
  - [1. 사전 요구 사항](#1-사전-요구-사항)
  - [2. 클론 및 의존성 설치](#2-클론-및-의존성-설치)
  - [3. 환경 변수 설정](#3-환경-변수-설정)
  - [4. 데이터 파일 준비](#4-데이터-파일-준비)
  - [5. 애플리케이션 실행](#5-애플리케이션-실행)
- [사용 방법](#사용-방법)
  - [인사이트 ETF (Insight ETF)](#인사이트-etf-insight-etf)
  - [채권 트래커 (Bond Tracker)](#채권-트래커-bond-tracker)
- [기여](#기여)
- [라이선스](#라이선스)

---

## 특징

### 인사이트 ETF (Insight ETF)

- **뉴스 기사 분석:** 사용자가 제공한 뉴스 기사 URL을 분석하여 관련 해외 ETF 정보를 제공합니다.
- **키워드 분석:** 한개 이상의 키워드로 연관 ETF 를 추천받을 수 있습니다.
- **다중 기사 및 하이라이트 분석:** 단일 또는 다중 기사, 그리고 기사 내 하이라이트된 부분을 분석할 수 있습니다.
- **실시간 데이터 통합:** OpenAI의 ChatGPT를 사용하여 기사와 관련된 ETF를 추출하고, Yahoo Finance를 통해 최신 종가를 제공합니다.
- **신뢰할 수 있는 정보 출처:** RAG(Retrieval-Augmented Generation)를 활용하여 Yahoo Finance, Bloomberg, Morningstar 등의 글로벌 금융 데이터 제공처에서 정보를 수집합니다.
- **상세 ETF 정보 제공:** 보유 종목 비중, 수수료, 운용보수, 거래 패턴, 글로벌 거래량 등 상세 정보를 함께 제공합니다.

### 채권 트래커 (Bond Tracker)

- **산업별 채권 데이터 분석:** 선택한 산업에 따른 기업의 채권 발행 내역을 분석하여 만기 도래 기업을 모니터링합니다.
- **데이터 기반 잠재 고객 발굴:** IB 전문 인력들이 데이터 기반으로 잠재 고객을 발굴하고, 효율적인 영업 활동을 수행할 수 있도록 지원합니다.
- **규제 모니터링:** 사모사채 및 메자닌 관련 규제를 실시간으로 모니터링하여 신속하게 대응할 수 있습니다.
- **업무 효율화:** 생성형 AI를 통해 수기 기록을 자동화하고, 데이터 기반의 영업 전략 수립을 지원합니다.

---

## 기술 스택

- **백엔드:**
  - [Flask](https://flask.palletsprojects.com/) - 파이썬 기반의 웹 프레임워크
  - [OpenAI API](https://openai.com/api/) - 생성형 AI 모델(ChatGPT)과의 통신
  - [yfinance](https://pypi.org/project/yfinance/) - 야후 파이낸스 API를 통한 실시간 주식 데이터 수집
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - 웹 스크래핑
- **프론트엔드:**
  - HTML5, CSS3 - 웹 페이지 구조 및 스타일링
- **데이터베이스:**
  - CSV 파일 - 뉴스 기사, ETF 목록, 채권 데이터 저장
- **환경 관리:**
  - [python-dotenv](https://pypi.org/project/python-dotenv/) - 환경 변수 관리

---

## 프로젝트 구조

```
etf_bond_mvp/
│
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── insight_etf.html
│   └── bond_tracker.html
├── static/
│   ├── style.css
│   └── spinner.gif
├── data/
│   ├── news_articles.csv
│   ├── etf_list.csv
│   ├── bond_data.csv
│   ├── business_quarterly_report.csv
│   ├── issue_decision_regular_report.csv
│   └── market_info.csv
├── utils/
│   ├── openai_utils.py
│   ├── etf_utils.py
│   └── bond_utils.py
├── .env
└── .gitignore
```

- **app.py:** Flask 애플리케이션의 메인 파일로, 라우트 정의 및 뷰 로직을 포함합니다.
- **requirements.txt:** 프로젝트에 필요한 파이썬 패키지 목록.
- **templates/:** HTML 템플릿 파일들이 위치하는 디렉토리.
- **static/:** CSS, 이미지 등 정적 파일들이 위치하는 디렉토리.
- **data/:** CSV 파일을 통한 데이터 저장소.
- **utils/:** 유틸리티 스크립트들로, OpenAI API 통신, 데이터 처리 등을 담당.
- **.env:** 환경 변수 파일로, API 키 및 비밀 키를 저장.
- **.gitignore:** Git에서 추적하지 않을 파일 및 디렉토리 목록.

---

## 설치 및 실행

### 1. 사전 요구 사항

- **Python 3.7 이상:** [다운로드 및 설치](https://www.python.org/downloads/)
- **pip:** Python 패키지 관리자 (Python 설치 시 기본 포함)
- **가상 환경 (선택 사항):** 프로젝트 의존성을 격리하기 위해 사용

### 2. 클론 및 의존성 설치

```bash
# 저장소 클론ㄴ
git clone https://github.com/yourusername/etf_bond_mvp.git
cd etf_bond_mvp

# 가상 환경 생성 (선택 사항)
python3 -m venv venv
source venv/bin/activate  # 윈도우: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 3. 환경 변수 설정

**.env** 파일을 프로젝트 루트에 생성하고 다음과 같이 설정합니다:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
SECRET_KEY=your_secure_flask_secret_key_here
```

- **OPENAI_API_KEY:** OpenAI에서 발급받은 API 키.
- **SECRET_KEY:** Flask 세션 관리를 위한 비밀 키. 안전하고 예측 불가능한 값으로 설정.

**비밀 키 생성 예시:**

```python
import secrets
print(secrets.token_hex(16))
```

### 4. 데이터 파일 준비

**data/*.csv** 모든 데이터 파일은 예시 데이터를 생성하여 활용하였습니다. 필요에 따라 추가적인 데이터를 입력하거나 업데이트할 수 있습니다.

### 5. 애플리케이션 실행

```bash
python app.py
```

**애플리케이션 접근:**

웹 브라우저에서 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)으로 이동합니다.

---

## 사용 방법

### 인사이트 ETF (Insight ETF)

1. **인사이트 ETF 페이지로 이동:**
   - 네비게이션 바에서 "인사이트 ETF"를 클릭합니다.

2. **뉴스 기사 URL 입력:**
   - 하나 이상의 뉴스 기사 URL을 쉼표로 구분하여 입력합니다.
   - 예시:
     ```
     https://n.news.naver.com/mnews/article/003/0012813808
     ```

3. **추천 ETF 받기:**
   - "Get ETF Recommendations" 버튼을 클릭하여 분석을 시작합니다.
   - 분석 결과로 관련 ETF 목록, 보유 종목, 설명, 최신 종가 등이 테이블 형식으로 표시됩니다.

### 채권 트래커 (Bond Tracker)

1. **채권 트래커 페이지로 이동:**
   - 네비게이션 바에서 "채권트래커"를 클릭합니다.

2. **산업 선택:**
   - 드롭다운 메뉴에서 관심 있는 산업을 선택합니다 (예: 금융, 헬스케어).

3. **채권 발행 내역 모니터링:**
   - "채권 발행 내역 모니터링" 버튼을 클릭하여 선택한 산업과 관련된 채권 발행 내역을 확인합니다.
   - 결과는 테이블 형식으로 표시되며, 회사명, 산업, 발행일, 만기일, 금액 등이 포함됩니다.

---

## 기여

기여는 언제나 환영입니다! 프로젝트를 개선하거나 새로운 기능을 추가하고 싶다면, 다음 단계를 따라주세요:

1. **저장소 포크:**
   - GitHub에서 이 저장소를 포크합니다.

2. **새 브랜치 생성:**
   ```bash
   git checkout -b feature/새로운-기능
   ```

3. **변경 사항 커밋:**
   ```bash
   git commit -m "Add 새로운 기능"
   ```

4. **브랜치 푸시:**
   ```bash
   git push origin feature/새로운-기능
   ```

5. **풀 리퀘스트 생성:**
   - GitHub에서 풀 리퀘스트를 생성하여 변경 사항을 제출합니다.

---

## 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE)를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참고하세요.
