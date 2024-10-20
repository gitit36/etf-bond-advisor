# ETF & Bond Advisor

**Smart investment analysis tool for ETFs and bonds, delivering personalized recommendations and insights.**

The **ETF & Bond Advisor MVP** is a web application that leverages generative AI to support investment decision-making in the financial industry and maximize the efficiency of internal sales activities. This project is built on Flask and integrates real-time financial data through OpenAI's ChatGPT API and Yahoo Finance to provide users with valuable information.

---

## Table of Contents

- [Demo Video](#demo-video)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation and Execution](#installation-and-execution)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone and Install Dependencies](#2-clone-and-install-dependencies)
  - [3. Set Environment Variables](#3-set-environment-variables)
  - [4. Prepare Data Files](#4-prepare-data-files)
  - [5. Run the Application](#5-run-the-application)
- [Usage](#usage)
  - [Insight ETF](#insight-etf)
  - [Bond Tracker](#bond-tracker)
- [Contributing](#contributing)
- [License](#license)

---

## Demo Video

[<img src="https://img.youtube.com/vi/kUqeLoRT7-U/maxresdefault.jpg" width="50%">](https://youtu.be/kUqeLoRT7-U)

## Features

### Insight ETF

- **News Article Analysis:** Analyzes news article URLs provided by users to deliver relevant international ETF information.
- **Keyword Analysis:** Allows users to receive ETF recommendations based on one or more keywords.
- **Multi-Article and Highlight Analysis:** Analyzes single or multiple articles, including highlighted sections within the articles.
- **Real-Time Data Integration:** Uses OpenAI's ChatGPT to extract ETFs related to the articles and provides the latest closing prices through Yahoo Finance.
- **Reliable Information Sources:** Utilizes RAG (Retrieval-Augmented Generation) to gather information from global financial data providers such as Yahoo Finance, Bloomberg, and Morningstar.
- **Detailed ETF Information:** Provides detailed information including holdings weight, fees, expense ratios, trading patterns, and global trading volume.

### Bond Tracker

- **Industry-Specific Bond Data Analysis:** Analyzes corporate bond issuance history based on selected industries to monitor upcoming maturities.
- **Data-Driven Lead Generation:** Supports investment banking professionals in identifying potential clients based on data and conducting efficient sales activities.
- **Regulatory Monitoring:** Monitors regulations related to private placements and mezzanine financing in real-time for quick responses.
- **Operational Efficiency:** Automates manual record-keeping through generative AI and supports data-driven sales strategy development.

---

## Technology Stack

- **Backend:**
  - [Flask](https://flask.palletsprojects.com/) - Python-based web framework
  - [OpenAI API](https://openai.com/api/) - Communication with generative AI model (ChatGPT)
  - [yfinance](https://pypi.org/project/yfinance/) - Real-time stock and ETF data collection via Yahoo Finance API
  - [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) - Web scraping
- **Frontend:**
  - HTML5, CSS3 - Web page structure and styling
- **Database:**
  - CSV files - Storage for news articles, ETF lists, and bond data
- **Environment Management:**
  - [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

---

## Project Structure

```plaintext
etf_bond_mvp/
│
├── app.py
├── test_openai.py
├── requirements.txt
├── README.md
├── README.en.md
├── LICENSE
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


- **app.py:** Main file for the Flask application, containing route definitions and view logic.
- **requirements.txt:** List of Python packages required for the project.
- **templates/:** Directory containing HTML template files.
- **static/:** Directory for static files like CSS and images.
- **data/:** Data repository through CSV files.
- **utils/:** Utility scripts handling OpenAI API communication and data processing.
- **.env:** Environment variable file for storing API keys and secret keys.
- **.gitignore:** List of files and directories to be excluded from Git tracking.

---

## Installation and Execution

### 1. Prerequisites

- **Python 3.7 or higher:** [Download and Install](https://www.python.org/downloads/)
- **pip:** Python package manager (included by default with Python installation)
- **Virtual Environment (Optional):** Used to isolate project dependencies

### 2. Clone and Install Dependencies

```bash
# Clone the repository
git clone https://github.com/yourusername/etf_bond_mvp.git
cd etf_bond_mvp

# Create a virtual environment (Optional)
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 3. Set Environment Variables

Create a **.env** file in the project root and set it up as follows:

```env
OPENAI_API_KEY=your_actual_openai_api_key_here
SECRET_KEY=your_secure_flask_secret_key_here
```

- **OPENAI_API_KEY:** The API key issued by OpenAI.
- **SECRET_KEY:** A secret key for Flask session management. Set to a secure and unpredictable value.

**Example of Generating a Secret Key:**

```python
import secrets
print(secrets.token_hex(16))
```

### 4. Prepare Data Files

All data files in **data/*.csv** are populated with example data. You can input or update additional data as needed.

### 5. Run the Application

```bash
python app.py
```

**Accessing the Application:**

Navigate to [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your web browser.

---

## Usage

### Insight ETF

1. **Navigate to the Insight ETF Page:**
   - Click "Insight ETF" in the navigation bar.

2. **Input News Article URLs:**
   - Enter one or more news article URLs, separated by commas.
   - Example:
     ```
     https://n.news.naver.com/mnews/article/003/0012813808
     ```

3. **Get Recommended ETFs:**
   - Click the "Find Related Overseas ETFs by URL" button to start the analysis.
   - The results will display a list of related ETFs, holdings, descriptions, and the latest closing prices in table format.

4. **Input Keywords:**
   - Enter one or more keywords in the keyword input section.
   - Example:
     ```
     technology, healthcare
     ```

5. **Get Keyword-Based ETF Recommendations:**
   - Click the "Find Related Overseas ETFs by Keywords" button to start the analysis.
   - The results will display a list of related ETFs along with detailed information.

### Bond Tracker

1. **Navigate to the Bond Tracker Page:**
   - Click "Bond Tracker" in the navigation bar.

2. **Select Industry:**
   - Choose an industry of interest from the dropdown menu (e.g., finance, healthcare).

3. **Monitor Bond Issuance History:**
   - Click the "Monitor Bond Issuance History" button to view the bond issuance history related to the selected industry.
   - The results will be displayed in table format, including company name, industry, issuance date, maturity date, and amount.

4. **Regulatory Monitoring:**
   - Monitor private placement and mezzanine-related regulations in real-time to stay updated.

5. **Data-Driven Lead Generation:**
   - Use the analysis results to identify potential clients and support efficient sales activities.

---

## Contributing

Contributions are always welcome! If you would like to improve the project or add new features, please follow these steps:

1. **Fork the Repository:**
   - Fork this repository on GitHub.

2. **Create a New Branch:**
   ```bash
   git checkout -b feature/new-feature
   ```

3. **Commit Your Changes:**
   ```bash
   git commit -m "Add new feature"
   ```

4. **Push Your Branch:**
   ```bash
   git push origin feature/new-feature
   ```

5. **Create a Pull Request:**
   - Go to your forked repository on GitHub and click on the "Pull Requests" tab.
   - Click the "New Pull Request" button.
   - Select your feature branch and compare it with the main repository.
   - Provide a description of your changes and submit the pull request for review.

---

## License

This project is licensed under the [MIT License](LICENSE). For more details, refer to the [LICENSE](LICENSE) file.

