# utils/openai_utils.py

import os
import openai
from dotenv import load_dotenv
import time
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    logger.error("OPENAI_API_KEY is not set. Please check your .env file.")
    raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

# Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Assume we have a collection of documents about ETFs and market trends
etf_documents = [
    """
    Markets broadened as we anticipated, and our Decathlon strategies were able to take advantage.

    With Central Banks around the world lowering their policy interest rates, the third quarter saw a broad-based rally in all asset-classes and geographies with the strongest performance taking place in previously lagging categories, like small capitalization and value stocks, emerging markets (especially China), and all types and maturities of bonds.
    Our strategies performed well in the quarter. We rotated from the market leading technology sectors into the laggard small-capitalization and value categories shortly before the herd and rode the trend for the majority of the rally.
    We are entering this quarter with the signal to tap the brakes a bit.
    After the strong rally, our investment systems are currently advising a reduction in risk-appetite. As a result, we recently positioned our portfolios more cautiously, lowering our equity exposure by 10% in our more aggressive models. For the near term, our models favor a significantly narrower selection of equity categories than three months ago, with the most attractive ETFs being in the financial services, healthcare (biotech), and electrical/ industrial sectors as well as India. Bonds of all categories are ranked highly, offering a modest incremental return if we have a market pause, and possibly, significant protection and relative performance if there is an adverse geopolitical event or market correction.
    The third quarter became a story of China and the Federal Reserve (Fed).
    Despite titling our last quarterly letter “Betting on a Broadening”, we never could have anticipated the near perfect reversal of market leadership we saw just days later at the onset of the third quarter. While the kindling was in place, it was Jerome Powell’s July speech, all but assuring imminent rate cuts, that provided the spark. Value outperformed growth, small caps outperformed large, and international outperformed domestic stocks[1]. This was a welcome reprieve against what had become a historically concentrated equity market. After China’s recent surge it is now amazingly far and away the best-performing equity market of the year.
    ￼
    Source: Bloomberg data 9/30/2019 through 9/30/2024. “Small caps” is referencing the S&P 600 Index; “Large cap” is referencing the S&P 500 Index; “Value” is referencing the CRSP US Large Cap Value Index; “Growth” is referencing CRSP US Large Cap Growth Index; U.S. is referencing S&P 500 Index; and International” is referencing MSCI All-Country World Ex-US Index
    The reversal in leadership did not come at the cost of weak performance for the prior leaders as nearly all assets did well in the quarter, driven by an expectation of better odds for a “soft landing” scenario. Economic growth and employment were weaker, but not too weak, while inflation has been subdued for some time. We suggested that a broadening would be welcome for our strategies, and it was, as performance amongst our higher-risk models was particularly strong. This quarter, the equities in our investment pool provided a tailwind to our performance as the average equity outperformed market cap indices. Most of our best-performing picks were within the financial sector, a sector whose fate is inextricably linked to economic health and interest rate policy. Energy provided the lone material performance detractor for the higher-risk models, while a general lack of fixed income duration, and the usage of alternatives, impacted the conservative model versus its target benchmark.
    We expect investors to continue to shift their focus from inflation to growth.
    At the beginning of the year, we stated that we believed investors’ focus would shift from inflation to growth. Now that Fed policy has shifted, we expect this shift to continue into 2025. While we believe the Fed could have lowered rates sooner, we are much more comfortable with the Fed’s narrative around interest rate policy after the last meeting. It is obvious that, on the margin, they are more focused on the risks to growth than inflation. It remains to be seen if the effects of higher rates will ultimately continue to cause pain on a lag or whether rate cuts will provide an immediate improvement in interest-sensitive areas such as housing and autos, which have had their demand substantially curtailed.
    As the market has moved considerably towards the prospect of a soft landing, we don’t see a compelling risk/reward for substantial equity risk. We believe, however, that intra-sector dispersion will increase, since the economy going forward is likely to be very different from the one of the past few years, potentially creating new relative winners and losers.
    While large-cap companies (particularly the very largest) have been able to navigate higher inflation with relative ease, the impact on smaller companies has been far more pronounced. Analogous to how top quintile households had an easier time absorbing inflation than bottom quintile. With inflation now in the rearview mirror, it’s possible a larger array of companies stands to benefit from a more predictable price environment.

    Source: Bloomberg, Russell 1000 and 2000 indices for 5-year period 9/30/2019 through 9/30/2024
    Negatives
    * Equities have run ahead of fundamentals. Investors have made large gains in a short time in many asset categories without having to wait for the underlying companies or economic facts to deliver.
    Stocks prices have increased despite near-term earnings expectations decreasing, with investors pushing out improvements into the less predictable future.

    Source: Bloomberg, earnings estimates from 9/30/2021 through 9/3/2024, large cap index is the S&P 500
    This is even more acute for smaller-cap companies, posing a risk to the broadening thesis.

    Source: Bloomberg, earnings estimates from 9/30/2021 through 9/3/2024, small cap index is the S&P 600
    * High valuations: The entire S&P 500 trades for 23.8x 2024 earnings[2], but multiples for the highest quality companies are even higher. The largest, Apple, sports a P/E of 34.7[3], among the highest in the companies’ own history.
    * Economic fatigue: Hiring is slowing. Both consumer credit and auto loan delinquencies are on the rise. There’s been a noticeable weakening of the low-end consumer.

    Source: Bloomberg data for 3/31/2003 through 9/30/2024, for all consumer loans and auto loans.
    *On-going political uncertainty ahead of U.S. Elections.
    * U.S. Government finances continue to appear unsustainable. In rough per-person terms, at a $1.6 trillion annual deficit, the government is spending $400 per month more than it is collecting in taxes. Neither presidential candidate appears willing to tackle this issue as the prospective options all require near-term economic pain.
    * Geopolitical tensions remain extraordinary. Russia/Ukraine, Israel/Hamas (increasingly Israel/Hamas/Hezbollah/Iran), US economic restrictions on commerce with China.
    * Possible contagion in organized labor strikes resulting in economic disruption and broad wage and generalized inflation. Wages are now firmly ahead of overall inflation which is good for aggregate consumers but may also create future inflation if it persists.

    Source: Bloomberg data for 5-year period 9/30/2019 through 9/30/2024, wage growth derived from Atlanta Fed wage tracker and CPI is US CPI for urban consumers.
    Positives
    * Wealth effect: Household wealth is increasing significantly with equity markets and house prices at all-time highs and bond prices largely recovered.
    *Future monetary policy looks more predictable. At current interest levels, retirees can earn a meaningful return on their savings. So middle- and upper-class households are in great economic shape. If the Federal Reserve lowers interest rates as projected, lower income families will have a tail wind as well.
    * Earnings from market leading companies have been great. There seems to be a new cost-conscious mindset across companies of all sizes, resulting in margin improvement and highly profitable growth. The application of new AI technologies to improve productivity may accelerate the trend.
    *Return of the Chinese growth engine. China’s late quarter market surge was driven by its relatively unprecedented stimulus plans. Thus far the Chinese economy has been a drag on global growth, albeit also potentially detracting from inflation. Should these measures help restore consumer confidence, which has been deeply shaken from the country’s weak housing market, China could be a very large contributor to Global growth.
    Concluding Thoughts
    Despite our own lukewarm outlook, our tactical strategies are poised to capitalize on any change to the prevailing market narrative. Our strategies’ strong trailing one-year performance, despite a narrow market environment for much of the period, gives us confidence that we are well-positioned to benefit disproportionately if the market broadening observed this quarter marks the start of a longer-term trend aligned with the new interest rate (cutting) cycle. The average investor’s portfolio, through inertia alone, is likely anchored heavily to recent winners, offering a unique opportunity to increase exposure to more tactical strategies or those that favor some overlooked sectors should the trend of a broader market continue.
    """,
    """
    The Multifaceted Impact of Federal Reserve Easing Cycles on Financial Markets
    Stringer Asset Management Oct 10, 2024
    Few entities wield as much influence on the economy as the U.S. Federal Reserve (Fed). When the Fed embarks on an easing cycle and lowers its target interest rate, the ripple effects are felt across various sectors of the economy. This piece explores the nuanced impacts of such a cycle with a particular focus on mortgages, consumer credit, and cash investments.
    Mortgage Market Dynamics
    The mortgage market is significantly affected by Fed policy, albeit in varying degrees depending on the type of mortgage. Two primary categories dominate the landscape: adjustable-rate mortgages (ARMs) and fixed-rate mortgages (FRMs).
    ARMs are generally more responsive to Fed rate cuts in the short term. These mortgages are typically benchmarked to shorter-term rates, which are more directly influenced by the Fed’s target rate. However, the impact on existing ARM holders can be delayed or muted due to several factors:
    1. Reset periods: Some ARMs may not adjust for several years after origination.
    2. Rate caps and floors: Many ARMs have built-in limits on how much rates can change, both per adjustment period and over the life of the loan.
    These features can create a lag between Fed action and mortgage rate adjustment. Conversely, new ARM borrowers might see more immediate benefits as initial rates begin to reflect the lower Fed funds rate.
    FRMs, or fixed-rate mortgages, march to a different drum. They are primarily impacted by the 10-year Treasury yield. These longer-term rates are more influenced by broader economic factors like economic growth expectations and inflation forecasts rather than short-term Fed policy. Consequently, FRM rates may not move in lockstep with Fed cuts, and borrowers might wait longer to see meaningful rate reductions.
    It’s crucial to note that all mortgages carry a premium over their benchmark rate. This premium is influenced by various factors including risk perceptions and the supply and demand dynamics of mortgage-backed securities. Thus, even when benchmark rates fall, mortgage rates may not always follow suit proportionally or immediately.
    Consumer Credit and Short-Term Borrowing
    While mortgages show a complex response to Fed easing, other forms of consumer credit tend to be more directly reactive. Short-term borrowing vehicles, such as home equity lines of credit (HELOCs) and some auto loans, are often tied to the prime rate. This rate typically moves in close correlation with the federal funds rate.
    Historical data shows that the prime rate often shadows Fed funds rate movements more closely than long-term mortgage rates. This means that consumers with prime-linked loans might see more immediate relief during an easing cycle. For instance, HELOC borrowers could experience lower monthly payments relatively quickly after Fed rate cuts.

    Cash Investments and Short-Term Savings
    For savers and conservative investors, a Fed easing cycle presents a different set of challenges. Cash and cash-equivalent investments, which have enjoyed higher yields in recent years, face downward pressure on returns during easing periods.
    Money market funds and high-yield savings accounts are typically quick to reflect lower rates, often adjusting within days or weeks of a Fed cut. Certificate of Deposit (CD) holders might be temporarily insulated if they’re locked into a fixed rate, but upon renewal, they’ll likely face lower yields.
    Short-term Treasury investors will also feel the impact as their securities mature and they are forced to reinvest at lower rates. This scenario underscores the importance of a diversified investment strategy that can weather various interest rate environments.
    Broader Economic Implications
    While the direct effects on borrowing and saving rates are significant, it’s important to consider the broader economic context of Fed easing cycles. These cycles are typically implemented to stimulate economic growth during periods of slowdown or in response to economic shocks.
    Lower interest rates can encourage business investment, consumer spending, and overall economic activity. This can lead to job creation and wage growth, potentially offsetting the reduced yields for savers. Additionally, lower rates often support asset prices including stocks and real estate, which can benefit investors and homeowners.
    However, prolonged periods of low interest rates can also have unintended consequences, such as inflating asset bubbles or encouraging excessive risk-taking in search of yield. This delicate balance highlights the challenging role the Fed plays in managing monetary policy.
    A Fed easing cycle sets in motion a complex series of adjustments across financial markets. While some effects like lower rates on certain consumer loans, may be relatively straightforward, others, such as the impact on mortgage rates and long-term investments, are subject to a variety of influencing factors. As a result, individuals and businesses must remain vigilant and adaptable by understanding that the full impact of Fed policy changes often unfolds gradually and unevenly across different sectors of the economy.
    """,
    """
    Markets Are Demonstrating Cautious Optimism
    Canterbury Investment Management October 10, 2024
    In this edition of Investor Insights, our commentary will explore current investor sentiment, comment on market participation, and show the current ranking of market sectors. The Chart of the Week featured is the Energy sector.
    Investor Sentiment
    As we approach the Presidential election, it’s an opportune time to gauge investor sentiment. The American Association of Individual Investors (AAII) publishes a weekly Investor Sentiment Survey, categorizing respondents as bullish, neutral, or bearish to provide insight into retail investors’ collective mood. Importantly, AAII views market sentiment as a contrarian indicator, suggesting that markets often move counter to prevailing expectations.
    The latest survey reveals that 46% of respondents are bullish on the markets, while 27% are bearish, with the remaining 27% neutral. Although bullish readings are on the higher end of historical averages, this represents a 5% decrease from the previous two weeks.
    This week’s survey included a bonus question regarding the November elections’ impact on stock market expectations. Only 11% of respondents indicated increased optimism, while 52% reported greater caution, and 33% saw no impact.
    In summary, while investors express caution heading into November, a high degree of market optimism persists. It’s worth noting that although sentiment is viewed as a contrarian indicator, and there is currently a high degree of optimism, it is good that investors are cautious. Remember, bull markets “crawl a wall of worry.” While the AAII Sentiment Survey offers valuable insights into market psychology, it should be considered alongside other indicators rather than in isolation when making investment decisions.
    Market Participation and Sector Ranks
    Global events currently appear to have little impact on the broad markets. The market’s Advance/Decline Line, which measures market breadth by comparing the number of advancing securities to declining ones, is at a high. This indicates that as market indexes rise, the proverbial rising tide is lifting most ships.
    Of the eleven US sector ETFs (State Street Select Sector SPDRs), all eleven are in one of Canterbury’s technical bull Market States. Canterbury also ranks the sectors according to their Volatility-Weighted-Relative Strength (VWRS). VWRS is risk adjusted strength. The current sector rankings are displayed in the table below. Note that the S&P 500’s largest sector, Information Technology, is ranked second-to-last on a risk-adjusted basis. While in a bull Market State, the sector has been twice as volatile as the first ranked sector, Utilities.
    VWRS Rank	Sector
    1	Utilities
    2	Real Estate
    3	Financials
    4	Industrials
    5	Communications
    6	Staples
    7	Discretionary
    8	Health Care
    9	Basic Materials
    10	Information Technology
    11	Energy
    Source: Canterbury Investment Management
    Chart of the Week: Energy
    This commentary’s Chart of the Week is interestingly the last ranked sector: Energy. The Energy sector is composed of stocks such as Exxon, Chevron, ConocoPhillips, and Schlumberger. The chart below shows the Energy sector ETF, XLE, along with some key points. Energy has gained some momentum in the news due to global events over the past week. From a technical perspective, it is beginning to break out of a technical channel and attempting to begin a new uptrend. See chart and points below.

    Source: Canterbury Investment Management. Chart created using Optuma Technical Analysis Software
    1. Downward Channel (March-Present):
        * Energy has been in a downward trending “channel” (established a sloped series of lower highs and lower lows).
        * 50-day and 200-day moving averages of price converged during this period.
    2. Recent Breakout:
        * In the last week, Energy sector prices have risen, and broken out of the upper end of the channel (resistance)
    3. MoneyFlow:
        * MoneyFlow: A “smart money” indicator based on volume and daily price movements
        * Ideal scenario: Strong upward moves on rising MoneyFlow, declines on flat/increasing MoneyFlow
        * Breakout occurred on strong, rising MoneyFlow
        * Note: While Energy was in the downward sloping channel, MoneyFlow was flat (positive divergence)
    This breakout, coupled with strong MoneyFlow, suggests a potential shift in the Energy sector’s trend. Canterbury continues to monitor for confirmation of a new uptrend.
    Bottom Line
    The general feeling right now is that investors are worried about the upcoming election and what impact it may or may not have on the markets. As shown by the sentiment survey, most investors are proceeding with caution, yet remain optimistic. Right now, markets are not reflecting any emotional environment. That could change but has not happened yet. Market participation has been strong, and rising tide is lifting most ships.
    Keep in mind that if volatility does decide to rear its ugly head, Adaptive Portfolio Management is a comprehensive process designed to navigate market volatility. Instead of buying, holding, and rebalancing, an Adaptive portfolio will rotate and adjust to accommodate whichever market environment comes next- bull or bear.
    """,
    """
    What Are Junk Bonds Saying About the Economy?
    Horizon Investments Oct 09, 2024
    By Mike Dickson, Ph.D.
    Lowest spreads in nearly three years
    Want more evidence that the U.S. economy is in good shape? Just take a look at the junk bond market.
    The spread between risky high-yield (or junk) corporate bonds and “safe” U.S. Treasuries has fallen to just 2.84 percentage points, as seen in the chart below. That’s the smallest spread since late 2021 and nearly a full percentage point lower than where it was in early August when there was labor market weakness and heightened recession fears.
    Here’s why that matters: When high-yield bond investors demand yields that are only slightly higher than the yield on comparable Treasuries, investors may see an economic environment with a low risk of corporate bond defaults. Given the string of positive news about the economy lately — strong consumer spending, robust GDP growth, and a surprising number of new jobs created — it makes sense that high-yield bonds have rallied this year (up 7.8% versus 3.4% for the bond market overall*). What’s more, high-yield bonds’ relatively short durations have helped the asset class avoid some of the recent volatility in the bond market.
    """,
    """
    Notes from the Desk: Yields Rise as Strong Labor Markets, "Goldilocks" Economy in Focus
    Sage Advisory   Oct 08, 2024
    Long-term US Treasury yields rose last week as investors digested mixed economic data that reinforced the idea of a “Goldilocks” economy — not too hot to require the Fed to pause/hike rates, not too cold to signal an imminent recession.
    The Bureau of Labor Statistics (BLS)’s September jobs report released on Friday exceeded expectations. Nonfarm payrolls increased by 254k jobs, well above the 159k expected, with positive revisions in each of the past two months. The unemployment rate cooled to 4.1% — better than expected – and job openings (JOLTS) stood at 8 million (vs. 7.7 million expected), signaling that jobs are still plentiful. Although job growth has cooled from the heady gains of the past two years, the job market remains robust enough to avoid raising recession alarms.

    The ISM Services Purchasing Managers’ Index (PMI) release from Thursday also suggested a continued economic expansion. September’s ISM Services PMI came in at 54.9, an improvement from the prior month’s 51.5 and better than expectations of 51.7. While the ISM employment survey results were weaker in both ISM Services and Manufacturing, they were overshadowed by strong labor data from the BLS.
    The combination of resurgent labor data and market pricing of a dovish Fed pushed Treasury yields higher last week, with the 10-year yield approaching 4%, among the highest levels since the onset of growth fears in late July. This is reflective of growing investor expectations that despite the outsized Fed cut in September, economic data may not warrant continued aggressive Fed action at the November and December meetings.
    A moderate economic expansion and an accommodative Fed is supportive of risk assets like equities, investment grade corporates, and high yield bonds, as the “Fed Put” limits the downside risk while risk assets continue to benefit if the economy is healthy. Corporate credit spreads are now at the lowest point of the year as investors price out the risk of default in a world of central bank easing and a lack of major economic recessions.

    The resilience of the labor market supports the notion of a “Goldilocks” economy — steady job creation and solid economic growth without the runaway inflation in the near term that could cause the Federal Reserve to pause or reverse its rate cuts. In the context of the current cutting cycle, this stability is especially favorable for asset markets as it allows the Fed to continue its rate reductions without having to deal with an imminent recession. The Fed’s ongoing cuts signal a supportive monetary policy stance, reinforcing the “Fed put” — the idea that the central bank will act to support markets, particularly in times of economic uncertainty, let alone in an environment where the economy remains in a solid expansion.
    """,
    """
    Q3 Recap: Value Begins to Take Leadership
    RiverFront Investment Group October 8, 2024

    By Dan Zolet, CFA, Associate Portfolio Manager
    SUMMARY
    * China led the way in Q3 after a huge rally in the final week.
    * US Value and International outpaced US Growth.
    * Weak US dollar improved developed international returns.
    Quarterly Recap: Fed Rate Cut and Chinese Stimulus Take the Spotlight
    The third quarter of 2024 culminated a year-long ‘pivot’ from the Federal Reserve (‘the Fed’). After quarters of speculation, the Fed surprised markets with a 50-basis point (basis point = 1/100th of a percent) cut, double what the market was expecting. As the Fed began their cutting in earnest, markets reacted with a bit of a ‘value’ rotation (as can be seen in both US Sector returns and International Returns).

    This entire quarterly market recap may have been focused solely on the Fed, if not for the People’s Bank of China (PBOC) announcing stimulus to help their ailing economy. In a single week, Chinese equities returned 21.2%, catapulting them, and thus emerging markets to the top of asset class returns for the quarter as shown in Table 1, below. With both Fed and PBOC stimulus, we believe it is important to look at the market’s reaction to determine where we expect things to shake out further.
    ￼
    US Sectors: ‘Value’ and Rate Sensitive Sectors Lead the WayTable 2 below shows sector performance. In the first 2 quarters, ‘growth’ sectors such as Technology and Communication Services posted the top returns, but in Q3 they trailed the S&P500. At the top of the third quarter return table were the rate sensitive sectors of Utilities and Real Estate. Both of these were boosted by the by the  well anticipated Fed cuts, in our view.

    Joining Utilities and Real Estate this quarter on the leader board were the ‘value’ sectors of Industrials, Financials, and Materials. The strong quarter from this trio could be pointing towards the start of a ‘value’ rotation, as discussed in last month’s Weekly View. Similar to Utilities and Real Estate, we believe lower rates should begin to bolster the fundamentals of these three sectors.
    On the other hand, Energy posted its second consecutive negative quarter with falling oil prices applying downward pressure to the sector. We remain bullish on Energy stocks, believing that oil prices will remain above the breakeven rates for US Energy companies, allowing them to continue to produce attractive free cash flows. Furthermore, looking forward, increasing tensions in the Middle East as well as recent Chinese stimulus could provide some upside to oil prices.
    International Stocks: China Rallies, Japan LagsMoving to Table 3 below, China had far and away the highest quarterly returns of any major market. As mentioned above, the majority of these returns came in the last week of the quarter, after the announcement of their stimulus package. While this economic stimulus could be a significant tailwind for the next few months or quarters, we still have reservations regarding China’s long-term prospects. Both their domestic and global political agenda place their equity markets in a tough spot for international investors. Despite this caution, we will continue to keep an eye on the effects of the stimulus.
    When looking at developed markets, the thing that sticks out is how strong currency returns were in the third quarter relative to the US dollar. This led to much higher returns for US-based investors versus local investors. While the weakening dollar boosted returns this quarter, this could create a headwind for more export driven markets in the future. For these exporters, a stronger domestic currency makes exports more expensive, and less attractive for foreigners. Specifically, we believe a strong yen and pound can often hurt the local returns of Japanese and UK equities.
    While we have been underweight International relative to our policy benchmarks because we believe in diversification for times like these, our portfolios maintained an allocation to international markets. Once again, we will have to determine whether this is another short-term relative bounce or a more sustainable trend.

    Looking Forward: Portfolio Positioning Remains Mostly the Same
    One positioning point of note is our cyclical exposure across our portfolios. In both the short and long horizon portfolios we hold international value and large cap industrials, while the long horizon portfolios also hold small cap equities. These positions give us exposure to the burgeoning value rotation we were seeing in both returns and earnings. Additionally, we still hold positions in large cap technology across our portfolios.
    We remain underweight Chinese equites in any portfolio with China exposure in the benchmark. However, as mentioned above, we will look for effects of the Chinese stimulus, specifically on company earnings, in order to determine if we need to pivot our positioning. We could do this either through direct China exposure or using Chinese-sensitive securities headquartered outside of China.
    Finally, our portfolios are underweight fixed income relative to our asset allocation benchmarks. While the Fed began their cutting with a surprising 50 bps (bps = basis point) cut, we believe the rate path won’t be as clear cut as the market believes. Specifically, we believe that there is some upside to 10-year yields, and we will wait for higher long-term rates or a shift in economic and monetary fundamentals before we add duration back to our portfolios.
    """,
    """
    It's a Volatile World
    GLOBALT Investments   Oct 08, 2024
    By Kimberly Woody, Senior Portfolio Manager
    Our Outlook
    Supply chain disruptions related to the port workers’ strike loom, the impacts of which we know can be incredibly destructive. The not-so-distant memory of supply chain dislocation will hopefully spur action on the part of the administration to facilitate a swift resolution. War in the Ukraine and unrest in the Middle East threaten oil supply at the minimum, but we can only hope that cooler heads have no appetite for world or even worse nuclear war. Finally, there seems to be no desire for economic austerity in Washington. Given the current peace and economic stability, running such a substantial deficit is inexcusable.
    Domestically, we seem to be navigating the elusive soft landing. Inflation is cooling in response to rate policy while not crushing growth. Corporate earnings are projected to resume double digit growth, employment imbalances have eased, and mortgage rates are moderating to a level that will ideally promote neither excess nor stagnation. The Fed’s measured and methodical approach to the economic fallout from the pandemic seems to have dodged the economic calamity envisioned by many.
    With most central banks cutting rates and bond yields declining around the world, gold has been the major beneficiary. The gold uptrend is especially decisive, and we would posit “don’t fight the tape.” Gold’s 50-day and 200-day moving averages are both rising. A broader explanation for the dollar’s weakness is that above and beyond policy rates, the aggregate picture is one of rate differentials that have narrowed. So even while the Bank of Japan tightened rates and both the European Central Bank and the Bank of England ease, the Pound, Euro and Yen have appreciated versus the US dollar. A stronger dollar threatens gold’s run, but given tepid pessimism for the dollar (i.e. not oversold) we don’t see strength meaningful enough to halt the strength in gold.
    Amid America’s deep political divide, the rhetoric endemic to each election suggests the outcome will impact the cultural, economic, and social survival or collapse of America as we know it. Perhaps this enmity is what keeps opposing powers in check as we continue to wade through the turmoil ultimately finding ourselves in the greatest country, without question, on earth.
    Third Quarter Review
    The stock market experienced a mixture of volatility and steady growth throughout the third quarter. The S&P 500 ended the quarter with a healthy gain of 5.9%. In the first half of 2024, technology and communication services had been the primary drivers of returns but lagged as investor preference shifted to more dividend focused, rate sensitive companies in the real estate and utilities sectors. The rotation was also likely motivated by both sectors’ relative underperformance in the first six months of the year. Despite some third quarter consolidation, the technology and communication services sectors are still up 30.3% and 28.8% through the end of the quarter, respectively, with utilities completely closing the former gap with the Magnificent Seven-heavy sectors up 30.6 year to date.
    Also closing a performance gap in the third quarter were the broader growth and value indexes. The Russell 1000 Growth had outperformed the value index by 14.1% through the first half of 2024 after beating it by 31.2% in 2023 leaving a historically wide disparity in performance. This was likely driven by bullishness surrounding real and forecasted earnings strength related to everything AI, but also the compounding effects of index publishers pushing concentrations of those stocks to record levels. The Magnificent 7 peaked mid-July at 56.9% of the Russell 1000 Growth. Compare this to the weight of those same stocks beginning 2023 at 35.3% and 2004 at 47.2% and currently at 54.1%. Similarly, small caps as measured by the Russell 2000, asserted themselves in the third quarter, making up some ground but still lagging year to date versus the large and mid-cap indexes.
    First half headwinds for fixed income reversed during the third quarter, as demand picked up primarily due to moderating gauges of inflationary pressures and the Fed’s mildly dovish stance. US Treasury yields fell dramatically over the quarter, reflecting expectations of easing monetary policy from the Federal Reserve. The 10-year Treasury yield ended the quarter at 3.79%, down from 4.37% to start the quarter. But yield curve moves are rarely linear, and the real action occurred in the 1–3-year tenor. For example, the 1-year rate dropped almost 110 basis points in just 90 days. The chart below shows absolute changes in the curve during the quarter. What is not shown is the volatility in yields to which we’ve become accustomed since the arrival of the pandemic. Despite smaller moves in the long bond, given their larger duration, the Bloomberg 20+ Year Treasury index was up 8.0% in the quarter versus the 1-3 Year Treasury index which was up 4.1%, highlighting not only duration risk associated with short term bonds, but also reinvestment risk.

    International markets were customarily mixed. After a strong 2023, Mexico and Brazil posted double digit negative year to date returns of -18.5% and -12.9%. Conversely, China’s markets are raging in response to the government’s most aggressive stimulus package since COVID. The announcements came ahead of China’s Golden Week holiday, exacerbating trading volumes and likely speculation. The stimulus package included interest rate cuts and support for the beleaguered real estate and stock markets. The objective is to spur economic activity in hopes of achieving the People’s Bank of China’s (PBOC) 5% percent growth target for the year. Also introduced were tools to support capital markets allowing funds, insurers and brokers easier access to funding in order to buy stocks.
    China’s household spending is less than 40% of annual economic output, some 20 percentage points below the global average. Investment, by comparison, is 20 percentage points above. For reference, it took Japan 17 years to raise the consumption share of its economic output by 10 percentage points from its bottom in 1991. The PBOC’s stimulus measures, while not insignificant, are ostensibly handouts involving more debt and more money supply and likely to create lasting impact. The strategy has shown little success in creating real, sustainable economic growth.
    With the third quarter behind us we focus on the upcoming earnings season. In aggregate, analysts are more pessimistic about earnings than usual, while companies are less so as measured by revisions to earnings outlooks. Analysts estimates for Q3 2024 have dropped by 3.8% per share since June 30, which exceeds the 5-year and 10-year averages of -3.3%. But 55% of S&P 500 companies have issued negative EPS guidance, below the 5-year average of 58% and the 10-year average of 62%. While S&P 500 earnings for Q3 are now lower than they were at the beginning of the quarter, analysts predict double-digit earnings growth beginning in Q4 2024. Specifically, earnings growth rates are projected to be 14.9% for Q4 2024, and over 14% for the first two quarters of 2025. Annual earnings growth for 2024 and 2025 is expected at 10.0% and 15.1%, respectively. In terms of winners, eight of eleven sectors are forecasted to report earnings growth, with information technology, health care, and communication services showing double-digit growth. The energy sector is predicted to see a double-digit decline. In terms of valuation, the forward 12-month P/E ratio is 21.6, above the 5-year average of 19.5 and the 10-year average of 18.0, as well as the end of Q2’s ratio of 21.0.
    Source: Factset
    """,
    """
    ETF Industry KPI – 10/7/2024
    Tidal Financial Group   Oct 07, 2024
    During the first week of October, the ETF industry saw 23 new launches, 12 fund closures, and 1 ticker change.
    * The current 1 Year ETF Open-to-Close ratio sits at 3.17.
    * The total number of US ETFs has risen to 3,793.
    Our Toroso ETF Industry Index, which tracks companies generating revenue from the ETF ecosystem, rose 0.67% last week, underperforming the S&P Financial Select Sector Index, which rose by 1.14%.
    ETF activity from the past week includes:
    * Tradr ETFs Launches Industry’s First Quarterly Reset Leveraged ETFs: Tradr ETFs has launched the first leveraged ETFs with a quarterly reset, offering the longest leveraged investment horizon in the ETF industry. The company also expanded its Calendar Reset Leveraged ETF lineup to 12 products by introducing a new monthly reset ETF. These new offers target quarterly performance of SPY, QQQ, and TLT. (SPYQ, QQQP, TLTQ).
    * 2 Issuers Launch Debut Products
    * 3EDGE Asset Management makes a strong entrance into the market with the debut of four active ETFs: focusing on Fixed Income, Hard Assets, International Equity, and US Equity. 3EDGE manages around $1.8 billion in assets for individuals, family offices, and institutional clients.
    * Eventide Asset Management, an investment adviser managing US$7 billion in assets as of Dec 2023, has launched its first ETF, the Eventide High Dividend ETF (ELCV), an actively managed fund with a 0.49% expense ratio. ELCV focuses on dividend-paying securities to provide income, income growth, and long-term capital appreciation, aiming to exceed the average yield of the Bloomberg US 3000 Total Return Index.
    * Defined Outcome ETFs continue to launch: New ETF launches in 2024 continue to be dominated by Buffered and Alternative Protection ETFs, with last week seeing a surge in these products. Innovator led the way with three Defined Protection ETFs, while Calamos continued to expand its lineup of alternative protection offerings. Approximately 30% of last week’s ETF launches were in this category.

    ETF Launches
    3EDGE Dynamic Fixed Income ETF (ticker: EDGF)3EDGE Dynamic Hard Assets ETF (ticker: EDGH)3EDGE Dynamic International Equity ETF (ticker: EDGI)3EDGE Dynamic US Equity ETF (ticker: EDGU)AllianzIM U.S. Equity Buffer15 Uncapped Oct ETF (ticker: OCTU)Astoria US Quality Growth Kings ETF (ticker: GQQQ)Calamos Russell 2000 Structured Alt Protection ETF – October (ticker: CPRO)Calamos S&P 500 Structured Alt Protection ETF (ticker: CPSO)Direxion Daily NFLX Bull 2X Shares ETF (ticker: NFXL)Direxion Daily TSM Bear 1X Shares ETF (ticker: TSMZ)Direxion Daily TSM Bull 2X Shares ETF (ticker: TSMX)Eventide High Dividend ETF (ticker: ELCV)First Trust New Constucts Core Earnings Leaders ETF (ticker: FTCE)Innovator Equity Defined Protection ETF – 1 Year October (ticker: ZOCT)Innovator Equity Defined Protection ETF – 2 Year to October 2026 (ticker: AOCT)Innovator Equity Defined Protection ETF – 6 Month April/October (ticker: APOC)iShares Large Cap Max Buffer Sep ETF (ticker: SMAX)NEOS Enhanced Income Credit Select ETF (ticker: HYBI)Roundhill China Dragons ETF (ticker: DRAG)Tradr 1.75X Long TLT Quarterly ETF (ticker: TLTQ)Tradr 2X Long SPY Quarterly ETF (ticker: SPYQ)Tradr 2X Long TLT Monthly ETF (ticker: TLTM)Tradr 2X Long Triple Q Quarterly ETF (ticker: QQQP)

    ETF Closures
    Blue Horizon BNE ETF (ticker: BNE)Goose Hollow Enhanced Equity ETF (ticker: GHEE)Mohr Industry Nav ETF (ticker: INAV)KraneShares S&P Pan Asia Dividend Arisotcrats ETF (ticker: KDIV)KraneShares CICC China 5G & Semiconductor ETF (ticker: KFVG)KraneShares Electification Metals Strategy ETF (ticker: KMET)Natixis Loomis Sayles Short Duration Income ETF (ticker: LSST)Mohr Growth ETF (ticker: MOHR)Syntax Stratified MidCap ETF (ticker: SMDY)Syntax Stratified SmallCap ETF (ticker: SSLY)Syntax Stratified Total Market II ETF (ticker: SYII)Syntax Stratified U.S. Total Market ETF (ticker: SYUS)

    Fund/Ticker Changes
    Ecofin Global Water ETG Fund (ticker: EBLU)became Tortoise Global Water ETG Fund (ticker: TBLU)
    Syntax Stratified MidCap ETF (ticker: SMDY), Syntax Statified SmallCap ETF (ticker: SSLY), Syntax Stratified Total Market II ETF (ticker: SYII) & Syntax Statified U.S. Total Market ETF (ticker: SYUS)became Stratified LargeCap Index ETF (ticker: SSPY)
    """,
    """
    China Takes Off
    Horizon Investments   Oct 04, 2024
    Long-suffering Chinese stocks broke out of their doldrums last week following the announcement of a massive monetary stimulus plan aimed at helping the slumping Chinese economy hit its economic growth target.

    The FTSE China 50 soared 16.2% for the week—the index’s biggest weekly return since 2007 (see the chart). Other China stock indices have also posted similar historical gains.

    Weekly Return of FTSE China 50 Index
    FTSE China 50 Index
    Source: Bloomberg, calculations by Horizon Investments, data as of 09/30/2024. Indices are unmanaged and do not have fees or expense charges, both of which would lower returns. It is not possible to invest directly in an unmanaged index.
    The Chinese government made several key decisions that fueled the rally, including:

    A roughly 50-basis-point interest rate cut on existing mortgage loans.
    A reduction in the amount of cash reserves banks must set aside.
    Fewer restrictions on borrowing money to invest in equities.
    Given China’s position as the world’s second-largest economy, these and other stimulus measures helped boost international equities in Europe, emerging markets, and in particular, sectors that stand to benefit from financially stronger Chinese consumers (such as luxury goods, metals and mining, and materials).

    China’s big gains last week—which have continued into this week thus far—are another reminder of potential opportunities for investors looking beyond U.S. stocks. But while intriguing, the current situation does demand some caution. Chinese stocks have generally declined since peaking in late 2020, and there have been several false starts since then. We will be searching for signals that this most recent stimulus round has truly taken root in market pricing and is being fully appreciated by investors.

    By Mike Dickson, Ph.D.
    """
]

# Encode the documents
document_embeddings = model.encode(etf_documents)

def retrieve_relevant_documents(query, top_k=3):
    """Retrieve the most relevant documents for the given query."""
    print("QUERY: ", query)
    
    # Check if query is a string
    if not isinstance(query, str):
        print(f"Warning: Query is not a string. Type: {type(query)}")
        # Try to convert to string if possible
        try:
            query = str(query)
        except:
            print("Error: Could not convert query to string.")
            return []
    
    # Check if query is empty
    if not query.strip():
        print("Warning: Query is empty.")
        return []
    
    try:
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, document_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [etf_documents[i] for i in top_indices]
    except Exception as e:
        print(f"Error in retrieve_relevant_documents: {e}")
        return []

def generate_etf_recommendations(text, retries=3, backoff_factor=2):
    """
    Sends a prompt to OpenAI API and returns the response.
    Implements exponential backoff for handling RateLimitError.
    Incorporates RAG using vector-based retrieval.
    """
    print(f"Input text type: {type(text)}")
    text = str(text)
    print(f"Input text type2: {type(text[:100])}")
    print(f"Input text: {text[:100]}...")  # Print first 100 characters
    
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(text)
    print(f"Number of relevant docs retrieved: {len(relevant_docs)}")
    
    # Construct the prompt with retrieved information
    rag_context = "\n".join(relevant_docs)
    
    prompt = f"""
    Here is the news article or text input: '{text}'.

    Additionally, here is relevant context about ETFs and market trends:
    {rag_context}

    Based on this input and the provided context, perform the following tasks:

    1. Identify the most relevant industries, sectors, or trends that could impact the U.S. stock markets (S&P 500, NASDAQ, Dow Jones). Focus on macro-level economic impacts derived from the article. For example, an article about conflict in the Middle East may affect not only the defense sector but also gold, energy, or the broader U.S. market due to hedging movements.
    2. For each identified industry or trend, list specific ETF tickers and their top 5 holdings (stocks) that are directly related to these sectors. Ensure the ETFs you suggest are relevant, established in U.S. markets, and very closely tied to the article's context.
    3. Include at least one leveraged ETF (e.g., 2x or 3x) if applicable.

    Return the response strictly and only in the following JSON format:

    {{
        "ticker": "<ETF Ticker>",
        "top5": ["<Top 1 stock>", "<Top 2 stock>", "<Top 3 stock>", "<Top 4 stock>", "<Top 5 stock>"],
        "explanation": "<Detailed explanation of how this ETF relates to the news article, including its relation to the identified trend or industry, and a brief description of the ETF itself>",
        "holdings_weight": "<The weight of the each stocks in the ETF holdings>",
        "expense_ratio": "<The expense ratio of the ETF>"
    }}

    Important notes:

    - If the article has absolutely no relevance to U.S. stock market-listed ETFs, return the response: NO_RELEVANT_ETFS_FOUND
    - The entire response must be written in Korean with a professional, expert tone.
    - Ensure that the explanations fully describe the relationship between the inferred ETFs and the input text.
    - Include a detailed description of each ETF, focusing on how the ETF is impacted by the identified trend or industry.
    - Make sure that the recommended ETFs are stocks that one would actually invest in (BUY), and are not just general market or sector funds.
    - Make sure that the recommended ETFs and their justifications are acceptable to a professional investor.

    Strictly follow the syntax and instructions provided.
    ONLY RETURN THE JSON OUTPUT.
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",  # Corrected model name
                messages=[
                    {"role": "system", "content": "You are a financial expert and stock market analyst."},
                    {"role": "user", "content": prompt}
                ],
                n=1,
                stop=None,
                temperature=0
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer
        except openai.error.RateLimitError as e:
            logger.warning(f"RateLimitError encountered: {e}. Attempt {attempt + 1} of {retries}.")
            sleep_time = backoff_factor ** attempt
            logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            break  # Do not retry on other OpenAI errors
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            break  # Do not retry on non-OpenAI errors

    logger.error("All retries failed. Unable to get ETF recommendations.")
    return None


def get_news_content(text, retries=3, backoff_factor=2):
    """
    Sends a prompt to OpenAI API and returns the response.
    Implements exponential backoff for handling RateLimitError.
    """
    # print(text)
    # print("LENGTH: ", len(text))

    prompt = f"""
    You are an expert at extracting information from HTML content.

    Here is a news article HTML:

    {text}

    Please identify and extract the **news article title** and **content** from the provided HTML. 

    Return the output in the following exact format without any additional text or explanations:

    News article title - [Insert Title Here]
    News article content - [Insert Content Here]
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Corrected model name
                messages=[
                    {"role": "system", "content": "You are a an experienced news article scraper."},
                    {"role": "user", "content": prompt}
                ],
                n=1,
                stop=None,
                temperature=0.1
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer
        except openai.error.RateLimitError as e:
            logger.warning(f"RateLimitError encountered: {e}. Attempt {attempt + 1} of {retries}.")
            sleep_time = backoff_factor ** attempt
            logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            break  # Do not retry on other OpenAI errors
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            break  # Do not retry on non-OpenAI errors

    logger.error("All retries failed. Unable to get ETF recommendations.")
    return None

def generate_etf_recommendations_for_keywords(keywords, retries=3, backoff_factor=2):
    """
    Sends a prompt to OpenAI API and returns ETF recommendations based on keywords.
    Implements exponential backoff for handling RateLimitError.
    """
    keywords_text = ", ".join(keywords)
    
    prompt = f"""
    Here is a list of keywords: '{keywords_text}'.

    Based on these keywords and the relevant context of any keywords combinations, perform the following tasks:

    1. Identify the most relevant industries, sectors, or trends that could impact the U.S. stock markets (S&P 500, NASDAQ, Dow Jones) related to these keywords.
    2. For each identified industry or trend, list specific ETF tickers and their top 5 holdings (stocks) that are directly related to these sectors. Ensure the ETFs you suggest are relevant, established in U.S. markets, and very closely tied to the keywords.
    3. Include at least one leveraged ETF (e.g., 2x or 3x) if applicable.

    Return the response strictly and only in the following JSON format:

    {{
        "ticker": "<ETF Ticker>",
        "top5": ["<Top 1 stock>", "<Top 2 stock>", "<Top 3 stock>", "<Top 4 stock>", "<Top 5 stock>"],
        "explanation": "<Detailed explanation of how this ETF relates to the keywords, including its relation to the identified trend or industry, and a brief description of the ETF itself>",
        "holdings_weight": "<The weight of the each stocks in the ETF holdings>",
        "expense_ratio": "<The expense ratio of the ETF>"
    }}

    Important notes:

    - If the keywords have absolutely no relevance to U.S. stock market-listed ETFs, return the response: NO_RELEVANT_ETFS_FOUND
    - The entire response must be written in Korean with a professional, expert tone.
    - Ensure that the explanations fully describe the relationship between the inferred ETFs and the keywords.
    - Include a detailed description of each ETF, focusing on how the ETF is impacted by the identified trend or industry.
    - Make sure that the recommended ETFs are stocks that one would actually invest in (BUY), and are not just general market or sector funds.
    - Make sure that the recommended ETFs and their justifications are acceptable to a professional investor.
    - Preferrably, recommend more than 1 ETF.

    Strictly follow the syntax and instructions provided.
    YOU MUST ONLY RETURN THE JSON OUTPUT. NO OTHER TEXT OR EXPLANATIONS ARE ACCEPTED.
    """

    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial expert and stock market analyst."},
                    {"role": "user", "content": prompt}
                ],
                n=1,
                stop=None,
                temperature=0
            )
            answer = response['choices'][0]['message']['content'].strip()
            return answer
        except openai.error.RateLimitError as e:
            logger.warning(f"RateLimitError encountered: {e}. Attempt {attempt + 1} of {retries}.")
            sleep_time = backoff_factor ** attempt
            logger.info(f"Sleeping for {sleep_time} seconds before retrying...")
            time.sleep(sleep_time)
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI API Error: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected Error: {e}")
            break

    logger.error("All retries failed. Unable to get ETF recommendations.")
    return None