from typing import Annotated, List
from math import sqrt, log, exp
import numpy as np 
from langchain.tools import tool
from scipy.stats import norm

# 1. Future Value of Investment
@tool
def get_future_value(
    present_value: Annotated[float, "Initial amount invested."],
    interest_rate: Annotated[float, "Annual interest rate (as a decimal)."],
    periods: Annotated[int, "Number of periods (years)."],
    compounding_frequency: Annotated[int, "Times interest is compounded per period."]
) -> float:
    """Calculates the future value of an investment using compound interest. This function helps investors estimate how much their initial investment will grow over a specific period at a given interest rate and compounding frequency. It's useful for planning long-term financial goals and understanding the impact of compound interest on investments."""
    fv = present_value * (1 + interest_rate / compounding_frequency) ** (compounding_frequency * periods)
    return fv

# 2. Present Value of Future Cash Flow
@tool
def get_present_value(
    future_value: Annotated[float, "Future amount to be received or paid."],
    discount_rate: Annotated[float, "Discount rate (as a decimal)."],
    periods: Annotated[int, "Number of periods until payment."]
) -> float:
    """Determines the current worth of a future amount of money by discounting it at a specific rate over a number of periods. This function is essential for assessing the value of future cash flows in today's terms, helping in investment decisions and comparing cash flows occurring at different times."""
    pv = future_value / (1 + discount_rate) ** periods
    return pv

# 3. Internal Rate of Return (IRR)
@tool
def get_internal_rate_of_return(
    cash_flows: Annotated[List[float], "Sequence of cash flows starting with initial investment (negative value)."]
) -> float:
    """Computes the Internal Rate of Return (IRR) for a series of cash flows, which is the discount rate that makes the net present value (NPV) of all cash flows equal to zero. This function is useful for evaluating the profitability of potential investments or projects, especially when comparing multiple options with different cash flow patterns."""
    try:
        irr = np.irr(cash_flows)
        return irr
    except ImportError:
        raise ImportError("NumPy is required to compute IRR. Please install NumPy.")

# 4. Payback Period
@tool
def get_payback_period(
    initial_investment: Annotated[float, "Initial investment amount (negative value)."],
    cash_flows: Annotated[List[float], "Sequence of net cash inflows."]
) -> float:
    """Calculates the time required to recover the initial investment from the net cash inflows generated by the project. This function helps assess the liquidity and risk of an investment by indicating how quickly the invested capital can be recouped. It's particularly useful when evaluating projects where cash flow timing is critical."""
    cumulative_cash_flow = initial_investment
    for i, cash_flow in enumerate(cash_flows):
        cumulative_cash_flow += cash_flow
        if cumulative_cash_flow >= 0:
            return i + 1  # Payback period in years
    return float('inf')  # Investment is not recovered within the provided cash flows

# 5. Return on Investment (ROI)
@tool
def get_return_on_investment(
    gain_from_investment: Annotated[float, "Total gain from the investment."],
    cost_of_investment: Annotated[float, "Total cost of the investment."]
) -> float:
    """Determines the Return on Investment (ROI), which measures the profitability and efficiency of an investment by calculating the percentage return relative to its cost. This function is useful for comparing the efficiency of several investments and making informed financial decisions."""
    roi = (gain_from_investment - cost_of_investment) / cost_of_investment
    return roi

# 6. Earnings Per Share (EPS)
@tool
def get_earnings_per_share(
    net_income: Annotated[float, "Net income after taxes and preferred dividends."],
    preferred_dividends: Annotated[float, "Dividends paid to preferred shareholders."],
    average_outstanding_shares: Annotated[float, "Average number of common shares outstanding."]
) -> float:
    """Calculates the Earnings Per Share (EPS), representing the portion of a company's profit allocated to each outstanding share of common stock. This metric is important for investors to assess a company's profitability on a per-share basis and compare it with peers or industry benchmarks."""
    eps = (net_income - preferred_dividends) / average_outstanding_shares
    return eps

# 7. Price to Earnings Ratio (P/E Ratio)
@tool
def get_price_to_earnings_ratio(
    market_price_per_share: Annotated[float, "Current market price per share."],
    earnings_per_share: Annotated[float, "Earnings per share (EPS)."]
) -> float:
    """Calculates the Price-to-Earnings (P/E) Ratio, which compares a company's share price to its earnings per share. This ratio helps investors determine the market's valuation of a company's profitability and is useful for comparing valuation levels across companies and industries."""
    pe_ratio = market_price_per_share / earnings_per_share
    return pe_ratio

# 8. Dividend Yield
@tool
def get_dividend_yield(
    annual_dividends_per_share: Annotated[float, "Total annual dividends per share."],
    price_per_share: Annotated[float, "Current market price per share."]
) -> float:
    """Determines the Dividend Yield, which shows how much a company pays out in dividends each year relative to its stock price. This function is valuable for income-focused investors who are interested in stocks that provide a steady income stream through dividends."""
    dividend_yield = annual_dividends_per_share / price_per_share
    return dividend_yield

# 9. Compound Annual Growth Rate (CAGR)
@tool
def get_compound_annual_growth_rate(
    beginning_value: Annotated[float, "Initial investment value."],
    ending_value: Annotated[float, "Ending investment value."],
    periods: Annotated[int, "Number of periods (years)."]
) -> float:
    """Computes the Compound Annual Growth Rate (CAGR), which represents the mean annual growth rate of an investment over a specified time period longer than one year. This function helps investors understand how different investments have performed over time and is useful for comparing the growth rates of various investments."""
    cagr = (ending_value / beginning_value) ** (1 / periods) - 1
    return cagr

# 10. Loan Payment Calculator (Amortization)
@tool
def get_loan_payment(
    principal: Annotated[float, "Total loan amount borrowed."],
    annual_interest_rate: Annotated[float, "Annual interest rate (as a decimal)."],
    periods: Annotated[int, "Total number of payment periods."]
) -> float:
    """Calculates the periodic payment required to amortize a loan over a specified number of periods at a given annual interest rate. This function helps borrowers plan their repayment schedules and understand the financial commitment involved in taking on a loan."""
    monthly_rate = annual_interest_rate / 12
    payment = principal * (monthly_rate * (1 + monthly_rate) ** periods) / ((1 + monthly_rate) ** periods - 1)
    return payment

# 11. Debt to Equity Ratio
@tool
def get_debt_to_equity_ratio(
    total_liabilities: Annotated[float, "Company's total liabilities."],
    shareholder_equity: Annotated[float, "Total shareholder's equity."]
) -> float:
    """Computes the Debt-to-Equity Ratio, which measures a company's financial leverage by comparing its total liabilities to shareholders' equity. This ratio is useful for assessing a company's risk level and financial stability, indicating how much debt is used to finance assets relative to equity."""
    ratio = total_liabilities / shareholder_equity
    return ratio

# 12. Current Ratio
@tool
def get_current_ratio(
    current_assets: Annotated[float, "Company's current assets."],
    current_liabilities: Annotated[float, "Company's current liabilities."]
) -> float:
    """Calculates the Current Ratio, a liquidity ratio that measures a company's ability to pay short-term obligations with its current assets. This function is essential for evaluating a company's short-term financial health and operational efficiency."""
    ratio = current_assets / current_liabilities
    return ratio

# 13. Quick Ratio
@tool
def get_quick_ratio(
    current_assets: Annotated[float, "Company's current assets."],
    inventory: Annotated[float, "Value of inventory."],
    current_liabilities: Annotated[float, "Company's current liabilities."]
) -> float:
    """Determines the Quick Ratio, also known as the acid-test ratio, which assesses a company's ability to meet its short-term obligations with its most liquid assets. By excluding inventory, this ratio provides a more stringent measure of liquidity than the current ratio."""
    quick_assets = current_assets - inventory
    ratio = quick_assets / current_liabilities
    return ratio

# 14. Interest Coverage Ratio
@tool
def get_interest_coverage_ratio(
    ebit: Annotated[float, "Earnings before interest and taxes."],
    interest_expense: Annotated[float, "Total interest expense."]
) -> float:
    """Calculates the Interest Coverage Ratio, which evaluates a company's ability to pay interest expenses on outstanding debt with its operating income. A higher ratio indicates greater ease in meeting interest obligations, which is crucial for assessing financial health and creditworthiness."""
    ratio = ebit / interest_expense
    return ratio

# 15. Gross Profit Margin
@tool
def get_gross_profit_margin(
    revenue: Annotated[float, "Total revenue or sales."],
    cogs: Annotated[float, "Cost of goods sold."]
) -> float:
    """Computes the Gross Profit Margin, indicating the percentage of revenue that exceeds the cost of goods sold (COGS). This metric helps assess a company's production efficiency and pricing strategy, providing insight into financial performance and competitiveness."""
    gross_profit = revenue - cogs
    margin = gross_profit / revenue
    return margin

# 16. Net Profit Margin
@tool
def get_net_profit_margin(
    net_income: Annotated[float, "Net income after all expenses."],
    revenue: Annotated[float, "Total revenue or sales."]
) -> float:
    """Determines the Net Profit Margin, which measures how much net income is generated as a percentage of revenue. This ratio reflects a company's overall profitability and efficiency in managing expenses, and is useful for comparing profitability across companies and industries."""
    margin = net_income / revenue
    return margin

# 17. Operating Profit Margin
@tool
def get_operating_profit_margin(
    operating_income: Annotated[float, "Operating income (EBIT)."],
    revenue: Annotated[float, "Total revenue or sales."]
) -> float:
    """Calculates the Operating Profit Margin, assessing the proportion of revenue left after covering operating expenses but before interest and taxes. This metric provides insight into a company's operational efficiency and core business profitability, excluding the effects of financing and tax structures."""
    margin = operating_income / revenue
    return margin

# 18. Inventory Turnover Ratio
@tool
def get_inventory_turnover_ratio(
    cogs: Annotated[float, "Cost of goods sold."],
    average_inventory: Annotated[float, "Average inventory value."]
) -> float:
    """Computes the Inventory Turnover Ratio, which measures how many times a company's inventory is sold and replaced over a period. This ratio helps assess inventory management efficiency and can indicate whether a company has excessive inventory relative to its sales levels."""
    ratio = cogs / average_inventory
    return ratio

# 19. Accounts Receivable Turnover Ratio
@tool
def get_accounts_receivable_turnover_ratio(
    net_credit_sales: Annotated[float, "Total net credit sales."],
    average_accounts_receivable: Annotated[float, "Average accounts receivable."]
) -> float:
    """Determines the Accounts Receivable Turnover Ratio, evaluating how efficiently a company collects on its credit sales. A higher ratio indicates effective credit and collection policies, and this metric is useful for assessing liquidity and operational efficiency."""
    ratio = net_credit_sales / average_accounts_receivable
    return ratio

# 20. Average Collection Period
@tool
def get_average_collection_period(
    accounts_receivable_turnover: Annotated[float, "Accounts receivable turnover ratio."]
) -> float:
    """Calculates the Average Collection Period, representing the average number of days it takes for a company to collect payments from its credit sales. This function is useful for evaluating the effectiveness of a company's credit policies and cash flow management."""
    period = 365 / accounts_receivable_turnover
    return period

# 21. Economic Order Quantity (EOQ)
@tool
def get_economic_order_quantity(
    demand_rate: Annotated[float, "Annual demand in units."],
    setup_cost: Annotated[float, "Cost per order/setup."],
    holding_cost: Annotated[float, "Holding cost per unit per year."]
) -> float:
    """Computes the Economic Order Quantity (EOQ), which determines the optimal order size that minimizes the total inventory costs, including ordering and holding costs. This function helps businesses manage inventory levels efficiently and reduce associated costs."""
    eoq = sqrt((2 * demand_rate * setup_cost) / holding_cost)
    return eoq

# 22. Weighted Average Cost of Capital (WACC)
@tool
def get_weighted_average_cost_of_capital(
    equity: Annotated[float, "Market value of equity."],
    debt: Annotated[float, "Market value of debt."],
    cost_of_equity: Annotated[float, "Cost of equity (as decimal)."],
    cost_of_debt: Annotated[float, "Cost of debt (as decimal)."],
    tax_rate: Annotated[float, "Corporate tax rate (as decimal)."]
) -> float:
    """Calculates the Weighted Average Cost of Capital (WACC), representing the average rate a company is expected to pay to finance its assets. WACC is essential for evaluating investment opportunities and serves as a hurdle rate in capital budgeting decisions."""
    total_value = equity + debt
    wacc = ((equity / total_value) * cost_of_equity) + ((debt / total_value) * cost_of_debt * (1 - tax_rate))
    return wacc

# 23. Capital Asset Pricing Model (CAPM)
@tool
def get_capital_asset_pricing_model(
    risk_free_rate: Annotated[float, "Risk-free rate (as decimal)."],
    beta: Annotated[float, "Beta of the security."],
    market_return: Annotated[float, "Expected market return (as decimal)."]
) -> float:
    """Uses the Capital Asset Pricing Model (CAPM) to calculate the expected return of an asset based on its systematic risk (beta). This function is useful for estimating the cost of equity and making informed investment decisions by comparing expected returns with required returns."""
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    return expected_return

# 24. Beta of a Stock
@tool
def get_beta(
    covariance: Annotated[float, "Covariance of stock and market returns."],
    variance: Annotated[float, "Variance of market returns."]
) -> float:
    """Determines the Beta of a stock, which measures its volatility or systematic risk relative to the overall market. This metric is key in portfolio management and the CAPM, helping investors understand how a stock might respond to market movements."""
    beta = covariance / variance
    return beta

# 25. Sharpe Ratio
@tool
def get_sharpe_ratio(
    portfolio_return: Annotated[float, "Average portfolio return (as decimal)."],
    risk_free_rate: Annotated[float, "Risk-free rate (as decimal)."],
    standard_deviation: Annotated[float, "Standard deviation of portfolio's excess return."]
) -> float:
    """Calculates the Sharpe Ratio, which evaluates the risk-adjusted return of an investment portfolio by comparing its excess return to its volatility. This function helps investors understand the return of an investment compared to its risk, facilitating better portfolio optimization."""
    sharpe_ratio = (portfolio_return - risk_free_rate) / standard_deviation
    return sharpe_ratio

# 26. Treynor Ratio
@tool
def get_treynor_ratio(
    portfolio_return: Annotated[float, "Average portfolio return (as decimal)."],
    risk_free_rate: Annotated[float, "Risk-free rate (as decimal)."],
    beta: Annotated[float, "Beta of the portfolio."]
) -> float:
    """Computes the Treynor Ratio, measuring the risk-adjusted return of a portfolio using its beta as the risk measure. This function is useful for assessing how well a portfolio compensates investors for taking on market risk, and comparing portfolios with different levels of systematic risk."""
    treynor_ratio = (portfolio_return - risk_free_rate) / beta
    return treynor_ratio

# 27. Jensen's Alpha
@tool
def get_jensens_alpha(
    portfolio_return: Annotated[float, "Portfolio return (as decimal)."],
    expected_return: Annotated[float, "Expected return using CAPM (as decimal)."]
) -> float:
    """Calculates Jensen's Alpha, which represents the excess returns of a portfolio over the expected return predicted by the Capital Asset Pricing Model (CAPM). This metric helps investors evaluate a portfolio manager's performance by indicating the value added through active management."""
    alpha = portfolio_return - expected_return
    return alpha

# 28. Sortino Ratio
@tool
def get_sortino_ratio(
    portfolio_return: Annotated[float, "Average portfolio return (as decimal)."],
    risk_free_rate: Annotated[float, "Risk-free rate (as decimal)."],
    downside_deviation: Annotated[float, "Standard deviation of negative returns."]
) -> float:
    """Determines the Sortino Ratio, which assesses the risk-adjusted return of an investment by focusing only on downside volatility. This function provides a more accurate evaluation of an investment's performance by penalizing only harmful volatility, making it useful for risk-averse investors."""
    sortino_ratio = (portfolio_return - risk_free_rate) / downside_deviation
    return sortino_ratio

# 29. Risk-Adjusted Return on Capital (RAROC)
@tool
def get_raroc(
    net_income: Annotated[float, "Net income from investment."],
    economic_capital: Annotated[float, "Economic capital at risk."]
) -> float:
    """Calculates the Risk-Adjusted Return on Capital (RAROC), which evaluates the profitability of an investment relative to the economic capital at risk. This metric helps in comparing investments with different risk profiles by standardizing returns against risk exposure."""
    raroc = net_income / economic_capital
    return raroc

# 30. Value at Risk (VaR)
@tool
def get_value_at_risk(
    portfolio_value: Annotated[float, "Total portfolio value."],
    confidence_level: Annotated[float, "Confidence level (e.g., 0.95 for 95%)."],
    standard_deviation: Annotated[float, "Standard deviation of portfolio returns."]
) -> float:
    """Computes the Value at Risk (VaR), estimating the maximum potential loss of a portfolio over a given time frame at a specified confidence level. This function is crucial for risk management, helping investors and institutions understand the extent of potential losses under normal market conditions."""
    z_score = norm.ppf(1 - confidence_level)
    var = portfolio_value * z_score * standard_deviation
    return var

# 31. Black-Scholes Option Pricing
@tool
def get_black_scholes_option_price(
    stock_price: Annotated[float, "Current stock price."],
    strike_price: Annotated[float, "Option strike price."],
    time_to_expiration: Annotated[float, "Time to expiration in years."],
    risk_free_rate: Annotated[float, "Risk-free interest rate (as decimal)."],
    volatility: Annotated[float, "Stock volatility (as decimal)."]
) -> float:
    """Calculates the theoretical price of a European call option using the Black-Scholes model. This function is essential for options valuation, helping traders and investors estimate the fair value of options and make informed trading decisions."""
    d1 = (log(stock_price / strike_price) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * sqrt(time_to_expiration))
    d2 = d1 - volatility * sqrt(time_to_expiration)
    call_price = stock_price * norm.cdf(d1) - strike_price * exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)
    return call_price

# 32. Put-Call Parity
@tool
def get_put_price_using_put_call_parity(
    call_price: Annotated[float, "Price of the call option."],
    strike_price: Annotated[float, "Option strike price."],
    risk_free_rate: Annotated[float, "Risk-free interest rate (as decimal)."],
    time_to_expiration: Annotated[float, "Time to expiration in years."],
    stock_price: Annotated[float, "Current stock price."]
) -> float:
    """Determines the price of a European put option using the put-call parity theorem. This function ensures consistency between put and call option pricing and is useful for arbitrage strategies and verifying option prices in the market."""
    present_value_strike = strike_price * exp(-risk_free_rate * time_to_expiration)
    put_price = call_price + present_value_strike - stock_price
    return put_price

# 33. Modified Internal Rate of Return (MIRR)
@tool
def get_modified_internal_rate_of_return(
    cash_flows: Annotated[List[float], "Cash flows starting with initial investment."],
    finance_rate: Annotated[float, "Finance rate (cost of investment)."],
    reinvestment_rate: Annotated[float, "Reinvestment rate (return on investment)."]
) -> float:
    """Computes the Modified Internal Rate of Return (MIRR), which adjusts the IRR to account for differences in the reinvestment rate of positive cash flows and the financing cost of negative cash flows. This function provides a more accurate reflection of a project's profitability and is useful when the reinvestment rate differs from the project's internal rate of return."""
    try:
        mirr = np.mirr(cash_flows, finance_rate, reinvestment_rate)
        return mirr
    except ImportError:
        raise ImportError("NumPy is required to compute MIRR. Please install NumPy.")

# 34. Annuity Payment Calculation
@tool
def get_annuity_payment(
    present_value: Annotated[float, "Present value of the annuity."],
    interest_rate: Annotated[float, "Interest rate per period (as decimal)."],
    periods: Annotated[int, "Number of periods."]
) -> float:
    """Calculates the periodic payment amount required to pay off an annuity over a specified number of periods at a given interest rate. This function is useful for planning loan repayments, retirement savings, and any financial scenario involving regular payments over time."""
    payment = (present_value * interest_rate) / (1 - (1 + interest_rate) ** -periods)
    return payment

# 35. Effective Annual Rate (EAR)
@tool
def get_effective_annual_rate(
    nominal_rate: Annotated[float, "Nominal interest rate (as decimal)."],
    compounding_periods: Annotated[int, "Number of compounding periods per year."]
) -> float:
    """Determines the Effective Annual Rate (EAR), which reflects the true annual interest rate accounting for compounding periods. This function helps investors and borrowers compare the annual interest between loans or investments with different compounding frequencies."""
    ear = (1 + nominal_rate / compounding_periods) ** compounding_periods - 1
    return ear

# 36. Duration of a Bond
@tool
def get_bond_duration(
    cash_flows: Annotated[List[float], "List of bond cash flows."],
    yields: Annotated[List[float], "Yield for each period."]
) -> float:
    """Calculates the Macaulay Duration of a bond, measuring the weighted average time until cash flows are received. This function helps investors understand a bond's sensitivity to interest rate changes and manage interest rate risk in their portfolios."""
    try:
        weighted_cash_flows = [cf / (1 + y) ** t for t, (cf, y) in enumerate(zip(cash_flows, yields), 1)]
        duration = sum(t * wcf for t, wcf in enumerate(weighted_cash_flows, 1)) / sum(weighted_cash_flows)
        return duration
    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")

# 37. Debt Service Coverage Ratio (DSCR)
@tool
def get_debt_service_coverage_ratio(
    net_operating_income: Annotated[float, "Net operating income."],
    total_debt_service: Annotated[float, "Total debt service."]
) -> float:
    """Computes the Debt Service Coverage Ratio (DSCR), which evaluates a company's ability to service its debt obligations with its net operating income. A DSCR greater than 1 indicates sufficient income to cover debt payments, important for lenders assessing credit risk."""
    dscr = net_operating_income / total_debt_service
    return dscr

# 38. Return on Equity (ROE)
@tool
def get_return_on_equity(
    net_income: Annotated[float, "Net income."],
    shareholder_equity: Annotated[float, "Shareholder's equity."]
) -> float:
    """Calculates the Return on Equity (ROE), indicating how effectively a company uses shareholders' equity to generate profits. This function is useful for comparing the profitability of companies within the same industry and assessing management effectiveness."""
    roe = net_income / shareholder_equity
    return roe

# 39. Return on Assets (ROA)
@tool
def get_return_on_assets(
    net_income: Annotated[float, "Net income."],
    total_assets: Annotated[float, "Total assets."]
) -> float:
    """Determines the Return on Assets (ROA), measuring how efficiently a company utilizes its assets to generate net income. This metric helps investors assess asset efficiency and compare companies regardless of their capital structures."""
    roa = net_income / total_assets
    return roa

# 40. Debt Ratio
@tool
def get_debt_ratio(
    total_liabilities: Annotated[float, "Total liabilities."],
    total_assets: Annotated[float, "Total assets."]
) -> float:
    """Computes the Debt Ratio, indicating the proportion of a company's assets that are financed through debt. This function helps evaluate a company's financial leverage and risk level, with higher ratios suggesting greater reliance on debt financing."""
    debt_ratio = total_liabilities / total_assets
    return debt_ratio

# 41. Dividend Payout Ratio
@tool
def get_dividend_payout_ratio(
    dividends_per_share: Annotated[float, "Dividends per share."],
    earnings_per_share: Annotated[float, "Earnings per share (EPS)."]
) -> float:
    """Calculates the Dividend Payout Ratio, showing the percentage of earnings distributed to shareholders as dividends. This function is useful for investors assessing a company's dividend policy and sustainability of dividend payments."""
    payout_ratio = dividends_per_share / earnings_per_share
    return payout_ratio

# 42. Retention Ratio (Plowback Ratio)
@tool
def get_retention_ratio(
    dividends_per_share: Annotated[float, "Dividends per share."],
    earnings_per_share: Annotated[float, "Earnings per share (EPS)."]
) -> float:
    """Determines the Retention Ratio, also known as the Plowback Ratio, indicating the portion of earnings retained in the business for growth and expansion. This function complements the dividend payout ratio and helps in analyzing a company's reinvestment strategy."""
    retention_ratio = 1 - (dividends_per_share / earnings_per_share)
    return retention_ratio

# 43. Operating Cash Flow (OCF)
@tool
def get_operating_cash_flow(
    net_income: Annotated[float, "Net income."],
    depreciation: Annotated[float, "Depreciation expense."],
    change_in_working_capital: Annotated[float, "Change in working capital."]
) -> float:
    """Computes the Operating Cash Flow (OCF), measuring the cash generated from a company's regular business operations. This function is crucial for assessing liquidity, financial flexibility, and the ability to sustain operations without external financing."""
    ocf = net_income + depreciation - change_in_working_capital
    return ocf

# 44. Free Cash Flow (FCF)
@tool
def get_free_cash_flow(
    operating_cash_flow: Annotated[float, "Operating cash flow."],
    capital_expenditures: Annotated[float, "Capital expenditures."]
) -> float:
    """Calculates the Free Cash Flow (FCF), indicating the cash a company generates after accounting for cash outflows to support operations and maintain capital assets. This function is important for valuation, as it represents the cash available for distribution to stakeholders."""
    fcf = operating_cash_flow - capital_expenditures
    return fcf

# 45. Price to Book Ratio (P/B Ratio)
@tool
def get_price_to_book_ratio(
    market_price_per_share: Annotated[float, "Market price per share."],
    book_value_per_share: Annotated[float, "Book value per share."]
) -> float:
    """Determines the Price-to-Book (P/B) Ratio, comparing a company's market value to its book value. This ratio helps investors identify undervalued or overvalued stocks by assessing how much shareholders are paying for the net assets of the company."""
    pb_ratio = market_price_per_share / book_value_per_share
    return pb_ratio

# 46. Market Capitalization
@tool
def get_market_capitalization(
    shares_outstanding: Annotated[float, "Total shares outstanding."],
    market_price_per_share: Annotated[float, "Market price per share."]
) -> float:
    """Calculates the Market Capitalization of a company, reflecting its total market value based on current share price and the total number of outstanding shares. This metric is used to classify companies by size and is a key indicator of market perception."""
    market_cap = shares_outstanding * market_price_per_share
    return market_cap

# 47. Enterprise Value (EV)
@tool
def get_enterprise_value(
    market_capitalization: Annotated[float, "Market capitalization."],
    total_debt: Annotated[float, "Total debt."],
    cash_and_equivalents: Annotated[float, "Cash and cash equivalents."]
) -> float:
    """Computes the Enterprise Value (EV), measuring a company's total value, including debt and excluding cash. This function is useful for comparing companies with different capital structures and is often used in valuation metrics like EV/EBITDA."""
    ev = market_capitalization + total_debt - cash_and_equivalents
    return ev

# 48. EV to EBITDA Ratio
@tool
def get_ev_to_ebitda_ratio(
    enterprise_value: Annotated[float, "Enterprise Value (EV)."],
    ebitda: Annotated[float, "Earnings before interest, taxes, depreciation, and amortization."]
) -> float:
    """Calculates the EV/EBITDA Ratio, a valuation metric that compares a company's Enterprise Value to its EBITDA. This ratio is widely used to assess a company's value and compare it with peers, as it considers both equity and debt while excluding the effects of non-cash expenses."""
    ratio = enterprise_value / ebitda
    return ratio

# 49. Interest Rate Swap Valuation
@tool
def get_interest_rate_swap_valuation(
    fixed_rate: Annotated[float, "Fixed interest rate (as decimal)."],
    floating_rate: Annotated[float, "Current floating interest rate (as decimal)."],
    notional_amount: Annotated[float, "Notional principal amount."],
    time_to_maturity: Annotated[float, "Time to maturity in years."]
) -> float:
    """Determines the value of an Interest Rate Swap by calculating the net present value of the fixed and floating rate cash flows. This function helps assess the financial benefit or cost of swapping fixed and floating interest rates, which is essential for managing interest rate risk in financial contracts."""
    swap_value = (fixed_rate - floating_rate) * notional_amount * time_to_maturity
    return swap_value

# 50. Foreign Exchange Forward Rate
@tool
def get_fx_forward_rate(
    spot_rate: Annotated[float, "Current spot exchange rate."],
    domestic_interest_rate: Annotated[float, "Domestic interest rate (as decimal)."],
    foreign_interest_rate: Annotated[float, "Foreign interest rate (as decimal)."],
    time_to_maturity: Annotated[float, "Time to maturity in years."]
) -> float:
    """Calculates the Forward Exchange Rate using the interest rate parity formula. This function helps businesses and investors forecast future exchange rates based on the interest rate differential between two countries. It's essential for hedging currency risk in international trade and investment decisions."""
    forward_rate = spot_rate * ((1 + domestic_interest_rate * time_to_maturity) / (1 + foreign_interest_rate * time_to_maturity))
    return forward_rate