"""Gold agent node factories (Phase 5).

Mirrors of the upstream node functions with gold-specific prompts and
deterministic, pre-fetched context.  State contracts (which keys each node
reads and writes) are IDENTICAL to upstream, so the shared graph wiring,
conditional logic, checkpoints, memory log, CLI display and reports all work
unchanged:

* gold market/technical analyst → ``market_report``
* gold macro analyst            → ``fundamentals_report`` (macro replaces equity fundamentals)
* gold news analyst             → ``news_report``
* gold sentiment analyst        → ``sentiment_report`` (structured SentimentReport)
* bull / bear researchers       → ``investment_debate_state``
* research manager              → ``investment_plan`` (+ debate judge)
* gold trader                   → ``trader_investment_plan`` (GoldTraderProposal)
* risk debators                 → ``risk_debate_state``
* portfolio manager             → ``final_trade_decision``

The analysts are the pre-fetch pattern (upstream sentiment-analyst precedent):
deterministic blocks are assembled by ``GoldContextBuilder`` and placed in the
prompt; the LLM interprets and never computes or fetches.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.context import get_language_instruction
from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.gold.agents.schemas import (
    GoldTraderProposal,
    render_gold_trader_proposal,
)

if TYPE_CHECKING:
    from tradingagents.gold.agents.context import GoldRunContext

#: trade_date -> GoldRunContext (provided by ``GoldContextBuilder.build``).
ContextProvider = Callable[[str], "GoldRunContext"]


# ---------------------------------------------------------------------------
# Analysts (pre-fetch pattern)
# ---------------------------------------------------------------------------


def _analyst_node(llm, context_provider: ContextProvider, report_key: str, build_prompt):
    def node(state) -> dict:
        ctx = context_provider(state["trade_date"])
        system_message = build_prompt(state, ctx)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what the supplied data supports; another agent decides the trade."
                    " Today's date is {current_date}; treat it as 'now' for all analysis."
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=state["trade_date"])
        formatted = prompt.format_messages(messages=state["messages"])
        report = llm.invoke(formatted).content
        return {"messages": [AIMessage(content=report)], report_key: report}

    return node


def create_gold_market_analyst(llm, context_provider: ContextProvider):
    """Technical/market analyst: interprets the deterministic snapshot."""

    def build_prompt(state, ctx: GoldRunContext) -> str:
        error_note = ""
        if ctx.has_data_error:
            error_note = (
                "\nDATA QUALITY ERRORS (from the deterministic quality gate):\n"
                + "\n".join(f"- {e}" for e in ctx.data_errors)
                + "\nIf these errors make the market data unusable, your report MUST begin "
                  "with 'DATA ERROR' and list what failed — do NOT analyse unavailable data "
                  "and do NOT invent substitute values.\n"
            )
        return f"""You are the GOLD MARKET / TECHNICAL ANALYST in a multi-agent gold research system. Your task is to produce a technical analysis report for {state["company_of_interest"]} (XAUUSD) as of {state["trade_date"]}.

{ctx.technical_block}

{ctx.market_data_block}{error_note}
## How to work

1. Every number you cite MUST come from the deterministic snapshot above. Never compute, adjust or invent values.
2. Interpret the market regime ({ "see snapshot" }): what does it imply for gold positioning over the next sessions?
3. Report the multi-timeframe picture explicitly: H4 context, H1 structure, M15 trigger. If timeframes disagree, state the conflict plainly — the system must allow H4 bullish + H1/M15 bearish without forcing a single direction.
4. Highlight actionable structure: nearest support/resistance, breakout state, session timing (Asia/London/NY), volatility conditions for stops.
5. Sentiment, macro and news are other agents' jobs; reference them only if already present in your inputs.
6. End with a short "Technical Read" paragraph: what the deterministic evidence says, and what would change the read (invalidation levels).

{get_language_instruction()}"""

    return _analyst_node(llm, context_provider, "market_report", build_prompt)


def create_gold_macro_analyst(llm, context_provider: ContextProvider):
    """GOLD MACRO ANALYST — replaces the equity fundamentals analyst (spec §9)."""

    def build_prompt(state, ctx: GoldRunContext) -> str:
        return f"""You are the GOLD MACRO ANALYST in a multi-agent gold research system. Your task is to produce a macro analysis report for {state["company_of_interest"]} (XAUUSD) as of {state["trade_date"]}. You REPLACE the equity fundamentals analyst: gold has no earnings — its fundamentals are macroeconomic.

{ctx.macro_block}

## How to work

1. Every level and date you cite MUST come from the macro snapshot above (FRED/ALFRED, vintage-pinned). Never invent or round values.
2. Explain what the data means for gold NOW: real yields (opportunity cost of holding gold), the dollar (the index shown is a DXY proxy — treat it as directional, not as the ICE DXY), Fed policy path, inflation prints, labor momentum, growth and stress gauges.
3. Separate FACTS (published numbers), INTERPRETATIONS (your reading), and UNCERTAINTIES (missing, stale or unparseable series) — label each section.
4. Flag scheduled-release risk only from data present in the snapshot; do not guess calendar dates you were not given.
5. If key series are unavailable, say so explicitly and lower the confidence of your conclusions accordingly.

{get_language_instruction()}"""

    return _analyst_node(llm, context_provider, "fundamentals_report", build_prompt)


def create_gold_news_analyst(llm, context_provider: ContextProvider):
    """GOLD & MACRO NEWS ANALYST (spec §10)."""

    def build_prompt(state, ctx: GoldRunContext) -> str:
        return f"""You are the GOLD & MACRO NEWS ANALYST in a multi-agent gold research system. Your task is to produce a news analysis report for {state["company_of_interest"]} (XAUUSD) as of {state["trade_date"]}.

{ctx.news_block}

## How to work

1. Work ONLY from the pre-fetched, window-clamped headlines above; never invent articles.
2. Classify each relevant item: gold-specific, USD/Fed, inflation/employment, central banks, geopolitical. Note each item's publication timestamp.
3. Identify which items are fresh catalysts vs context, and which contradict the technical or macro picture (if visible in your inputs).
4. Preserve point-in-time discipline: your window is exactly the one shown; do not speculate about later developments.
5. End with "News Read": net news pressure on gold (bullish / bearish / mixed / unavailable) and why.

{get_language_instruction()}"""

    return _analyst_node(llm, context_provider, "news_report", build_prompt)


def create_gold_sentiment_analyst(llm, context_provider: ContextProvider):
    """GOLD SENTIMENT ANALYST — structured SentimentReport, evidence only (spec §11)."""

    structured_llm = bind_structured(llm, SentimentReport, "Gold Sentiment Analyst")

    def node(state) -> dict:
        ctx = context_provider(state["trade_date"])
        system_message = f"""You are the GOLD SENTIMENT ANALYST. Produce a sentiment report for {state["company_of_interest"]} (XAUUSD) as of {state["trade_date"]} from the pre-fetched social chatter below.

{ctx.sentiment_block}

## How to work

1. Report only what the supplied sources show; if they are unavailable or thin, say so and reflect it in `confidence`.
2. Gold sentiment sources talk about gold, the dollar, the Fed and macro fear — read them as supporting evidence.
3. Sentiment must NOT produce a BUY/SELL signal by itself. Frame it as one input the trader weighs alongside technicals, macro and news.
4. Past chatter is not predictive; frame conclusions as current positioning/mood.

## Output fields

- **overall_band**: exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish.
- **overall_score**: 0 (max bearish) to 10 (max bullish); 5 neutral; consistent with the band.
- **confidence**: low / medium / high based on data quality and sample size.
- **narrative**: source-by-source breakdown, divergences, dominant themes, and what they mean for gold positioning.

{get_language_instruction()}"""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Report what the supplied data supports; another agent decides the trade."
                    " Today's date is {current_date}; treat it as 'now' for all analysis."
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=state["trade_date"])
        formatted = prompt.format_messages(messages=state["messages"])
        report_text = invoke_structured_or_freetext(
            structured_llm, llm, formatted, render_sentiment_report, "Gold Sentiment Analyst",
        )
        return {"messages": [AIMessage(content=report_text)], "sentiment_report": report_text}

    return node


# ---------------------------------------------------------------------------
# Bull / Bear debate (spec §12): disagreement is the point.
# ---------------------------------------------------------------------------

_BULL_PROMPT = """You are the Bull Researcher for GOLD (XAUUSD) in a multi-agent research system. Build the STRONGEST evidence-based bullish case for gold as of {trade_date}.

Your job:
- Supporting evidence: cite specific numbers ONLY from the analysts' reports below (technical levels, macro prints, news, sentiment).
- Catalysts: what could drive gold higher from here, given the reported regime and session context.
- Invalidation conditions: state explicitly what would prove this bull case WRONG (price levels, regime flips, macro surprises). A bull case without invalidation conditions is incomplete.
- Rebuttal: engage the bear analyst's last argument directly; neither side is instructed to agree with the other. Disagreement is the purpose of this debate.

Resources:
{instrument_context}
Market/technical report: {market_report}
Macro report: {fundamentals_report}
News report: {news_report}
Sentiment report: {sentiment_report}
Debate history: {history}
Last bear argument: {opponent}

Deliver a compelling gold bull argument with the sections above.""" + get_language_instruction()

_BEAR_PROMPT = """You are the Bear Researcher for GOLD (XAUUSD) in a multi-agent research system. Build the STRONGEST evidence-based bearish case against gold as of {trade_date}.

Your job:
- Opposing evidence: cite specific numbers ONLY from the analysts' reports below (technical levels, macro prints, news, sentiment).
- Risks: what could drive gold lower from here, given the reported regime and session context.
- Invalidation conditions: state explicitly what would prove this bear case WRONG (price levels, regime flips, macro surprises). A bear case without invalidation conditions is incomplete.
- Rebuttal: engage the bull analyst's last argument directly; neither side is instructed to agree with the other. Disagreement is the purpose of this debate.

Resources:
{instrument_context}
Market/technical report: {market_report}
Macro report: {fundamentals_report}
News report: {news_report}
Sentiment report: {sentiment_report}
Debate history: {history}
Last bull argument: {opponent}

Deliver a compelling gold bear argument with the sections above.""" + get_language_instruction()


def _gold_debator(llm, role_prompt: str, side: str):
    def node(state) -> dict:
        debate = state["investment_debate_state"]
        opponent_key = "current_response" if side == "bull" else "current_response"
        opponent = debate.get(opponent_key, "") or "(no opposing argument yet — open the debate)"
        prompt = role_prompt.format(
            trade_date=state["trade_date"],
            instrument_context=state.get("instrument_context", ""),
            market_report=state.get("market_report") or "(absent)",
            fundamentals_report=state.get("fundamentals_report") or "(absent)",
            news_report=state.get("news_report") or "(absent)",
            sentiment_report=state.get("sentiment_report") or "(absent)",
            history=debate.get("history", "") or "(empty)",
            opponent=opponent,
        )
        response = llm.invoke(prompt)
        argument = f"{'Bull' if side == 'bull' else 'Bear'} Researcher: {response.content}"
        return {
            "investment_debate_state": {
                "history": debate.get("history", "") + "\n" + argument,
                "bull_history": debate.get("bull_history", "") + (("\n" + argument) if side == "bull" else ""),
                "bear_history": debate.get("bear_history", "") + (("\n" + argument) if side == "bear" else ""),
                "current_response": argument,
                "count": debate["count"] + 1,
            }
        }

    return node


def create_gold_bull_researcher(llm):
    return _gold_debator(llm, _BULL_PROMPT, "bull")


def create_gold_bear_researcher(llm):
    return _gold_debator(llm, _BEAR_PROMPT, "bear")


# ---------------------------------------------------------------------------
# Research Manager (spec §13): facts vs interpretations vs uncertainties.
# ---------------------------------------------------------------------------


def create_gold_research_manager(llm):
    from tradingagents.agents.schemas import ResearchPlan, render_research_plan

    structured_llm = bind_structured(llm, ResearchPlan, "Gold Research Manager")

    def node(state) -> dict:
        debate = state["investment_debate_state"]
        prompt = f"""You are the GOLD RESEARCH MANAGER. Evaluate the bull/bear debate on XAUUSD as of {state["trade_date"]} and deliver a clear, actionable research plan for the trader.

{state.get("instrument_context", "")}

**Debate history:**
{debate.get("history", "")}

## How to judge

1. Distinguish FACTS (numbers present in the analysts' reports) from INTERPRETATIONS (agents' readings) from UNCERTAINTIES (missing, conflicting or low-confidence inputs). Your rationale MUST contain a short section for each, plus an explicit "Conflicting evidence" list.
2. The debate always contains conflicting arguments; deciding which side is stronger is the job, so conflict alone is not a reason to Hold. Commit to the side with the stronger evidence, sized by how decisively it wins. Choose Hold only when the evidence is genuinely balanced or too thin.
3. The deterministic market regime from the technical report is authoritative — do not contradict it without citing specific conflicting evidence.

## Output

- **Recommendation**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Rationale**: which arguments decided it — include FACTS / INTERPRETATIONS / UNCERTAINTIES and Conflicting evidence sections
- **Strategic Actions**: concrete steps for the trader (direction, key levels from the reports, what to watch)

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm, llm, prompt, render_research_plan, "Gold Research Manager",
        )
        return {
            "investment_debate_state": {
                "judge_decision": investment_plan,
                "history": debate.get("history", ""),
                "bear_history": debate.get("bear_history", ""),
                "bull_history": debate.get("bull_history", ""),
                "current_response": investment_plan,
                "count": debate["count"],
            },
            "investment_plan": investment_plan,
        }

    return node


# ---------------------------------------------------------------------------
# Trader (spec §14): structured gold proposal, prices only from supplied data.
# ---------------------------------------------------------------------------


def create_gold_trader(llm):
    structured_llm = bind_structured(llm, GoldTraderProposal, "Gold Trader")

    def node(state, name):
        ctx_market = (state.get("market_report") or "").strip()
        prompt = f"""You are the GOLD TRADER. Turn the research plan into a concrete XAUUSD transaction proposal as of {state["trade_date"]}.

{state.get("instrument_context", "")}

**Research plan:**
{state["investment_plan"]}

**Deterministic technical snapshot (source of all price levels):**
{ctx_market or "(market report absent — you cannot state price levels)"}

## Rules

1. Action: exactly one of Buy / Sell / Hold / Review. Overweight from the research manager is a Buy; Underweight is a Sell. Use Review ONLY when the evidence is genuinely insufficient (missing/contradictory data) — never to avoid committing when the evidence points somewhere.
2. NEVER invent market data. Entry, stop-loss and take-profit MUST be absolute USD-per-ounce levels drawn from the technical snapshot (support/resistance, ATR-based distances, current price). If you cannot ground a level in that snapshot, omit the field.
3. For a Buy: stop_loss < entry_price < take_profit. For a Sell: take_profit < entry_price < stop_loss. State nothing you cannot justify from the reports.
4. **Confidence** is your stated conviction that this action correctly reads the supplied evidence (0–1). It is NOT a probability of profit and is not a risk input — deterministic code owns risk limits.
5. **Invalidation**: the specific condition that kills this trade idea.

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        proposal_text = invoke_structured_or_freetext(
            structured_llm, llm, prompt, render_gold_trader_proposal, "Gold Trader",
        )
        return {
            "messages": [AIMessage(content=proposal_text)],
            "trader_investment_plan": proposal_text,
            "sender": name,
        }

    return functools.partial(node, name="Trader")


# ---------------------------------------------------------------------------
# Risk debators (spec §16): LLM risk analysis; code enforces final limits.
# ---------------------------------------------------------------------------

_RISK_ROLES = {
    "aggressive": (
        "Aggressive Risk Analyst",
        "Champion the upside of the trader's proposal: missed-opportunity cost, regime "
        "momentum, catalysts. Push back on excessive caution, but stay concrete: cite the "
        "reports' levels and data. You argue FOR taking the trade.",
    ),
    "conservative": (
        "Conservative Risk Analyst",
        "Attack the trader's proposal from capital preservation: spread and slippage at the "
        "reported volatility, overnight/weekend gap risk, event risk around scheduled macro "
        "releases, the futures-proxy basis, data-quality errors if any were reported. Demand "
        "explicit invalidation and stop placement. You argue AGAINST taking the trade as proposed.",
    ),
    "neutral": (
        "Neutral Risk Analyst",
        "Weigh both sides on probabilities given the deterministic regime and volatility. "
        "Identify what each side overstates, what each ignores, and under what conditions "
        "each would be right. Do not advocate a direction; advocate clarity.",
    ),
}


def _gold_risk_debator(llm, role: str):
    title, stance = _RISK_ROLES[role]
    key = f"current_{role}_response"
    history_key = f"{role}_history"

    def node(state) -> dict:
        debate = state["risk_debate_state"]
        opponents = "\n".join(
            f"Last {other} argument: {debate.get(f'current_{other}_response', '') or '(none yet)'}"
            for other in ("aggressive", "conservative", "neutral") if other != role
        )
        prompt = f"""You are the {title} for a GOLD (XAUUSD) research system as of {state["trade_date"]}.

{stance}

**Trader's proposal:**
{state["trader_investment_plan"]}

**Resources:**
{state.get("instrument_context", "")}
{state.get("portfolio_context", "")}
Market/technical report: {state.get("market_report") or "(absent)"}
Macro report: {state.get("fundamentals_report") or "(absent)"}
News report: {state.get("news_report") or "(absent)"}
Sentiment report: {state.get("sentiment_report") or "(absent)"}

{opponents}

Debate history: {debate.get("history", "") or "(empty)"}

Respond conversationally, engaging the other analysts directly. Final risk limits are enforced by deterministic code after you — your job is the strongest honest analysis, not the final gate.""" + get_language_instruction()

        response = llm.invoke(prompt)
        argument = f"{title}: {response.content}"
        speaker = {"aggressive": "Aggressive", "conservative": "Conservative", "neutral": "Neutral"}[role]
        updates = {
            "history": debate.get("history", "") + "\n" + argument,
            "conservative_history": debate.get("conservative_history", ""),
            "aggressive_history": debate.get("aggressive_history", ""),
            "neutral_history": debate.get("neutral_history", ""),
            "latest_speaker": speaker,
            "current_aggressive_response": debate.get("current_aggressive_response", ""),
            "current_conservative_response": debate.get("current_conservative_response", ""),
            "current_neutral_response": debate.get("current_neutral_response", ""),
            "count": debate["count"] + 1,
        }
        updates[key] = argument
        updates[history_key] = debate.get(history_key, "") + "\n" + argument
        return {"risk_debate_state": updates}

    return node


def create_gold_aggressive_debator(llm):
    return _gold_risk_debator(llm, "aggressive")


def create_gold_conservative_debator(llm):
    return _gold_risk_debator(llm, "conservative")


def create_gold_neutral_debator(llm):
    return _gold_risk_debator(llm, "neutral")


# ---------------------------------------------------------------------------
# Portfolio Manager: final structured decision (5-tier contract preserved).
# ---------------------------------------------------------------------------


def create_gold_portfolio_manager(llm):
    from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision

    structured_llm = bind_structured(llm, PortfolioDecision, "Gold Portfolio Manager")

    def node(state) -> dict:
        debate = state["risk_debate_state"]
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{state.get('past_context', '')}\n"
            if state.get("past_context")
            else ""
        )
        prompt = f"""You are the GOLD PORTFOLIO MANAGER. Synthesize the risk debate and deliver the final XAUUSD decision as of {state["trade_date"]}.

{state.get("instrument_context", "")}

{state.get("portfolio_context", "")}

**Research plan:** {state["investment_plan"]}
**Trader's proposal:** {state["trader_investment_plan"]}
{lessons_line}
**Risk debate history:**
{debate.get("history", "")}

## Rules

1. Rating: exactly one of Buy / Overweight / Hold / Underweight / Sell (the 5-tier scale downstream systems parse).
2. The deterministic market regime and the data-quality gate are authoritative. If the reports carry unresolved DATA ERROR flags, the only defensible rating is Hold (no action) with that stated as the reason.
3. Ground every conclusion in specific evidence from the analysts; note conflicting evidence explicitly.
4. Your decision feeds a deterministic risk gate and paper-trading engine — state the decision and the thesis; code owns position sizing and final limits.

## Output

- **Rating**: exactly one of Buy / Overweight / Hold / Underweight / Sell
- **Executive Summary**: the call and how to act on it
- **Investment Thesis**: the evidence that decided it, and what would change it

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        decision = invoke_structured_or_freetext(
            structured_llm, llm, prompt, render_pm_decision, "Gold Portfolio Manager",
        )
        return {
            "risk_debate_state": {
                "judge_decision": decision,
                "history": debate.get("history", ""),
                "aggressive_history": debate.get("aggressive_history", ""),
                "conservative_history": debate.get("conservative_history", ""),
                "neutral_history": debate.get("neutral_history", ""),
                "latest_speaker": "Judge",
                "current_aggressive_response": debate.get("current_aggressive_response", ""),
                "current_conservative_response": debate.get("current_conservative_response", ""),
                "current_neutral_response": debate.get("current_neutral_response", ""),
                "count": debate["count"],
            },
            "final_trade_decision": decision,
        }

    return node
