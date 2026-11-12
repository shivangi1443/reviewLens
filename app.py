from pathlib import Path
import re
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="ReviewLens | Customer Intelligence", page_icon="📊", layout="wide")
DATA_PATH = Path(__file__).parent / "Customer_Sentiment.csv"
analyzer = SentimentIntensityAnalyzer()
COLORS = {"positive": "#168C72", "neutral": "#E8A838", "negative": "#D85656"}
SENTIMENT_ORDER = ["positive", "neutral", "negative"]


@st.cache_data(show_spinner="Loading and analyzing customer feedback…")
def load_data(file_bytes):
    frame = pd.read_csv(DATA_PATH if file_bytes is None else BytesIO(file_bytes), low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    required = {"customer_id", "gender", "age_group", "region", "product_category",
                "purchase_channel", "platform", "customer_rating", "review_text",
                "sentiment", "response_time_hours", "issue_resolved", "complaint_registered"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))

    for col in ["gender", "age_group", "region", "product_category", "purchase_channel",
                "platform", "sentiment", "issue_resolved", "complaint_registered"]:
        frame[col] = frame[col].fillna("unknown").astype(str).str.strip().str.lower()
    frame["customer_id"] = frame["customer_id"].astype(str).str.strip()
    frame["review_text"] = frame["review_text"].fillna("").astype(str).str.strip()
    frame["customer_rating"] = pd.to_numeric(frame["customer_rating"], errors="coerce")
    frame["response_time_hours"] = pd.to_numeric(frame["response_time_hours"], errors="coerce")
    # Keep the dataset's supplied labels as the primary sentiment field; VADER is an independent comparison.
    scores = frame["review_text"].map(analyzer.polarity_scores)
    frame["vader_score"] = scores.map(lambda score: score["compound"])
    frame["vader_sentiment"] = frame["vader_score"].map(
        lambda score: "positive" if score >= 0.05 else ("negative" if score <= -0.05 else "neutral")
    )
    frame["sentiment_match"] = frame["sentiment"] == frame["vader_sentiment"]
    return frame


def pct(numerator, denominator):
    return 100 * numerator / denominator if denominator else 0.0


st.title("📊 ReviewLens")
st.caption("Customer sentiment, satisfaction, and service operations in one interactive dashboard.")
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a compatible CSV", type=["csv"])
    st.caption("This dashboard uses the supplied sentiment labels. VADER is shown as a separate text-based comparison.")
    st.divider()
    st.header("Filter the analysis")

try:
    data = load_data(uploaded.getvalue() if uploaded else None)
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

with st.sidebar:
    categories = sorted(data["product_category"].unique())
    platforms = sorted(data["platform"].unique())
    regions = sorted(data["region"].unique())
    ages = sorted(data["age_group"].unique())
    sentiments = [s for s in SENTIMENT_ORDER if s in data["sentiment"].unique()]
    selected_categories = st.multiselect("Product category", categories, default=categories)
    selected_platforms = st.multiselect("Platform", platforms, default=platforms)
    selected_regions = st.multiselect("Region", regions, default=regions)
    selected_ages = st.multiselect("Age group", ages, default=ages)
    selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    min_star, max_star = st.slider("Customer rating", 1, 5, (1, 5))
    complaint_filter = st.selectbox("Complaint registered", ["All", "Yes", "No"])
    resolved_filter = st.selectbox("Issue resolved", ["All", "Yes", "No"])

df = data[
    data["product_category"].isin(selected_categories)
    & data["platform"].isin(selected_platforms)
    & data["region"].isin(selected_regions)
    & data["age_group"].isin(selected_ages)
    & data["sentiment"].isin(selected_sentiments)
    & data["customer_rating"].between(min_star, max_star)
].copy()
if complaint_filter != "All":
    df = df[df["complaint_registered"] == complaint_filter.lower()]
if resolved_filter != "All":
    df = df[df["issue_resolved"] == resolved_filter.lower()]

if df.empty:
    st.info("No records match the current filters. Broaden the selection in the sidebar.")
    st.stop()

n = len(df)
avg_rating = df["customer_rating"].mean()
positive_pct = pct((df["sentiment"] == "positive").sum(), n)
complaints = df[df["complaint_registered"] == "yes"]
resolution_pct = pct((complaints["issue_resolved"] == "yes").sum(), len(complaints))
avg_response = df["response_time_hours"].mean()
median_response = df["response_time_hours"].median()

tab_overview, tab_feedback, tab_service, tab_segments, tab_customer, tab_data = st.tabs(
    ["Overview", "Sentiment & feedback", "Service operations", "Customer segments", "Customer lookup", "Data & model notes"]
)

with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{n:,}")
    c2.metric("Unique customer IDs", f"{df['customer_id'].nunique():,}")
    c3.metric("Average rating", f"{avg_rating:.2f} / 5")
    c4.metric("Positive labels", f"{positive_pct:.1f}%")
    c5.metric("Complaint resolution", f"{resolution_pct:.1f}%", help="Share of complaint records marked resolved.")

    left, right = st.columns(2)
    with left:
        st.subheader("Customer rating distribution")
        rating_counts = df["customer_rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0).rename_axis("Stars").reset_index(name="Records")
        fig = px.bar(rating_counts, x="Stars", y="Records", color="Stars", color_continuous_scale=["#D85656", "#E8A838", "#168C72"])
        fig.update_layout(showlegend=False, coloraxis_showscale=False, xaxis=dict(dtick=1), margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Dataset sentiment labels")
        sent_counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0).rename_axis("Sentiment").reset_index(name="Records")
        fig = px.pie(sent_counts, names="Sentiment", values="Records", hole=0.55, color="Sentiment", color_discrete_map=COLORS)
        fig.update_layout(margin=dict(t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("At-a-glance interpretation")
    negative_pct = pct((df["sentiment"] == "negative").sum(), n)
    low_rating_pct = pct((df["customer_rating"] <= 2).sum(), n)
    if negative_pct >= 40:
        tone = "A substantial share of records is labeled negative."
    elif negative_pct >= 20:
        tone = "Negative feedback is present and worth monitoring."
    else:
        tone = "Negative labels are a smaller share of this selection."
    st.markdown(
        f"{tone} The selected records average **{avg_rating:.2f}/5**; **{low_rating_pct:.1f}%** carry a 1–2 star rating. "
        f"Among records with a registered complaint, **{resolution_pct:.1f}%** are marked resolved. "
        f"Average response time is **{avg_response:.1f} hours** (median **{median_response:.1f} hours**)."
    )
    st.info("These are descriptive summaries of the selected CSV. They show association, not proof that a platform, demographic group, or service action caused an outcome.")

with tab_feedback:
    left, right = st.columns(2)
    with left:
        st.subheader("Sentiment by product category")
        summary = df.groupby(["product_category", "sentiment"]).size().reset_index(name="Records")
        summary["Share"] = summary["Records"] / summary.groupby("product_category")["Records"].transform("sum") * 100
        fig = px.bar(summary, x="product_category", y="Share", color="sentiment", barmode="stack",
                     category_orders={"sentiment": SENTIMENT_ORDER}, color_discrete_map=COLORS,
                     labels={"product_category": "Category", "Share": "Share of category (%)"})
        fig.update_layout(xaxis_tickangle=-35, margin=dict(t=20, b=10), legend_title_text="Dataset label")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Rating and sentiment")
        cross = pd.crosstab(df["customer_rating"], df["sentiment"], normalize="index").reindex(index=[1, 2, 3, 4, 5], columns=SENTIMENT_ORDER, fill_value=0) * 100
        fig = px.imshow(cross, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                        labels=dict(x="Dataset sentiment", y="Star rating", color="Share (%)"))
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("VADER comparison")
    vader_agreement = pct(df["sentiment_match"].sum(), n)
    v1, v2 = st.columns([1, 2])
    v1.metric("Label agreement with VADER", f"{vader_agreement:.1f}%")
    v1.caption("Agreement is not model accuracy: the dataset labels may have been created by a different rule.")
    compare = pd.crosstab(df["sentiment"], df["vader_sentiment"], normalize="index").reindex(index=SENTIMENT_ORDER, columns=SENTIMENT_ORDER, fill_value=0) * 100
    fig = px.imshow(compare, text_auto=".1f", aspect="auto", color_continuous_scale="Teal",
                    labels=dict(x="VADER prediction", y="CSV sentiment label", color="Share (%)"))
    fig.update_layout(margin=dict(t=20, b=10))
    v2.plotly_chart(fig, use_container_width=True)

    st.subheader("Common words in negative feedback")
    negative_text = " ".join(df.loc[df["sentiment"] == "negative", "review_text"].astype(str)).lower()
    stop_words = {"the", "and", "for", "with", "this", "that", "was", "very", "not", "but", "after", "few", "its", "are", "product"}
    tokens = [token for token in re.findall(r"[a-z']{3,}", negative_text) if token not in stop_words]
    if tokens:
        words = pd.Series(tokens).value_counts().head(15).rename_axis("Word").reset_index(name="Mentions")
        st.bar_chart(words.set_index("Word"), color="#D85656")
    else:
        st.write("No negative review text in this selection.")

    st.subheader("Review explorer")
    query = st.text_input("Search review wording", placeholder="Search phrases such as delivery, quality, price…")
    reviews = df.copy()
    if query.strip():
        reviews = reviews[reviews["review_text"].str.contains(re.escape(query.strip()), case=False, na=False)]
    reviews["vader_score"] = reviews["vader_score"].round(3)
    review_columns = ["review_text", "sentiment", "customer_rating", "vader_sentiment", "vader_score", "product_category", "platform", "region"]
    st.dataframe(reviews[review_columns].rename(columns={"review_text": "Review", "sentiment": "CSV label", "customer_rating": "Stars", "vader_sentiment": "VADER label", "vader_score": "VADER score", "product_category": "Category"}), use_container_width=True, hide_index=True)
    st.download_button("Download filtered feedback", reviews[review_columns].to_csv(index=False).encode("utf-8"), "reviewlens_filtered_feedback.csv", "text/csv")

with tab_service:
    complaints_only = df[df["complaint_registered"] == "yes"]
    resolutions = complaints_only["issue_resolved"].value_counts().reindex(["yes", "no"], fill_value=0)
    response = df.dropna(subset=["response_time_hours"]).copy()
    left, right = st.columns(2)
    with left:
        st.subheader("Complaint outcomes")
        outcomes = resolutions.rename_axis("Outcome").reset_index(name="Records")
        outcomes["Outcome"] = outcomes["Outcome"].map({"yes": "Resolved", "no": "Unresolved"})
        fig = px.pie(outcomes, names="Outcome", values="Records", hole=0.5,
                     color="Outcome", color_discrete_map={"Resolved": "#168C72", "Unresolved": "#D85656"})
        fig.update_layout(margin=dict(t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{len(complaints_only):,} complaint records in current selection.")
    with right:
        st.subheader("Average response time by sentiment")
        response_summary = response.groupby("sentiment").agg(
            Records=("response_time_hours", "size"),
            Average=("response_time_hours", "mean"),
            Median=("response_time_hours", "median"),
            P90=("response_time_hours", lambda values: values.quantile(0.90)),
        ).reindex(SENTIMENT_ORDER).dropna(subset=["Average"]).reset_index()
        if not response_summary.empty:
            fig = px.bar(response_summary, x="sentiment", y="Average", color="sentiment",
                         category_orders={"sentiment": SENTIMENT_ORDER}, color_discrete_map=COLORS,
                         labels={"sentiment": "Dataset sentiment", "Average": "Average response time (hours)"})
            fig.update_layout(showlegend=False, margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(response_summary.rename(columns={"sentiment": "Sentiment", "Records": "Records", "Average": "Average (hours)", "Median": "Median (hours)", "P90": "90th percentile (hours)"}).round(2), use_container_width=True, hide_index=True)
            spread_minutes = (response_summary["Average"].max() - response_summary["Average"].min()) * 60
            st.caption(f"The averages differ by {spread_minutes:.1f} minutes across sentiment groups in this selection. Very similar values mean the groups will look alike; the table also shows their medians and 90th percentiles.")
    st.subheader("Service metrics by platform")
    platform_summary = df.groupby("platform").agg(
        Records=("customer_id", "size"),
        AvgRating=("customer_rating", "mean"),
        NegativeShare=("sentiment", lambda s: (s == "negative").mean() * 100),
        ComplaintShare=("complaint_registered", lambda s: (s == "yes").mean() * 100),
        ResolutionAmongComplaints=("issue_resolved", lambda s: pct((s[df.loc[s.index, "complaint_registered"] == "yes"] == "yes").sum(), (df.loc[s.index, "complaint_registered"] == "yes").sum())),
        AvgResponseHours=("response_time_hours", "mean")
    ).sort_values("Records", ascending=False)
    show = platform_summary.rename(columns={"AvgRating": "Avg rating", "NegativeShare": "Negative (%)", "ComplaintShare": "Complaints (%)", "ResolutionAmongComplaints": "Complaint resolution (%)", "AvgResponseHours": "Avg response (hours)"})
    for col in ["Avg rating", "Negative (%)", "Complaints (%)", "Complaint resolution (%)", "Avg response (hours)"]:
        show[col] = show[col].round(2)
    st.dataframe(show, use_container_width=True)
    st.caption("Platform comparisons are descriptive. Differences may reflect the composition of records in this CSV.")

with tab_segments:
    st.subheader("Compare customer groups")
    segment_field = st.selectbox("Break down by", ["product_category", "platform", "region", "age_group", "gender", "purchase_channel"])
    group = df.groupby(segment_field).agg(
        Records=("customer_id", "size"),
        AvgRating=("customer_rating", "mean"),
        PositiveShare=("sentiment", lambda s: (s == "positive").mean() * 100),
        NegativeShare=("sentiment", lambda s: (s == "negative").mean() * 100),
        ComplaintShare=("complaint_registered", lambda s: (s == "yes").mean() * 100),
        AvgResponseHours=("response_time_hours", "mean")
    ).sort_values("Records", ascending=False).reset_index()
    fig = px.scatter(group, x="AvgRating", y="NegativeShare", size="Records", color=segment_field,
                     hover_name=segment_field, hover_data={"Records": True, "PositiveShare": ":.1f", "ComplaintShare": ":.1f", "AvgResponseHours": ":.1f"},
                     labels={"AvgRating": "Average rating", "NegativeShare": "Negative feedback (%)", "Records": "Records"})
    fig.update_layout(margin=dict(t=20, b=10), legend_title_text=segment_field.replace("_", " ").title())
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(group.rename(columns={"AvgRating": "Average rating", "PositiveShare": "Positive (%)", "NegativeShare": "Negative (%)", "ComplaintShare": "Complaints (%)", "AvgResponseHours": "Avg response (hours)"}).round(2), use_container_width=True, hide_index=True)
    st.caption("Demographic cuts are aggregate comparisons only. Avoid treating them as causal explanations or as claims about an individual.")

with tab_customer:
    st.subheader("Customer record lookup")
    st.write("Search by the dataset's customer ID to inspect one record. The ID is used only as a lookup key; the dashboard does not infer personal traits.")
    customer_id = st.text_input("Customer ID", placeholder="Enter an ID from the CSV")
    if customer_id.strip():
        record = data[data["customer_id"] == customer_id.strip()]
        if record.empty:
            st.warning("No matching customer ID was found.")
        else:
            row = record.iloc[0]
            a, b, c, d = st.columns(4)
            a.metric("Rating", f"{row['customer_rating']:.0f} / 5")
            b.metric("CSV sentiment", row["sentiment"].title())
            c.metric("Response time", f"{row['response_time_hours']:g} hours")
            d.metric("Complaint", "Yes" if row["complaint_registered"] == "yes" else "No")
            st.write(f"**Review:** {row['review_text']}")
            info1, info2 = st.columns(2)
            with info1:
                st.write(f"**Category:** {row['product_category'].title()}")
                st.write(f"**Platform:** {row['platform'].title()}")
                st.write(f"**Channel:** {row['purchase_channel'].title()}")
                st.write(f"**Region:** {row['region'].title()}")
            with info2:
                st.write(f"**Age group:** {row['age_group']}")
                st.write(f"**Gender field:** {row['gender'].title()}")
                st.write(f"**Issue resolved:** {row['issue_resolved'].title()}")
                st.write(f"**VADER comparison:** {row['vader_sentiment'].title()} ({row['vader_score']:.3f})")
    else:
        st.info("Enter a customer ID to view that record. Available records are not listed by ID here.")

with tab_data:
    st.subheader("Dataset profile")
    missing_total = int(data.isna().sum().sum())
    duplicates = int(data.duplicated().sum())
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Rows", f"{len(data):,}")
    d2.metric("Columns", f"{len(data.columns)}")
    d3.metric("Missing cells", f"{missing_total:,}")
    d4.metric("Exact duplicate rows", f"{duplicates:,}")
    st.write("**Fields used:** customer ID; gender and age group; region; product category; purchase channel and platform; 1–5 customer rating; review text; supplied sentiment label; response time; complaint and resolution status.")
    st.write("**Date coverage:** this file has no date or timestamp field, so the dashboard cannot show changes over time or claim to be a live feed.")
    st.write("**Sentiment approach:** the CSV's sentiment field is treated as the supplied label. VADER runs independently on text so you can compare outputs. VADER is a general English lexicon model, not a classifier trained on this CSV.")
    st.write("**Data interpretation:** the repeated, short review phrases suggest this file may be templated or synthetic. Confirm its source and labeling method before using its findings as evidence about real customers.")
    st.write("**Privacy:** customer IDs are excluded from aggregate charts and review tables; use the lookup tab only when you have a legitimate reason to inspect a record.")
    st.subheader("CSV columns")
    st.dataframe(pd.DataFrame({"Column": data.columns, "Example": [str(data[c].iloc[0])[:120] for c in data.columns], "Data type": [str(data[c].dtype) for c in data.columns]}), use_container_width=True, hide_index=True)

st.caption("ReviewLens is an exploratory analytics prototype. Sentiment and dashboard summaries should be checked against the data source and business context.")

