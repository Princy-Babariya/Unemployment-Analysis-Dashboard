from __future__ import annotations

from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


BASE_DIR = Path(__file__).resolve().parent
DATA_FILES = [
    BASE_DIR / "data" / "Unemployment in India.csv",
    BASE_DIR / "data" / "Unemployment_Rate_upto_11_2020.csv",
]


def _load_dataset() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for csv_path in DATA_FILES:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        if "Region.1" in df.columns:
            df = df.rename(columns={"Region.1": "zone"})
        elif list(df.columns).count("Region") > 1:
            renamed: list[str] = []
            seen_region = 0
            for column in df.columns:
                if column == "Region":
                    seen_region += 1
                    renamed.append("zone" if seen_region == 2 else "Region")
                else:
                    renamed.append(column)
            df.columns = renamed

        df = df.rename(
            columns={
                "Region": "region",
                "Date": "date",
                "Estimated Unemployment Rate (%)": "unemployment_rate",
                "Estimated Employed": "estimated_employed",
                "Estimated Labour Participation Rate (%)": "labour_participation_rate",
                "Area": "area",
            }
        )
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.dropna(subset=["region", "date", "unemployment_rate"]).copy()
    dataset["date"] = pd.to_datetime(dataset["date"], dayfirst=True, errors="coerce")
    dataset["unemployment_rate"] = pd.to_numeric(dataset["unemployment_rate"], errors="coerce")
    dataset = dataset.dropna(subset=["date", "unemployment_rate"]).copy()
    dataset["year"] = dataset["date"].dt.year.astype(int)
    dataset["month"] = dataset["date"].dt.month.astype(int)
    dataset["month_label"] = dataset["date"].dt.strftime("%b %Y")
    return dataset.sort_values("date").reset_index(drop=True)


DATASET = _load_dataset()


def _round(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), 2)


def _forecast_series(series: pd.Series) -> list[dict[str, float | str | None]]:
    series = series.sort_index().asfreq("ME").ffill()

    actual_points = [
        {
            "date": index.strftime("%b %Y"),
            "actual": _round(value),
            "forecast": None,
        }
        for index, value in series.tail(8).items()
    ]

    if len(series) < 4 or series.nunique() <= 1:
        return actual_points

    try:
        model_fit = ARIMA(series, order=(1, 1, 1)).fit()
        forecast = model_fit.forecast(steps=6)
        forecast_points = [
            {
                "date": index.strftime("%b %Y"),
                "actual": None,
                "forecast": _round(value),
            }
            for index, value in forecast.items()
        ]
        return actual_points + forecast_points
    except Exception:
        return actual_points


def _build_insights(filtered_df: pd.DataFrame, grouped_by_region: pd.DataFrame) -> list[dict[str, str]]:
    peak_row = filtered_df.loc[filtered_df["unemployment_rate"].idxmax()]
    low_row = filtered_df.loc[filtered_df["unemployment_rate"].idxmin()]
    top_region = grouped_by_region.iloc[0] if not grouped_by_region.empty else None

    insights = [
        {
            "title": "Peak unemployment period",
            "description": (
                f"{peak_row['region']} recorded the highest observed rate at "
                f"{_round(peak_row['unemployment_rate'])}% in {peak_row['date'].strftime('%b %Y')}."
            ),
            "tone": "alert",
        },
        {
            "title": "Lowest unemployment period",
            "description": (
                f"The lowest rate in the selected data was {_round(low_row['unemployment_rate'])}% "
                f"for {low_row['region']} in {low_row['date'].strftime('%b %Y')}."
            ),
            "tone": "positive",
        },
    ]

    if top_region is not None:
        insights.append(
            {
                "title": "Region with highest average",
                "description": (
                    f"{top_region['region']} has the highest average unemployment in the current view "
                    f"at {_round(top_region['rate'])}%."
                ),
                "tone": "info",
            }
        )

    return insights


def run_analysis(selected_region: str | None = None, selected_year: str | None = None) -> dict:
    df = DATASET.copy()
    all_regions = sorted(df["region"].dropna().unique().tolist())
    all_years = sorted(int(year) for year in df["year"].dropna().unique().tolist())

    if selected_region:
        df = df[df["region"] == selected_region]

    if selected_year:
        df = df[df["year"] == int(selected_year)]

    if df.empty:
        return {
            "summary": {
                "average_rate": 0.0,
                "max_rate": 0.0,
                "min_rate": 0.0,
                "latest_rate": 0.0,
                "records": 0,
            },
            "regions": all_regions,
            "years": all_years,
            "selected_region": selected_region or "",
            "selected_year": int(selected_year) if selected_year else None,
            "charts": {
                "trend": [],
                "covid": [],
                "regions": [],
                "seasonal": [],
                "heatmap": [],
                "forecast": [],
            },
            "insights": [
                {
                    "title": "No matching data",
                    "description": "Try a different filter combination to view unemployment analytics.",
                    "tone": "info",
                }
            ],
        }

    trend_df = (
        df.groupby("date", as_index=False)["unemployment_rate"]
        .mean()
        .sort_values("date")
        .assign(label=lambda frame: frame["date"].dt.strftime("%b %Y"))
    )
    covid_df = (
        df[df["year"] >= 2020]
        .groupby("year", as_index=False)["unemployment_rate"]
        .mean()
        .rename(columns={"unemployment_rate": "rate"})
        .sort_values("year")
    )
    region_df = (
        df.groupby("region", as_index=False)["unemployment_rate"]
        .mean()
        .rename(columns={"unemployment_rate": "rate"})
        .sort_values("rate", ascending=False)
        .head(10)
    )
    seasonal_df = (
        df.groupby("month", as_index=False)["unemployment_rate"]
        .mean()
        .sort_values("month")
        .assign(label=lambda frame: pd.to_datetime(frame["month"], format="%m").dt.strftime("%b"))
        .rename(columns={"unemployment_rate": "rate"})
    )

    heatmap_source = (
        df.groupby(["month", "year"])["unemployment_rate"]
        .mean()
        .reset_index()
        .sort_values(["month", "year"])
    )
    heatmap_pivot = heatmap_source.pivot(index="month", columns="year", values="unemployment_rate").fillna(0)
    heatmap = []
    for month, row in heatmap_pivot.iterrows():
        row_dict: dict[str, float | str] = {
            "month": pd.to_datetime(month, format="%m").strftime("%b")
        }
        for year, value in row.items():
            row_dict[str(int(year))] = _round(value)
        heatmap.append(row_dict)

    forecast = _forecast_series(
        df.groupby("date")["unemployment_rate"].mean().sort_index()
    )

    latest_row = trend_df.iloc[-1]
    summary = {
        "average_rate": _round(df["unemployment_rate"].mean()),
        "max_rate": _round(df["unemployment_rate"].max()),
        "min_rate": _round(df["unemployment_rate"].min()),
        "latest_rate": _round(latest_row["unemployment_rate"]),
        "records": int(len(df)),
        "date_range": {
            "start": df["date"].min().strftime("%d %b %Y"),
            "end": df["date"].max().strftime("%d %b %Y"),
        },
    }

    return {
        "summary": summary,
        "regions": all_regions,
        "years": all_years,
        "selected_region": selected_region or "",
        "selected_year": int(selected_year) if selected_year else None,
        "charts": {
            "trend": [
                {"date": row["label"], "rate": _round(row["unemployment_rate"])}
                for _, row in trend_df.iterrows()
            ],
            "covid": [
                {"year": str(int(row["year"])), "rate": _round(row["rate"])}
                for _, row in covid_df.iterrows()
            ],
            "regions": [
                {"region": row["region"], "rate": _round(row["rate"])}
                for _, row in region_df.iterrows()
            ],
            "seasonal": [
                {"month": row["label"], "rate": _round(row["rate"])}
                for _, row in seasonal_df.iterrows()
            ],
            "heatmap": heatmap,
            "forecast": forecast,
        },
        "insights": _build_insights(df, region_df),
    }
