import pandas as pd

def feature_engineering(df):

    # Tenure Group
    df["TenureGroup"] = pd.cut(
    df["Tenure Months"],
    bins=[0, 12, 24, 48, 72],
    labels=["New", "Mid", "Long", "Very Long"]
    )

    # Spend Category
    df["SpendCategory"] = pd.cut(
        df["Monthly Charges"],
        bins=[0, 35, 70, 100, 150],
        labels=["Low", "Medium", "High", "Very High"]
    )

    # Avg Spend
    df["AvgSpendPerMonth"] = df["Total Charges"] / (df["Tenure Months"] + 1)

    # Risk Features
    df["HighRiskCustomer"] = (
        (df["Contract"] == "Month-to-month") &
        (df["Tenure Months"] < 12)
    ).astype(int)

    # Service Count
    service_cols = [
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support"
    ]
    df["ServiceCount"] = (df[service_cols] == "Yes").sum(axis=1)

    # Streaming Count
    stream_cols = ["Streaming TV", "Streaming Movies"]
    df["StreamingCount"] = (df[stream_cols] == "Yes").sum(axis=1)

    # Payment Flags
    df["IsElectronicCheck"] = (df["Payment Method"] == "Electronic check").astype(int)

    df["IsAutoPay"] = df["Payment Method"].isin([
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]).astype(int)

    # Engagement
    df["TotalEngagement"] = df["ServiceCount"] + df["StreamingCount"] + df["IsAutoPay"]

    # Interaction feature
    df["Contract_Tenure_Risk"] = (
        (df["Contract"] == "Month-to-month").astype(int) *
        (df["Tenure Months"] < 12).astype(int)
    )

    return df