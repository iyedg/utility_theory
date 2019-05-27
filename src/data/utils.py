def get_indicator(df, indicators, year, values_only=True):
    if isinstance(indicators, str):
        indicators = [indicators]
    df = df.pipe(
        lambda df: df.loc[(df["date"] == year) & (df["indicator"].isin(indicators))]
    ).pipe(lambda df: df.drop(columns=["date"]))
    if values_only:
        return df.loc[:, "value"]
    else:
        if len(indicators) > 1:
            return df
        else:
            return df.drop(columns=["indicator"])
