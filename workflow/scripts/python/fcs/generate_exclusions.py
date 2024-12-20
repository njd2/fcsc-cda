import pandas as pd
from pathlib import Path
from typing import Any


# TODO this would be much better with pydantic
def generate_exclusion(x: dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    def go(col: str, _df: pd.DataFrame) -> pd.DataFrame:
        if col in x:
            return _df[_df[col] == x[col]]
        return _df

    df = go("org", df)
    df = go("machine", df)
    df = go("sop", df)
    df = go("eid", df)
    df = go("rep", df)
    df = go("material", df)
    df["comment"] = x["comment"]
    return df.copy()


def main(smk: Any) -> None:
    sp = smk.params
    xs = sp["exclusions"]
    combos_in = Path(smk.input[0])
    exclusions_out = Path(smk.output[0])

    df_combos = pd.read_table(combos_in)

    df_exclusions = pd.concat([generate_exclusion(x, df_combos) for x in xs])

    df_exclusions.to_csv(exclusions_out, sep="\t", index = False)


main(snakemake)  # type: ignore
