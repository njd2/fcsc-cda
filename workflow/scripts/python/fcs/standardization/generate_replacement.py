import pandas as pd
from pathlib import Path
from typing import Any


# TODO this would be much better with pydantic
def generate_exclusion(x: dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    def go(col: str, _df: pd.DataFrame) -> pd.DataFrame:
        if col in x:
            xcol = x[col]
            if isinstance(xcol, list):
                return pd.concat([_df[_df[col] == xc] for xc in xcol])
            else:
                return _df[_df[col] == xcol]
        return _df

    # TODO add filepath to this to make downstream ops easy

    # get stuff to be replaced in one dataframe
    df = go("org", df)
    df = go("machine", df)
    df = go("sop", df)
    df = go("eid", df)
    df = go("rep", df)
    df = go("material", df)
    # the only thing that changes is the machine, which means that we can only
    # replace data b/t machines that are the same in every other way
    df["machine_replace"] = x["replacement"]
    df["comment"] = x["comment"]
    return df.copy()


def main(smk: Any) -> None:
    sp = smk.params
    xs = sp["replacements"]
    combos_in = Path(smk.input[0])
    replacments_out = Path(smk.output[0])

    df_combos = pd.read_table(combos_in)

    df_replace = pd.concat([generate_exclusion(x, df_combos) for x in xs])

    df_replace.to_csv(replacments_out, sep="\t", index=False)


main(snakemake)  # type: ignore
