
from datasets import Dataset

dataset = Dataset.from_pandas(df)

dataset.push_to_hub(
    "mangalachanda99/tourism-dataset"
)
from datasets import load_dataset

df = load_dataset(
    "mangalachanda99/tourism-dataset"
)["train"].to_pandas()

df.drop(
    columns=[
        "Unnamed: 0",
        "CustomerID"
    ],
    inplace=True
)
