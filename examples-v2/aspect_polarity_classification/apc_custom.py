from pathlib import Path
from typing import Any

import pandas as pd
from pyabsa.utils import prep_text_for_apc
from pyabsa.tasks.AspectPolarityClassification import SentimentClassifier
from sklearn.metrics import classification_report

cwd: Path = Path(__file__).parents[2]

in_path: Path = cwd / 'unit_test' / 'data'
print(in_path.exists())

model = SentimentClassifier(
    checkpoint='english', # replace with local path to saved model or checkpoint
    verbose=False,
)

data: pd.DataFrame = pd.read_csv(in_path / 'test_data.csv')\
    .sample(frac=1.0, random_state=99)\
    .reset_index(drop=True)

# format data for aspect-based sentiment analysis
data['aspect_data'] = data.apply(
    lambda x: prep_text_for_apc(
        text=x['input'],
        aspect=x['target'],
        clean_aspect=True,
        allow_aspect_substring=False,
        allow_fuzzy=True,
    ), 
    axis=1
)

# prepare and clean the data
data = data.dropna(subset=['aspect_data'])
data['aspect_data'] = data['aspect_data'].apply(lambda x: x[0])
data[['inference_input', 'target', 'start', 'end']] = pd.DataFrame(data['aspect_data'].tolist())
data = data.dropna(subset=['inference_input'], ignore_index=True)

print(data.shape)

# run inference
predictions: list[dict[str, Any]] = model.predict(
    data['inference_input'].tolist(),
    print_result=False,
    ignore_error=False,
    eval_batch_size=32,
    merge_results=False, # important: this ensures outputs have the same length in case there are duplicate texts
)

data['sentiment_pred'] = [pred['sentiment'].lower() for pred in predictions]

print(classification_report(data['sentiment'], data['sentiment_pred']))