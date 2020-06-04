import langdetect
import pandas as pd
from langdetect import DetectorFactory
DetectorFactory.seed = 0

test_data = pd.read_csv("test_data.csv")
predictions = []

for i in range(20):
    pred = langdetect.detect_langs(test_data['tweets'].iloc[i])
    predictions.append(pred)
print(test_data[:20])

for i in range(20):
    print(predictions[i])
