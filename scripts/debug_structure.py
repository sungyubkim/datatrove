import pandas as pd
import numpy as np

df = pd.read_parquet('./output/orz-math-cleaned/000_00000.parquet')
row = df.iloc[0]
metadata = row['metadata']
prompt = metadata.get('prompt', [])

print('Prompt type:', type(prompt))
print('Is numpy array:', isinstance(prompt, np.ndarray))

if isinstance(prompt, np.ndarray):
    print('Array shape:', prompt.shape)
    print('Array dtype:', prompt.dtype)
    print('First element type:', type(prompt[0]))
    print('First element:', prompt[0])
    print('First element keys:', prompt[0].keys() if hasattr(prompt[0], 'keys') else 'N/A')
