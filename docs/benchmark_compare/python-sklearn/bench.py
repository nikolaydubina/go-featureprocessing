import argparse
import random
import time

import numpy as np
import pandas as pd

from sklearn.pipeline import *
from sklearn.compose import *
from sklearn.preprocessing import *
from sklearn.feature_extraction.text import *

"""
Example from Go:

// Employee is example from readme
type Employee struct {
	Age         int     `feature:"identity"`
	Salary      float64 `feature:"minmax"`
	Kids        int     `feature:"maxabs"`
	Weight      float64 `feature:"standard"`
	Height      float64 `feature:"quantile"`
	City        string  `feature:"onehot"`
	Car         string  `feature:"ordinal"`
	Income      float64 `feature:"kbins"`
	Description string  `feature:"tfidf"`
	SecretValue float64
}
"""
    
parser = argparse.ArgumentParser(description='Benchmarking feature preprocessing from structs for sklearn')
parser.add_argument('--nsamples', type=int, default=100000, help='Number of samples')
parser.add_argument('--ntrials', type=int, default=20, help='Number of trials')
parser.add_argument('--ntrialsgroup', type=int, default=20, help='Number of trials')
args = parser.parse_args()

nsamples = args.nsamples
ntrials = args.ntrials
    
setupstartt = time.perf_counter_ns()

samples = [
    {
        'age': int(random.uniform(1, 100)),
        'salary': random.uniform(0, 9000),
        'kids': int(random.uniform(1, 10)),
        'weight': random.uniform(1, 200),
        'height': random.uniform(1, 200),
        'city': random.choice(["seoul", "pangyo", "daejeon", "busan", "something_else"]),
        'car': random.choice(["bmw", "tesla", "volvo", "hyndai", "something_else"]),
        'income': random.uniform(1, 200),
        'description': "some very long description here some very long description here some very long description here some very long description here ",
        'secret': 42.1,
    }
    for i in range(nsamples)
]
df = pd.DataFrame.from_records(samples, nrows=nsamples)

corpus = ['this is the first document', 'this document is the second document', 'and this is the third one', 'is this the first document']
vocabulary = ['this', 'document', 'first', 'is', 'second', 'the', 'and', 'one']
pipeTfidf = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)), ('tfid', TfidfTransformer())])

preprocessor = ColumnTransformer(
    transformers=[
        ('age', StandardScaler(), ["age"]),
        ('salary', MinMaxScaler(), ["salary"]),
        ('kids', MaxAbsScaler(), ["kids"]),
        ('weight', StandardScaler(), ["weight"]),
        ('height', Normalizer(), ["height"]),
        ('city', OneHotEncoder(), ["city"]),
        ('car', OrdinalEncoder(), ["car"]),
        ('income', KBinsDiscretizer(), ["income"]),
        #('description', pipeTfidf, ["description"]), #cant not run it
    ],
)
tr = preprocessor.fit(df)
    
setupendt = time.perf_counter_ns()

def benchmark():
    data = tr.transform(df)

# evaluate
# perf_counter_ns ~ 83ns precision
# monotonic_ns ~ 83ns precision
# process_time_ns ~ 2ms precision
# https://www.python.org/dev/peps/pep-0564/
runs = np.zeros(ntrials)
for i in range(ntrials):
    tic = time.perf_counter_ns()
    for j in range(args.ntrialsgroup):
        data = tr.transform(df)
    toc = time.perf_counter_ns()
    runs[i] = (toc - tic) / args.ntrialsgroup

print(f"nsamples={nsamples}\t ntrials={ntrials}\t ntrialsgroup={args.ntrialsgroup}\t avg={int(np.mean(runs))} ns\t min={int(np.min(runs))} ns\t max={int(np.max(runs))} ns\t samples_dataframe_size={df.memory_usage(index=False, deep=True)[1].sum()} B setuptook={int(setupendt - setupstartt)} ns ")
