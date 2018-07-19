# feature_extraction

## Introduction
Feature engineering is often an important part of conducting a successful machine learning project.
Features are the attributes that you believe will successfully discriminate between the classes in
your data set. Your data set might contain suitable features already, or you may have so much data
that you can create and detect useful features automatically with deep neural networks, but in many
cases with limited data, you will need to use domain expertise to derive meaningful features from
the raw data. Some examples of features:
* Mean and standard deviation of a biometric time series
* Ratio of two nitrogen isotopes in a tissue sample
* Peak frequency of a waveform

In some applications, knowing which features to extract from the raw data is the tricky part, and in
others deciding which to use among an abundance of features in the model building step – feature
selection/reduction – is harder. This example will not go into detail about those challenges, but it
will demonstrate an efficient method for applying functions for feature extraction to a DataFrame
containing multiple time series. Specifically, this work demonstrates vectorized feature generation
with pandas and follows with a collection of functions for features commonly extracted from
accelerometer data.

## Prerequisites

This method assumes that:
1. You have imported and cleaned your data set and it is contained in a single pandas DataFrame
2. Samples from different subjects, observations, and/or repetitions are stacked in a flat format
where the columns hold the different sensor streams and axes
3. Any relevant metadata (e.g. for SubjectID, RepetitionID, Gender, etc.) are contained in their own
columns
4. There is a column holding the Label or Class

The example file contains data from a tri-axial accelerometer worn by 16 subjects who each performed
14 Activities of Daily Living (ADLs)<sup>1</sup>. The data come from the public [UCI Machine Learning
Repository](https://archive.ics.uci.edu/ml/index.php) and I have combined them into a file that when
imported into a DataFrame, has the format described above. I have included the conversion in this
GitHub project.

## Methodology

If you use Python to develop code for feature generation, then the functions you write to operate on
single data streams or time series can be easily adapted to work in a bulk processing paradigm.
Vectorizing your algorithms for rapid application to a DataFrame can not only speed up the calculations,
but it can also provide a convenient and extensible framework for integrating additional feature
functions as you develop more.

The pandas groupby operation lets you apply a function to subsets of a DataFrame without using a for
loop. This is very similar to applying an aggregate function with a group by clause in SQL.

For the ADL data set, we want to calculate a set of features for every sensor axis in every file. We
will apply a groupby operation that aggregates by timestamp because it is unique to every file. Since
we want the other indicators and the label to be in the output, and these are all guaranteed to be
constant for a given timestamp, we will include those columns in the groupby specification. The order
does not matter. After grouping, we can easily generate a feature like the mean for every group in the
set:
```
grouped_data = data.groupby(['SubjectID', 'Gender', 'Timestamp', 'ADL'])
grouped_data[['X', 'Y', 'Z']].mean()
```

With the agg() function, you can generate many different features in one pass through the original
DataFrame.
```
from numpy import mean, std, min, max
features = grouped_data['X'].agg({
    'MeanX': mean,
    'StdX': std,
    'MinX': min,
    'MaxX': max,
    'LenX': len
})
```

Say we would like to do this for every accelerometer axis with only a single pass through the DataFrame.
Defining a set of [functions](https://github.com/mhbuehler/feature_extraction/blob/master/features.py)
will allow us to do that while also maintaining control over the named columns. This might seem like a lot
of overhead for such readily available built-in functions like mean and std, but it will be become very
useful when you write your own feature extraction algorithms. If you find yourself repeatedly analyzing
time series data, this technique will come in handy over and over again.

## Code Tutorial
For an ipython notebook code tutorial, see
[Vectorized Feature Extraction.ipynb](https://github.com/mhbuehler/feature_extraction/blob/master/Vectorized%20Feature%20Extraction.ipynb) included in this repository.

## Citation
1. [Dataset for ADL Recognition with Wrist-worn Accelerometer](https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer):
Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.