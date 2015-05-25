from numpy import mean, median, std, percentile, fft, abs, argmax

def Mean(data):
    """Returns the mean of a time series"""
    return data.mean()

def Median(data):
    """Returns the median of a time series"""
    return data.median()

def Std(data):
    """Returns the standard deviation a time series"""
    return data.std()

def IQR(data):
    """Returns the interquartile range a time series"""
    return percentile(data, 75) - percentile(data, 25)

def Min(data):
    """Returns the minimum value of a time series"""
    return data.min()

def Max(data):
    """Returns the maximum value of a time series"""
    return data.max()

def Length(data):
    """Returns the number of samples in a time series"""
    return len(data)

def DominantFrequency(data):
    """Returns the dominant frequency of a time series with 32 samples per second"""
    w = fft.fft(data)
    freqs = fft.fftfreq(len(data))
    i = argmax(abs(w))
    dom_freq = freqs[i]
    dom_freq_hz = abs(dom_freq * 32.0)
    return dom_freq_hz