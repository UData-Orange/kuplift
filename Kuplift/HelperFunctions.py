from math import log

_Log_Fact_Table = []

def log_fact(n):
    """
    Compute log(fact(n))
    :param n:
    :return: value of log(fact(n))
    """
    # Use approximation for large n
    if n > 1e6:
        raise ValueError("Value of n is too large")
    # computation of values, tabulation in private array
    else:
        s = len(_Log_Fact_Table)
        if n >= s:
            if s == 0:
                _Log_Fact_Table.append(0)
            size = len(_Log_Fact_Table)
            while size <= n:
                # print('%d<=%d' %(size,n))
                _Log_Fact_Table.append(log(size) + _Log_Fact_Table[size - 1])
                size = size + 1
        return _Log_Fact_Table[n]

def log_2_star(k: int):
    """
    Computes the term log_2*(k)=log_2(k) + log_2(log_2(k)) + ...  of Rissanen's code for integers
    so long as the terms are positive
    :param k:
    :return:
    """
    d_log2 = log(2.0)
    d_cost = 0.0
    d_logI = log(1.0 * k) / d_log2

    if k < 1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        while d_logI > 0:
            d_cost += d_logI
            d_logI = log(d_logI) / d_log2

        return d_cost    


def universal_code_natural_numbers(k: int):
    """
    Compute the universal code for integers presented by Rissanen in
    'A Universal Prior for Integers and Estimation by Minimum Description Length', Rissanen 1983
    :param k:
    :return:
    """
    dC0 = 2.86511  # First value computed following the given estimation formula, as e(3)=65536 + d_log2^5 / (1-d_log2)
    d_log2 = log(2.0)

    if k < 1:
        raise ValueError("Universal code is defined for natural numbers over 1")
    else:
        # Initialize code length cost to log_2(dC0)
        d_cost = log(dC0) / d_log2

        # Add log_2*(k)
        d_cost += log_2_star(k)

        # Go back to the natural log
        d_cost *= d_log2

        return d_cost 
    
def preprocessData(Data_features, treatment_col='segment', y_col='visit'):
    cols = Data_features.columns
    num_cols = list(Data_features._get_numeric_data().columns)

    num_cols.remove(treatment_col)
    num_cols.remove(y_col)
    for num_col in num_cols:
        if len(Data_features[num_col].value_counts())<(Data_features.shape[0]/100):
            num_cols.remove(num_col)
        else:
            Data_features[num_col] = Data_features[num_col].fillna(Data_features[num_col].mean())

    categoricalCols=list(set(cols) - set(num_cols))
    if treatment_col in categoricalCols:
        categoricalCols.remove(treatment_col)
    if y_col in categoricalCols:
        categoricalCols.remove(y_col)
    for catCol in categoricalCols:
        Data_features[catCol] = Data_features[catCol].fillna(Data_features[catCol].mode()[0])
        DictValVsUplift={}
        for val in Data_features[catCol].value_counts().index:
            dataset_slice=Data_features[Data_features[catCol]==val]
            t0j0=dataset_slice[(dataset_slice[treatment_col]==0)&(dataset_slice[y_col]==0)].shape[0]
            t0j1=dataset_slice[(dataset_slice[treatment_col]==0)&(dataset_slice[y_col]==1)].shape[0]
            t1j0=dataset_slice[(dataset_slice[treatment_col]==1)&(dataset_slice[y_col]==0)].shape[0]
            t1j1=dataset_slice[(dataset_slice[treatment_col]==1)&(dataset_slice[y_col]==1)].shape[0]

            if (t1j1+t1j0)==0:
                UpliftInThisSlice=-1
            elif (t0j1+t0j1)==0:
                UpliftInThisSlice=0
            else:
                UpliftInThisSlice=(t1j1/(t1j1+t1j0))-(t0j1/(t0j1+t0j1))
            DictValVsUplift[val]=UpliftInThisSlice
        OrderedDict={k: v for k, v in sorted(DictValVsUplift.items(), key=lambda item: item[1])}
        encoded_i=0
        for k,v in OrderedDict.items():
            Data_features[catCol] = Data_features[catCol].replace([k],encoded_i)
            encoded_i+=1
    Data_features[treatment_col]=Data_features[treatment_col].astype(str)
    return Data_features
    
def log_binomial_coefficient(n: int, k: int):
    """
    Computes the log of the binomial coefficient  (n
                                                   k)
    (log of the total number of combinations of k elements from n)
    :param n: Total number of elements
    :param k: Number of selected elements
    :return:
    """
    
    global _Log_Fact_Table
    
    global binomialFunctionAccessCount
    binomialFunctionAccessCount+=1
    try:
        nf = _Log_Fact_Table[n]
        kf = _Log_Fact_Table[k]
        nkf = _Log_Fact_Table[n - k]
    except:
        print("length of log_fact table is ",len(_Log_Fact_Table))
        print("n is ",n)
        raise
    return (nf - nkf) - kf
