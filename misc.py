import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import entropy
import statsmodels.api as sm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from collections import Counter

FRAMES = ['Economic',
         'Capacity_and_resources',
         'Morality',
         'Fairness_and_equality',
         'Legality_jurisdiction',
         'Policy_prescription',
         'Crime_and_punishment',
         'Security_and_defense',
         'Health_and_safety',
         'Quality_of_life',
         'Cultural_identity',
         'Public_sentiment',
         'Political',
         'External_regulation',
         'Other']


def read_article_data(data_file, first_year, rename_frames=False):
    """
    Read in tone predictions for an issue and return a pandas DataFrame
    :param data_file (.csv file): file containing the predictions
    :param first_year (int): Exclude all rows prior to this year
    :return: pandas DataFrame
    """

    # read the data into a pandas data frame
    data = pd.read_csv(data_file, header=0, index_col=0)

    # exclude articles marked as "irrelevant" and those from before 1980
    data = data.loc[(data['Irrelevant']==0) & (data['Year'] >= first_year)]

    # combine year, month, day into a single date variable
    data['date'] = data.apply(lambda row: pd.Timestamp(dt.date(row['Year'], row['Month'], row['Day'])), axis=1)

    if rename_frames:
        columns = list(data.columns)
        for f_i, f in enumerate(FRAMES):
            col_index = columns.index('p' + str(f_i))
            columns[col_index] = f
        data.columns = columns

    return data


def convert_dates(df, first_year):
    df['year'] = [d.year for d in df['date']]
    df['month'] = [d.month for d in df['date']]
    df['day'] = [d.day for d in df['date']]
    df['quarter'] = [d.quarter for d in df['date']]
    df['p_month'] = [(d.year - first_year)*12 + (d.month - 1) for d in df['date']]
    df['p_quarter'] = [(d.year - first_year)*4 + (d.quarter - 1) for d in df['date']]
    return df


def get_f_dates(data, first_year, group_by):
    # create a grouping of articles by year/quarter
    if group_by == 'month':
        data['f_date'] = data['year'] + (data['month'] - 1) / 12.0
    elif group_by == 'quarter':
        data['f_date'] = data['year'] + (data['quarter'] - 1) / 4.0
    else:
        sys.exit('group_by not recognized')
    data['f_date_0'] = data['f_date'] - first_year
    return data


def group_article_data(data, group_by, first_year, group_tone=False, group_frames=False):
    """
    Group the data in a DataFrame by either month or quarter
    :param data (DataFrame): the data frame to group
    :param group_by (str): either 'month' or 'quarter'
    :return A new DataFrame grouped accordingly
    """

    # create a dummy variable = 1 for all articles
    data['stories'] = 1

    data = get_f_dates(data, first_year, group_by)
      
    if group_by == 'quarter':
        groups = data.groupby('p_quarter')
    elif group_by == 'month':
        groups = data.groupby('p_month')
    else:
        sys.exit('group_by not recognized')

      
    # create a new dataframe "grouped", which is what we will work with
    grouped = pd.DataFrame()
    grouped['f_date'] = groups.aggregate(np.mean)['f_date']
    grouped['f_date_0'] = groups.aggregate(np.mean)['f_date_0']

    # add up the total number of articles per quarter and store in grouped
    grouped['stories'] = groups.aggregate(np.sum)['stories']

    if group_tone:
        grouped['tone'] = groups.aggregate(np.mean)['tone']
        grouped['Pro'] = groups.aggregate(np.mean)['Pro']
        grouped['Neutral'] = groups.aggregate(np.mean)['Anti']
        grouped['Anti'] = groups.aggregate(np.mean)['Anti']

    if group_frames:
        for c in FRAMES:
            grouped[c] = groups.aggregate(np.mean)[c]

    log_stories = np.log(grouped['stories'].values)
    grouped['logStories'] = log_stories - float(np.mean(log_stories))

    return grouped


def compute_entropy(df):
    for i, index in enumerate(df.index):
        row = df.loc[index]
        frame_vals = np.array([row[f] for f in FRAMES])    
        df.loc[index, 'entropy'] = entropy(frame_vals)
    return df


def load_polls(filename, first_year, last_date=None, subcode=None):
    """
    Load the polling data from a .csv file
    :param filename (str): The file containing the polling data
    :param first_year (int): Exclude polls before this year
    :param last_date (date): Exclude polls after this date
    :param subcode (int): If not None, only include polls with this subcode
    :return A DataFrame of polling data
    """

    df = pd.read_csv(filename)
    nRows, nCols = df.shape

    #dates = [pd.Timestamp(dt.datetime.strptime(str(d), '%m/%d/%Y')) for d in df['Date']]
    df['date'] = [pd.Timestamp(d) for d in df['Date']]

    df = convert_dates(df, first_year)
    
    df['value'] = df['Index'].as_matrix() / 100.0

    # transform poll values in range (0-1) to R
    df['transformed'] = np.log(df.value / (1 - df.value))

    # only include polls with N > 0
    df = df[df['N'] > 0]

    # filter by date
    df = df[df['date'] >= pd.datetime(first_year, 1, 1)]
    if last_date is not None:
        df = df[df['date'] <= last_date]  

    # sort the polls by date
    df = df.sort_values(by='date')
    df = convert_dates(df, first_year)
    
    return df


def calculate_weighted_average(polls, temperature=500):
    polls = polls.sort('date')
    nRows, nCols = polls.shape

    running_average = []

    for i in range(nRows):
        date_i = polls.iloc[i]['date']
        n_i = polls.iloc[i]['N']
        weight_sum = 0
        running_sum = 0
        for j in range(0, i):
            date_j = polls.iloc[j]['date']
            date_diff = date_i - date_j
            n_j = polls.iloc[j]['N']
            weight = np.exp(-date_diff.days/float(temperature)) * n_j
            running_sum += polls.iloc[j]['Index'] * weight
            weight_sum += weight
        if weight_sum  > 0:
            running_average.append(running_sum / weight_sum)
        else:
            running_average.append(np.NaN)
        #print i, running_sum, weight_sum

    return running_average


def combine_polls_and_tone(polls, grouped):
    nRows, nColls = polls.shape

    # get the dates of the polls
    dates = [dt.datetime.strptime(str(d), '%m/%d/%Y') for d in polls['Date']]
    values = polls['Index'].ravel() / 100.0
    n = polls['N'].ravel()
    varnames = polls['Varname'].values

    # create a new dataframe
    combined = pd.DataFrame(columns=grouped.columns)

    for i in range(nRows):  
        year = dates[i].year
        quarter = int((dates[i].month-1)/3.0) + 1 
        combined.loc[i] = grouped.loc[year, quarter]
        combined.ix[i, 'Year'] = int(year)
        combined.ix[i, 'Quarter'] = int(quarter)
      

    combined['date'] = dates
    combined['Value'] = values
    combined['N'] = n
    combined['Varname'] = varnames
    combined['running_average'] = polls['running_average'].values / 100.0
    combined['predict'] = polls['predict'].values
    combined['prev_value'] = polls['prev_value'].values

    max_N = np.max(combined['N'])
    combined['N'] = combined['N'] / float(max_N)

    varname_vals = set(polls['Varname'].ravel())
    for varname in varname_vals:
        combined[varname] = 0

    for row in combined.index:
        varname = combined.loc[row]['Varname']
        combined.ix[row, varname] = 1

    return combined


def combine_polls_with_preceeding_articles(polls, data, n_days=30, use_directness=True, use_frames=True):

    combined = pd.DataFrame(columns=['Pro', 'Neutral', 'Anti', 'directness', 'stories'], dtype=float)

    for i, index in enumerate(polls.index):
        date_i = polls.loc[index, 'date']
        articles = data[(data['date'] <= date_i) & (data['date'] > date_i - dt.timedelta(n_days))]
        n_articles, _ = articles.shape
        article_mean = articles.mean(axis=0)
        combined.loc[index, 'Pro'] = float(article_mean['Pro'])
        combined.loc[index, 'Neutral'] = article_mean['Neutral']
        combined.loc[index, 'Anti'] = article_mean['Anti']
        if use_directness:
            combined.loc[index, 'directness'] = article_mean['directness']
        combined.loc[index, 'stories'] = n_articles
        if use_frames:
            combined.loc[index, 'entropy'] = entropy(get_frame_vals(article_mean))
            frames = ['p' + str(i) for i in range(15)]
            for f in frames:
                combined.loc[index, f] = article_mean[f]


    combined['tone'] = combined['Pro'] - combined['Anti']
    dates = [dt.datetime.strptime(str(d), '%m/%d/%Y') for d in polls['Date']]
    combined['date'] = dates
    combined['Value'] = polls['Index'] / 100.0
    combined['normalized'] = polls['normalized']
    combined['N'] = polls['N']
    combined['Varname'] = polls['Varname']
    combined['running_average'] = polls['running_average'].values / 100.0
    if use_frames:
        combined['linear'] = polls['linear'].values
    #combined['predict'] = polls['predict']
    #combined['prev_value'] = polls['prev_value'] / 100.0

    max_N = np.max(combined['N'])
    combined['N'] = combined['N'] / float(max_N)

    varname_vals = set(polls['Varname'].ravel())
    for varname in varname_vals:
        combined[varname] = 0

    for row in combined.index:
        varname = combined.loc[row]['Varname']
        combined.ix[row, varname] = 1

    return combined


def get_top_poll_questions(polls, n=10):
    varname_counts = Counter()
    varname_counts.update(polls['Varname'].ravel())
    top_varnames = [k for k, c in varname_counts.most_common(n=n)]
    return top_varnames

