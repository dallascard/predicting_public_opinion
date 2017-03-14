import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import entropy

from collections import Counter

def read_tone_data(data_file, first_year):
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

    # determine the quarter (1/4 year) from the date
    data['quarter'] = data.apply(lambda row: row['date'].quarter, axis=1)

    # compute the net tone
    data['tone'] = data['Pro'] - data['Anti']
    data['directness'] = data['Explicit']

    return data


def get_frame_vals(row):
    frames = ['p' + str(i) for i in range(15)]
    frame_vals = np.array([row[f] for f in frames])    
    return frame_vals


def group_tone_data(data, group_by):
    """
    Group the data in a DataFrame by either month or quarter
    :param data (DataFrame): the data frame to group
    :param group_by (str): either 'month' or 'quarter'
    :return A new DataFrame grouped accordingly
    """

    # create a dummy variable = 1 for all articles
    data['stories'] = 1
      
    # create a grouping of articles by year/quarter
    if group_by == 'quarter':
        groups = data.groupby(['Year', 'quarter'])
    elif group_by == 'month':
        groups = data.groupby(['Year', 'Month'])
    else:
        sys.exit('group_by not recognized')
      
    # create a new dataframe "grouped", which is what we will work with
    grouped = pd.DataFrame()

    # add up the total number of articles per quarter and store in grouped
    grouped['stories'] = groups.aggregate(np.sum)['stories']

    grouped['tone'] = groups.aggregate(np.mean)['tone']
    tone = grouped['tone'].values
    N = grouped['stories'].values
    tone_sd = np.sqrt(tone * (1-tone) / N)
    grouped['tone_sd'] = tone_sd

    grouped['directness'] = groups.aggregate(np.mean)['directness']
    directness = grouped['directness'].values
    directness_sd = np.sqrt(directness * (1-directness) / N)
    grouped['directness_sd'] = directness_sd

    frames = ['p' + str(i) for i in range(15)]
    for c in frames:
        grouped[c] = groups.aggregate(np.mean)[c]

    for i, index in enumerate(grouped.index):
        row = grouped.loc[index]
        frame_vals = get_frame_vals(row)
        grouped.loc[index, 'entropy'] = entropy(frame_vals)

    if group_by == 'quarter':
        grouped['x'] = [i[0] + (i[1]-1)*0.25 for i in grouped.index]
    elif group_by == 'month':
        grouped['x'] = [i[0] + (i[1]-1)/12.0 for i in grouped.index]
    else:
        grouped['x'] = 0

    return grouped


def load_polls(filename, first_year, last_date, subcode=None):
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

  dates = [dt.datetime.strptime(str(d), '%m/%d/%Y') for d in df['Date']]
  values = df['Index'].as_matrix() / 100.0
  n = df['N'].as_matrix()
  varnames = df['Varname'].ravel()
  df['date'] = dates

  # only include polls with N > 0
  df = df[df['N'] > 0]
  df = df[df['date'] > dt.date(first_year, 1, 1)]
  df = df[df['date'] <= last_date]  

  if subcode is not None:
    df = df[df['Subcode'] == subcode]

  df = df.sort('date')

  return df


def store_previous_poll_result(polls):
    varnames = set(polls['Varname'].ravel())

    dfs = []
    
    for varname in varnames:
        df_v = polls[polls['Varname'] == varname]
        df_v.sort_values(by='date')
        for i, index in enumerate(df_v.index):
            if i == 0:
                df_v.loc[index, 'predict'] = 0
            else:
                df_v.loc[index, 'prev_value'] = df_v.loc[df_v.index[i-1], 'Index']
                df_v.loc[index, 'predict'] = 1

        dfs.append(df_v)

    df = pd.concat(dfs)
    df = df.sort('date')
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


def combine_polls_with_preceeding_articles(polls, data, n_days=30):

    combined = pd.DataFrame(columns=['Pro', 'Neutral', 'Anti', 'directness', 'stories'], dtype=float)

    for i, index in enumerate(polls.index):
        date_i = polls.loc[index, 'date']
        articles = data[(data['date'] <= date_i) & (data['date'] > date_i - dt.timedelta(n_days))]
        n_articles, _ = articles.shape
        article_mean = articles.mean(axis=0)
        combined.loc[index, 'Pro'] = float(article_mean['Pro'])
        combined.loc[index, 'Neutral'] = article_mean['Neutral']
        combined.loc[index, 'Anti'] = article_mean['Anti']
        combined.loc[index, 'directness'] = article_mean['directness']
        combined.loc[index, 'stories'] = n_articles
        combined.loc[index, 'entropy'] = entropy(get_frame_vals(article_mean))

    combined['tone'] = combined['Pro'] - combined['Anti']
    combined['date'] = polls['date']
    combined['Value'] = polls['Index'] / 100.0
    combined['N'] = polls['N']
    combined['Varname'] = polls['Varname']
    combined['running_average'] = polls['running_average'].values / 100.0
    combined['predict'] = polls['predict']
    combined['prev_value'] = polls['prev_value'] / 100.0

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