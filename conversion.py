import calendar
import pandas as pd

""" Functions for converting between various ways of representing dates and times"""

# dates = pd.Timestamp dates
# periods = periods of days, months, years, etc, starting form 0 = first_year/first_month...
# f_dates = partial year as floats, e.g. July 1, 1999 = 1999.5

def ymd_to_dates(years, months, days):
    """
    Take three vectors (years, months, and days) and return a vector of pd.Timestamep dates
    """    
    assert len(years) == len(months) == len(days)
    n_times = len(years)
    dates = []
    for t in range(n_times):
        dates.append(pd.Timestamp(year=years[t], month=months[t], day=days[t]))
    return dates

def dates_to_periods(dates, first_year, first_month=1, first_week=1, first_day=1, period='day'):
    if period == 'day':
        periods = dates_to_days(dates, first_year, first_month, first_day)
    elif period == 'week':
        periods = dates_to_weeks(dates, first_year, first_week)
    elif period == 'month':
        periods = dates_to_months(dates, first_year, first_month)
    elif period == 'quarter':    
        periods = dates_to_quarters(dates, first_year)
    elif period == 'year':
        periods = dates_to_years(dates, first_year)
    else:
        raise("period not recognized")
    return periods

def dates_to_days(dates, first_year, first_month=1, first_day=1):
    """ Convert a list of pd.Timestamp dates to a list of days, where first_year/1/1 = 0 """
    day_0 = pd.Timestamp(year=first_year, month=first_month, day=first_day)
    return [(d - day_0).days for d in dates]

def dates_to_weeks(dates, first_year, first_week=1):
    return [int((d.year - first_year) * 52 + (d.weekofyear - first_week)) for d in dates]

def dates_to_months(dates, first_year, first_month=1):
    """ Convert a list of pd.Timestamp dates to a list of month ids, where first_year/1/1 = 0 """
    return [int((d.year - first_year) * 12 + (d.month - first_month)) for d in dates]

def dates_to_quarters(dates, first_year):
    return [int((d.year - first_year) * 4 + (d.month-1)/3) for d in dates]    

def dates_to_years(dates, first_year):
    return [int(d.year - first_year) for d in dates]        

def days_to_dates(days, first_year, first_month=1, first_day=1):
    """ Convert a list of days to dates where day_0 = first_year/jan/1 """
    day_0 = pd.Timestamp(year=first_year, month=first_month, day=first_day)
    return [day_0 + pd.Timedelta(days=d) for d in days]

def periods_to_f_dates(periods, first_year, first_month=1, first_week=1, first_day=1, period='day'):
    """ Convert a list of periods (0-based int indices) to float dates (e.g. 1999.5) """
    if period == 'day':
        f_dates = dates_to_f_dates(days_to_dates(periods, first_year, first_month, first_day))
    elif period == 'week':
        f_dates = weeks_to_f_dates(periods, first_year, first_week)
    elif period == 'month':
        f_dates = months_to_f_dates(periods, first_year, first_month)
    elif period == 'quarter':
        f_dates = quarters_to_f_dates(periods, first_year)
    elif period == 'year':
        f_dates = years_to_f_dates(periods, first_year)
    return f_dates

def dates_to_f_dates(dates):
    """ Convert dates to a list of floats which represent frational years, i.e. 2017.5 """
    f_dates = []
    for d in dates:
        year = d.year
        if calendar.isleap(year):
            f_dates.append(year + (d.dayofyear-1) / 366.0)
        else:
            f_dates.append(year + (d.dayofyear-1) / 365.0)
    return f_dates

def weeks_to_f_dates(weeks, first_year, first_week=1):
    return [(w + first_week - 1) / 52.0 + first_year for w in weeks]

def months_to_f_dates(months, first_year, first_month=1):
    """
    Convert a list of integers to floats representing partial years
    :param months: a list of integers where 0 = first_year/first_month, 1 = first_year/(first_month+1), etc.
    :param first_year: year corresponding to month 0
    :param first_month: month corresponding to month 0 (January = 1)
    :return: a list of floats indicating partial years, i.e. July 1999 = 1999.5
    """
    return [(m + first_month - 1) / 12.0 + first_year for m in months]

def quarters_to_f_dates(quarters, first_year):
    return [q / 4.0 + first_year for q in quarters]

def years_to_f_dates(years, first_year):
    return [int(y + first_year) for y in years]
