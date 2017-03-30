import pystan
from optparse import OptionParser

import numpy as np
import pandas as pd

import misc
import plotting

model = """
data {
  int<lower=1> N;  # number of polls
  int<lower=1> J;  # number of questions
  int<lower=1,upper=J> question[N];
  vector[N] y;
  vector[N] x;
}
parameters {
  vector[J] a; # question intercepts
  real mu_a;   # mean of intercept prior 
  real<lower=0> sigma_a;  # variance of intercept prior
  real beta;  # slope
  real<lower=0> sigma_beta; # variance of slope prior
  real<lower=0> sigma_y;  # variance of observations
}
model {
  vector[N] y_hat;
  for (i in 1:N)
    y_hat[i] = a[question[i]] + beta * x[i];

  mu_a ~ normal(0, 1);
  sigma_a ~ cauchy(0, 2.5);
  sigma_beta ~ cauchy(0, 2.5);
  sigma_y ~ cauchy(0, 2.5);
  a ~ normal (mu_a, sigma_a);
  beta ~ normal(0, sigma_beta);
  y ~ normal(y_hat, sigma_y);
}
"""

def main():
    usage = "%prog poll_filename first_year output_filename"
    parser = OptionParser(usage=usage)
    parser.add_option('--gap', dest='gap', default=30,
                      help='Number of days gap before each poll: default=%default')
    parser.add_option('--transformed', action="store_true", dest="transformed", default=False,
                      help='Used transformed values: default=%default')

    (options, args) = parser.parse_args()
    filename = args[0]
    first_year = int(args[1])
    output_filename = args[2]
    gap = int(options.gap)
    transformed = options.transformed

    polls = misc.load_polls(filename, first_year, last_date=None)
    polls = polls[polls['Varname'] != 'IMMLEGAL']
	
    # try looping through all polls forward in time
	# take all polls before t - 30 days and fit a Bayesian model
	# use it to predict public opinion at t-30, and record that value
	# then use that and tone, etc. to predict the outcome of the corresponding poll
    
    time_diff = 30
    fits = {}
	# first, select the data
    n_polls = len(polls.index)
    for target in range(8, n_polls):
        print target
        target_date = polls['date'].iloc[target] - pd.Timedelta(days=time_diff)
        polls['x'] = [d.days/365.0 for d in polls['date'] - target_date]
        polls_copy = polls[(polls['date'] < target_date) & (polls['date'] > target_date - pd.Timedelta(days=365*5))].copy()

        varnames = polls_copy['Varname'].values
        questions = list(set(varnames))
        questions.sort()
        for q_i, q in enumerate(questions):
            polls_copy.loc[polls_copy['Varname']==q, 'question'] = q_i

        # fit the model
        N = len(polls_copy.index)
        n_questions = len(questions)
        question_index = np.array(polls_copy['question'].values + 1, dtype=int)
        if transformed:
            y = polls_copy['transformed'].values
        else:
            y = polls_copy['value'].values
        x = polls_copy['x'].values
        data = {'N': N, 'J': n_questions, 'question': question_index, 'y': y, 'x': x}

        # fit the model
        fit = pystan.stan(model_code=model, data=data, iter=1000, chains=4)
        fits[target] = fit
        	   
    print "Done"

    polls['mu_a'] = np.NaN
    polls['beta'] = np.NaN
    for target in range(8, n_polls):
        polls['mu_a'].iloc[target] = np.mean(fits[target].extract('mu_a')['mu_a'])
        polls['beta'].iloc[target] = np.mean(fits[target].extract('beta')['beta'])
        
    polls.to_csv(output_filename)


if __name__ == '__main__':
    main()
