import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

from collections import Counter

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):  
    r, g, b = tableau20[i]  
    tableau20[i] = [r / 255., g / 255., b / 255.]

    
markers = {0: "o", #circle
           1: "v", #triangle_down
           2: "^", #triangle_up
           3: "<", #triangle_left
           4: ">", #triangle_right
           5: "s", #square
           6: "p", #pentagon
           7: "D", #diamond
           8: "h", #hexagon
           9: "8", #octagon
           }

def plot_variables(grouped):

  # plot the basic variables of interest
  fig, ax = plt.subplots(4, sharex=True)
  ax1, ax2, ax3, ax4 = ax

  x = grouped['x']
  tone = grouped['tone']
  tone_sd = grouped['tone_sd']

  ax1.fill_between(x,  tone+tone_sd*2, tone-tone_sd*2, facecolor='grey', edgecolor='white', alpha=0.6)
  ax1.plot(x, tone.ravel(), label='Net tone')
  ax1.set_ylim(0, 1)
  ax1.legend(loc='upper left')

  directness = grouped['directness']
  directness_sd = grouped['directness_sd']

  ax2.fill_between(x,  directness+directness_sd*2, directness-directness_sd*2, facecolor='grey', edgecolor='white', alpha=0.6)
  ax2.plot(x, directness, label='Directness')
  ax2.set_ylim(0, 1)
  ax2.legend(loc='upper left')

  ax3.plot(x, grouped['stories'], label='Number of stories')
  ax3.legend(loc='upper left') 

  ax4.plot(x, grouped['entropy'], label='Entropy')  


def plot_frames(grouped):
    f, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16)) = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8,6))
    axes = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15)
    reorder = ['Economic',
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

    frames = ['p' + str(i) for i in range(15)]
    for k, frame in enumerate(frames):
        x = grouped['x']
        n = grouped['stories']
        y = grouped[frame].as_matrix()
        y_sd = np.sqrt(y * (1-y) / n)
        globalMean = np.mean(y)
        axes[k].fill_between(x,  y+y_sd*2, y-y_sd*2, facecolor='grey', edgecolor='white', alpha=0.6)
        axes[k].plot(x, y, c='blue')
        #axes[k].plot(x, np.zeros(len(y)), 'k--')
        #axes[k].plot(x, np.ones(len(y))*globalMean, 'k--')
        axes[k].set_ylim(0, 0.6)
        #axes[k].set_xlim(1980, 2015)
        axes[k].text(np.min(x)+1, 0.5, reorder[k])
        if k > 12:
            axes[k].set_xticks([1990, 2000, 2010])


    ax16.axis('off')
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)  


def plot_polling_data(polls):

  varname_vals = set(polls['Varname'].ravel())
  varname_counts = Counter()
  varname_counts.update(polls['Varname'].ravel())
  top_varnames = [k for k, c in varname_counts.most_common()]
  #top_varnames = ['IMMKEPT', 'IMMIGRAT']
  #top_varnames = ['IMMKEPT', 'IMMIGRAT', 'IMMLEGAL', 'PSRAIMM', 'CNNIMM', 'SPIMMIG', 'IMMRATE']

  n_colours = 10
  varname_index = dict(zip(top_varnames[:n_colours], range(n_colours)))  

  #house_vals = set(polls['House'].ravel())
  #house_counts = Counter()
  #house_counts.update(polls['House'].ravel())
  #top_houses = [k for k, c in house_counts.most_common(n=10)]
  #house_index = dict(zip(top_houses, range(len(top_houses))))

  max_N = np.max(polls['N'])
  fig, ax = plt.subplots(figsize=(8, 6))

  print "Question\tResponses"
  for v_i, varname in enumerate(top_varnames):
      print "%8s\t%d" % (varname, varname_counts[varname])
      polls_v = polls[polls['Varname'] == varname]
      size = [max(1, 150*s/float(max_N)) for s in polls_v['N'].ravel()]
      if varname in varname_index:
          facecolor=tableau20[v_i * 2 + 1]
          edgecolor=tableau20[v_i * 2]
      else:
          facecolor='white'
          edgecolor='black'
              
      dates = polls_v['date'].ravel()
      # deal with an annoying bug in plt.scatter that will plot 3 points in different colours
      if len(dates) == 3 and len(facecolor) == 3: 
        facecolor = facecolor + [1]
        edgecolor = edgecolor + [1]
      ax.scatter(dates, polls_v['Value'].ravel(), s=size, facecolor=facecolor, edgecolor=edgecolor, label=varname, alpha=0.5)

  plt.legend(loc='upper right', scatterpoints=1, bbox_to_anchor=(1.3,1))
  first_year = polls['date'].min().year-2
  last_year = polls['date'].max().year+2
  plt.xlim(dt.date(first_year, 1, 1), dt.date(last_year, 1, 1))
  #plt.plot(polls['date'].values, polls['entropy'].values /, 'k')
  #plt.plot(polls['date'].values, 0.3 + polls['tone'].values, 'k')
  #plt.plot(polls['date'].values, 0.3 + polls['directness'].values * polls['tone'].values, 'g')
  #plt.plot(polls['date'].values, 0.5 +  polls['tone'].values * polls['stories'].values / polls['stories'].max(), 'r')
  #plt.plot(polls['date'].values, polls['stories'].values / polls['stories'].max(), 'b')
  #plt.scatter(polls['date'].values, polls['tone'].values, c='k')
  plt.ylim(0,1)


