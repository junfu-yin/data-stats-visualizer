# -*- coding: utf-8 -*-
"""
Created on Mon Mar 06 14:12:19 2017

@author: operjyu
"""

import pandas as pd
import numpy as np
#import collections
import matplotlib.pyplot as plt
import operator
import random

from scipy.stats import gaussian_kde

import sys, getopt


#from os import listdir
#from os.path import isfile

import os

def data_stats_generator(input_file, output_path):
    
#    print('Number of arguments:' + str(len(sys.argv)) + 'arguments.')
#    print('Argument List:' + str(sys.argv))
    
    #files = [f for f in listdir('.') if isfile(f)]
    #for f in files:
    #    if '.csv' in f:
    #        print(f)
            
#    path = r'D:/'
#    outputpath = path
#    filename = 'LandTaxCompanies.csv'

    pathname = input_file
    outputpath = output_path
    
    filename = os.path.basename(pathname)
    
    data = pd.read_csv(pathname, header=0, dtype=str, delimiter = ',', quotechar = '"')
    
    
    
    #print properties.head()
    
    
    #data['Ref_ID'].astype(float).plot()
    
    #data['ZONE_DESCRIPTION'].value_counts().plot(kind='pie')
    
    data =  data.fillna(np.nan)
    
    
    first_line = data.columns.values
    
    
    number_of_columns = len(data.columns)
    my_data = data.as_matrix()
    #my_data = data.values.tolist()
    
    #values = np.zeros(columns.size)
    #
    ##print np.concatenate((columns, values))
    ##print np.append(columns, values)
    #recs = np.vstack((columns, values)).T

    
    
    #data['VALUATION_AMOUNT'].plot().hist(orientation='vertical', cumulative = True)
    
    
    #recs = (columns, values)
    #print recs[0]
    
    color_palette = ['aliceblue','antiquewhite','aqua','aquamarine','azure','beige','bisque','black','blanchedalmond'
    ,'blue','blueviolet','brown','burlywood','cadetblue','chartreuse','chocolate','coral','cornflowerblue'
    ,'cornsilk','crimson','cyan','darkblue','darkcyan','darkgoldenrod','darkgray','darkgreen','darkkhaki'
    ,'darkmagenta','darkolivegreen','darkorange','darkorchid','darkred','darksalmon','darkseagreen','darkslateblue','darkslategray'
    ,'darkturquoise','darkviolet','deeppink','deepskyblue','dimgray','dodgerblue','firebrick','floralwhite','forestgreen'
    ,'fuchsia','gainsboro','ghostwhite','gold','goldenrod','gray','green','greenyellow','honeydew'
    ,'hotpink','indianred','indigo','ivory','khaki','lavender','lavenderblush','lawngreen','lemonchiffon'
    ,'lightblue','lightcoral','lightcyan','lightgoldenrodyellow','lightgreen','lightgray','lightpink','lightsalmon','lightseagreen'
    ,'lightskyblue','lightslategray','lightsteelblue','lightyellow','lime','limegreen','linen','magenta','maroon'
    ,'mediumaquamarine','mediumblue','mediumorchid','mediumpurple','mediumseagreen','mediumslateblue','mediumspringgreen','mediumturquoise','mediumvioletred'
    ,'midnightblue','mintcream','mistyrose','moccasin','navajowhite','navy','oldlace','olive','olivedrab'
    ,'orange','orangered','orchid','palegoldenrod','palegreen','paleturquoise','palevioletred','papayawhip','peachpuff'
    ,'peru','pink','plum','powderblue','purple','red','rosybrown','royalblue','saddlebrown'
    ,'salmon','sandybrown','seagreen','seashell','sienna','silver','skyblue','slateblue','slategray'
    ,'snow','springgreen','steelblue','tan','teal','thistle','tomato','turquoise','violet'
    ,'wheat','white','whitesmoke','yellow','yellowgreen']
    
    #counter = collections.Counter()
    
    for i in range(number_of_columns):
        current_column_name = first_line[i]
        print('current_column_name = ' + current_column_name)
        
        current_column_content = (my_data.T)[i]
        print('current_column_content = ' + str(current_column_content))
        
        
        
    #    Hist = np.histogram(current_column_content, bins = 'auto')
    #    current_column_content
    #    Hist = dict((x, current_column_content.count(x)) for x in current_column_content)
    
    
    
    #    counter.clear()
    #    Hist = counter
    #    Hist.update(current_column_content)
    
        Hist = {}
        
        threshold = 15
        for v in current_column_content:
            if v is np.nan:
                v = 'nan'
            else:
                v = str(v)
            if v not in Hist:
                Hist[v] = 0
            Hist[v] += 1
            if len(Hist.keys()) >= threshold:
                break    
    
        if len(Hist.keys()) < threshold:
            print('Drawing PIE')
        
            colors = ['yellowgreen', 'red', 'gold', 'lightskyblue', 
                      'lightcoral','blue','pink', 'darkgreen', 
                      'yellow','grey','violet','magenta','cyan']
    #        Hist = collections.OrderedDict(sorted(Hist.items()))        
            sorted_x = sorted(Hist.items(), key=operator.itemgetter(1))
            Hist = dict(sorted_x)
            
            LABELS = Hist.keys()  
#            print (LABELS)
            
            picsize = len(Hist.keys())
            if picsize < 10:
                picsize = 10
            
            fig = plt.figure(figsize=(1.5 * picsize , 1.3 * picsize))
    #        plt.bar(x, Hist.values())
    
            values = [float(v) for v in Hist.values()]
            keys = [s for s in LABELS]
            patches, plt_labels, junk  = plt.pie(values, labels=keys,
                                        autopct='%1.1f%%', 
                                        shadow=True, 
                                        startangle=90,
                                        colors = colors
                                        )
                                        # The default startangle is 0, which would start
                                        # the Frogs slice on the x-axis.  With startangle=90,
                                        # everything is rotated counter-clockwise by 90 degrees,
                                        # so the plotting starts on the positive y-axis.
                            
            percent = 100.* np.array(values)/np.array(values).sum()
            labels_per = ['{0} - {1:1.2f}% - {2}'.format(itp,jtp,ktp) for itp,jtp,ktp in zip(np.array(keys), percent, np.array(values))]
    
            sort_legend = True
            if sort_legend:
                patches, labels_per, dummy =  zip(*sorted(zip(patches, labels_per, list(values)),
                                                      key=lambda x: x[2],
                                                      reverse=True))
    #        plt.legend(patches, labels_per, bbox_to_anchor=(1.25, 1))     
            plt.legend(patches, labels_per, loc = 'best')     
               
               
                            
            plt.title(filename + '~' + current_column_name, bbox={'facecolor':'0.8', 'pad':5})
    
    #        plt.xticks([offsetx+0.5 for offsetx in x], LABELS,  rotation='vertical')
    #        fig.autofmt_xdate()
    
            
            plt.savefig(outputpath + current_column_name + '.jpg')
            
            plt.close(fig)
            continue
            
    
            
    #------------------------------------------------------------------------------
    #-------------This is for the real values (float and double)-------------------
    #------------------------------------------------------------------------------
            
            
        isvalues = True
        for v in current_column_content:
            try:
                i = float(v)
            except ValueError:
    #            print "not a number\n"
                isvalues = False
                break
        
    #    if 'ID' not in current_column_name and isvalues:
        if isvalues:
            print('Drawing Hist')

    #        current_column_content.sort()
    #        x = [v for v in range(len(current_column_content))]
    #        plt.scatter(x, current_column_content, s = 1, alpha = 0.5)
    #        continue
        
            current_column_content = [float(v) for v in current_column_content]
    
    #        The following line is not working!!!!
    #        current_column_content.sort()
    
    
            current_column_content = np.array(current_column_content)
            current_column_content = current_column_content[np.argsort(current_column_content)]
    
    

            picsize = 15
            
            fig = plt.figure(figsize=(1.5 * picsize , 1.3 * picsize))
    #        plt.bar(x, Hist.values())
            
    #        plt.scatter(x, current_column_content, s = 1, alpha = 0.5, c = random.choice(color_palette))
            
            chosencolor = random.choice(color_palette)
            
            ret_n,ret_bins,ret_patches = plt.hist(current_column_content[~np.isnan(current_column_content)],  
                                            bins = 20,normed = 1, 
                                            facecolor=chosencolor,
                                            alpha = 0.75,
                                            edgecolor = 'none')
    
    
            vals = ret_n * max(current_column_content[~np.isnan(current_column_content)])
            labels_per = ['{1:1.2f}% - {1:1.2f}%'.format(itp,jtp) for itp,jtp in zip(np.array(vals), ret_n)]
            for rect, label in zip(ret_patches, labels_per):
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    
            
            current_column_content_NOnan = current_column_content[~np.isnan(current_column_content)]
            
            density = gaussian_kde(current_column_content_NOnan)
            bw = density.covariance_factor()
            density2 = gaussian_kde(current_column_content_NOnan, bw_method=bw)
            
    
            
            
            xs = np.linspace(min(current_column_content_NOnan),max(current_column_content_NOnan),1000)
    #        density.covariance_factor = lambda : .25
    #        density._compute_covariance()
            
            
            
            plt.plot(xs, density2.evaluate(xs))
            
    
    #        plt.xscale('log')
            
    
                            
            plt.title(filename + '~' + current_column_name + '~bar:'+chosencolor, bbox={'facecolor':'0.8', 'pad':5})
    
    #        plt.xticks([offsetx+0.5 for offsetx in x], LABELS,  rotation='vertical')
    #        fig.autofmt_xdate()
    
            
            plt.savefig(outputpath + current_column_name + '.jpg')
            
            plt.close(fig)
            continue
    
    
def main(argv):
    inputfile = ''
    outputfile = ''
       
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
     
    data_stats_generator(input_file = inputfile, output_path = outputfile)
#   print('Input file is "', inputfile)
#   print('Output file is "', outputfile)
         

if __name__ == "__main__":
    main(sys.argv[1:])
