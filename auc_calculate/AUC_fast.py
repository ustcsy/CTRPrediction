#!/usr/bin/env python

import sys
import multiprocessing as mp

def scoreClickAUC(num_clicks, num_impressions, predicted_ctr):
    """
    Calculates the area under the ROC curve (AUC) for click rates

    Parameters
    ----------
    num_clicks : a list containing the number of clicks

    num_impressions : a list containing the number of impressions

    predicted_ctr : a list containing the predicted click-through rates

    Returns
    -------
    auc : the area under the ROC curve (AUC) for click rates
    """
    minCTR = min(predicted_ctr)
    maxCTR = max(predicted_ctr)
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i],
                      reverse=True)
    auc_temp = 0.0
    click_sum = 0.0
    old_click_sum = 0.0
    no_click = 0.0
    no_click_sum = 0.0

    # treat all instances with the same predicted_ctr as coming from the
    # same bucket
    last_ctr = predicted_ctr[i_sorted[0]] + 1.0

    for i in range(len(predicted_ctr)):
        if last_ctr != predicted_ctr[i_sorted[i]]:
            auc_temp += (click_sum+old_click_sum) * no_click / 2.0
            old_click_sum = click_sum
            no_click = 0.0
            last_ctr = predicted_ctr[i_sorted[i]]
        no_click += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        no_click_sum += num_impressions[i_sorted[i]] - num_clicks[i_sorted[i]]
        click_sum += num_clicks[i_sorted[i]]
    auc_temp += (click_sum+old_click_sum) * no_click / 2.0
    auc = auc_temp / (click_sum * no_click_sum)
    return auc

def getTure(infile):
    infile = open(infile,'r')
    clicks = []
    impress = []
    for line in infile:
        lines = line.split(' ')
        clicks.append(float(lines[0]))
        impress.append(float(lines[1]))
    return clicks,impress

def getPred(infile):
    infile = open(infile,'r')
    preds = []
    for line in infile:
        #print line
        preds.append(float(line[0:-1]))
    return preds

if len(sys.argv)<3:
    print 'Use: pred_file test_file result_file'
    exit(-1)

#print sys.argv
pred_file = sys.argv[1]
test_file = sys.argv[2]
result_file = sys.argv[3]

print 'read test..'
clicks,impress = getTure(test_file)
print 'merge instance: '+str(len(impress))
#print 'all instance: '+str(sum(impress))
print 'read pred...'
#print clicks,impress
preds = getPred(pred_file)
print 'pred len:'+str(len(preds))
print 'get auc...'
auc = scoreClickAUC(clicks,impress,preds)
print 'AUC:'+str(auc)
print 'write result...'
outfile = open(result_file,'w')
outfile.write(str(auc)+'\n')
