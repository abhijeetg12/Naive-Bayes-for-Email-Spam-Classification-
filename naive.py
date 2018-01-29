import numpy as np
import os
from scipy.stats import beta

legittxt=[]
legitd=[]
spamtxt=[]
spamd=[]
legitc=0
spamc=0
cwd = os.getcwd()
path1 = os.path.join(cwd, "../Dataset/2_NaiveBayes/")
Yval=[]
listy=os.listdir(path1)
#print listy
for i in listy:
    add1=os.listdir(path1+i)
    #print add1
    for k in add1:
        if 'legit' in k:

            legitc=legitc+1
            read1=open(path1+i+'/'+k,'r')
            cont=''.join(read1)
            contspl=cont.split() #split the content into individual words now
            #print contspl
            contspl=contspl[1:] #we need to remove the Subject: from the data, and it is the first element of the contspl
            for w in contspl:# checking every indidvidual word in constspl
                if w not in legittxt:
                    legittxt.append(w)
        if 'spm' in k:
            spamc=spamc+1
            read1=open(path1+i+'/'+k,'r')
            cont=''.join(read1)
            contspl=cont.split() #split the content into individual words now
            #print contspl
            contspl=contspl[1:] #we need to remove the Subject: from the data, and it is the first element of the contspl
            for w in contspl:# checking every indidvidual word in constspl
                if w not in spamtxt:
                    spamtxt.append(w)

print len(legittxt)
print len(spamtxt)

total=[]
total.extend(spamtxt)
for i in legittxt:
    if i not in total:
        total.append(i)
print len(total)
Matr=np.empty(len(total))
Emp=np.zeros(len(total))
print Matr

#now we need to keep the track for the frequncy of individual word in each of the datasets

for i in listy:
    add1=os.listdir(path1+i)
    #print add1
    for k in add1:
        if 'legit' in k:
            Yval.append(0)
            read1=open(path1+i+'/'+k,'r')
            cont=''.join(read1)
            contspl=cont.split() #split the content into individual words now
            #print contspl
            contspl=contspl[1:] #we need to remove the Subject: from the data, and it is the first element of the contspl
            Emp1=np.zeros(len(total))
            for w in contspl:# checking every indidvidual word in constspl
                alp=total.index(w)
                #print alp
                Emp1[alp]=Emp1[alp]+1
            '''    if w not in legittxt:
                    legittxt.append(w)
            '''
        if 'spm' in k:
            Yval.append(1)
            read1=open(path1+i+'/'+k,'r')
            cont=''.join(read1)
            contspl=cont.split() #split the content into individual words now
            #print contspl
            contspl=contspl[1:] #we need to remove the Subject: from the data, and it is the first element of the contspl
            Emp1=np.zeros(len(total))
            for w in contspl:# checking every indidvidual word in constspl
                alp=total.index(w)

                Emp1[alp]=Emp1[alp]+1
                '''if w not in spamtxt:
                    spamtxt.append(w)
                '''

        #print Emp1
        #Matr.append(Emp1)

        Matr= np.vstack([Matr,Emp1])
#print (Matr)
#Matr=Matr+1
print Matr.shape
Matr=Matr[1:]
print len(Yval)

#   y=beta.pdf(x, a, b)
from sklearn.metrics import precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.cross_validation import train_test_split
a_train, a_test, b_train, b_test = train_test_split(Matr, Yval, test_size=0.2, random_state=42)
clf = MultinomialNB()
clf2= BernoulliNB()
clf.fit(a_train, b_train)
clf2.fit(a_train, b_train)
Ax=clf.predict(a_test)
Bx=clf2.predict(a_test)
from sklearn.metrics import f1_score

print f1_score(b_test, Ax, average='macro')
print f1_score(b_test, Bx, average='macro')

import matplotlib.pyplot as plt

precision, recall, _ = precision_recall_curve(b_test, Bx)

plt.step(recall, precision, color='b', alpha=0.2,where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
#plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()
