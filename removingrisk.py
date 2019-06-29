"""risk = open('risk.csv','r')
risk = risk.readlines()
risk = risk[0]
risk = risk.strip('\ufeff').strip('\n').split(",")
output=open('headers.csv','w+')
for i in risk:
    i=i.split(".")[0]
    output.write("{},".format(i))
output.write("\n")
"""
master=open('Master1-4.csv','r')
master= master.readlines()
newfile=open('sort_jun_17.csv','r')
newfile=newfile.readlines()
new_ids={}
ids={}
for i in master:
    i=i.strip("\n").split(",")
    ids[i[0]]=i[-3:]
for i in newfile:
    i=i.strip("\n").split(",")
    new_ids[i[0]]=i[1:]
count = 0
new_data={}
for i,m in ids.items():
    for j,n in new_ids.items():
        if i == j:
            new_data[j]=m+n
            count+=1
cleandata = 'cleandata.csv'
output=open(cleandata,'w+')
for i,j in new_data.items():
    output.write("{},{}\n".format(i,j))
