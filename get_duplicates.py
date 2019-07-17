batchone=open('cleandata_batch_one.csv','r')
batchtwo=open('cleandata_batch_two.csv','r')
next(batchtwo)
duplicates=open('Duplicates.csv','r')
next(duplicates)

batchone=batchone.readlines()
batchtwo=batchtwo.readlines()

batch_one={}
batch_two={}
for sample in batchone:
    sample=sample.strip("\n").split(",")
    batch_one[sample[0].split("_")[1]]=sample[1:]
for sample in batchtwo:
    sample=sample.strip("\n").split(",")
    #print(sample[0])
    batch_two[sample[0].split("_")[1]]=sample[1:]

batchone_duplicates_id = []
batchtwo_duplicates_id = []


for id in duplicates:
    id = id.strip("\n").split(",")

    batchone_duplicates_id.append(id[0])
    batchtwo_duplicates_id.append(id[1])

#for i,j in zip(batchone_duplicates_id,batchtwo_duplicates_id):
#    print(i,j)

batchone_duplicates={}
for k in batchone_duplicates_id:
    for i,j in batch_one.items():
        if k==i:
            val = [m for m in j[3:]]
            batchone_duplicates[k]=val
batchtwo_duplicates={}
for k in batchtwo_duplicates_id:
    for i,j in batch_two.items():
        if k==i:
            val = [m for m in j[3:]]
            batchtwo_duplicates[k]=val
cleanbatchduplone=open('clean_batchone_dupl.csv','w+')
for i,j in batchone_duplicates.items():
    cleanbatchduplone.write("{},{}\n".format(i,', '.join(j)))
cleanbatchduplotwo=open('clean_batchtwo_dupl.csv','w+')
for i,j in batchtwo_duplicates.items():
    cleanbatchduplotwo.write("{},{}\n".format(i,', '.join(j)))
