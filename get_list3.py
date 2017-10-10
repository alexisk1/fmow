import boto3

s3 = boto3.resource('s3')

for bucket in s3.buckets.all():
    print(bucket.name)

client = boto3.client('s3')

sum_ftr=0
result= client.list_objects_v2(Bucket='fmow-rgb',RequestPayer='requester',Prefix="train/")
#print(result['NextContinuationToken'])
for o in result['Contents']:
    sum_ftr=sum_ftr+o['Size']
    #print (o['Key'])

prev ='' 
while ( result['NextContinuationToken']!=prev):
   prev=result['NextContinuationToken']
   result= client.list_objects_v2(Bucket='fmow-rgb',RequestPayer='requester',ContinuationToken=result['NextContinuationToken'],Prefix="train/")
#   print(result['NextContinuationToken'])
   print('train=',sum_ftr)
   for o in result['Contents']:
      sum_ftr=sum_ftr+o['Size']
      #print(o['Key'])
