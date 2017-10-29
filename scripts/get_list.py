import boto
from boto.s3.key import Key
import boto.s3.connection

Bucketname = 'fmow-rgb'       


conn = boto.connect_s3()


bucket = conn.get_bucket(Bucketname, headers={'x-amz-request-payer': 'requester'})


for key in bucket.list(headers={'x-amz-request-payer': 'requester'}):
   print( key.name,
          key.size,
key.last_modified)
