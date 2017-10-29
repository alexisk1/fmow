import boto3
import pickle


s3client = boto3.client('s3')

s3client.download_file('fmow-rgb', 'test/0000000/0000000_0_msrgb.jpg', './0_msrgb.jpg',{'RequestPayer':'requester'})
s3client.download_file('fmow-rgb', 'test/0000000/0000000_0_msrgb.json', './0_msrgb.json',{'RequestPayer':'requester'})


