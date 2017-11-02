import re
import json
from pprint import pprint

with open('/home/alex/fmow/manifest.json') as data_file:    
    data = json.load(data_file)

train =0
train_jpg = 0 
train_json = 0
tr_classes=[]
tr_classes_keys=[]
val=0
val_classes=[]
val_classes_keys=[]
test=0
test_classes= []
test_classes_keys =[]
#print(data[1:10])

for i in range(len(data)):
      if 'train' in data[i]['key'][0:5]:
          train= train+1
          pattern = re.compile(r"train/(?P<path>[a-zA-Z0-9_-]+)/.*\.(?P<ending>[a-z]+)")
          m = pattern.search(data[i]['key'])
          if m.group('ending') == 'json':
             train_json = train_json +1
          if m.group('ending') == 'jpg':
             train_jpg = train_jpg +1
          if m.group('path') not in tr_classes:
             tr_classes.append(m.group('path'))
             n=0
             for x in range(10):
                try:
                 while( m.group('ending') != 'jpg'):
                    n=n+1
                    m = pattern.search(data[i+n]['key'])
                except:
                  print(data[i+n]['key'],  m.group('ending')) 
                tr_classes_keys.append(data[i+n]['key'])
                n=n+1
                m = pattern.search(data[i+n]['key'])
      if 'test' in data[i]['key'][0:4]:
          test= test+1
          pattern = re.compile(r"test/(?P<path>[a-zA-Z0-9]+)*")
          m = pattern.search(data[i]['key'])
          if m!=None: 
            if m.group('path') not in test_classes:
             test_classes.append(m.group('path'))
             test_classes_keys.append(data[i]['key'])
      if 'val' in data[i]['key'][0:3]:
          val= val+1
            
          pattern = re.compile(r"val/(?P<path>[a-zA-Z0-9-_]+)/.*\.(?P<ending>[a-z]+)")
          m = pattern.search(data[i]['key'])
          if m != None:
           keyssss =m.group('path')
           if m.group('path') not in val_classes:
             val_classes.append(m.group('path'))
             n=0
             for x in range(10):
                found = False
                try:
                   while( m.group('ending') != 'jpg' and i+n<len(data) ):
                       n=n+1
                       m = pattern.search(data[i+n]['key'])
                       if (m.group('ending')!='jpg' and i+n<len(data) and m.group('path')==keyssss and 'val' in data[i+n]['key'][0:3]):
                           found = True
                   if  found == True:
                      val_classes_keys.append(data[i+n]['key'])
                      #n=n+1
                      found = True
                      m = pattern.search(data[i+n]['key'])
                except :
                 if i+n<len(data) and found == True:
                    print (data[i]['key'][0:6])
                    print ("INDEX ERROR:",data[i+n]['key'],i+n,'val' in data[i]['key'][0:3], i,data[i]['key'])
                    print ("INDEX ERROR+++:",data[i+n-1]['key'],i+n-1,'val' in data[i]['key'][0:3], i,data[i]['key'])

print("DATA",len(data))
print("TR",train,len(tr_classes))
print("TR_json_jpg",train_json,train_jpg)
print("Te",test,len(test_classes))
print("Val",val,len(val_classes))

tr_classes_keys = sorted(tr_classes_keys)
val_classes_keys = sorted(val_classes_keys)

tr_classes = sorted(tr_classes)
val_classes = sorted(val_classes)


print("Train keys",len(tr_classes_keys))
print("Val jeys",len(val_classes_keys))

print("Train classes =",(tr_classes))
print("Val classes =",(val_classes))
exit()
input()
import boto3

s3client = boto3.client('s3')



#for i in tr_classes_keys:
#          pattern = re.compile(r".*/(?P<name>[a-zA-Z0-9_]+).*$")
#          pattern2 = re.compile(r"(?P<name>[a-zA-Z0-9_/]+).*$")
#          m = pattern.search(i)
#          m2 = pattern2.search(i)
#          fname = m.group('name')
#          pall = m2.group('name')
#          print(pall, fname)
#          try:
#            s3client.download_file('fmow-full', pall+'.json' , 'train_fl/'+fname+'.json',{'RequestPayer':'requester'})
#            s3client.download_file('fmow-full', pall+'.tif'  , 'train_fl/'+fname+'.tif',{'RequestPayer':'requester'})
#          except:
#            print("not:",pall)

