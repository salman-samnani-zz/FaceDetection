# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:50:10 2020

@author: UKEPG-1
"""

import socket                   # Import socket module
import os
import time

format=".jpg"
myDir_1="D:\\daynial\\david\\mtcnn-master\\images"
myDir_2="D:\\daynial\\david\\mtcnn-master\\New" 
myDir="images"


fileList=[]

def fetch_files():
    fileList = []
    for root, dirs, files in os.walk(myDir, topdown=False):
            for name in files:
                if name.endswith(format):
                    fullName = os.path.join(root, name)
                    fileList.append(fullName)
                    
    #print(len(fileList))
    if len(fileList) == 0:
        pass
    else:
        print("sending files")
        for f in range(len(fileList)):
            print(fileList[f])
            
            send_img(fileList[f])
            
    
def send_img(img):
    
    s = socket.socket()             # Create a socket object
    host = '10.194.2.45' #'192.168.0.118'     # Get local machine name
    port = 60000                    # Reserve a port for your service.
    s.connect((host, port)) # connect to server
    time.sleep(0.2)
    filename = img
    # open file
    with open(filename, 'rb') as f:
        while True:
            dataToSend = f.read(1024)  # read 1024 bytes
            if not dataToSend: # if no data break
                break
            s.send(dataToSend) # if there is data send data
            
            
            #print('Sent ', repr(dataToSend),'size=',len(dataToSend)) #prin data and its size
    os.remove(img)
    s.close() # close connection
    print('the is send file Successfully') # end of transmition
    
if __name__ == '__main__':  
    while True:
        try:
            fetch_files()
            time.sleep(1)
            
        except:
            print("In Except")
            