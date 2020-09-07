import socket                   # Import socket module
import os
import time
import time
#s = socket.socket()             # Create a socket object
host = '10.194.2.38' #'192.168.0.118'     # Get local machine name
port = 60000                    # Reserve a port for your service.
#s.connect((host, port)) # connect to server
filename = 'logo_1.png'
format = ".jpg"
myDir = 'images'
fileList = []
for root,dirs ,files in os.walk(myDir, topdown=False):
    for name in files:
        if name.endswith(format):
            fullName = os.path.join(root, name)
            fileList.append(fullName)

for f in range(len(fileList)):
    # open file
    print("------------------------------------------------------------------------------")
    print(f)
    print(fileList[f])
    print(time.asctime(time.localtime(time.time())))
    s = socket.socket()
    s.connect((host, port))
    time.sleep(1)

    print("------------------------------------------------------------------------------")
    with open(fileList[f], 'rb') as y:
        while True:
            dataToSend = y.read(1024)  # read 1024 bytes
            if not dataToSend: # if no data break

                break
            s.send(dataToSend) # if there is data send data
            print('Sent ', repr(dataToSend),'size=',len(dataToSend)) #prin data and its size
    os.remove(fileList[f])
    print("File Removed!")
    s.close() # close connection
    print('Successfully get the file') # end of transmition
