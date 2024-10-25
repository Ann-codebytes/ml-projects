Apache server setup:

https://mid.as/kb/00143/install-configure-apache-on-windows#download-apache

Download and edit the following files:


A. httpd.conf in the Apache/conf path --> change as listed below
-------------------------------------
1. Update the path to point to Apache root directory
===================================================
Define SRVROOT "D:/Apache/Apache24"

2. Add port as
===============
Listen 80 on the file

3. Add these or uncomment:
=======================
LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so
LoadModule headers_module modules/mod_headers.so

4. Add server name:
=====================
ServerName localhost:80



5. Add the Directory folder:
============================
<Directory "D://ann_app_dev_ogph//chatbot_app//client">
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, POST, OPTIONS"
    Header set Access-Control-Allow-Headers "Content-Type"
</Directory>

6. Add the following include:
=========================
Include conf/extra/chatbot_proxy.conf



B. chatbot_proxy.conf in the Apache/conf/extra path --> change as follows
---------------------------------------------------------------------- 
1. DocumentRoot and DirectoryPath on this to point to the client folder on the platform




C. Running the Apache Server:
========================

Open power shell as administrator:
Navigate to the bin directory in apache path ------> run the following

 .\httpd.exe -k restart

You can also launch ApacheMonitor. Didn't work for me.

Additional note ------------ To kill any Apache service:
TASKKILL /F /IM ApacheMonitor.exe

D. To start the server:
==========================

Either open the main.py on pycharm and execute run.
OR
Open a command prompt - >  navigate to  D:\ann_app_dev_ogph\chatbot_app\server -> activate the venv
-> then execute the following to turn on the server side

uvicorn main:app --host 127.0.0.1 --port 8000



E. To verify if the server service is working:
=========================================
Navigate to server folder path
Activate the venv
Execute:
curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d "{\"text\": \"Hello\"}"



