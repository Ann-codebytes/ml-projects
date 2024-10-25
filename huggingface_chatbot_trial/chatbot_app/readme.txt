Apache server setup:

https://mid.as/kb/00143/install-configure-apache-on-windows#download-apache

Download and edit the following files:

httpd.conf in the Apache/conf path --> change as listed below
-------------------------------------
Update the path to point to Apache root directory
====
Define SRVROOT "D:/Apache/Apache24"

Add port as
====
Listen 80 on the file

Add these or uncomment:
=====
LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so
LoadModule headers_module modules/mod_headers.so

Add server name:
=========
ServerName localhost:80



Add the Directory folder:
===================
<Directory "D://ann_app_dev_ogph//chatbot_app//client">
    Header set Access-Control-Allow-Origin "*"
    Header set Access-Control-Allow-Methods "GET, POST, OPTIONS"
    Header set Access-Control-Allow-Headers "Content-Type"
</Directory>

Add the following include:
=========================
Include conf/extra/chatbot_proxy.conf



chatbot_proxy.conf in the Apache/conf/extra path --> change as follows
---------------------------------------------------------------------- 
DocumentRoot and DirectoryPath on this to point to the client folder on the platform



