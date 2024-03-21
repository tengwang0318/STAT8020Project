import zipfile

zip_file = zipfile.ZipFile("feedback-prize-2021.zip")
zip_list = zip_file.namelist() # 得到压缩包里所有文件

for f in zip_list:
    zip_file.extract(f, "feedback-prize-2021") # 循环解压文件到指定目录
 
zip_file.close() # 关闭文件，必须有，释放内存