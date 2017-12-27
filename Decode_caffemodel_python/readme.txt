使用说明：在提取caffemodel的文件夹下需要 创建一个weight文件夹，用来存放解析好的数据。

cmd命令栏下，命令格式如下
python Decode_caffemodel.py your.prototxt your.caffemodel output.txt
pause
其中output.txt，为直观显示解析出来的数据，供参考。

示例
python Decode_caffemodel.py Gender_age_forDecode.prototxt Gender_age.caffemodel test.txt
pause