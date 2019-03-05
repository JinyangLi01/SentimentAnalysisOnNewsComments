基于词典的方法
程序：run.ipynb
其中all表示对前多少条评论进行打分，可以自己设置
ratio1, ratio2, ratio3, ratio4分别表示以正负1，正负2，正负3
和正负4作为分界，对评论打出-1,0,1三个分数，
四种情况的运行结果分别在res_ratio1, res_ratio2, res_ratio3, res_ratio4文件中。
comments.xls是从网上爬取的评论，分数是手动标注。
运行方法：用Jupyter Notebook打开run.ipynb，直接运行，
正确率体现在程序控制台，运行结束会打印finish。
结果写入result文件。

