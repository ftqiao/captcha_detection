# import matplotlib.pyplot as plt
#
#
# plt.figure(figsize=(20, 10), dpi=1000)
#
# game = ['300', '500', '1000', '1500', '2000']
# weighted_average = [0.295, 0.369, 0.452, 0.037, 0.063]
# simple_average = [0.307, 0.363, 0.44, 0.033, 0.067]
# # assists = [16, 7, 8, 10, 10, 7, 9, 5, 9, 7, 12, 4, 11, 8, 10, 9, 9, 8, 8, 7, 10]
# plt.plot(game, weighted_average, c='red', label="precision1")
# plt.plot(game, simple_average, c='green', linestyle='--', label="precision2")
# # plt.plot(game, assists, c='blue', linestyle='-.', label="助攻")
# plt.scatter(game, weighted_average, c='red')
# plt.scatter(game, simple_average, c='green')
# # plt.scatter(game, assists, c='blue')
# plt.legend(loc='best')
# plt.yticks(range(0, 50, 1))
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xlabel("赛程", fontdict={'size': 16})
# plt.ylabel("数据", fontdict={'size': 16})
# plt.title("average precision", fontdict={'size': 20})
# plt.show()
# encoding=utf-8
from pylab import *  # 支持中文

mpl.rcParams['font.sans-serif'] = ['SimHei']
names = ['500(997)', '1000(1998)', '1500(2998)', '2000(3998)']
x = range(len(names))
weighted_average = [3.86, 45.26, 3.18, 3.7]
simple_average = [4.15, 43.96, 3.11, 3.67]

# plt.plot(x, y, 'ro-')
# plt.plot(x, y1, 'bo-')
# pl.xlim(-1, 11) # 限定横轴的范围
# pl.ylim(-1, 110) # 限定纵轴的范围
plt.plot(x, weighted_average, marker='o', mec='r', mfc='w', label=u'加权平均')
plt.plot(x, simple_average, marker='^', ms=6, label=u'简单平均')
plt.legend()  # 让图例生效
plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"数据集大小(张)")  # X轴标签
plt.ylabel("识别准确率(%)")  # Y轴标签
plt.title("平均识别准确率")  # 标题

plt.show()
