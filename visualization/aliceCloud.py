# coding:utf8
import os
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


# 更新默认路径为当前目录
print("当前路径为:", os.getcwd())
os.chdir("D:/GitHub-project/Machine-Learning-Algorithm/visualization")
print("路径更新为:", os.getcwd())

file = open('爱丽丝梦游仙境_英文版.txt', encoding="utf-8").read()
print(type(file))

# 基本的词云
# wordcloud = WordCloud(background_color="white", width=1000, height=860, margin=2).generate(file)
#
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()
# wordcloud.to_file("test1.png")

alice_mask = np.array(Image.open("mask.jpg"))# 基于掩码图片的词云
#
# wordcloud = WordCloud(background_color="white", max_words=1000, mask=alice_mask,
#                       max_font_size=40, random_state=42)
# wordcloud.generate(file)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()
# wordcloud.to_file("test2.png")

# 从掩码图片获取颜色策略
alice_mask = np.array(Image.open("mask.jpg"))
image_colors = ImageColorGenerator(alice_mask)
wordcloud = WordCloud(background_color="white", max_words=1000, mask=alice_mask,
                      max_font_size=50, random_state=42)
wordcloud.generate(file)
wordcloud.recolor(color_func=image_colors)
wordcloud.to_file("test3.png")
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.imshow(alice_mask, cmap=plt.cm.gray, interpolation="bilinear")
plt.axis("off")
plt.show()