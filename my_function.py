import random
import numpy as np


def random_choose(all, choose_count):
    chosen_set = set()
    unchosen_list = []
    count = 0
    while count != choose_count:
        chosen_number = random.randint(0, all - 1)
        if chosen_number in chosen_set:
            continue
        chosen_set.add(chosen_number)
        count += 1
    else:
        for i in range(all):
            if i not in chosen_set:
                unchosen_list.append(i)
    return list(chosen_set), unchosen_list


# 在rects数组中找到最大的一块脸部区域，并作为数组返回
def primary_facial_area(rects):
    biggest_area_value = 0
    for x, y, w, h in rects:
        area_now = w * h
        if area_now > biggest_area_value:
            biggest_area_value = area_now
            biggest_area = [x, y, w, h]
    return biggest_area


# 输入一个opencv识别出来的区域的x,y,w,h，返回相比原来正方形各边变长，中心不变的更大的正方形的x,y,w,h
def bigger_area(x, y, w, h, longer):
    center = (x + w / 2, y + h / 2)
    side = max(w, h) + longer
    x = center[0] - side / 2
    y = center[1] - side / 2
    return x, y, side
