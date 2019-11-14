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
    return list(chosen_set),unchosen_list


# print(np.zeros((3,3,3)))
# array2 = []
# for i in range(103):
#     array1 = np.zeros((100,100))
#     array2.append(array1)
# print(array2)
# print(np.array(array2))