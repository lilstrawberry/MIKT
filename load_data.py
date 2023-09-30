import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        ans_list = []
        split_char = ','

        read = open(self.path, 'r')
        for index, line in enumerate(read):
            if index % 3 == 0:
                pass

            elif index % 3 == 1:
                problems = line.strip().split(split_char)
                # 由于列表problems每个元素都是char 需要变为int
                problems = list(map(int, problems))
                problem_list.append(problems)

            elif index % 3 == 2:
                ans = line.strip().split(split_char)
                # 由于列表ans每个元素都是char 需要变为int
                ans = list(map(float, ans))
                ans = [int(x) for x in ans]
                ans_list.append(ans)

        read.close()
        return problem_list, ans_list

class KT_Dataset(data.Dataset):

    def __init__(self, problem_max, problem_list, ans_list, skill_list, min_problem_num, max_problem_num):
        self.problem_max = problem_max
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num
        self.problem_list, self.ans_list, self.skill_list = [], [], []
        # 个人定义，少于 min_problem_num 丢弃
        # 根据论文 多于 max_problem_num  的分成多个 max_problem_num
        for (problem, ans, skill) in zip(problem_list, ans_list, skill_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]

                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])
                    self.skill_list.append(skill[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill = now_skill[i * max_problem_num:(i + 1) * max_problem_num]

                    self.problem_list.append(item_problem)
                    self.ans_list.append(item_ans)
                    self.skill_list.append(item_skill)
            else:
                item_problem = problem
                item_ans = ans
                item_skill = skill
                self.problem_list.append(item_problem)
                self.ans_list.append(item_ans)
                self.skill_list.append(item_skill)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        now_problem = self.problem_list[index]
        now_problem = np.array(now_problem)

        now_ans = self.ans_list[index]

        # 由于需要统一格式
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[-num:] = now_problem
        use_ans[-num:] = now_ans

        next_ans = use_ans[1:]
        next_problem = use_problem[1:]

        last_ans = use_ans[:-1]
        last_problem = use_problem[:-1]

        use_mask[-num:] = 1
        next_mask = use_mask[1:]

        last_problem = torch.from_numpy(last_problem).to(device).long()

        next_problem = torch.from_numpy(next_problem).to(device).long()
        last_ans = torch.from_numpy(last_ans).to(device).long()
        next_ans = torch.from_numpy(next_ans).to(device).float()

        res_mask = torch.from_numpy(next_mask != 0).to(device)

        return last_problem, last_ans, next_problem, next_ans, res_mask

def getLoader(problem_max, skill_path, path, batch_size, is_train, min_problem_num, max_problem_num):
    read_data = getReader(path)
    problem_list, ans_list = read_data.readData()

    skill_read = getReader(skill_path)
    skill_list, ans_list = skill_read.readData()

    dataset = KT_Dataset(problem_max, problem_list, ans_list, skill_list, min_problem_num, max_problem_num)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader