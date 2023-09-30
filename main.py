import pandas as pd
import torch
import torch.nn as nn

import glo
from model import MIKT
from run import run_epoch

if __name__ == '__main__':

    mp2path = {
        'assist09': {
            'ques_skill_path': 'data/assist09/assistment2009_ques_skill.csv',
            'train_path': 'data/assist09/assistment2009_train_question.txt',
            'test_path': 'data/assist09/assistment2009_test_question.txt',
            'train_skill_path': 'data/assist09/assistment2009_train_skill.txt',
            'test_skill_path': 'data/assist09/assistment2009_test_skill.txt',
            'skill_max': 167
        }
    }

    glo._init()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = 'assist09'
    d = 64
    state_d = 64
    p = 0.4
    learning_rate = 0.002
    epochs = 200
    batch_size = 80
    min_seq = 3
    max_seq = 200
    grad_clip = 15.0
    patience = 30

    ques_skill_path = mp2path[dataset]['ques_skill_path']
    train_path = mp2path[dataset]['train_path']
    test_path = mp2path[dataset]['test_path']
    train_skill_path = mp2path[dataset]['train_skill_path']
    test_skill_path = mp2path[dataset]['test_skill_path']
    skill_max = mp2path[dataset]['skill_max']

    pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
    skill_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 1])

    pro2skill = torch.zeros((pro_max, skill_max)).to(device)

    for (x, y) in zip(pd.read_csv(ques_skill_path).values[:, 0], pd.read_csv(ques_skill_path).values[:, 1]):
        pro2skill[x][y] = 1

    glo.set_value('state_d', state_d)
    glo.set_value('max_seq', max_seq)
    glo.set_value('pro2skill', pro2skill)

    ############################ model training ##################################3

    avg_auc = 0
    avg_acc = 0

    for now_step in range(5):

        best_acc = 0
        best_auc = 0
        state = {'auc': 0, 'acc': 0, 'loss': 0}

        criterion = nn.BCELoss()
        classify = nn.CrossEntropyLoss()

        model = MIKT(skill_max, pro_max, d, p).to(device)

        ccc = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)

        one_p = 0

        for epoch in range(epochs):

            one_p += 1

            train_loss, train_acc, train_auc = run_epoch(classify, train_skill_path, model, optimizer,
                                                         pro_max, train_path, batch_size,
                                                         True, min_seq, max_seq, criterion, device,
                                                         grad_clip)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

            test_loss, test_acc, test_auc = run_epoch(classify, test_skill_path, model, optimizer, pro_max,
                                                      test_path, batch_size, False,
                                                      min_seq, max_seq, criterion, device, grad_clip)

            print(
                f'epoch: {epoch}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')

            if test_auc >= best_auc:
                one_p = 0
                best_auc = test_auc
                best_acc = test_acc
                torch.save(model.state_dict(), f"./MIKT_{dataset}_{now_step}_model.pkl")
                state['auc'] = test_auc
                state['acc'] = test_acc
                state['loss'] = test_loss
                torch.save(state, f'./MIKT_{dataset}_{now_step}_state.ckpt')

        print(f'*******************************************************************************')
        print(f'best_acc: {best_acc:.4f}, best_auc: {best_auc:.4f}')
        print(f'*******************************************************************************')

        avg_auc += best_auc
        avg_acc += best_acc

    avg_auc = avg_auc / 5
    avg_acc = avg_acc / 5
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
