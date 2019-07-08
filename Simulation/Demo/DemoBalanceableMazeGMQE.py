from ast import literal_eval


def main():
    q_history = []
    with open('balanceable_maze_mp_gm_q_history.txt', 'r') as f:  # 'balanceable_maze_q_history.txt'
        for line in f:
            q_history = literal_eval(line.replace('][', ', '))
    print(len(q_history))
    q_sum = []
    for p in range(3240):
        _slice = q_history[p*100:p*100+100]
        q_sum += [sum(_slice)/len(_slice)]

    len(q_sum)
    lst = [(i*100, p) for i, p in enumerate(q_sum)]
    for e in lst:
        print(str(e) + ' ', end='')


if __name__ == "__main__":
    main()