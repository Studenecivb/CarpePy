##############################
#### Mathematica2Python
### Author: Stuart J.E. Baird
###############################

import itertools
import multiprocessing as mp
import numpy as np


def Split(lst, test):  # code credit: BPL @ StackOverflow
    res = []
    sublst = []

    for x, y in zip(lst, lst[1:]):
        sublst.append(x)
        if not test(x, y):
            res.append(sublst)
            sublst = []

    if len(lst) > 1:
        sublst.append(lst[-1])
        if not test(lst[-2], lst[-1]):
            res.append(sublst)
            sublst = []

    if sublst:
        res.append(sublst)

    return res


def SplitBy(lst, test):  # code credit: BPL @ StackOverflow
    return Split(lst, lambda x, y: test(x) == test(y))


# code credit SJEB (my tools ported from Mathematica)
def Characters(s): return [*s]


def Position(lst, elem):
    i = -1
    pos = []
    for l in lst:
        i += 1
        if l == elem:
            pos.append(i)
    return pos


def FirstPosition(lst, elem):
    i = -1
    pos = []
    for l in lst:
        i += 1
        if l == elem:
            pos.append(i)
            break
    return pos


def Tally(lst):  # single pass so in principle fast ( O(n) ) but answers unsorted
    states = []
    tally = []
    for x in lst:
        p = FirstPosition(states, x)
        if p == []:
            states.append(x)
            tally.append([x, 1])
        else:
            tally[p[0]][1] += 1
    return tally


def Commonest(lst):
    t = Tally(lst)  # single pass so in principle fast ( O(n) ) but answers unsorted
    maxn = 0
    maxlist = []
    for prs in t:
        if prs[1] > maxn:
            maxn = prs[1]
            maxlist = [prs[0]]
        elif prs[1] == maxn:
            maxlist.append(prs[0])
    return maxlist


def Select(lst, test):
    passlist = []
    for i in lst:
        if test(i):
            passlist.append(i)
    return passlist


def RLE(lst):
    slst = Split(lst, lambda x, y: y == x)
    states = []
    lengths = []
    for i in slst:
        states.append(i[0])
        lengths.append(len(i))
    return [states, lengths]


def RichRLE(lst):
    slst = Split(lst, lambda x, y: y == x)
    cumstart = 0
    states = []
    lengths = []
    starts = []
    ends = []
    for i in slst:
        leni = len(i)
        states.append(i[0])
        lengths.append(leni)
        starts.append(cumstart)
        cumstart += leni
        ends.append(cumstart - 1)
    return [states, lengths, starts, ends]


def BackgroundedRLE(lst):
    rRLE = RichRLE(lst)
    background = Commonest(rRLE[0])[0]  # does not matter which of commonest states is backgrounded
    ans = [[background, 0, len(lst) - 1]]
    i = -1
    for state in rRLE[0]:
        i += 1
        if state != background:
            ans.append([state, rRLE[2][i], rRLE[1][i]])  # states, starts, lengths (1 not 3) (suited to matlibplot)
    return ans


def BackgroundedRLEcompression(bRLE):
    l = len(bRLE)
    if l == 0:
        ans = "NA"
    else:
        ans = l / bRLE[0][3]  # RLE items per original list length
    return ans


def Map(f, lst): return list(map(f, lst))


def ParallelMap(f, lst):
    pool = mp.Pool()
    return list(pool.map(f, lst))


def Flatten(lstOlists): return list(itertools.chain.from_iterable(lstOlists))


def StringJoin(slst):
    separator = ''
    return separator.join(slst)


def SJ(lst):
    return StringJoin(Map(str, lst))


def Transpose(mat): return list(np.array(mat).T)  # care here - hidden type casting on heterogeneous 'mat'rices


def StringTranspose(slst): return Map(StringJoin, Transpose(Map(Characters, slst)))


def First(lst):  return lst[0]


def Second(lst): return lst[1]


def Third(lst):  return lst[2]


def TakeCols(lstOlsts, cols):
    ans = []
    for lst in lstOlsts:
        ans.append(list(np.array(lst)[cols]))
    return ans


def Join(lst1, lst2): return lst1 + lst2


def Take(lst, n):
    if n > 0:
        ans = lst[:n]
    elif n == 0:
        ans = lst
    else:
        ans = lst[n:]
    return ans


def Drop(lst, n):
    if n > 0:
        ans = lst[n:]
    elif n == 0:
        ans = lst
    else:
        ans = lst[:n]
    return ans


def Total(lst): return sum(lst)


def Accumulate(lst): return list(itertools.accumulate(lst))


def Length(x): return len(x)