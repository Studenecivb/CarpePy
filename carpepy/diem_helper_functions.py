##############################
#### From DIEMPy
### Author: Stuart J.E. Baird
###############################

import numpy as np
from collections import Counter

from .mathematica2python import *


StringReplace20_dict = str.maketrans('02', '20')

def StringReplace20(text):
    """will _!simultaneously!_ replace 2->0 and 0->2"""
    return text.translate(StringReplace20_dict)


def sStateCount(s):
    counts = Map(Second, Tally(Join(["0", "1", "2"], Characters(s))))
    nU = Total(
        Drop(counts, 3)
    )  # only the three 'call' chars above are not U encodings!
    counts = list(np.array(Take(counts, 3)) - 1)
    return Join([nU], counts)


def csStateCount(cs):
    """
    This function counts diem chars; input cs is char list or string; output is an np.array
    """
    ans = Counter("_012")
    ans.update(cs)
    return np.array(list(ans.values())) - 1


def pHetErrOnString(s):
    sCount = sStateCount(s)
    callTotal = Total(Drop(sCount, 1))
    if callTotal > 0:
        ans = (
            Total(np.array(sCount) * [0, 0, 1, 2]) / (2 * callTotal),
            sCount[2] / callTotal,
            sCount[0] / Total(sCount),
        )
    else:  # no calls... are there any Us?
        if sCount[0] > 0:
            pErr = 1
        else:
            pErr = "NA"
        ans = ("NA", "NA", pErr)
    return ans


#  pHetErrOnStateCount([1 2 3 4]) == array([0.61111111, 0.33333333, 0.1]) == [11/18,3/9,1/10]
def pHetErrOnStateCount(csCount):
    """
    This function counts diem state pState pHet pErr ; csCount is np.array return of csStateCount
    """
    sumcsCount = sum(csCount)
    if sumcsCount > 0:
        err = csCount[0] / sumcsCount
    else:
        err = np.nan
    CallTotal = sumcsCount - csCount[0]
    if CallTotal > 0:
        ans = np.array(
            [
                sum(csCount * np.array([0, 0, 1, 2])) / (2 * CallTotal),
                csCount[2] / CallTotal,
                err,
            ]
        )
    else:
        ans = np.array([np.nan, np.nan, err])
    return ans


def pHetErrOnStringChars(cs):
    return pHetErrOnStateCount(csStateCount(cs))
