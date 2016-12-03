"Function to diff example output and allow small numerical errors"
import sys


def diff(output1, output2, tol=1e-3):
    "check that output1 and output2 are same up to small errors in numbers"
    strs1 = output1.split(" ")
    strs2 = output2.split(" ")
    if len(strs1) != len(strs2):
        return False
    for s1, s2 in zip(strs1, strs2):
        try:
            v1 = float(s1)
            v2 = float(s2)
            if abs((v1 - v2) / (v1 + v2)) >= tol:
                return False
        except ValueError:
            if s1 != s2:
                return False
    return True


if __name__ == "__main__":
    assert len(sys.argv) == 3
    if not diff(sys.argv[1], sys.argv[2]):
        raise RuntimeWarning("Arguments were not the same")
