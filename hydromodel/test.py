def d2b(_d, _acc=23):
    if isinstance(_d, int):
        return int(bin(_d)[2:])
    else:
        _float = _d - int(_d)
        _int = int(_d)
        _bin = ''
        while _acc:
            _float *= 2
            _bin += '1' if _float > 1. else '0'
            _float -= int(_float)
            _acc -= 1
        return bin(_int)[2:]+'.'+_bin


def b2d(_b):
    if isinstance(_b, int):
        return int(str(_b), 2)
    else:
        _int, _float = str(_b).split('.')
        _b = str(_b).replace('.', '')
        i = len(_int)-1
        _d = 0
        for j in _b:
            _d += int(j)*2**i
            i -= 1
        return _d


if __name__ == '__main__':
    print(d2b(122.45678))
    print(b2d(d2b(122.45678)))

