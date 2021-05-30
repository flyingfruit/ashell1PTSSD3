#changescore for dcf

import sys
def main():
    with open('scores','r')as f:
        spksE = []
        uttsE = []
        with open('trials', 'w')as f2:
            for line in f.readlines():
                [a, b,_] = line.split()
                spksE.append(b)
                uttsE.append(a)
                if a==b[0:5]:
                    target='target'
                else:
                    target='nontarget'
                f2.write(a + ' ' + b + ' ' + target + '\n')


if __name__ == '__main__':
    main()