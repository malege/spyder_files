# -*- coding: utf-8 -*-
import math
file = 'e:/wiki_0.txt'
x = 25
with open('e:/wiki_0_new_0.txt', 'w', encoding = 'utf-8') as f:
    with open(file, 'r', encoding = 'utf-8') as f1:
        lines = f1.readlines()
    for s in lines:
        s = s.strip().replace('\0x00', '')
        if len(s) == 0:
            continue
        elif len(s)>x:
            m = math.ceil(len(s)/x)
            n = math.ceil(len(s)/m)
            for i in range(m-1):
                text = s[i*n : (i+1)*n]
                if text[0] in '，。、；：’”！？.':  
                    f.writelines(text[1:] + '\n')  # text = text.replace(text[0], '', 1)
                else:
                    f.writelines(text + '\n')
            text = s[(m-1)*n : ]
            if text[0] in '，。、；：’”！？.':  
                f.writelines(text[1:] + '\n')      # text = text.replace(text[0], '', 1)
            else:
                f.writelines(text + '\n')
        else:
            f.writelines(s + '\n')