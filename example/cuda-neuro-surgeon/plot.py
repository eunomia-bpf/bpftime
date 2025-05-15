import re
import numpy as np
import matplotlib.pyplot as plt

# Full log data provided by the user
log_data = """
2025-05-15 00:11:02,531 INFO Measuring split (1,2)
(1,2) copy=0.001s mid=0.173s fin=0.032s total=0.205s
2025-05-15 00:11:02,748 INFO Measuring split (1,3)
(1,3) copy=0.000s mid=0.002s fin=0.031s total=0.033s
2025-05-15 00:11:02,783 INFO Measuring split (1,4)
(1,4) copy=0.000s mid=0.003s fin=0.030s total=0.034s
2025-05-15 00:11:02,817 INFO Measuring split (1,5)
(1,5) copy=0.000s mid=0.005s fin=0.029s total=0.034s
2025-05-15 00:11:02,852 INFO Measuring split (1,6)
(1,6) copy=0.000s mid=0.006s fin=0.028s total=0.034s
2025-05-15 00:11:02,887 INFO Measuring split (1,7)
(1,7) copy=0.000s mid=0.007s fin=0.027s total=0.034s
2025-05-15 00:11:02,922 INFO Measuring split (1,8)
(1,8) copy=0.000s mid=0.008s fin=0.026s total=0.034s
2025-05-15 00:11:02,957 INFO Measuring split (1,9)
(1,9) copy=0.000s mid=0.009s fin=0.025s total=0.034s
2025-05-15 00:11:02,992 INFO Measuring split (1,10)
(1,10) copy=0.000s mid=0.010s fin=0.024s total=0.034s
2025-05-15 00:11:03,027 INFO Measuring split (1,11)
(1,11) copy=0.000s mid=0.011s fin=0.023s total=0.034s
2025-05-15 00:11:03,062 INFO Measuring split (1,12)
(1,12) copy=0.000s mid=0.012s fin=0.022s total=0.034s
2025-05-15 00:11:03,097 INFO Measuring split (1,13)
(1,13) copy=0.000s mid=0.014s fin=0.021s total=0.035s
2025-05-15 00:11:03,132 INFO Measuring split (1,14)
(1,14) copy=0.000s mid=0.014s fin=0.020s total=0.034s
2025-05-15 00:11:03,167 INFO Measuring split (1,15)
(1,15) copy=0.000s mid=0.015s fin=0.018s total=0.034s
2025-05-15 00:11:03,202 INFO Measuring split (1,16)
(1,16) copy=0.000s mid=0.017s fin=0.017s total=0.034s
2025-05-15 00:11:03,237 INFO Measuring split (1,17)
(1,17) copy=0.000s mid=0.018s fin=0.016s total=0.034s
2025-05-15 00:11:03,272 INFO Measuring split (1,18)
(1,18) copy=0.000s mid=0.019s fin=0.015s total=0.034s
2025-05-15 00:11:03,308 INFO Measuring split (1,19)
(1,19) copy=0.000s mid=0.020s fin=0.014s total=0.034s
2025-05-15 00:11:03,343 INFO Measuring split (1,20)
(1,20) copy=0.000s mid=0.021s fin=0.013s total=0.034s
2025-05-15 00:11:03,378 INFO Measuring split (1,21)
(1,21) copy=0.000s mid=0.022s fin=0.012s total=0.034s
2025-05-15 00:11:03,413 INFO Measuring split (1,22)
(1,22) copy=0.000s mid=0.023s fin=0.011s total=0.034s
2025-05-15 00:11:03,448 INFO Measuring split (1,23)
(1,23) copy=0.000s mid=0.024s fin=0.010s total=0.034s
2025-05-15 00:11:03,483 INFO Measuring split (1,24)
(1,24) copy=0.000s mid=0.025s fin=0.009s total=0.034s
2025-05-15 00:11:03,518 INFO Measuring split (1,25)
(1,25) copy=0.000s mid=0.026s fin=0.008s total=0.034s
2025-05-15 00:11:03,553 INFO Measuring split (1,26)
(1,26) copy=0.000s mid=0.027s fin=0.007s total=0.034s
2025-05-15 00:11:03,588 INFO Measuring split (1,27)
(1,27) copy=0.000s mid=0.028s fin=0.006s total=0.034s
2025-05-15 00:11:03,623 INFO Measuring split (1,28)
(1,28) copy=0.000s mid=0.030s fin=0.004s total=0.034s
2025-05-15 00:11:03,659 INFO Measuring split (1,29)
(1,29) copy=0.000s mid=0.031s fin=0.003s total=0.034s
2025-05-15 00:11:03,694 INFO Measuring split (1,30)
(1,30) copy=0.000s mid=0.032s fin=0.002s total=0.034s
2025-05-15 00:11:03,729 INFO Measuring split (1,31)
(1,31) copy=0.000s mid=0.033s fin=0.001s total=0.034s
2025-05-15 00:11:03,764 INFO Measuring split (1,32)
(1,32) copy=0.000s mid=0.034s fin=0.000s total=0.034s
2025-05-15 00:11:03,799 INFO Measuring split (2,3)
(2,3) copy=0.000s mid=0.001s fin=0.032s total=0.033s
2025-05-15 00:11:03,833 INFO Measuring split (2,4)
(2,4) copy=0.000s mid=0.002s fin=0.030s total=0.033s
2025-05-15 00:11:03,867 INFO Measuring split (2,5)
(2,5) copy=0.000s mid=0.003s fin=0.029s total=0.033s
2025-05-15 00:11:03,901 INFO Measuring split (2,6)
(2,6) copy=0.000s mid=0.005s fin=0.029s total=0.033s
2025-05-15 00:11:03,935 INFO Measuring split (2,7)
(2,7) copy=0.000s mid=0.006s fin=0.027s total=0.033s
2025-05-15 00:11:03,969 INFO Measuring split (2,8)
(2,8) copy=0.000s mid=0.007s fin=0.026s total=0.033s
2025-05-15 00:11:04,003 INFO Measuring split (2,9)
(2,9) copy=0.000s mid=0.008s fin=0.025s total=0.033s
2025-05-15 00:11:04,037 INFO Measuring split (2,10)
(2,10) copy=0.000s mid=0.009s fin=0.024s total=0.032s
2025-05-15 00:11:04,071 INFO Measuring split (2,11)
(2,11) copy=0.000s mid=0.010s fin=0.023s total=0.033s
2025-05-15 00:11:04,107 INFO Measuring split (2,12)
(2,12) copy=0.000s mid=0.011s fin=0.022s total=0.033s
2025-05-15 00:11:04,141 INFO Measuring split (2,13)
(2,13) copy=0.000s mid=0.012s fin=0.021s total=0.033s
2025-05-15 00:11:04,175 INFO Measuring split (2,14)
(2,14) copy=0.000s mid=0.013s fin=0.020s total=0.033s
2025-05-15 00:11:04,209 INFO Measuring split (2,15)
(2,15) copy=0.000s mid=0.014s fin=0.019s total=0.033s
2025-05-15 00:11:04,243 INFO Measuring split (2,16)
(2,16) copy=0.000s mid=0.016s fin=0.018s total=0.033s
2025-05-15 00:11:04,277 INFO Measuring split (2,17)
(2,17) copy=0.000s mid=0.017s fin=0.016s total=0.033s
2025-05-15 00:11:04,312 INFO Measuring split (2,18)
(2,18) copy=0.000s mid=0.018s fin=0.015s total=0.033s
2025-05-15 00:11:04,345 INFO Measuring split (2,19)
(2,19) copy=0.000s mid=0.019s fin=0.014s total=0.033s
2025-05-15 00:11:04,380 INFO Measuring split (2,20)
(2,20) copy=0.000s mid=0.020s fin=0.013s total=0.033s
2025-05-15 00:11:04,414 INFO Measuring split (2,21)
(2,21) copy=0.000s mid=0.021s fin=0.013s total=0.034s
2025-05-15 00:11:04,449 INFO Measuring split (2,22)
(2,22) copy=0.000s mid=0.022s fin=0.011s total=0.032s
2025-05-15 00:11:04,482 INFO Measuring split (2,23)
(2,23) copy=0.000s mid=0.023s fin=0.010s total=0.033s
2025-05-15 00:11:04,516 INFO Measuring split (2,24)
(2,24) copy=0.000s mid=0.024s fin=0.009s total=0.033s
2025-05-15 00:11:04,550 INFO Measuring split (2,25)
(2,25) copy=0.000s mid=0.025s fin=0.008s total=0.033s
2025-05-15 00:11:04,584 INFO Measuring split (2,26)
(2,26) copy=0.000s mid=0.026s fin=0.007s total=0.033s
2025-05-15 00:11:04,618 INFO Measuring split (2,27)
(2,27) copy=0.000s mid=0.027s fin=0.006s total=0.033s
2025-05-15 00:11:04,652 INFO Measuring split (2,28)
(2,28) copy=0.000s mid=0.028s fin=0.004s total=0.033s
2025-05-15 00:11:04,686 INFO Measuring split (2,29)
(2,29) copy=0.000s mid=0.030s fin=0.003s total=0.033s
2025-05-15 00:11:04,720 INFO Measuring split (2,30)
(2,30) copy=0.000s mid=0.031s fin=0.002s total=0.033s
2025-05-15 00:11:04,755 INFO Measuring split (2,31)
(2,31) copy=0.000s mid=0.032s fin=0.001s total=0.033s
2025-05-15 00:11:04,789 INFO Measuring split (2,32)
(2,32) copy=0.000s mid=0.033s fin=0.000s total=0.033s
2025-05-15 00:11:04,823 INFO Measuring split (3,4)
(3,4) copy=0.000s mid=0.001s fin=0.031s total=0.032s
2025-05-15 00:11:04,856 INFO Measuring split (3,5)
(3,5) copy=0.000s mid=0.002s fin=0.029s total=0.032s
2025-05-15 00:11:04,888 INFO Measuring split (3,6)
(3,6) copy=0.000s mid=0.004s fin=0.028s total=0.032s
2025-05-15 00:11:04,921 INFO Measuring split (3,7)
(3,7) copy=0.000s mid=0.005s fin=0.027s total=0.032s
2025-05-15 00:11:04,954 INFO Measuring split (3,8)
(3,8) copy=0.000s mid=0.006s fin=0.026s total=0.032s
2025-05-15 00:11:04,987 INFO Measuring split (3,9)
(3,9) copy=0.000s mid=0.007s fin=0.025s total=0.032s
2025-05-15 00:11:05,020 INFO Measuring split (3,10)
(3,10) copy=0.000s mid=0.008s fin=0.024s total=0.032s
2025-05-15 00:11:05,053 INFO Measuring split (3,11)
(3,11) copy=0.000s mid=0.009s fin=0.023s total=0.032s
2025-05-15 00:11:05,086 INFO Measuring split (3,12)
(3,12) copy=0.000s mid=0.010s fin=0.022s total=0.032s
2025-05-15 00:11:05,119 INFO Measuring split (3,13)
(3,13) copy=0.000s mid=0.011s fin=0.021s total=0.032s
2025-05-15 00:11:05,152 INFO Measuring split (3,14)
(3,14) copy=0.000s mid=0.012s fin=0.020s total=0.032s
2025-05-15 00:11:05,185 INFO Measuring split (3,15)
(3,15) copy=0.000s mid=0.013s fin=0.019s total=0.032s
2025-05-15 00:11:05,218 INFO Measuring split (3,16)
(3,16) copy=0.000s mid=0.014s fin=0.017s total=0.032s
2025-05-15 00:11:05,251 INFO Measuring split (3,17)
(3,17) copy=0.000s mid=0.017s fin=0.016s total=0.034s
2025-05-15 00:11:05,286 INFO Measuring split (3,18)
(3,18) copy=0.000s mid=0.017s fin=0.015s total=0.032s
2025-05-15 00:11:05,319 INFO Measuring split (3,19)
(3,19) copy=0.000s mid=0.018s fin=0.014s total=0.032s
2025-05-15 00:11:05,351 INFO Measuring split (3,20)
(3,20) copy=0.000s mid=0.019s fin=0.013s total=0.032s
2025-05-15 00:11:05,384 INFO Measuring split (3,21)
(3,21) copy=0.000s mid=0.020s fin=0.012s total=0.032s
2025-05-15 00:11:05,417 INFO Measuring split (3,22)
(3,22) copy=0.000s mid=0.021s fin=0.011s total=0.032s
2025-05-15 00:11:05,451 INFO Measuring split (3,23)
(3,23) copy=0.000s mid=0.022s fin=0.010s total=0.032s
2025-05-15 00:11:05,484 INFO Measuring split (3,24)
(3,24) copy=0.000s mid=0.023s fin=0.009s total=0.032s
2025-05-15 00:11:05,517 INFO Measuring split (3,25)
(3,25) copy=0.000s mid=0.024s fin=0.008s total=0.032s
2025-05-15 00:11:05,550 INFO Measuring split (3,26)
(3,26) copy=0.000s mid=0.025s fin=0.007s total=0.032s
2025-05-15 00:11:05,583 INFO Measuring split (3,27)
(3,27) copy=0.000s mid=0.027s fin=0.006s total=0.032s
2025-05-15 00:11:05,616 INFO Measuring split (3,28)
(3,28) copy=0.000s mid=0.027s fin=0.004s total=0.032s
2025-05-15 00:11:05,649 INFO Measuring split (3,29)
(3,29) copy=0.000s mid=0.028s fin=0.003s total=0.032s
2025-05-15 00:11:05,682 INFO Measuring split (3,30)
(3,30) copy=0.000s mid=0.030s fin=0.002s total=0.032s
2025-05-15 00:11:05,715 INFO Measuring split (3,31)
(3,31) copy=0.000s mid=0.030s fin=0.001s total=0.032s
2025-05-15 00:11:05,748 INFO Measuring split (3,32)
(3,32) copy=0.000s mid=0.032s fin=0.000s total=0.032s
2025-05-15 00:11:05,780 INFO Measuring split (4,5)
(4,5) copy=0.000s mid=0.001s fin=0.030s total=0.031s
2025-05-15 00:11:05,813 INFO Measuring split (4,6)
(4,6) copy=0.000s mid=0.002s fin=0.028s total=0.031s
2025-05-15 00:11:05,844 INFO Measuring split (4,7)
(4,7) copy=0.000s mid=0.003s fin=0.028s total=0.031s
2025-05-15 00:11:05,877 INFO Measuring split (4,8)
(4,8) copy=0.000s mid=0.005s fin=0.026s total=0.031s
2025-05-15 00:11:05,908 INFO Measuring split (4,9)
(4,9) copy=0.000s mid=0.006s fin=0.025s total=0.031s
2025-05-15 00:11:05,940 INFO Measuring split (4,10)
(4,10) copy=0.000s mid=0.007s fin=0.024s total=0.031s
2025-05-15 00:11:05,972 INFO Measuring split (4,11)
(4,11) copy=0.000s mid=0.008s fin=0.023s total=0.031s
2025-05-15 00:11:06,004 INFO Measuring split (4,12)
(4,12) copy=0.000s mid=0.009s fin=0.022s total=0.031s
2025-05-15 00:11:06,036 INFO Measuring split (4,13)
(4,13) copy=0.000s mid=0.010s fin=0.021s total=0.031s
2025-05-15 00:11:06,068 INFO Measuring split (4,14)
(4,14) copy=0.000s mid=0.011s fin=0.020s total=0.031s
2025-05-15 00:11:06,100 INFO Measuring split (4,15)
(4,15) copy=0.000s mid=0.012s fin=0.019s total=0.031s
2025-05-15 00:11:06,132 INFO Measuring split (4,16)
(4,16) copy=0.000s mid=0.013s fin=0.017s total=0.031s
2025-05-15 00:11:06,163 INFO Measuring split (4,17)
(4,17) copy=0.000s mid=0.014s fin=0.016s total=0.031s
2025-05-15 00:11:06,195 INFO Measuring split (4,18)
(4,18) copy=0.000s mid=0.015s fin=0.015s total=0.031s
2025-05-15 00:11:06,227 INFO Measuring split (4,19)
(4,19) copy=0.000s mid=0.017s fin=0.014s total=0.031s
2025-05-15 00:11:06,259 INFO Measuring split (4,20)
(4,20) copy=0.000s mid=0.018s fin=0.013s total=0.031s
2025-05-15 00:11:06,291 INFO Measuring split (4,21)
(4,21) copy=0.000s mid=0.019s fin=0.012s total=0.031s
2025-05-15 00:11:06,323 INFO Measuring split (4,22)
(4,22) copy=0.000s mid=0.020s fin=0.011s total=0.031s
2025-05-15 00:11:06,354 INFO Measuring split (4,23)
(4,23) copy=0.000s mid=0.021s fin=0.010s total=0.031s
2025-05-15 00:11:06,386 INFO Measuring split (4,24)
(4,24) copy=0.000s mid=0.022s fin=0.009s total=0.031s
2025-05-15 00:11:06,418 INFO Measuring split (4,25)
(4,25) copy=0.000s mid=0.023s fin=0.008s total=0.031s
2025-05-15 00:11:06,450 INFO Measuring split (4,26)
(4,26) copy=0.000s mid=0.024s fin=0.007s total=0.031s
2025-05-15 00:11:06,482 INFO Measuring split (4,27)
(4,27) copy=0.000s mid=0.025s fin=0.006s total=0.031s
2025-05-15 00:11:06,514 INFO Measuring split (4,28)
(4,28) copy=0.000s mid=0.026s fin=0.004s total=0.031s
2025-05-15 00:11:06,546 INFO Measuring split (4,29)
(4,29) copy=0.000s mid=0.028s fin=0.003s total=0.031s
2025-05-15 00:11:06,578 INFO Measuring split (4,30)
(4,30) copy=0.000s mid=0.028s fin=0.002s total=0.031s
2025-05-15 00:11:06,610 INFO Measuring split (4,31)
(4,31) copy=0.000s mid=0.029s fin=0.001s total=0.031s
2025-05-15 00:11:06,642 INFO Measuring split (4,32)
(4,32) copy=0.000s mid=0.031s fin=0.000s total=0.031s
2025-05-15 00:11:06,674 INFO Measuring split (5,6)
(5,6) copy=0.000s mid=0.001s fin=0.028s total=0.030s
2025-05-15 00:11:06,705 INFO Measuring split (5,7)
(5,7) copy=0.000s mid=0.002s fin=0.027s total=0.030s
2025-05-15 00:11:06,736 INFO Measuring split (5,8)
(5,8) copy=0.000s mid=0.003s fin=0.026s total=0.030s
2025-05-15 00:11:06,766 INFO Measuring split (5,9)
(5,9) copy=0.000s mid=0.005s fin=0.025s total=0.030s
2025-05-15 00:11:06,797 INFO Measuring split (5,10)
(5,10) copy=0.000s mid=0.006s fin=0.024s total=0.030s
2025-05-15 00:11:06,828 INFO Measuring split (5,11)
(5,11) copy=0.000s mid=0.007s fin=0.023s total=0.030s
2025-05-15 00:11:06,859 INFO Measuring split (5,12)
(5,12) copy=0.000s mid=0.008s fin=0.022s total=0.030s
2025-05-15 00:11:06,889 INFO Measuring split (5,13)
(5,13) copy=0.000s mid=0.009s fin=0.021s total=0.030s
2025-05-15 00:11:06,920 INFO Measuring split (5,14)
(5,14) copy=0.000s mid=0.010s fin=0.020s total=0.030s
2025-05-15 00:11:06,951 INFO Measuring split (5,15)
(5,15) copy=0.000s mid=0.011s fin=0.019s total=0.030s
2025-05-15 00:11:06,981 INFO Measuring split (5,16)
(5,16) copy=0.000s mid=0.012s fin=0.018s total=0.030s
2025-05-15 00:11:07,013 INFO Measuring split (5,17)
(5,17) copy=0.000s mid=0.013s fin=0.016s total=0.030s
2025-05-15 00:11:07,043 INFO Measuring split (5,18)
(5,18) copy=0.000s mid=0.015s fin=0.016s total=0.030s
2025-05-15 00:11:07,074 INFO Measuring split (5,19)
(5,19) copy=0.000s mid=0.016s fin=0.014s total=0.030s
2025-05-15 00:11:07,105 INFO Measuring split (5,20)
(5,20) copy=0.000s mid=0.017s fin=0.013s total=0.030s
2025-05-15 00:11:07,136 INFO Measuring split (5,21)
(5,21) copy=0.000s mid=0.018s fin=0.012s total=0.030s
2025-05-15 00:11:07,166 INFO Measuring split (5,22)
(5,22) copy=0.000s mid=0.019s fin=0.011s total=0.030s
2025-05-15 00:11:07,197 INFO Measuring split (5,23)
(5,23) copy=0.000s mid=0.020s fin=0.010s total=0.030s
2025-05-15 00:11:07,228 INFO Measuring split (5,24)
(5,24) copy=0.000s mid=0.021s fin=0.009s total=0.030s
2025-05-15 00:11:07,261 INFO Measuring split (5,25)
(5,25) copy=0.000s mid=0.022s fin=0.008s total=0.030s
2025-05-15 00:11:07,292 INFO Measuring split (5,26)
(5,26) copy=0.000s mid=0.023s fin=0.007s total=0.030s
2025-05-15 00:11:07,323 INFO Measuring split (5,27)
(5,27) copy=0.000s mid=0.024s fin=0.006s total=0.030s
2025-05-15 00:11:07,354 INFO Measuring split (5,28)
(5,28) copy=0.000s mid=0.025s fin=0.004s total=0.030s
2025-05-15 00:11:07,384 INFO Measuring split (5,29)
(5,29) copy=0.000s mid=0.026s fin=0.003s total=0.030s
2025-05-15 00:11:07,415 INFO Measuring split (5,30)
(5,30) copy=0.000s mid=0.027s fin=0.002s total=0.030s
2025-05-15 00:11:07,446 INFO Measuring split (5,31)
(5,31) copy=0.000s mid=0.029s fin=0.001s total=0.030s
2025-05-15 00:11:07,477 INFO Measuring split (5,32)
(5,32) copy=0.000s mid=0.030s fin=0.000s total=0.030s
2025-05-15 00:11:07,507 INFO Measuring split (6,7)
(6,7) copy=0.000s mid=0.001s fin=0.027s total=0.029s
2025-05-15 00:11:07,537 INFO Measuring split (6,8)
(6,8) copy=0.000s mid=0.002s fin=0.026s total=0.029s
2025-05-15 00:11:07,567 INFO Measuring split (6,9)
(6,9) copy=0.000s mid=0.004s fin=0.025s total=0.029s
2025-05-15 00:11:07,596 INFO Measuring split (6,10)
(6,10) copy=0.000s mid=0.005s fin=0.024s total=0.029s
2025-05-15 00:11:07,626 INFO Measuring split (6,11)
(6,11) copy=0.000s mid=0.006s fin=0.023s total=0.029s
2025-05-15 00:11:07,656 INFO Measuring split (6,12)
(6,12) copy=0.000s mid=0.007s fin=0.022s total=0.029s
2025-05-15 00:11:07,685 INFO Measuring split (6,13)
(6,13) copy=0.000s mid=0.008s fin=0.021s total=0.029s
2025-05-15 00:11:07,715 INFO Measuring split (6,14)
(6,14) copy=0.000s mid=0.009s fin=0.020s total=0.029s
2025-05-15 00:11:07,745 INFO Measuring split (6,15)
(6,15) copy=0.000s mid=0.010s fin=0.020s total=0.030s
2025-05-15 00:11:07,776 INFO Measuring split (6,16)
(6,16) copy=0.000s mid=0.011s fin=0.018s total=0.029s
2025-05-15 00:11:07,805 INFO Measuring split (6,17)
(6,17) copy=0.000s mid=0.012s fin=0.017s total=0.029s
2025-05-15 00:11:07,835 INFO Measuring split (6,18)
(6,18) copy=0.000s mid=0.013s fin=0.015s total=0.029s
2025-05-15 00:11:07,865 INFO Measuring split (6,19)
(6,19) copy=0.000s mid=0.014s fin=0.014s total=0.029s
2025-05-15 00:11:07,895 INFO Measuring split (6,20)
(6,20) copy=0.000s mid=0.015s fin=0.014s total=0.030s
2025-05-15 00:11:07,925 INFO Measuring split (6,21)
(6,21) copy=0.000s mid=0.017s fin=0.012s total=0.029s
2025-05-15 00:11:07,955 INFO Measuring split (6,22)
(6,22) copy=0.000s mid=0.018s fin=0.011s total=0.029s
2025-05-15 00:11:07,985 INFO Measuring split (6,23)
(6,23) copy=0.000s mid=0.019s fin=0.010s total=0.029s
2025-05-15 00:11:08,014 INFO Measuring split (6,24)
(6,24) copy=0.000s mid=0.020s fin=0.009s total=0.029s
2025-05-15 00:11:08,044 INFO Measuring split (6,25)
(6,25) copy=0.000s mid=0.021s fin=0.008s total=0.029s
2025-05-15 00:11:08,074 INFO Measuring split (6,26)
(6,26) copy=0.000s mid=0.022s fin=0.007s total=0.029s
2025-05-15 00:11:08,104 INFO Measuring split (6,27)
(6,27) copy=0.000s mid=0.023s fin=0.006s total=0.029s
2025-05-15 00:11:08,133 INFO Measuring split (6,28)
(6,28) copy=0.000s mid=0.024s fin=0.004s total=0.029s
2025-05-15 00:11:08,163 INFO Measuring split (6,29)
(6,29) copy=0.000s mid=0.025s fin=0.003s total=0.029s
2025-05-15 00:11:08,193 INFO Measuring split (6,30)
(6,30) copy=0.000s mid=0.026s fin=0.002s total=0.029s
2025-05-15 00:11:08,222 INFO Measuring split (6,31)
(6,31) copy=0.000s mid=0.028s fin=0.001s total=0.029s
2025-05-15 00:11:08,252 INFO Measuring split (6,32)
(6,32) copy=0.000s mid=0.030s fin=0.000s total=0.030s
2025-05-15 00:11:08,283 INFO Measuring split (7,8)
(7,8) copy=0.000s mid=0.001s fin=0.026s total=0.028s
2025-05-15 00:11:08,312 INFO Measuring split (7,9)
(7,9) copy=0.000s mid=0.002s fin=0.025s total=0.027s
2025-05-15 00:11:08,340 INFO Measuring split (7,10)
(7,10) copy=0.000s mid=0.003s fin=0.024s total=0.027s
2025-05-15 00:11:08,369 INFO Measuring split (7,11)
(7,11) copy=0.000s mid=0.005s fin=0.023s total=0.028s
2025-05-15 00:11:08,397 INFO Measuring split (7,12)
(7,12) copy=0.000s mid=0.006s fin=0.022s total=0.028s
2025-05-15 00:11:08,426 INFO Measuring split (7,13)
(7,13) copy=0.000s mid=0.007s fin=0.021s total=0.027s
2025-05-15 00:11:08,454 INFO Measuring split (7,14)
(7,14) copy=0.000s mid=0.008s fin=0.020s total=0.028s
2025-05-15 00:11:08,482 INFO Measuring split (7,15)
(7,15) copy=0.000s mid=0.009s fin=0.019s total=0.028s
2025-05-15 00:11:08,511 INFO Measuring split (7,16)
(7,16) copy=0.000s mid=0.010s fin=0.018s total=0.028s
2025-05-15 00:11:08,539 INFO Measuring split (7,17)
(7,17) copy=0.000s mid=0.011s fin=0.016s total=0.028s
2025-05-15 00:11:08,568 INFO Measuring split (7,18)
(7,18) copy=0.000s mid=0.012s fin=0.015s total=0.028s
2025-05-15 00:11:08,596 INFO Measuring split (7,19)
(7,19) copy=0.000s mid=0.013s fin=0.014s total=0.028s
2025-05-15 00:11:08,625 INFO Measuring split (7,20)
(7,20) copy=0.000s mid=0.014s fin=0.013s total=0.028s
2025-05-15 00:11:08,653 INFO Measuring split (7,21)
(7,21) copy=0.000s mid=0.015s fin=0.012s total=0.028s
2025-05-15 00:11:08,682 INFO Measuring split (7,22)
(7,22) copy=0.000s mid=0.017s fin=0.011s total=0.028s
2025-05-15 00:11:08,710 INFO Measuring split (7,23)
(7,23) copy=0.000s mid=0.018s fin=0.010s total=0.028s
2025-05-15 00:11:08,739 INFO Measuring split (7,24)
(7,24) copy=0.000s mid=0.019s fin=0.009s total=0.028s
2025-05-15 00:11:08,767 INFO Measuring split (7,25)
(7,25) copy=0.000s mid=0.020s fin=0.008s total=0.028s
2025-05-15 00:11:08,796 INFO Measuring split (7,26)
(7,26) copy=0.000s mid=0.021s fin=0.007s total=0.028s
2025-05-15 00:11:08,824 INFO Measuring split (7,27)
(7,27) copy=0.000s mid=0.022s fin=0.006s total=0.028s
2025-05-15 00:11:08,853 INFO Measuring split (7,28)
(7,28) copy=0.000s mid=0.023s fin=0.004s total=0.028s
2025-05-15 00:11:08,882 INFO Measuring split (7,29)
(7,29) copy=0.000s mid=0.024s fin=0.003s total=0.028s
2025-05-15 00:11:08,910 INFO Measuring split (7,30)
(7,30) copy=0.000s mid=0.025s fin=0.002s total=0.028s
2025-05-15 00:11:08,939 INFO Measuring split (7,31)
(7,31) copy=0.000s mid=0.026s fin=0.001s total=0.028s
2025-05-15 00:11:08,967 INFO Measuring split (7,32)
(7,32) copy=0.000s mid=0.027s fin=0.000s total=0.028s
2025-05-15 00:11:08,996 INFO Measuring split (8,9)
(8,9) copy=0.000s mid=0.001s fin=0.025s total=0.026s
2025-05-15 00:11:09,023 INFO Measuring split (8,10)
(8,10) copy=0.000s mid=0.002s fin=0.024s total=0.027s
2025-05-15 00:11:09,051 INFO Measuring split (8,11)
(8,11) copy=0.000s mid=0.003s fin=0.023s total=0.026s
2025-05-15 00:11:09,078 INFO Measuring split (8,12)
(8,12) copy=0.000s mid=0.005s fin=0.023s total=0.028s
2025-05-15 00:11:09,107 INFO Measuring split (8,13)
(8,13) copy=0.000s mid=0.006s fin=0.021s total=0.027s
2025-05-15 00:11:09,134 INFO Measuring split (8,14)
(8,14) copy=0.000s mid=0.007s fin=0.020s total=0.027s
2025-05-15 00:11:09,161 INFO Measuring split (8,15)
(8,15) copy=0.000s mid=0.008s fin=0.019s total=0.027s
2025-05-15 00:11:09,189 INFO Measuring split (8,16)
(8,16) copy=0.000s mid=0.009s fin=0.018s total=0.027s
2025-05-15 00:11:09,216 INFO Measuring split (8,17)
(8,17) copy=0.000s mid=0.010s fin=0.016s total=0.027s
2025-05-15 00:11:09,244 INFO Measuring split (8,18)
(8,18) copy=0.000s mid=0.011s fin=0.016s total=0.027s
2025-05-15 00:11:09,272 INFO Measuring split (8,19)
(8,19) copy=0.000s mid=0.012s fin=0.014s total=0.027s
2025-05-15 00:11:09,299 INFO Measuring split (8,20)
(8,20) copy=0.000s mid=0.013s fin=0.013s total=0.027s
2025-05-15 00:11:09,326 INFO Measuring split (8,21)
(8,21) copy=0.000s mid=0.014s fin=0.012s total=0.026s
2025-05-15 00:11:09,354 INFO Measuring split (8,22)
(8,22) copy=0.000s mid=0.015s fin=0.011s total=0.026s
2025-05-15 00:11:09,381 INFO Measuring split (8,23)
(8,23) copy=0.000s mid=0.017s fin=0.010s total=0.027s
2025-05-15 00:11:09,409 INFO Measuring split (8,24)
(8,24) copy=0.000s mid=0.018s fin=0.009s total=0.027s
2025-05-15 00:11:09,437 INFO Measuring split (8,25)
(8,25) copy=0.000s mid=0.019s fin=0.008s total=0.027s
2025-05-15 00:11:09,464 INFO Measuring split (8,26)
(8,26) copy=0.000s mid=0.020s fin=0.007s total=0.027s
2025-05-15 00:11:09,491 INFO Measuring split (8,27)
(8,27) copy=0.000s mid=0.021s fin=0.006s total=0.027s
2025-05-15 00:11:09,519 INFO Measuring split (8,28)
(8,28) copy=0.000s mid=0.022s fin=0.004s total=0.027s
2025-05-15 00:11:09,546 INFO Measuring split (8,29)
(8,29) copy=0.000s mid=0.023s fin=0.003s total=0.026s
2025-05-15 00:11:09,574 INFO Measuring split (8,30)
(8,30) copy=0.000s mid=0.024s fin=0.002s total=0.026s
2025-05-15 00:11:09,601 INFO Measuring split (8,31)
(8,31) copy=0.000s mid=0.025s fin=0.001s total=0.027s
2025-05-15 00:11:09,629 INFO Measuring split (8,32)
(8,32) copy=0.000s mid=0.026s fin=0.000s total=0.026s
2025-05-15 00:11:09,656 INFO Measuring split (9,10)
(9,10) copy=0.000s mid=0.001s fin=0.024s total=0.025s
2025-05-15 00:11:09,682 INFO Measuring split (9,11)
(9,11) copy=0.000s mid=0.002s fin=0.023s total=0.026s
2025-05-15 00:11:09,709 INFO Measuring split (9,12)
(9,12) copy=0.000s mid=0.004s fin=0.022s total=0.025s
2025-05-15 00:11:09,735 INFO Measuring split (9,13)
(9,13) copy=0.000s mid=0.005s fin=0.021s total=0.025s
2025-05-15 00:11:09,761 INFO Measuring split (9,14)
(9,14) copy=0.000s mid=0.006s fin=0.020s total=0.026s
2025-05-15 00:11:09,788 INFO Measuring split (9,15)
(9,15) copy=0.000s mid=0.007s fin=0.019s total=0.026s
2025-05-15 00:11:09,814 INFO Measuring split (9,16)
(9,16) copy=0.000s mid=0.008s fin=0.018s total=0.026s
2025-05-15 00:11:09,840 INFO Measuring split (9,17)
(9,17) copy=0.000s mid=0.009s fin=0.017s total=0.026s
2025-05-15 00:11:09,867 INFO Measuring split (9,18)
(9,18) copy=0.000s mid=0.010s fin=0.015s total=0.026s
2025-05-15 00:11:09,893 INFO Measuring split (9,19)
(9,19) copy=0.000s mid=0.011s fin=0.014s total=0.025s
2025-05-15 00:11:09,920 INFO Measuring split (9,20)
(9,20) copy=0.000s mid=0.012s fin=0.013s total=0.026s
2025-05-15 00:11:09,946 INFO Measuring split (9,21)
(9,21) copy=0.000s mid=0.013s fin=0.012s total=0.026s
2025-05-15 00:11:09,973 INFO Measuring split (9,22)
(9,22) copy=0.000s mid=0.014s fin=0.011s total=0.025s
2025-05-15 00:11:09,999 INFO Measuring split (9,23)
(9,23) copy=0.000s mid=0.015s fin=0.010s total=0.026s
2025-05-15 00:11:10,025 INFO Measuring split (9,24)
(9,24) copy=0.000s mid=0.017s fin=0.009s total=0.026s
2025-05-15 00:11:10,052 INFO Measuring split (9,25)
(9,25) copy=0.000s mid=0.018s fin=0.008s total=0.026s
2025-05-15 00:11:10,078 INFO Measuring split (9,26)
(9,26) copy=0.000s mid=0.019s fin=0.007s total=0.026s
2025-05-15 00:11:10,105 INFO Measuring split (9,27)
(9,27) copy=0.000s mid=0.020s fin=0.006s total=0.025s
2025-05-15 00:11:10,131 INFO Measuring split (9,28)
(9,28) copy=0.000s mid=0.021s fin=0.004s total=0.025s
2025-05-15 00:11:10,158 INFO Measuring split (9,29)
(9,29) copy=0.000s mid=0.022s fin=0.003s total=0.026s
2025-05-15 00:11:10,184 INFO Measuring split (9,30)
(9,30) copy=0.000s mid=0.023s fin=0.002s total=0.026s
2025-05-15 00:11:10,211 INFO Measuring split (9,31)
(9,31) copy=0.000s mid=0.024s fin=0.001s total=0.026s
2025-05-15 00:11:10,237 INFO Measuring split (9,32)
(9,32) copy=0.000s mid=0.025s fin=0.000s total=0.025s
2025-05-15 00:11:10,263 INFO Measuring split (10,11)
(10,11) copy=0.000s mid=0.001s fin=0.023s total=0.024s
2025-05-15 00:11:10,289 INFO Measuring split (10,12)
(10,12) copy=0.000s mid=0.002s fin=0.022s total=0.025s
2025-05-15 00:11:10,314 INFO Measuring split (10,13)
(10,13) copy=0.000s mid=0.003s fin=0.021s total=0.024s
2025-05-15 00:11:10,339 INFO Measuring split (10,14)
(10,14) copy=0.000s mid=0.005s fin=0.020s total=0.024s
2025-05-15 00:11:10,364 INFO Measuring split (10,15)
(10,15) copy=0.000s mid=0.006s fin=0.019s total=0.024s
2025-05-15 00:11:10,389 INFO Measuring split (10,16)
(10,16) copy=0.000s mid=0.007s fin=0.018s total=0.025s
2025-05-15 00:11:10,415 INFO Measuring split (10,17)
(10,17) copy=0.000s mid=0.008s fin=0.016s total=0.024s
2025-05-15 00:11:10,440 INFO Measuring split (10,18)
(10,18) copy=0.000s mid=0.009s fin=0.015s total=0.025s
2025-05-15 00:11:10,466 INFO Measuring split (10,19)
(10,19) copy=0.000s mid=0.010s fin=0.014s total=0.024s
2025-05-15 00:11:10,491 INFO Measuring split (10,20)
(10,20) copy=0.000s mid=0.011s fin=0.013s total=0.024s
2025-05-15 00:11:10,516 INFO Measuring split (10,21)
(10,21) copy=0.000s mid=0.012s fin=0.012s total=0.024s
2025-05-15 00:11:10,541 INFO Measuring split (10,22)
(10,22) copy=0.000s mid=0.013s fin=0.011s total=0.024s
2025-05-15 00:11:10,567 INFO Measuring split (10,23)
(10,23) copy=0.000s mid=0.014s fin=0.010s total=0.025s
2025-05-15 00:11:10,592 INFO Measuring split (10,24)
(10,24) copy=0.000s mid=0.016s fin=0.009s total=0.024s
2025-05-15 00:11:10,617 INFO Measuring split (10,25)
(10,25) copy=0.000s mid=0.017s fin=0.008s total=0.024s
2025-05-15 00:11:10,643 INFO Measuring split (10,26)
(10,26) copy=0.000s mid=0.018s fin=0.007s total=0.025s
2025-05-15 00:11:10,668 INFO Measuring split (10,27)
(10,27) copy=0.000s mid=0.019s fin=0.006s total=0.025s
2025-05-15 00:11:10,693 INFO Measuring split (10,28)
(10,28) copy=0.000s mid=0.020s fin=0.004s total=0.024s
2025-05-15 00:11:10,719 INFO Measuring split (10,29)
(10,29) copy=0.000s mid=0.021s fin=0.003s total=0.024s
2025-05-15 00:11:10,744 INFO Measuring split (10,30)
(10,30) copy=0.000s mid=0.025s fin=0.002s total=0.027s
2025-05-15 00:11:10,772 INFO Measuring split (10,31)
(10,31) copy=0.000s mid=0.023s fin=0.001s total=0.024s
2025-05-15 00:11:10,797 INFO Measuring split (10,32)
(10,32) copy=0.000s mid=0.024s fin=0.000s total=0.024s
2025-05-15 00:11:10,822 INFO Measuring split (11,12)
(11,12) copy=0.000s mid=0.001s fin=0.022s total=0.024s
2025-05-15 00:11:10,846 INFO Measuring split (11,13)
(11,13) copy=0.000s mid=0.002s fin=0.021s total=0.023s
2025-05-15 00:11:10,870 INFO Measuring split (11,14)
(11,14) copy=0.000s mid=0.004s fin=0.020s total=0.023s
2025-05-15 00:11:10,895 INFO Measuring split (11,15)
(11,15) copy=0.000s mid=0.005s fin=0.019s total=0.023s
2025-05-15 00:11:10,919 INFO Measuring split (11,16)
(11,16) copy=0.000s mid=0.006s fin=0.017s total=0.023s
2025-05-15 00:11:10,943 INFO Measuring split (11,17)
(11,17) copy=0.000s mid=0.007s fin=0.017s total=0.023s
2025-05-15 00:11:10,967 INFO Measuring split (11,18)
(11,18) copy=0.000s mid=0.008s fin=0.015s total=0.023s
2025-05-15 00:11:10,991 INFO Measuring split (11,19)
(11,19) copy=0.000s mid=0.009s fin=0.014s total=0.023s
2025-05-15 00:11:11,015 INFO Measuring split (11,20)
(11,20) copy=0.000s mid=0.010s fin=0.013s total=0.023s
2025-05-15 00:11:11,039 INFO Measuring split (11,21)
(11,21) copy=0.000s mid=0.011s fin=0.012s total=0.023s
2025-05-15 00:11:11,063 INFO Measuring split (11,22)
(11,22) copy=0.000s mid=0.012s fin=0.013s total=0.025s
2025-05-15 00:11:11,089 INFO Measuring split (11,23)
(11,23) copy=0.000s mid=0.013s fin=0.010s total=0.023s
2025-05-15 00:11:11,113 INFO Measuring split (11,24)
(11,24) copy=0.000s mid=0.014s fin=0.009s total=0.023s
2025-05-15 00:11:11,137 INFO Measuring split (11,25)
(11,25) copy=0.000s mid=0.016s fin=0.008s total=0.023s
2025-05-15 00:11:11,161 INFO Measuring split (11,26)
(11,26) copy=0.000s mid=0.017s fin=0.007s total=0.024s
2025-05-15 00:11:11,185 INFO Measuring split (11,27)
(11,27) copy=0.000s mid=0.018s fin=0.006s total=0.023s
2025-05-15 00:11:11,209 INFO Measuring split (11,28)
(11,28) copy=0.000s mid=0.019s fin=0.004s total=0.023s
2025-05-15 00:11:11,233 INFO Measuring split (11,29)
(11,29) copy=0.000s mid=0.020s fin=0.003s total=0.023s
2025-05-15 00:11:11,257 INFO Measuring split (11,30)
(11,30) copy=0.000s mid=0.021s fin=0.002s total=0.023s
2025-05-15 00:11:11,281 INFO Measuring split (11,31)
(11,31) copy=0.000s mid=0.022s fin=0.001s total=0.024s
2025-05-15 00:11:11,306 INFO Measuring split (11,32)
(11,32) copy=0.000s mid=0.023s fin=0.000s total=0.023s
2025-05-15 00:11:11,330 INFO Measuring split (12,13)
(12,13) copy=0.000s mid=0.001s fin=0.021s total=0.022s
2025-05-15 00:11:11,353 INFO Measuring split (12,14)
(12,14) copy=0.000s mid=0.002s fin=0.020s total=0.022s
2025-05-15 00:11:11,376 INFO Measuring split (12,15)
(12,15) copy=0.000s mid=0.003s fin=0.019s total=0.022s
2025-05-15 00:11:11,399 INFO Measuring split (12,16)
(12,16) copy=0.000s mid=0.005s fin=0.018s total=0.022s
2025-05-15 00:11:11,422 INFO Measuring split (12,17)
(12,17) copy=0.000s mid=0.006s fin=0.016s total=0.022s
2025-05-15 00:11:11,445 INFO Measuring split (12,18)
(12,18) copy=0.000s mid=0.007s fin=0.015s total=0.022s
2025-05-15 00:11:11,468 INFO Measuring split (12,19)
(12,19) copy=0.000s mid=0.008s fin=0.014s total=0.022s
2025-05-15 00:11:11,491 INFO Measuring split (12,20)
(12,20) copy=0.000s mid=0.009s fin=0.013s total=0.022s
2025-05-15 00:11:11,514 INFO Measuring split (12,21)
(12,21) copy=0.000s mid=0.010s fin=0.012s total=0.022s
2025-05-15 00:11:11,537 INFO Measuring split (12,22)
(12,22) copy=0.000s mid=0.011s fin=0.011s total=0.022s
2025-05-15 00:11:11,560 INFO Measuring split (12,23)
(12,23) copy=0.000s mid=0.012s fin=0.010s total=0.022s
2025-05-15 00:11:11,583 INFO Measuring split (12,24)
(12,24) copy=0.000s mid=0.013s fin=0.009s total=0.022s
2025-05-15 00:11:11,606 INFO Measuring split (12,25)
(12,25) copy=0.000s mid=0.015s fin=0.008s total=0.022s
2025-05-15 00:11:11,629 INFO Measuring split (12,26)
(12,26) copy=0.000s mid=0.015s fin=0.007s total=0.022s
2025-05-15 00:11:11,652 INFO Measuring split (12,27)
(12,27) copy=0.000s mid=0.016s fin=0.006s total=0.022s
2025-05-15 00:11:11,675 INFO Measuring split (12,28)
(12,28) copy=0.000s mid=0.018s fin=0.004s total=0.023s
2025-05-15 00:11:11,698 INFO Measuring split (12,29)
(12,29) copy=0.000s mid=0.019s fin=0.003s total=0.022s
2025-05-15 00:11:11,721 INFO Measuring split (12,30)
(12,30) copy=0.000s mid=0.020s fin=0.002s total=0.022s
2025-05-15 00:11:11,744 INFO Measuring split (12,31)
(12,31) copy=0.000s mid=0.021s fin=0.001s total=0.022s
2025-05-15 00:11:11,768 INFO Measuring split (12,32)
(12,32) copy=0.000s mid=0.022s fin=0.000s total=0.022s
2025-05-15 00:11:11,790 INFO Measuring split (13,14)
(13,14) copy=0.000s mid=0.001s fin=0.020s total=0.022s
2025-05-15 00:11:11,813 INFO Measuring split (13,15)
(13,15) copy=0.000s mid=0.002s fin=0.019s total=0.021s
2025-05-15 00:11:11,835 INFO Measuring split (13,16)
(13,16) copy=0.000s mid=0.004s fin=0.018s total=0.021s
2025-05-15 00:11:11,857 INFO Measuring split (13,17)
(13,17) copy=0.000s mid=0.005s fin=0.017s total=0.021s
2025-05-15 00:11:11,879 INFO Measuring split (13,18)
(13,18) copy=0.000s mid=0.006s fin=0.015s total=0.021s
2025-05-15 00:11:11,901 INFO Measuring split (13,19)
(13,19) copy=0.000s mid=0.007s fin=0.014s total=0.021s
2025-05-15 00:11:11,923 INFO Measuring split (13,20)
(13,20) copy=0.000s mid=0.008s fin=0.015s total=0.024s
2025-05-15 00:11:11,947 INFO Measuring split (13,21)
(13,21) copy=0.000s mid=0.009s fin=0.012s total=0.022s
2025-05-15 00:11:11,969 INFO Measuring split (13,22)
(13,22) copy=0.000s mid=0.010s fin=0.011s total=0.021s
2025-05-15 00:11:11,991 INFO Measuring split (13,23)
(13,23) copy=0.000s mid=0.011s fin=0.010s total=0.021s
2025-05-15 00:11:12,013 INFO Measuring split (13,24)
(13,24) copy=0.000s mid=0.012s fin=0.009s total=0.021s
2025-05-15 00:11:12,035 INFO Measuring split (13,25)
(13,25) copy=0.000s mid=0.013s fin=0.008s total=0.021s
2025-05-15 00:11:12,057 INFO Measuring split (13,26)
(13,26) copy=0.000s mid=0.014s fin=0.007s total=0.021s
2025-05-15 00:11:12,079 INFO Measuring split (13,27)
(13,27) copy=0.000s mid=0.015s fin=0.006s total=0.021s
2025-05-15 00:11:12,101 INFO Measuring split (13,28)
(13,28) copy=0.000s mid=0.017s fin=0.005s total=0.021s
2025-05-15 00:11:12,123 INFO Measuring split (13,29)
(13,29) copy=0.000s mid=0.018s fin=0.003s total=0.021s
2025-05-15 00:11:12,145 INFO Measuring split (13,30)
(13,30) copy=0.000s mid=0.019s fin=0.002s total=0.021s
2025-05-15 00:11:12,167 INFO Measuring split (13,31)
(13,31) copy=0.000s mid=0.020s fin=0.001s total=0.021s
2025-05-15 00:11:12,189 INFO Measuring split (13,32)
(13,32) copy=0.000s mid=0.021s fin=0.000s total=0.021s
2025-05-15 00:11:12,211 INFO Measuring split (14,15)
(14,15) copy=0.000s mid=0.001s fin=0.019s total=0.020s
2025-05-15 00:11:12,231 INFO Measuring split (14,16)
(14,16) copy=0.000s mid=0.002s fin=0.017s total=0.020s
2025-05-15 00:11:12,252 INFO Measuring split (14,17)
(14,17) copy=0.000s mid=0.003s fin=0.016s total=0.020s
2025-05-15 00:11:12,273 INFO Measuring split (14,18)
(14,18) copy=0.000s mid=0.006s fin=0.015s total=0.021s
2025-05-15 00:11:12,294 INFO Measuring split (14,19)
(14,19) copy=0.000s mid=0.006s fin=0.014s total=0.020s
2025-05-15 00:11:12,315 INFO Measuring split (14,20)
(14,20) copy=0.000s mid=0.007s fin=0.013s total=0.020s
2025-05-15 00:11:12,336 INFO Measuring split (14,21)
(14,21) copy=0.000s mid=0.008s fin=0.012s total=0.020s
2025-05-15 00:11:12,357 INFO Measuring split (14,22)
(14,22) copy=0.000s mid=0.009s fin=0.011s total=0.020s
2025-05-15 00:11:12,377 INFO Measuring split (14,23)
(14,23) copy=0.000s mid=0.010s fin=0.010s total=0.020s
2025-05-15 00:11:12,398 INFO Measuring split (14,24)
(14,24) copy=0.000s mid=0.011s fin=0.009s total=0.020s
2025-05-15 00:11:12,419 INFO Measuring split (14,25)
(14,25) copy=0.000s mid=0.012s fin=0.008s total=0.020s
2025-05-15 00:11:12,440 INFO Measuring split (14,26)
(14,26) copy=0.000s mid=0.014s fin=0.007s total=0.020s
2025-05-15 00:11:12,461 INFO Measuring split (14,27)
(14,27) copy=0.000s mid=0.015s fin=0.006s total=0.020s
2025-05-15 00:11:12,482 INFO Measuring split (14,28)
(14,28) copy=0.000s mid=0.016s fin=0.005s total=0.020s
2025-05-15 00:11:12,503 INFO Measuring split (14,29)
(14,29) copy=0.000s mid=0.017s fin=0.003s total=0.020s
2025-05-15 00:11:12,523 INFO Measuring split (14,30)
(14,30) copy=0.000s mid=0.018s fin=0.002s total=0.020s
2025-05-15 00:11:12,545 INFO Measuring split (14,31)
(14,31) copy=0.000s mid=0.019s fin=0.001s total=0.020s
2025-05-15 00:11:12,565 INFO Measuring split (14,32)
(14,32) copy=0.000s mid=0.020s fin=0.000s total=0.020s
2025-05-15 00:11:12,586 INFO Measuring split (15,16)
(15,16) copy=0.000s mid=0.001s fin=0.018s total=0.019s
2025-05-15 00:11:12,606 INFO Measuring split (15,17)
(15,17) copy=0.000s mid=0.002s fin=0.019s total=0.021s
2025-05-15 00:11:12,628 INFO Measuring split (15,18)
(15,18) copy=0.000s mid=0.003s fin=0.015s total=0.019s
2025-05-15 00:11:12,647 INFO Measuring split (15,19)
(15,19) copy=0.000s mid=0.005s fin=0.014s total=0.019s
2025-05-15 00:11:12,667 INFO Measuring split (15,20)
(15,20) copy=0.000s mid=0.006s fin=0.013s total=0.019s
2025-05-15 00:11:12,686 INFO Measuring split (15,21)
(15,21) copy=0.000s mid=0.007s fin=0.012s total=0.019s
2025-05-15 00:11:12,706 INFO Measuring split (15,22)
(15,22) copy=0.000s mid=0.008s fin=0.011s total=0.019s
2025-05-15 00:11:12,725 INFO Measuring split (15,23)
(15,23) copy=0.000s mid=0.009s fin=0.010s total=0.019s
2025-05-15 00:11:12,745 INFO Measuring split (15,24)
(15,24) copy=0.000s mid=0.010s fin=0.009s total=0.019s
2025-05-15 00:11:12,764 INFO Measuring split (15,25)
(15,25) copy=0.000s mid=0.011s fin=0.008s total=0.019s
2025-05-15 00:11:12,784 INFO Measuring split (15,26)
(15,26) copy=0.000s mid=0.012s fin=0.007s total=0.019s
2025-05-15 00:11:12,804 INFO Measuring split (15,27)
(15,27) copy=0.000s mid=0.013s fin=0.006s total=0.019s
2025-05-15 00:11:12,824 INFO Measuring split (15,28)
(15,28) copy=0.000s mid=0.015s fin=0.005s total=0.019s
2025-05-15 00:11:12,843 INFO Measuring split (15,29)
(15,29) copy=0.000s mid=0.016s fin=0.003s total=0.019s
2025-05-15 00:11:12,863 INFO Measuring split (15,30)
(15,30) copy=0.000s mid=0.017s fin=0.002s total=0.019s
2025-05-15 00:11:12,883 INFO Measuring split (15,31)
(15,31) copy=0.000s mid=0.018s fin=0.001s total=0.019s
2025-05-15 00:11:12,903 INFO Measuring split (15,32)
(15,32) copy=0.000s mid=0.019s fin=0.000s total=0.019s
2025-05-15 00:11:12,922 INFO Measuring split (16,17)
(16,17) copy=0.000s mid=0.001s fin=0.016s total=0.018s
2025-05-15 00:11:12,941 INFO Measuring split (16,18)
(16,18) copy=0.000s mid=0.002s fin=0.015s total=0.018s
2025-05-15 00:11:12,959 INFO Measuring split (16,19)
(16,19) copy=0.000s mid=0.003s fin=0.014s total=0.018s
2025-05-15 00:11:12,977 INFO Measuring split (16,20)
(16,20) copy=0.000s mid=0.007s fin=0.013s total=0.020s
2025-05-15 00:11:12,998 INFO Measuring split (16,21)
(16,21) copy=0.000s mid=0.006s fin=0.012s total=0.018s
2025-05-15 00:11:13,016 INFO Measuring split (16,22)
(16,22) copy=0.000s mid=0.007s fin=0.011s total=0.018s
2025-05-15 00:11:13,035 INFO Measuring split (16,23)
(16,23) copy=0.000s mid=0.008s fin=0.010s total=0.018s
2025-05-15 00:11:13,053 INFO Measuring split (16,24)
(16,24) copy=0.000s mid=0.009s fin=0.009s total=0.018s
2025-05-15 00:11:13,072 INFO Measuring split (16,25)
(16,25) copy=0.000s mid=0.010s fin=0.008s total=0.018s
2025-05-15 00:11:13,090 INFO Measuring split (16,26)
(16,26) copy=0.000s mid=0.011s fin=0.007s total=0.018s
2025-05-15 00:11:13,109 INFO Measuring split (16,27)
(16,27) copy=0.000s mid=0.012s fin=0.006s total=0.018s
2025-05-15 00:11:13,127 INFO Measuring split (16,28)
(16,28) copy=0.000s mid=0.013s fin=0.005s total=0.018s
2025-05-15 00:11:13,146 INFO Measuring split (16,29)
(16,29) copy=0.000s mid=0.014s fin=0.003s total=0.018s
2025-05-15 00:11:13,164 INFO Measuring split (16,30)
(16,30) copy=0.000s mid=0.015s fin=0.002s total=0.018s
2025-05-15 00:11:13,183 INFO Measuring split (16,31)
(16,31) copy=0.000s mid=0.016s fin=0.001s total=0.018s
2025-05-15 00:11:13,201 INFO Measuring split (16,32)
(16,32) copy=0.000s mid=0.018s fin=0.000s total=0.018s
2025-05-15 00:11:13,220 INFO Measuring split (17,18)
(17,18) copy=0.000s mid=0.001s fin=0.015s total=0.017s
2025-05-15 00:11:13,237 INFO Measuring split (17,19)
(17,19) copy=0.000s mid=0.002s fin=0.014s total=0.017s
2025-05-15 00:11:13,254 INFO Measuring split (17,20)
(17,20) copy=0.000s mid=0.003s fin=0.013s total=0.017s
2025-05-15 00:11:13,271 INFO Measuring split (17,21)
(17,21) copy=0.000s mid=0.005s fin=0.012s total=0.017s
2025-05-15 00:11:13,288 INFO Measuring split (17,22)
(17,22) copy=0.000s mid=0.006s fin=0.011s total=0.017s
2025-05-15 00:11:13,306 INFO Measuring split (17,23)
(17,23) copy=0.000s mid=0.007s fin=0.010s total=0.017s
2025-05-15 00:11:13,323 INFO Measuring split (17,24)
(17,24) copy=0.000s mid=0.008s fin=0.009s total=0.017s
2025-05-15 00:11:13,340 INFO Measuring split (17,25)
(17,25) copy=0.000s mid=0.009s fin=0.008s total=0.017s
2025-05-15 00:11:13,358 INFO Measuring split (17,26)
(17,26) copy=0.000s mid=0.010s fin=0.007s total=0.017s
2025-05-15 00:11:13,375 INFO Measuring split (17,27)
(17,27) copy=0.000s mid=0.011s fin=0.005s total=0.017s
2025-05-15 00:11:13,392 INFO Measuring split (17,28)
(17,28) copy=0.000s mid=0.012s fin=0.004s total=0.017s
2025-05-15 00:11:13,409 INFO Measuring split (17,29)
(17,29) copy=0.000s mid=0.013s fin=0.003s total=0.017s
2025-05-15 00:11:13,426 INFO Measuring split (17,30)
(17,30) copy=0.000s mid=0.014s fin=0.002s total=0.017s
2025-05-15 00:11:13,444 INFO Measuring split (17,31)
(17,31) copy=0.000s mid=0.015s fin=0.001s total=0.017s
2025-05-15 00:11:13,461 INFO Measuring split (17,32)
(17,32) copy=0.000s mid=0.017s fin=0.000s total=0.017s
2025-05-15 00:11:13,478 INFO Measuring split (18,19)
(18,19) copy=0.000s mid=0.001s fin=0.014s total=0.016s
2025-05-15 00:11:13,494 INFO Measuring split (18,20)
(18,20) copy=0.000s mid=0.002s fin=0.013s total=0.016s
2025-05-15 00:11:13,510 INFO Measuring split (18,21)
(18,21) copy=0.000s mid=0.003s fin=0.012s total=0.016s
2025-05-15 00:11:13,527 INFO Measuring split (18,22)
(18,22) copy=0.000s mid=0.005s fin=0.011s total=0.015s
2025-05-15 00:11:13,543 INFO Measuring split (18,23)
(18,23) copy=0.000s mid=0.006s fin=0.010s total=0.016s
2025-05-15 00:11:13,559 INFO Measuring split (18,24)
(18,24) copy=0.000s mid=0.007s fin=0.009s total=0.016s
2025-05-15 00:11:13,575 INFO Measuring split (18,25)
(18,25) copy=0.000s mid=0.008s fin=0.008s total=0.016s
2025-05-15 00:11:13,591 INFO Measuring split (18,26)
(18,26) copy=0.000s mid=0.009s fin=0.007s total=0.016s
2025-05-15 00:11:13,607 INFO Measuring split (18,27)
(18,27) copy=0.000s mid=0.010s fin=0.006s total=0.016s
2025-05-15 00:11:13,623 INFO Measuring split (18,28)
(18,28) copy=0.000s mid=0.011s fin=0.004s total=0.016s
2025-05-15 00:11:13,639 INFO Measuring split (18,29)
(18,29) copy=0.000s mid=0.013s fin=0.003s total=0.016s
2025-05-15 00:11:13,656 INFO Measuring split (18,30)
(18,30) copy=0.000s mid=0.013s fin=0.002s total=0.016s
2025-05-15 00:11:13,673 INFO Measuring split (18,31)
(18,31) copy=0.000s mid=0.014s fin=0.001s total=0.016s
2025-05-15 00:11:13,689 INFO Measuring split (18,32)
(18,32) copy=0.000s mid=0.015s fin=0.000s total=0.015s
2025-05-15 00:11:13,705 INFO Measuring split (19,20)
(19,20) copy=0.000s mid=0.001s fin=0.013s total=0.015s
2025-05-15 00:11:13,720 INFO Measuring split (19,21)
(19,21) copy=0.000s mid=0.002s fin=0.012s total=0.015s
2025-05-15 00:11:13,735 INFO Measuring split (19,22)
(19,22) copy=0.000s mid=0.003s fin=0.011s total=0.014s
2025-05-15 00:11:13,750 INFO Measuring split (19,23)
(19,23) copy=0.000s mid=0.005s fin=0.010s total=0.014s
2025-05-15 00:11:13,765 INFO Measuring split (19,24)
(19,24) copy=0.000s mid=0.006s fin=0.009s total=0.014s
2025-05-15 00:11:13,780 INFO Measuring split (19,25)
(19,25) copy=0.000s mid=0.007s fin=0.008s total=0.014s
2025-05-15 00:11:13,795 INFO Measuring split (19,26)
(19,26) copy=0.000s mid=0.008s fin=0.007s total=0.015s
2025-05-15 00:11:13,810 INFO Measuring split (19,27)
(19,27) copy=0.000s mid=0.009s fin=0.005s total=0.014s
2025-05-15 00:11:13,825 INFO Measuring split (19,28)
(19,28) copy=0.000s mid=0.010s fin=0.004s total=0.014s
2025-05-15 00:11:13,840 INFO Measuring split (19,29)
(19,29) copy=0.000s mid=0.011s fin=0.003s total=0.014s
2025-05-15 00:11:13,855 INFO Measuring split (19,30)
(19,30) copy=0.000s mid=0.012s fin=0.002s total=0.015s
2025-05-15 00:11:13,870 INFO Measuring split (19,31)
(19,31) copy=0.000s mid=0.013s fin=0.001s total=0.014s
2025-05-15 00:11:13,885 INFO Measuring split (19,32)
(19,32) copy=0.000s mid=0.014s fin=0.000s total=0.014s
2025-05-15 00:11:13,900 INFO Measuring split (20,21)
(20,21) copy=0.000s mid=0.001s fin=0.012s total=0.013s
2025-05-15 00:11:13,914 INFO Measuring split (20,22)
(20,22) copy=0.000s mid=0.002s fin=0.011s total=0.013s
2025-05-15 00:11:13,928 INFO Measuring split (20,23)
(20,23) copy=0.000s mid=0.003s fin=0.010s total=0.013s
2025-05-15 00:11:13,942 INFO Measuring split (20,24)
(20,24) copy=0.000s mid=0.005s fin=0.009s total=0.013s
2025-05-15 00:11:13,956 INFO Measuring split (20,25)
(20,25) copy=0.000s mid=0.006s fin=0.008s total=0.013s
2025-05-15 00:11:13,970 INFO Measuring split (20,26)
(20,26) copy=0.000s mid=0.007s fin=0.007s total=0.013s
2025-05-15 00:11:13,983 INFO Measuring split (20,27)
(20,27) copy=0.000s mid=0.008s fin=0.006s total=0.013s
2025-05-15 00:11:13,997 INFO Measuring split (20,28)
(20,28) copy=0.000s mid=0.009s fin=0.004s total=0.013s
2025-05-15 00:11:14,011 INFO Measuring split (20,29)
(20,29) copy=0.000s mid=0.010s fin=0.003s total=0.013s
2025-05-15 00:11:14,025 INFO Measuring split (20,30)
(20,30) copy=0.000s mid=0.011s fin=0.002s total=0.013s
2025-05-15 00:11:14,039 INFO Measuring split (20,31)
(20,31) copy=0.000s mid=0.012s fin=0.001s total=0.013s
2025-05-15 00:11:14,053 INFO Measuring split (20,32)
(20,32) copy=0.000s mid=0.013s fin=0.000s total=0.013s
2025-05-15 00:11:14,067 INFO Measuring split (21,22)
(21,22) copy=0.000s mid=0.001s fin=0.011s total=0.012s
2025-05-15 00:11:14,079 INFO Measuring split (21,23)
(21,23) copy=0.000s mid=0.002s fin=0.010s total=0.012s
2025-05-15 00:11:14,092 INFO Measuring split (21,24)
(21,24) copy=0.000s mid=0.003s fin=0.009s total=0.012s
2025-05-15 00:11:14,105 INFO Measuring split (21,25)
(21,25) copy=0.000s mid=0.005s fin=0.008s total=0.012s
2025-05-15 00:11:14,118 INFO Measuring split (21,26)
(21,26) copy=0.000s mid=0.006s fin=0.007s total=0.012s
2025-05-15 00:11:14,130 INFO Measuring split (21,27)
(21,27) copy=0.000s mid=0.007s fin=0.005s total=0.012s
2025-05-15 00:11:14,143 INFO Measuring split (21,28)
(21,28) copy=0.000s mid=0.008s fin=0.004s total=0.012s
2025-05-15 00:11:14,156 INFO Measuring split (21,29)
(21,29) copy=0.000s mid=0.009s fin=0.003s total=0.012s
2025-05-15 00:11:14,169 INFO Measuring split (21,30)
(21,30) copy=0.000s mid=0.010s fin=0.002s total=0.012s
2025-05-15 00:11:14,181 INFO Measuring split (21,31)
(21,31) copy=0.000s mid=0.011s fin=0.001s total=0.012s
2025-05-15 00:11:14,194 INFO Measuring split (21,32)
(21,32) copy=0.000s mid=0.012s fin=0.000s total=0.012s
2025-05-15 00:11:14,207 INFO Measuring split (22,23)
(22,23) copy=0.000s mid=0.001s fin=0.010s total=0.011s
2025-05-15 00:11:14,219 INFO Measuring split (22,24)
(22,24) copy=0.000s mid=0.002s fin=0.009s total=0.011s
2025-05-15 00:11:14,230 INFO Measuring split (22,25)
(22,25) copy=0.000s mid=0.003s fin=0.008s total=0.011s
2025-05-15 00:11:14,242 INFO Measuring split (22,26)
(22,26) copy=0.000s mid=0.005s fin=0.007s total=0.011s
2025-05-15 00:11:14,254 INFO Measuring split (22,27)
(22,27) copy=0.000s mid=0.006s fin=0.006s total=0.011s
2025-05-15 00:11:14,265 INFO Measuring split (22,28)
(22,28) copy=0.000s mid=0.007s fin=0.004s total=0.011s
2025-05-15 00:11:14,277 INFO Measuring split (22,29)
(22,29) copy=0.000s mid=0.008s fin=0.003s total=0.011s
2025-05-15 00:11:14,289 INFO Measuring split (22,30)
(22,30) copy=0.000s mid=0.009s fin=0.002s total=0.011s
2025-05-15 00:11:14,300 INFO Measuring split (22,31)
(22,31) copy=0.000s mid=0.010s fin=0.001s total=0.011s
2025-05-15 00:11:14,312 INFO Measuring split (22,32)
(22,32) copy=0.000s mid=0.011s fin=0.000s total=0.011s
2025-05-15 00:11:14,324 INFO Measuring split (23,24)
(23,24) copy=0.000s mid=0.001s fin=0.009s total=0.010s
2025-05-15 00:11:14,334 INFO Measuring split (23,25)
(23,25) copy=0.000s mid=0.002s fin=0.008s total=0.010s
2025-05-15 00:11:14,345 INFO Measuring split (23,26)
(23,26) copy=0.000s mid=0.003s fin=0.007s total=0.010s
2025-05-15 00:11:14,355 INFO Measuring split (23,27)
(23,27) copy=0.000s mid=0.005s fin=0.005s total=0.010s
2025-05-15 00:11:14,366 INFO Measuring split (23,28)
(23,28) copy=0.000s mid=0.006s fin=0.004s total=0.010s
2025-05-15 00:11:14,376 INFO Measuring split (23,29)
(23,29) copy=0.000s mid=0.007s fin=0.003s total=0.010s
2025-05-15 00:11:14,387 INFO Measuring split (23,30)
(23,30) copy=0.000s mid=0.008s fin=0.002s total=0.010s
2025-05-15 00:11:14,397 INFO Measuring split (23,31)
(23,31) copy=0.000s mid=0.009s fin=0.001s total=0.010s
2025-05-15 00:11:14,408 INFO Measuring split (23,32)
(23,32) copy=0.000s mid=0.010s fin=0.000s total=0.010s
2025-05-15 00:11:14,418 INFO Measuring split (24,25)
(24,25) copy=0.000s mid=0.001s fin=0.008s total=0.009s
2025-05-15 00:11:14,428 INFO Measuring split (24,26)
(24,26) copy=0.000s mid=0.002s fin=0.007s total=0.009s
2025-05-15 00:11:14,437 INFO Measuring split (24,27)
(24,27) copy=0.000s mid=0.003s fin=0.006s total=0.009s
2025-05-15 00:11:14,446 INFO Measuring split (24,28)
(24,28) copy=0.000s mid=0.005s fin=0.004s total=0.009s
2025-05-15 00:11:14,456 INFO Measuring split (24,29)
(24,29) copy=0.000s mid=0.006s fin=0.003s total=0.009s
2025-05-15 00:11:14,465 INFO Measuring split (24,30)
(24,30) copy=0.000s mid=0.008s fin=0.002s total=0.011s
2025-05-15 00:11:14,476 INFO Measuring split (24,31)
(24,31) copy=0.000s mid=0.008s fin=0.001s total=0.009s
2025-05-15 00:11:14,486 INFO Measuring split (24,32)
(24,32) copy=0.000s mid=0.009s fin=0.000s total=0.009s
2025-05-15 00:11:14,495 INFO Measuring split (25,26)
(25,26) copy=0.000s mid=0.001s fin=0.007s total=0.008s
2025-05-15 00:11:14,503 INFO Measuring split (25,27)
(25,27) copy=0.000s mid=0.002s fin=0.006s total=0.008s
2025-05-15 00:11:14,511 INFO Measuring split (25,28)
(25,28) copy=0.000s mid=0.003s fin=0.004s total=0.008s
2025-05-15 00:11:14,520 INFO Measuring split (25,29)
(25,29) copy=0.000s mid=0.005s fin=0.003s total=0.008s
2025-05-15 00:11:14,528 INFO Measuring split (25,30)
(25,30) copy=0.000s mid=0.006s fin=0.002s total=0.008s
2025-05-15 00:11:14,536 INFO Measuring split (25,31)
(25,31) copy=0.000s mid=0.007s fin=0.001s total=0.008s
2025-05-15 00:11:14,545 INFO Measuring split (25,32)
(25,32) copy=0.000s mid=0.008s fin=0.000s total=0.008s
2025-05-15 00:11:14,553 INFO Measuring split (26,27)
(26,27) copy=0.000s mid=0.001s fin=0.006s total=0.007s
2025-05-15 00:11:14,560 INFO Measuring split (26,28)
(26,28) copy=0.000s mid=0.002s fin=0.004s total=0.007s
2025-05-15 00:11:14,567 INFO Measuring split (26,29)
(26,29) copy=0.000s mid=0.003s fin=0.003s total=0.007s
2025-05-15 00:11:14,574 INFO Measuring split (26,30)
(26,30) copy=0.000s mid=0.005s fin=0.002s total=0.007s
2025-05-15 00:11:14,582 INFO Measuring split (26,31)
(26,31) copy=0.000s mid=0.006s fin=0.001s total=0.007s
2025-05-15 00:11:14,589 INFO Measuring split (26,32)
(26,32) copy=0.000s mid=0.007s fin=0.000s total=0.007s
2025-05-15 00:11:14,596 INFO Measuring split (27,28)
(27,28) copy=0.000s mid=0.001s fin=0.004s total=0.006s
2025-05-15 00:11:14,602 INFO Measuring split (27,29)
(27,29) copy=0.000s mid=0.002s fin=0.003s total=0.006s
2025-05-15 00:11:14,608 INFO Measuring split (27,30)
(27,30) copy=0.000s mid=0.003s fin=0.002s total=0.006s
2025-05-15 00:11:14,614 INFO Measuring split (27,31)
(27,31) copy=0.000s mid=0.005s fin=0.001s total=0.006s
2025-05-15 00:11:14,620 INFO Measuring split (27,32)
(27,32) copy=0.000s mid=0.006s fin=0.000s total=0.006s
2025-05-15 00:11:14,626 INFO Measuring split (28,29)
(28,29) copy=0.000s mid=0.001s fin=0.003s total=0.005s
2025-05-15 00:11:14,631 INFO Measuring split (28,30)
(28,30) copy=0.000s mid=0.002s fin=0.002s total=0.005s
2025-05-15 00:11:14,636 INFO Measuring split (28,31)
(28,31) copy=0.000s mid=0.005s fin=0.001s total=0.006s
2025-05-15 00:11:14,642 INFO Measuring split (28,32)
(28,32) copy=0.000s mid=0.004s fin=0.000s total=0.005s
2025-05-15 00:11:14,647 INFO Measuring split (29,30)
(29,30) copy=0.000s mid=0.001s fin=0.002s total=0.004s
2025-05-15 00:11:14,651 INFO Measuring split (29,31)
(29,31) copy=0.000s mid=0.002s fin=0.001s total=0.004s
2025-05-15 00:11:14,654 INFO Measuring split (29,32)
(29,32) copy=0.000s mid=0.003s fin=0.000s total=0.003s
2025-05-15 00:11:14,658 INFO Measuring split (30,31)
(30,31) copy=0.000s mid=0.001s fin=0.001s total=0.003s
2025-05-15 00:11:14,661 INFO Measuring split (30,32)
(30,32) copy=0.000s mid=0.002s fin=0.000s total=0.002s
2025-05-15 00:11:14,663 INFO Measuring split (31,32)
(31,32) copy=0.000s mid=0.001s fin=0.000s total=0.001s
"""

# Regex to capture start layer, end layer, and total time
pattern = re.compile(r'Measuring split \((\d+),(\d+)\)[\s\S]*?total=([0-9.]+)s')

matches = pattern.findall(log_data)

# Initialize a 32x32 matrix for layers
n_layers = 32
heatmap = np.full((n_layers, n_layers), np.nan)

# Populate the matrix
for start, end, total in matches:
    i = int(start) - 1
    j = int(end) - 1
    heatmap[i, j] = float(total)

# Plot heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap, aspect='auto')
plt.colorbar(label='Total Time (s)')
plt.title('Full Split Total Time Heatmap')
plt.xlabel('Layer End (j)')
plt.ylabel('Layer Start (i)')
plt.xticks(np.arange(n_layers), np.arange(1, n_layers+1))
plt.yticks(np.arange(n_layers), np.arange(1, n_layers+1))
plt.tight_layout()
plt.savefig('heatmap.pdf')