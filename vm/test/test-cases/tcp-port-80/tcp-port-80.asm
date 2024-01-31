ldxb r2, [r1+12]
ldxb r3, [r1+13]
lsh r3, 0x8
or r3, r2
mov r0, 0x0
jne r3, 0x8, +12
ldxb r2, [r1+23]
jne r2, 0x6, +10
ldxb r2, [r1+14]
add r1, 0xe
and r2, 0xf
lsh r2, 0x2
add r1, r2
ldxh r2, [r1+2]
jeq r2, 0x5000, +2
ldxh r1, [r1]
jne r1, 0x5000, +1
mov r0, 0x1
exit
