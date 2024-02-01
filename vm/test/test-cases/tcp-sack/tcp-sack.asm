ldxb r2, [r1+12]
ldxb r3, [r1+13]
lsh r3, 0x8
or r3, r2
mov r0, 0x0
jne r3, 0x8, +37
ldxb r2, [r1+23]
jne r2, 0x6, +35
ldxb r2, [r1+14]
add r1, 0xe
and r2, 0xf
lsh r2, 0x2
add r1, r2
mov r0, 0x0
ldxh r4, [r1+12]
add r1, 0x14
rsh r4, 0x2
and r4, 0x3c
mov r2, r4
add r2, 0xffffffec
mov r5, 0x15
mov r3, 0x0
jgt r5, r4, +20
mov r5, r3
lsh r5, 0x20
arsh r5, 0x20
mov r4, r1
add r4, r5
ldxb r5, [r4]
jeq r5, 0x1, +4
jeq r5, 0x0, +12
mov r6, r3
jeq r5, 0x5, +9
ja +2
add r3, 0x1
mov r6, r3
ldxb r3, [r4+1]
add r3, r6
lsh r3, 0x20
arsh r3, 0x20
jsgt r2, r3, -18
ja +1
mov r0, 0x1
exit
