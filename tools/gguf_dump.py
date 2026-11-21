#!/usr/bin/env python3
"""Dump GGUF metadata and tensor table from the head of a (possibly truncated) file.
Usage: python3 -I gguf_dump.py FILE [--tensors] [--meta]"""
import struct, sys
T = {0:'u8',1:'i8',2:'u16',3:'i16',4:'u32',5:'i32',6:'f32',7:'bool',8:'str',9:'arr',10:'u64',11:'i64',12:'f64'}
F = {'u8':'<B','i8':'<b','u16':'<H','i16':'<h','u32':'<I','i32':'<i','f32':'<f','bool':'<?','u64':'<Q','i64':'<q','f64':'<d'}
GT = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',39:'MXFP4',40:'NVFP4'}
class R:
    def __init__(s,b): s.b=b; s.o=0
    def u(s,f):
        n=struct.calcsize(f); v=struct.unpack_from(f,s.b,s.o)[0]; s.o+=n; return v
    def st(s):
        n=s.u('<Q'); v=s.b[s.o:s.o+n].decode('utf-8','replace'); s.o+=n; return v
    def val(s,t):
        if t=='str': return s.st()
        if t=='arr':
            et=T[s.u('<I')]; n=s.u('<Q')
            if et in F and n>64:  # skip big numeric arrays quickly
                sz=struct.calcsize(F[et]); vals=list(struct.unpack_from('<'+F[et][1]*n,s.b,s.o)); s.o+=sz*n; return vals
            return [s.val(et) for _ in range(n)]
        return s.u(F[t])
def main():
    b=open(sys.argv[1],'rb').read(); r=R(b)
    assert b[:4]==b'GGUF', 'not gguf'; r.o=4
    ver=r.u('<I'); nt=r.u('<Q'); nk=r.u('<Q')
    print(f'# GGUF v{ver}, {nt} tensors, {nk} kv')
    kv={}
    for _ in range(nk):
        k=r.st(); t=T[r.u('<I')]; v=r.val(t); kv[k]=v
    if '--meta' in sys.argv or '--tensors' not in sys.argv:
        for k,v in kv.items():
            if k.startswith('tokenizer.ggml.') and isinstance(v,list) and len(v)>40:
                print(f'{k} = <{len(v)} items>'); continue
            s=repr(v)
            print(f'{k} = {s[:300]}')
    if '--tensors' in sys.argv:
        for _ in range(nt):
            n=r.st(); nd=r.u('<I'); dims=[r.u('<Q') for _ in range(nd)]; ty=r.u('<I'); off=r.u('<Q')
            print(f'{n}\t{dims}\t{GT.get(ty,ty)}\t{off}')
main()
