# 说明
项目主要是对BPE算法原理，进行了实现，实际中的库会更加复杂。       
参考代码链接为：https://github.com/owenliang/bpe-tokenizer  
(jupyter nbconvert --to markdown bpe.ipynb --output readme.md)  


```python
# 读取文件
cn=open("/home/minimind/xiaobo/dataset/train-cn.txt",mode="r",encoding="utf-8").read()
en=open("/home/minimind/xiaobo/dataset/train-en.txt",mode="r",encoding="utf-8").read()
```


```python
# 格式化数据 --字节列表 list(byte)
doc_byte_list=[]
cn_byte=[bytes([b]) for b in cn.encode("utf-8")]
en_byte=[bytes([b]) for b in en.encode("utf-8")]
doc_byte_list.extend(cn_byte)
doc_byte_list.extend(en_byte)
print(f"{len(cn_byte)},{len(en_byte)},{len(doc_byte_list)}")

```

    777269,185768,963037



```python
from tqdm import tqdm
import pickle
class BPE():
    def __init__(self):
        self.b2t={} # 字节转tokens
        self.t2b={} # tokens转字节
        self.lenToken=0 # tokens字典大小
    def _parse_states(self,arg_docByte):
        stats={}
        for i in range(len(arg_docByte)-1):
            new_tokens=arg_docByte[i]+arg_docByte[i+1]
            if new_tokens not in stats:
                stats[new_tokens]=1
            else:
                stats[new_tokens]+=1
        return stats
    def __megar_pair(self,arg_docByte,arg_token):
        megarTokens=[]
        i=0
        while i<len(arg_docByte):
            if i+1 < len(arg_docByte) and arg_docByte[i]+arg_docByte[i+1]==arg_token:
                megarTokens.append(arg_token)
                i+=2
            else:
                megarTokens.append(arg_docByte[i])
                i+=1
        return megarTokens
    
    def train(self,arg_docByte,arg_vocalSize):
        #单字节，基础token
        for i in range(256):
            self.b2t[bytes([i])]=i
        self.lenToken=len(self.b2t)
        processBar=tqdm(total=arg_vocalSize-self.lenToken)
        while True:
            if len(self.b2t) > arg_vocalSize:
                break
            #统计相邻token频率
            states=self._parse_states(arg_docByte)
            if len(states)==0:
                break
            #查找出现次数最多的组合
            newToken=max(states,key=states.get)
            arg_docByte=self.__megar_pair(arg_docByte,newToken)
            self.b2t[newToken]=len(self.b2t)
            processBar.update(1)
        self.t2b={v:k for k ,v in self.b2t.items()}
    def save(self,arg_fileName):
        with open(arg_fileName,'wb') as fp:
            fp.write(pickle.dumps((self.b2t,self.t2b)))
    def load(self,arg_fileName):
        with open(arg_fileName,'rb') as f : 
            self.b2t,self.t2b=pickle.loads(f.read())
    def encode(self,text):
        enc_tokens=[]
        enc_docByte=[]
        docByte=[bytes([b]) for b in text.encode("utf-8")]
        while True:
            #合并相邻token
            states = self._parse_states(docByte)
            newToken=None
            # 选择合并后id最小的pair合并（也就是优先合并短的）
            for merga_token in states:
                if merga_token in self.b2t and (newToken is None or self.b2t[merga_token]<self.b2t[newToken]):
                    newToken=merga_token
            if newToken ==None:
                break
            docByte=self.__megar_pair(docByte,newToken)
        enc_tokens.extend([self.b2t[tok] for tok in docByte])
        enc_docByte.extend(docByte)
        return enc_tokens,enc_docByte
    def decode(self,arg_tokens):
        byteList=[]
        for token in arg_tokens:
            byteList.append(self.t2b[token])
        return b''.join(byteList).decode("utf-8",errors="replace")
            
```


```python
bpe=BPE()
bpe.train(doc_byte_list,500)

```

      0%|          | 0/244 [00:00<?, ?it/s]245it [01:40,  2.44it/s]                         



```python
bpe.save("test_bpe")
```


```python
bpe=BPE()
bpe.load("test_bpe")
```


```python
text="hello world"
token,byteInfo=bpe.encode(text)
print(token)
print(byteInfo)
```

    [104, 101, 455, 32, 119, 111, 114, 108, 100]
    [b'h', b'e', b'll', b' ', b'w', b'o', b'r', b'l', b'd']



```python
tx=bpe.decode(token)
print(tx)
```

    hell world

