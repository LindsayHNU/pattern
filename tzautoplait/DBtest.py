import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
def DTWDistance(s1, s2):
    DTW = {}
    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])

def DTWDistance1(s1, s2, w):
    DTW={}
    w = max(w, abs(len(s1)-len(s2)))
    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0
    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])
    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

def gyh(datas,bl):
    v=bl*(datas-datas.min())/(datas.max()-datas.min())
    return v



def compute_S_init(datas,bl=1,perference='median',juli='oushi'):
    N, D = np.shape(datas)  
    l = D
    gyh_datas = gyh(datas,bl)
    tile_x = np.tile(np.expand_dims(datas, 1), [1, N, 1])
    tile_y = np.tile(np.expand_dims(datas, 0), [N, 1, 1])
    gyh_tile_x = np.tile(np.expand_dims(gyh_datas, 1), [1, N, 1])
    gyh_tile_y = np.tile(np.expand_dims(gyh_datas, 0), [N, 1, 1])
    if juli == 'oushi':
        dis = (tile_x - tile_y) * (tile_x - tile_y)
        S = -np.sqrt(np.sum(dis, axis=-1))

    if juli == 'gyh_oushi':
        dis = (gyh_tile_x - gyh_tile_y) * (gyh_tile_x - gyh_tile_y)
        S = -np.sqrt(np.sum(dis, axis=-1))

    if juli == 'cos':
        cz1 = tile_x[:, :, 1:l] - tile_x[:, :, 0:(l - 1)]
        cz2 = tile_y[:, :, 1:l] - tile_y[:, :, 0:(l - 1)]
        SO = []
        acos = 0
        count = 0
        for m, n in zip(cz1, cz2):
            for mm, nn in zip(m, n):
                for mmm, nnn in zip(mm, nn):
                    A = np.array([1, mmm])
                    B = np.array([1, nnn])
                    num = np.dot(A, B)
                    denom = np.linalg.norm(A) * np.linalg.norm(B)
                    cos = num / denom  # 余弦值
                    sim = 1 -  cos  # 每段归一化后用1减去
                    acos = acos + sim
                    count = count + 1
                    if count % (l - 1) == 0:
                        SO.append(acos)
                        acos = 0
        SS = -np.array(SO).reshape(N, N)
        S = np.around(SS, 3)

    if juli == 'oushi-cos':
        dis1 = (tile_x - tile_y) * (tile_x - tile_y)
        cz1 = tile_x[:, :, 1:l] - tile_x[:, :, 0:(l - 1)]
        cz2 = tile_y[:, :, 1:l] - tile_y[:, :, 0:(l - 1)]
        ecos = []
        for m, n in zip(cz1, cz2):
            for mm, nn in zip(m, n):
                for mmm, nnn in zip(mm, nn):
                    A = np.array([1, mmm])
                    B = np.array([1, nnn])
                    num = np.dot(A, B)
                    denom = np.linalg.norm(A) * np.linalg.norm(B)
                    cos = num / denom  # 余弦值
                    sim = 1 - cos  # 每段归一化后用1减去
                    ecos.append(math.exp(2*sim))
        CE = np.array(ecos).reshape(N, N,l-1)
        odce=[]
        for ods, ces in zip(dis1, CE):
            for od,ce in zip(ods,ces):
                hod=0.5*od
                jhod1=hod[0:len(hod)-1]
                jhod2=hod[1:len(hod)]
                jhod=jhod1+jhod2
                odce.append(sum(jhod*ce)+hod[0]+hod[-1])
        OC=np.array(odce).reshape(N, N)
        S=-np.sqrt(OC)


    if juli == 'gyhoushi-cos':
        dis1 = (gyh_tile_x - gyh_tile_y) * (gyh_tile_x - gyh_tile_y)
        cz1 = tile_x[:, :, 1:l] - tile_x[:, :, 0:(l - 1)]
        cz2 = tile_y[:, :, 1:l] - tile_y[:, :, 0:(l - 1)]
        ecos = []
        for m, n in zip(cz1, cz2):
            for mm, nn in zip(m, n):
                for mmm, nnn in zip(mm, nn):
                    A = np.array([1, mmm])
                    B = np.array([1, nnn])
                    num = np.dot(A, B)
                    denom = np.linalg.norm(A) * np.linalg.norm(B)
                    cos = num / denom  # 余弦值
                    sim = 1 - cos  # 每段归一化后用1减去
                    ecos.append(math.exp(2*sim))
        CE = np.array(ecos).reshape(N, N, l - 1)
        odce = []
        for ods, ces in zip(dis1, CE):
            for od, ce in zip(ods, ces):
                hod = 0.5 * od
                jhod1 = hod[0:len(hod) - 1]
                jhod2 = hod[1:len(hod)]
                jhod = jhod1 + jhod2
                odce.append(sum(jhod * ce) + hod[0] + hod[-1])
        OC = np.array(odce).reshape(N, N)
        S = -np.sqrt(OC)

    if juli == 'dtw':
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dis = DTWDistance(tile_x[i][0], tile_y[0][j])
                S[i][j] = -dis

    if juli == 'wdtw':
        S = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                dis = DTWDistance1(tile_x[i][0], tile_y[0][j], 2)
                S[i][j] = -dis


    indices=np.where(~np.eye(S.shape[0],dtype=bool))

    if perference=='median':
        m=np.median(S[indices])
        print(m)

    elif perference=='min':
        m = np.min(S[indices])
        print(m)
    elif perference=='min2':
        m = np.min(S[indices])/2
        print(m)

    elif perference=='mean':
        m = np.mean(S[indices])
        print(m)

    else:
        m=perference
        print(m)

    np.fill_diagonal(S,m)

    return S



def compute_R(S,R,A,dampfac):
    to_max=A+R
    N=np.shape(to_max)[0]  

    max_AS=np.zeros_like(S)
    for i in range(N):
        for k in range(N):
            if not i==k:
                temp=to_max[i,:].copy() 
                temp[k]=-np.inf
                max_AS[i,k]=max(temp)
            else:
                temp=S[i,:].copy()  
                temp[k]=-np.inf
                max_AS[i,k]=max(temp)
    r_res=(1-dampfac)*(S-max_AS)+dampfac*R
    return r_res


def compute_A(R,A,dampfac):
    max_R=np.zeros_like(R)
    N=np.shape(max_R)[0] 

    for i in range(N):
        for k in range(N):
            max_R[i,k]=np.max([0,R[i,k]])

    min_A=np.zeros_like(A)
    for i in range(N):
        for k in range(N):
            if not i==k:
                temp=max_R[:,k].copy()#取出第k列
                temp[i]=0
                min_A[i,k]=np.min([0,R[k,k]+np.sum(temp)])
            else:
                temp=max_R[:,k].copy()#取出第k列
                temp[k]=0
                min_A[i,k]=np.sum(temp)
    a_res=(1-dampfac)*min_A+dampfac*A
    return a_res


def cplot(datas,labels,str_tile=''):
    plt.cla() 
    index_center=np.unique(labels).tolist() 
    colors={}
    for i,each in zip(index_center,np.linspace(0,1,len(index_center))):
        colors[i]=plt.cm.Spectral(each)
    N,D=np.shape(datas)
    for i in range(N):
        i_center=labels[i]
        center=datas[i_center]
        data=datas[i]
        color=colors[i_center]
        plt.plot([center[0],data[0]],[center[1],data[1]],color=color)#在center和data之间划线
    plt.title(str_tile)


def affinity_prop(datas,bl=1,maxiter=200,perference='median',dampfac=0.7,juli='gyhoushi-cos',dispaly=False):

    message_thresh= 1e-5

    loc_thresh=10

    S=compute_S_init(datas,bl,perference,juli)


    A=np.zeros_like(S)
    R=np.zeros_like(S)

    #S=S+ 1e-12*np.random.normal(size=A.shape)*(np.max(S)-np.min(S))

    count_equal=0

    i=0

    converged=False

    while i<maxiter:

        E_old=R+A
        labels_old=np.argmax(E_old,axis=1)


        R=compute_R(S,R,A,dampfac)
        A=compute_A(R,A,dampfac)

        E_new=R+A
        labels_cur=np.argmax(E_new,axis=1)
        types =len(np.unique(labels_cur))

        if dispaly== True:
            cplot(datas, labels_cur,str_tile=str(i)+'epoch,'+str(types)+'clusters')
            plt.show()

        if np.all(labels_cur==labels_old):
            count_equal+=1
        else:
            count_equal=0
        if (message_thresh!=0 and np.allclose(E_old,E_new,atol=message_thresh)) or\
                (loc_thresh!=0 and count_equal>loc_thresh):
            converged=True
            break
        i=i+1



    E=R+A

    labels=np.argmax(E,axis=1)

    exemplars=np.unique(labels)

    centers=datas[exemplars]

    return labels,exemplars,centers,S

def all_equal(lst):
  return lst[1:] == lst[:-1]

def caldb(s,labels,exemplars):
    np.fill_diagonal(s, 0)
    ss = -s
    #计算每类簇的均值
    count = Counter(labels)
    s_i=np.zeros(len(exemplars))
    zxi=-1
    for zx in exemplars:
        zxi=zxi+1
        for i in range(len(labels)):
            if labels[i]==zx:
                s_i[zxi]=ss[zx][i]+s_i[zxi]
        s_i[zxi]=s_i[zxi]/count[zx]
    combined_s_i_j = s_i[:, None] + s_i
    #计算簇间距离
    centroid_dis=[]
    for i in exemplars:
        for j in exemplars:
            centroid_dis.append(ss[i][j])
    centroid_distances=np.array(centroid_dis).reshape(len(exemplars),len(exemplars))
    centroid_distances[centroid_distances == 0] = np.inf
    scores = np.max(combined_s_i_j/centroid_distances,axis=1)
    db=np.mean(scores)
    return db


df=pd.read_excel('ex1.xlsx',header=None)
e=np.array(df.iloc[:,0])
if all_equal(list(e)):
    res=e[0:24]
else:
    allday=int(len(e)/24)
    maxk=int(np.sqrt(0.5*allday))
    data=e.reshape(allday,24)
    labels,exemplars,centers,s=affinity_prop(data,bl=10,dampfac=0.5,perference='median',juli='gyhoushi-cos',dispaly=False)
    pdd=pd.DataFrame(s)
    p=round(s[0][0],1)
    k=len(exemplars)
    db=caldb(s,labels,exemplars)
    print(db)
    ks=[]
    dbs=[]
    ps=[]
    ps.append(p)
    ks.append(k)
    dbs.append(db)
    if k<=2:
        while(k<=maxk):
            p =round(p + 0.1,1)
            labels, exemplars, centers, s = affinity_prop(data, bl=10, dampfac=0.5, perference=p, juli='gyhoushi-cos',
                                                          dispaly=False)
            k = len(exemplars)
            db = caldb(s, labels, exemplars)
            ps.append(p)
            ks.append(k)
            dbs.append(db)
            if k>maxk:
                break
    elif k>2:
        while(k>2):
            p=round(p-1,1)
            labels, exemplars, centers, s = affinity_prop(data, bl=10, dampfac=0.5, perference=p, juli='gyhoushi-cos',
                                                          dispaly=False)
            k=len(exemplars)
            print(k)
            db = caldb(s, labels, exemplars)
            ps.append(p)
            ks.append(k)
            dbs.append(db)
            if k==2:
                break
    print(ps)
    print(ks)
    print(dbs)










        #db=caldb(s,labels,exemplars)

        # print(labels)
        # print(exemplars)
        # print(len(exemplars))



# for exe in exemplars:
#     plt.plot(np.linspace(1,len(data[exe]),len(data[exe])),data[exe])
# plt.show()
# print(centers)
# print(Counter(labels))
#print(centers)

# cplot(data,labels)
# plt.show()