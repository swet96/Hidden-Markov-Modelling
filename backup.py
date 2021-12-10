#l=0,b=1
'''
    fid=open("training_set.txt")
    data=fid.read()
    data=data.replace(' ','').replace('\n','')
    for i in data:
    if i.isdigit():
    data=data.replace(i,'')
    print(data)
    print(len(data))
    '''

#preparing the training set
#======================================================================
#>UniProtKB - B0C4Z0 (B0C4Z0_ACAM1)
data1='MCFETSLNKANFSFQSLLRVIMQFSLPHSQRIPNRLTLIAPFFLWGTAMVAMKGVLAHTTPLFMGGLRIAPAGLLVIGVALLLGKSQPKGWRAWLWILLFALVDVTLFQGFLALGLSHTSAGLGSVMIDSQPLVVALLALVLYGERIGLWGWLGLGIGVSGISCIGLPDEWILQLGSGEWLQQDWQQVFNWQYLNQLTQSGEFLMLLAALSMAVGTVMIRQVCRYADPISATGWHMVIGGLPLFIGSGLWESEQWVHLTQIDWLSIGYAAVFGSAVAYGLFFYFASKGNLTSLSSLTFLTPVFALIFGNLFLAEVLSPIQTLGVCLTLVSIYFVNQREQLPTLGHLWSSLELRMLGTKDSKTAEIPVRVDPKTTVK'

value1=[0,36,56,62,83,95,116,122,142,149,168,203,219,231,251,263,284,296,313,319,335,376]

mod_value1=''

for i in range(len(value1)-1):
    if i%2==0:
        mod_value1+='l'*(value1[i+1]-value1[i]-1)
    else:
        mod_value1+='b'*(value1[i+1]-value1[i]+1)
mod_value1+='l'


#>UniProtKB - A0A2W4WYD7 (A0A2W4WYD7_9CYAN)
data2='MAPRSSTDDPDTGNDPNDAPRIPIRQPLLALTSNPLILIAPFFLWGTAMVAMKGVMQETTPLFLAGVRLLPAGLLVVAVSMLLGKQQPKGWRAWLWISLFALVDGTLFQGFLAKGLERTGAGLGSVMIDSQPLAVAVMARFLFQEWIGPLGWIGLLIGLIGISFIGLPDEWIIGLFQGPISGPISGPISGPISGGGAASGPVITVEQEIWSGLFQQGEWLMLMAALSMAVGTILIRYVVRWADPVAATGWHMVIGGVPLMAAALLDWLSGNPAEYSDLAGGTMGSVIREALAGGTAPWQGISLSGWLEMSYATVFGSAIAYGLFFYIAAQGNLTSLSALTFLTPVFALLFSTLLLSESLSALQWGGVVLTLVSIYLINQRIQLANWLSDQLSISISAPPAAAEDPQKTT'
value2=[0,28,50,62,82,94,113,150,167,219,238,250,268,309,329,336,355,361,378,409]
mod_value2=''

for i in range(len(value2)-1):
    if i%2==0:
        mod_value2+='l'*(value2[i+1]-value2[i]-1)
    else:
        mod_value2+='b'*(value2[i+1]-value2[i]+1)
mod_value2+='l'

#>UniProtKB - P74436 (Y355_SYNY3)
data3='MQIESKTNTNIRSGLTLIAPFFLWGTAMVAMKGVLADTTPFFVATVRLIPAGILVLLWAMGQKRPQPQNWQGWGWIILFALVDGTLFQGFLAQGLERTGAGLGSVIIDSQPIAVALLSSWLFKEVIGGIGWLGLLLGVGGISLIGLPDEWFYQLWHLQGLSINWSGSALGSSGELWMLLASLSMAVGTVLIPFVSRRVDPVVATGWHMIIGGLPLLAIALVQDSEPWQNIDLWGWGNLAYATVFGSAIAYGIFFYLASKGNLTSLSSLTFLTPIFALSFSNLILEEQLSSLQWLGVAFTLVSIYLINQREQLKIQLRDIWSLVRKPVIND'
value3=[0,15,35,41,61,72,92,102,122,125,145,175,195,201,221,238,258,264,284,286,306,330]
mod_value3=''

for i in range(len(value3)-1):
    if i%2==0:
        mod_value3+='l'*(value3[i+1]-value3[i]-1)
    else:
        mod_value3+='b'*(value3[i+1]-value3[i]+1)
mod_value3+='l'


#following 4 are the input to the user defined supervised_estimate function
#==================================================================
obs_seq=data1+data2+data3
hidden_seq=mod_value1+mod_value2+mod_value3
obs_states=['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
hidden_states=['l','b']
#===================================================================

import pandas as pd
import numpy as np
from numpy import linalg as LA
tol=0.01
const=0.001

#finding out initial, transition and emission probability
#=====================================================================

def get_supervised_estimate(obs_seq,obs_states,hidden_seq,hidden_states):
    
    
    training_set=list(map(list,zip(hidden_seq,obs_seq)))
    
    
    init_prob=[]
    for i in hidden_states:
        init_prob.append(hidden_seq.count(i)/len(hidden_seq))
    init_prob=pd.Series(init_prob,index=hidden_states)


df=pd.DataFrame(0,columns=hidden_states,index=hidden_states)
for i in range(len(hidden_seq)-1):
    s1=hidden_seq[i]
    s2=hidden_seq[i+1]
    df.loc[s1,s2]= df.loc[s1,s2]+1
    df.replace(0,const)    #replaced the zero values incase it might give rise to zero division error
    df=df.div(df.sum(axis=1),axis=0)
    trans_prob=df
    
    
    
    df=pd.DataFrame(0,columns=obs_states,index=hidden_states)
    for s in training_set:
        df.loc[s[0],s[1]]= df.loc[s[0],s[1]]+1
    df.replace(0,const)    #replaced the zero values incase it might give rise to zero division error
    df=df.div(df.sum(axis=1),axis=0)
    emit_prob=df
    return [init_prob,trans_prob,emit_prob]
#=============================================================================
'''
    print(init_prob)
    print(trans_prob)
    print(emit_prob)
    '''

#this is the test data whose hidden states i.e., hydrophobic or hydrophilic sites, we want to know

#UniProtKB - B0CG09
#test_data='MSSSSPNPFRRSLKFVEQWYRETPQRALDGAYEAARAIEEIEKKHFKGQPVPLRIRTESVMTNYFQSEVQKNLQFIQTRLREFKSSSLVVEVADKLKPPSIPPAPTPLDTPNTIDFTDEYDVTSEEYSSELVSPSIDAQGSLDKLAFIDAVLKRYRSASIQREAAAAASKAARASAPKSGSEMKKNIPQPLPIQSAQNSLYESEFISDDITEDPSKLDSSSFIPRSILRTATRFRKELNPDPGTEDDILNDFRNSRVRTRAAVSFVLGLMIVPLLTQQVSKNLVIGPFVDKLKGPEQIEIRINPEIENEVLTELARFEERLKFESLTSPIPLSPAEIQFQLKAKAEDLKEEYQWDLRQPLKNAISDLFSLVALAIYFVLNRQKIAVLKSFFDEIIYGLSDSAKAFIIILFTDVFVGFHSPHGWEVIVESVLSHFGLPQDRNFINMFIATFPVMLDTVFKYWIFRYLNQISPSAVATYRNMNE'

#UniProtKB - B0C8V0
#test_data='MTQLKSTGKKVGSYGVKPNPYAFIALLLGAIGIAFAPIFVRLSELGPSATAFYRLGFAVPLLGVWLWTSRPATSFPVNEQRVTGLPLISAGCFFAADLAVWHWSIQFTSVANATLLANFAPIFVVLGGWLLFGETVSRWFLVAIALVLLGATLLVSASIDTQHVFGDALGLLTAIFYAGYILSVARLRLHFPTATIAFACSIVGATILFFIAWLSGEGFLPMTMTGWLSLVGLACISQVIGQSLIMFALAHLPSAFASVSLLLQPVTAAILAWRLFGEALTLQQGFGGVIVLAGIVLARWSDRKPVS'

#UniProtKB - B0CAT2
#test_data='MFRLDYPPLANWAGFSAALLYVVTLLPTILRVVFPSTKTTGIPKKLLIQRRLLGIIAFLLSVIHGYWMVSKRELDFLDLQTYWIYCQGIFTFLIFALLAITSNDWSVKKLKKSWKKLHKLTYLAMFLLLWHVIDKMWGHWTWVTPPSLFITGIITTLFVIRVIRENYVLDKSKAQSSQSKAPEATASKKD'

#UniProtKB - B0CDA5
test_data='MTSNLGQPQAFFKKTLVPVLADLRLAIVLLLAIALFSISGTVIEQGQSLEFYQANYPEEPALFGFLTWKVLVTIGLDHVYATWWFLSLLILFGTSLTACTFMRQLPALKAARSWQFYKKPRQFGKLALSTTLDPDQKPSLLKALEKNRYKVFEEDQSIYARKGITGRIGPIIVHASMILILLGSIWGSLTGFMAQEMIPSGTTAKVSNIVKSGPWSGAQIPRDWAVQVNRFWIDYTPEGQIDQFYSDLSIVDEDKNELDRQTIHVNQPLKHKGVTLYQADWSIAGVRVQLNNSPVLQLPMAPLEAAGGRIWGTWVPTKPDLSAGVTLLTTDLQGTVVVYDESGKLVSTVRTGMSTDVNDISLKLVELVGSTGLQIKSDPGIPWIYAGFGLLMIGVIMSYVSHSQIWLLTADDQLYVGGRTNRALLTFERELVEMIESSASDSALSPSTNPQPQEVA'







#along with test_data, the following three(panda dataframes, notice suitable column and row names are used for easy reference) are inputs to the user defined function viterbi
result=get_supervised_estimate(obs_seq,obs_states,hidden_seq,hidden_states)
init_prob=result[0]
trans_prob=result[1]
emit_prob=result[2]
#==============================================

'''
    DOCUMENTATION
    This function prints/returns the hidden states using the viterbi algorithm.
    Value_mat stores the probabilities and hidden_path stores the hidden states which gives the result by back tracing and
    hidden_seq stores the result.
    '''
'''
    #value_mat and hidden_path are dataframes but hidden_seq is a list
    def viterbi(test_data,init_prob,trans_prob,emit_prob):
    
    value_mat=pd.DataFrame(0,index=hidden_states,columns=range(len(test_data)))
    hidden_path=pd.DataFrame(0,index=hidden_states,columns=range(len(test_data)))
    
    
    for i in hidden_states:
    value_mat.loc[i,0]=init_prob[i]*emit_prob.loc[i,test_data[0]]
    hidden_path.loc[i,0]=1
    
    
    for j in range(1,len(test_data)):
    for i in hidden_states:
    buf =(value_mat[j-1].multiply(trans_prob[i])).idxmax()
    value_mat.loc[i,j]=value_mat.loc[buf,j-1]*trans_prob.loc[buf,i]*emit_prob.loc[i,test_data[j]]
    hidden_path.loc[i,j]=buf
    hidden_seq=[0]*len(test_data)
    
    hidden_seq[len(test_data)-1]=value_mat[len(test_data)-1].idxmax()
    
    for i in range(len(test_data)-2,-1,-1):
    hidden_seq[i]=hidden_path.loc[hidden_seq[i+1],i+1]
    
    hidden_seq=''.join(hidden_seq)
    return hidden_seq
    #=======================================================================
    '''


#since probabilities are multiplied, storing issues arise. Hence log of probability is used.
def log_viterbi(test_data,init_prob,trans_prob,emit_prob):
    init_prob=np.log(init_prob)
    trans_prob=np.log(trans_prob)
    emit_prob=np.log(emit_prob)
    
    value_mat=pd.DataFrame(0,index=hidden_states,columns=range(len(test_data)))
    hidden_path=pd.DataFrame(0,index=hidden_states,columns=range(len(test_data)))
    
    
    for i in hidden_states:
        value_mat.loc[i,0]=init_prob[i]+emit_prob.loc[i,test_data[0]]
        hidden_path.loc[i,0]=1
    
    
    for j in range(1,len(test_data)):
        for i in hidden_states:
            buf =(value_mat[j-1]+(trans_prob[i])).idxmax()
            value_mat.loc[i,j]=value_mat.loc[buf,j-1]+trans_prob.loc[buf,i]+emit_prob.loc[i,test_data[j]]
            hidden_path.loc[i,j]=buf
    hidden_seq=[0]*len(test_data)

hidden_seq[len(test_data)-1]=value_mat[len(test_data)-1].idxmax()

for i in range(len(test_data)-2,-1,-1):
    hidden_seq[i]=hidden_path.loc[hidden_seq[i+1],i+1]
    
    hidden_seq=''.join(hidden_seq)
    return hidden_seq



old_init_prob=init_prob
old_trans_prob=trans_prob
old_emit_prob=emit_prob

while True:
    old_init_prob=init_prob
    old_trans_prob=trans_prob
    old_emit_prob=emit_prob
    hidden_seq= log_viterbi(test_data,init_prob,trans_prob,emit_prob)
    init_prob,trans_prob,emit_prob= get_supervised_estimate(obs_seq,obs_states,hidden_seq,hidden_states)
    if LA.norm(init_prob-old_init_prob)>tol or LA.norm(trans_prob-old_trans_prob)>tol or LA.norm(emit_prob-old_emit_prob)<tol:
        break

print(init_prob)
print(trans_prob)
print(emit_prob)




#"log_viterbi" function gives the most likely hidden sequence of states corresponding to he input observation
hidden_seq=log_viterbi(test_data,init_prob,trans_prob,emit_prob)
print(hidden_seq)
#========================================

#printing the position of the hydrophobic sites
i=0
j=len(test_data)
if hidden_seq.startswith('b'):
    for i in range(len(test_data)):
        if hidden_seq[i]=='l':
            break
    print(1,end='\t')
    print(i)

if hidden_seq.endswith('b'):
    for j in range(len(test_data)-1,-1,-1):
        if hidden_seq[j]=='l':
            break
    print(j+2,end='\t')
    print(len(test_data))

for k in range(i,j):
    if hidden_seq[k:k+2]=='lb':
        start=k+2
        print(start,end='\t')
    if hidden_seq[k:k+2]=='bl':
        finish=k+1
        print(finish)
#===========================================


