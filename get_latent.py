from scipy import stats
import numpy as np
import h5py
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models,optimizers,callbacks,constraints
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.layers import Input, Dense, Conv2D,multiply,Lambda,Add,Concatenate, Multiply,Conv2DTranspose,Layer, Reshape, ZeroPadding2D,Flatten, MaxPooling2D, RepeatVector,UpSampling2D,TimeDistributed, BatchNormalization,LeakyReLU,LSTM,Embedding,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping,Callback,LearningRateScheduler,ModelCheckpoint
from sklearn.utils import shuffle
jobid=0#os.getenv('SLURM_ARRAY_TASK_ID')
jobid=int(jobid)
param={i:[] for i in range(11)}
param[0]=[15,15,16,16]
param[1]=[15,15,32,16]
param[2]=[15,15,32,32]

param[3]=[45,30,16,16]
param[4]=[30,30,32,16]
param[5]=[15,30,32,16]
param[6]=[45,30,32,16]
param[7]=[30,30,32,32]
param[8]=[15,30,32,32]
param[9]=[45,30,32,32]
param[10]=[45,45,32,32]


W=param[jobid][0]
F=param[jobid][1]
hh=param[jobid][2]
lstm_len=param[jobid][3]
allpose=np.load('train_pose_1.npy')
# newall=np.zeros((allpose.shape))
# newall[:len(allpose)-1]=allpose[1:]
# allpose=newall-allpose
# allpose=allpose[:-1]
scaler = MinMaxScaler()
allpose=scaler.fit_transform(allpose)
beh=np.load('train_behavior_1.npy')[:-1]


def get_data(allpose,W,F):
    length=len(allpose)-W-F
    future=np.zeros((length,W,allpose.shape[-1]))
    current=np.zeros((length,W,allpose.shape[-1]))
    for i in range (length):
        temp=allpose[i:i+W]
        fut=allpose[i+F:i+W+F]
        future[i]=temp
        current[i]=fut
    return future,current

current, future=get_data(allpose,W,F)
new_file='Dis_train_linear_'+str(jobid)+'_W_'+str(W)+'_F_'+str(F)+'.hdf5'
def get_dis(pose):
    new=np.zeros((pose.shape[0],16))

    new[:,0]=np.power((pose[:,0]-pose[:,1]),2)+np.power((pose[:,7]-pose[:,8]),2)
    new[:,1]=np.power(pose[:,0]-pose[:,2],2)+np.power((pose[:,7]-pose[:,9]),2)
    new[:,2]=np.power(pose[:,1]-pose[:,3],2)+np.power(pose[:,8]-pose[:,10],2)
    new[:,3]=np.power(pose[:,2]-pose[:,3],2)+np.power(pose[:,9]-pose[:,10],2)
    new[:,4]=np.power(pose[:,3]-pose[:,4],2)+np.power(pose[:,10]-pose[:,11],2)
    new[:,5]=np.power(pose[:,3]-pose[:,5],2)+np.power(pose[:,10]-pose[:,12],2)
    new[:,6]=np.power(pose[:,4]-pose[:,6],2)+np.power(pose[:,11]-pose[:,13],2)
    new[:,7]=np.power(pose[:,5]-pose[:,7],2)+np.power(pose[:,12]-pose[:,13],2)
    
    pose=pose[:,14:]
    new[:,8]=np.power(pose[:,0]-pose[:,1],2)+np.power(pose[:,7]-pose[:,8],2)
    new[:,9]=np.power(pose[:,0]-pose[:,2],2)+np.power(pose[:,7]-pose[:,9],2)
    new[:,10]=np.power(pose[:,1]-pose[:,3],2)+np.power(pose[:,8]-pose[:,10],2)
    new[:,11]=np.power(pose[:,2]-pose[:,3],2)+np.power(pose[:,9]-pose[:,10],2)
    new[:,12]=np.power(pose[:,3]-pose[:,4],2)+np.power(pose[:,10]-pose[:,11],2)
    new[:,13]=np.power(pose[:,3]-pose[:,5],2)+np.power(pose[:,10]-pose[:,12],2)
    new[:,14]=np.power(pose[:,4]-pose[:,6],2)+np.power(pose[:,11]-pose[:,13],2)
    new[:,15]=np.power(pose[:,5]-pose[:,7],2)+np.power(pose[:,12]-pose[:,13],2)
    return new

with h5py.File(new_file, 'w', libver='latest', swmr=True) as f:

    # enable single write, multi-read - needed for simultaneous model fitting
    f.swmr_mode = True
    for m in range(1,5):
        allpose=np.load('train_pose_'+str(m)+'.npy')
        allpose=get_dis(allpose)
        scaler = MinMaxScaler()
        allpose=scaler.fit_transform(allpose)
        beh=np.load('train_behavior_'+str(m)+'.npy')[:-1]

        group_i = f.create_group(str(m))
        # print(name)
        subgroup1=group_i.create_group('current')#pose
        subgroup2=group_i.create_group('future')#beh
        current, future=get_data(allpose,W,F)        
        
        for i in range (len(current)):
            subgroup1.create_dataset(str(i), data=current[i])
            subgroup2.create_dataset(str(i), data=future[i])#, dtype='uint8'
a = h5py.File(new_file, 'r')

total_len=0
total_test_len=0
for i in (list(a.keys())):
    total_len+=len(list(a[i]['current'].keys()))//2
    total_test_len+=len(list(a[i]['current'].keys()))-len(list(a[i]['current'].keys()))//2
    # print(len(list(a[i]['current'].keys())))
# total_len
trainindex={i:[] for i in range (total_len)}
testindex={i:[] for i in range (total_test_len)}

z=0
zz=0
for i in (list(a.keys())):
    for idx in list(a[i]['current'].keys())[:len(list(a[i]['current'].keys()))//2]:
        trainindex[z].append(i)
        trainindex[z].append(idx)
        z+=1
    # print(idx)
for i in (list(a.keys())):      
    for idx in list(a[i]['current'].keys())[len(list(a[i]['current'].keys()))//2:]:
        testindex[zz].append(i)
        testindex[zz].append(idx)
        zz+=1
# index
# trainindex=[index[i] for i in range (len(index)//2)]
# testindex=[index[i] for i in range (len(index)//2,len(index))]
trainindex=shuffle(trainindex)
# trainindex,testindex
from spektral.layers import GATConv
import random


def data_generator(idx, hdf5file,batch_size,if_train = True):
    i = 0
#     j=0 
    while True:
        X = {j:[] for j in range (2)}
        Y = {j:[] for j in range (2)}
        for b in range(batch_size):
            if i == len(idx):
                i = 0
            
            alll=idx[i]
                
            x = np.array(hdf5file[alll[0]]['current'][alll[1]])  # read dataset on the fly
            y = np.array(hdf5file[alll[0]]['future'][alll[1]]) # read dataset on the fly


            X[0].append(x[:,:8])
            Y[0].append(y[:,:8])
            X[1].append(x[:,8:])
            Y[1].append(y[:,8:])

            i += 1

        X = [np.asarray(X[j]) for j in X]
        Y = [np.asarray(Y[j]) for j in Y]

        A=np.ones((X[0].shape[0],hh,hh))
        yield [X,Y,A],[Y]
weight1=1
weight2=1    
class MSE_UNSUP1(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSE_UNSUP1, self).__init__(*args, **kwargs)
    def call(self,inputs):
        D,A=inputs
        # L=0
        # for d,a in zip(D,A):
            # print(D[d],A[a])
        tt=tf.keras.losses.mse(D,A)
        # print(tt)
        L=tf.reduce_mean(tt)
            
        self.add_loss(L*W*8*weight1, inputs=inputs)
        
        return inputs,L
class MSE_UNSUP2(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSE_UNSUP2, self).__init__(*args, **kwargs)
    def call(self,inputs):
        D,A=inputs
        # L=0
        # for d,a in zip(D,A):
            # print(D[d],A[a])
        tt=tf.keras.losses.mse(D[:,:,:7],A[:,:,:7])
        tt1=tf.keras.losses.mse(D[:,:,7:],A[:,:,7:])
        # print(tt)
        L=tf.reduce_mean(tt+tt1*weight2)
            
        self.add_loss(L*W*8, inputs=inputs)
        
        return inputs,L
    
input0 = [Input(shape=(W,8)) for i in range (2)]
input1 = [Input(shape=(W,8)) for i in range (2)]
adj_input = Input(shape=(hh,hh), sparse=True)

emb0={i: TimeDistributed(Dense(14, kernel_regularizer='l2'))(input0[i]) for i in range (len(input0))}
emb0={i: LSTM(lstm_len, kernel_regularizer='l2', return_sequences=True, return_state=True)(emb0[i]) for i in range (len(input0))}
# emb1={i: TimeDistributed(Dense(14))(input1[i]) for i in range (len(input1))}
emb10=Concatenate(axis=-1)([emb0[i][0] for i in range (len(emb0))])
lstm = keras.layers.LSTM(hh, return_sequences=True, return_state=True)
emb10,h,c = lstm(emb10)
encoder_states = [h,c]
emb10=tf.transpose(emb10,perm=[0,2,1])
gat_layer = GATConv(
    channels=W,              # Number of output units (set according to your needs)
    attn_heads=4,             # Number of attention heads
    concat_heads=False,        # Whether to concatenate or average the attention heads
    dropout_rate=0.5,         # Dropout rate for the attention coefficients
    activation='relu'         # Activation function
)

# gat_layer = GATConv(28)([tf.expand_dims(emb10[:,0,:],axis=2), adj_input])
# gat_layer=tf.expand_dims(gat_layer,axis=1)

# for Bi in range (W-1):
#     tz_B = GATConv(28)([tf.expand_dims(emb10[:,Bi,:],axis=2), adj_input])
#     tz_B=tf.expand_dims(tz_B,axis=1)
#     gat_layer=Concatenate(axis=1)([gat_layer,tz_B])
# gat_layer=TimeDistributed(GlobalAveragePooling1D())(gat_layer)

    
# gat_layer = GATConv(100)([emb10, adj_input])

emb10 = gat_layer([emb10, adj_input])
emb10=tf.transpose(emb10,perm=[0,2,1])

decoder_lstm = LSTM(hh, return_sequences=True, return_state=True)
emb10, _, _ = decoder_lstm(emb10,initial_state=encoder_states)

social=TimeDistributed(Dense(18, kernel_regularizer='l2'))(emb10)
latent1=Concatenate(axis=-1)([social,emb0[0][0]])#m1
latent2=Concatenate(axis=-1)([social,emb0[1][0]])#m2

decoder_lstm1 = LSTM(lstm_len, return_sequences=True, return_state=True)
latent1, _, _ = decoder_lstm1(latent1,initial_state=[emb0[0][1],emb0[0][2]])
decoder_lstm2 = LSTM(lstm_len, return_sequences=True, return_state=True)
latent2, _, _ = decoder_lstm2(latent2,initial_state=[emb0[1][1],emb0[1][2]])

output1=TimeDistributed(Dense(8, kernel_regularizer='l2'))(latent1)#m1
output2=TimeDistributed(Dense(8, kernel_regularizer='l2'))(latent2)#m2

[output2,_],loss1=MSE_UNSUP1()([output2,input1[1]])#m2->m1
[output1,_],loss2=MSE_UNSUP2()([output1,input1[0]])#m1->m2


linearmodel= Model(inputs=[input0,input1,adj_input], outputs=[[output1,output2]])

callbacks=[LearningRateScheduler(lambda epoch: 0.001 * 0.85 ** (epoch // 100))]
term=tf.keras.callbacks.TerminateOnNaN()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,clipvalue=1.0)
batch_size=64

linearmodel.add_metric(loss1, "lossM2")
linearmodel.add_metric(loss2, "lossM1")

linearmodel.compile( optimizer=optimizer)
# history1 =linearmodel.fit( data_generator(trainindex, a,batch_size,if_train = True),
#                        batch_size=64, epochs=200,
#                        verbose=1, steps_per_epoch=np.ceil((len(trainindex))/64-1),callbacks=callbacks)
linearmodel.load_weights('newRNNGATmodel_lstmlen_'+str(lstm_len)+'_hidden_unit_'+str(hh)+'_W_'+str(W)+'_F_'+str(F)+'.h5')

name=trainindex

outputs = [
    l.output for l in linearmodel.layers if isinstance(l, layers.Concatenate )
]

intermediate_layer_model = Model(inputs=linearmodel.input,
                                 outputs=outputs)###41 latent 42 label 43 image
for ni in range (1,5):
    latent1=np.zeros((len(list(a[str(ni)]['current'].keys()))//2,W,lstm_len*2))
    latent2=np.zeros((len(list(a[str(ni)]['current'].keys()))//2,W,18+hh))
    latent3=np.zeros((len(list(a[str(ni)]['current'].keys()))//2,W,18+hh))
    for n in range (len(list(a[str(ni)]['current'].keys()))//2):
        X = {j:[] for j in range (2)}
        Y = {j:[] for j in range (2)}


        x = np.array(a[str(ni)]['current'][str(n)])  # read dataset on the fly
        y = np.array(a[str(ni)]['future'][str(n)]) # read dataset on the fly



        X[0].append(x[:,:8])
        Y[0].append(y[:,:8])
        X[1].append(x[:,8:])
        Y[1].append(y[:,8:])
        

        X = [np.asarray(X[j]) for j in X]
        Y = [np.asarray(Y[j]) for j in Y]
        A=np.ones((X[0].shape[0],hh,hh))
        latent1[n],latent2[n],latent3[n]=intermediate_layer_model.predict([X,Y,A])

        np.save('latent1_lstmlen_'+str(lstm_len)+'_hidden_unit_'+str(hh)+'_W_'+str(W)+'_F_'+str(F)+'_data_'+str(ni),latent1)
        np.save('latent2_lstmlen_'+str(lstm_len)+'_hidden_unit_'+str(hh)+'_W_'+str(W)+'_F_'+str(F)+'_data_'+str(ni),latent2)
        np.save('latent3_lstmlen_'+str(lstm_len)+'_hidden_unit_'+str(hh)+'_W_'+str(W)+'_F_'+str(F)+'_data_'+str(ni),latent3)