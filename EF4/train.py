import os 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import h5py
import numpy as np
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from networks import TransformerNet,RegNet,ClassNet,OutNet,PoolingByMultiHeadAttention

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model_name = "test.h5"
epochs     = 1
batch_size = 64

data_dir_train = 'Data-MLtutorial/JetDataset/train/'
data_dir_val = 'Data-MLtutorial/JetDataset/val/'
target_onehot = np.array([])
jetList = np.array([])
jetImages = np.array([])
target_reg = np.array([])
features_names = dict()

#dirs = [data_dir_train,data_dir_val]
dirs = [data_dir_train]

for d in dirs:
    #datafiles =  os.listdir(d)
    datafiles = ["jetImage_1_100p_60000_70000.h5","jetImage_1_100p_70000_80000.h5"]
    for i_f,fileIN in enumerate(datafiles):
        print("Appending %s" %fileIN)
        f = h5py.File(d + fileIN)
        jetList_file = np.array(f.get("jetConstituentList"))
        if len(jetList_file.shape) != 3: continue
        target_file = np.array(f.get('jets')[0:,-6:-1])
        mass = np.array(f.get("jets")[0:,3])    
        jetImages_file = np.array(f.get('jetImage'))
        jetList = np.concatenate([jetList, jetList_file], axis=0) if jetList.size else jetList_file
        target_onehot = np.concatenate([target_onehot, target_file], axis=0) if target_onehot.size else target_file
        jetImages = np.concatenate([jetImages, jetImages_file], axis=0) if jetImages.size else jetImages_file
        target_reg = np.concatenate([target_reg,mass],axis=0)
        del jetList_file, target_file, jetImages_file, mass
        #save particles/nodes features names and their indecies in a dictionary
        if i_f==0:
            for feat_idx,feat_name in enumerate(list(f['particleFeatureNames'])[:-1]):
                features_names[feat_name.decode("utf-8").replace('j1_','')] = feat_idx
        f.close()

target = np.argmax(target_onehot, axis=1)
num_classes = len(np.unique(target))
label_names= ["gluon", "quark", "W", "Z", "top"]

print('Jets shape : ',jetList.shape)
print('Target/Labels shape : ',target.shape)
print('Particles/Nodes features : ',list(features_names.keys()))

labelCat = ["gluon", "quark", "W", "Z", "top"]
max_mass = np.max(target_reg)
target_reg /= max_mass

learning_rate=0.001
num_heads = 8
hidden_dimensions = 64

t_net = TransformerNet(num_heads=num_heads, hidden_units=hidden_dimensions)
r_net = RegNet(hidden_dimensions)
c_net = ClassNet(hidden_dimensions)

inputs = keras.Input(shape=(100,16), name='input')
output = layers.TimeDistributed(layers.Dense(hidden_dimensions))(inputs)
output = t_net(output)
output = layers.Lambda(lambda y: tf.reduce_sum(y, axis=1))(output)
output = [r_net(output), c_net(output)]

model = keras.models.Model(inputs=inputs, outputs=output)
model.summary()

model.compile(
    loss=["mse", keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=keras.metrics.SparseCategoricalAccuracy(),
    loss_weights=[0.5,0.5])

X_train, X_val, y_train_class, y_val_class, y_train_reg, y_val_reg, y_train_onehot, y_val_onehot = train_test_split(jetList, target,target_reg, target_onehot, test_size=0.1, shuffle=True)
print(X_train.shape, X_val.shape, y_train_class.shape, y_val_class.shape, y_train_reg.shape, y_val_reg.shape)
del jetList, target, target_onehot

history = model.fit(x=X_train, 
                    y=[y_train_reg,y_train_class], 
                    validation_data=(X_val,[y_val_reg,y_val_class]), 
                    batch_size=batch_size, 
                    epochs=epochs,
                    verbose=1,
                    callbacks = [ EarlyStopping(monitor='val_loss', patience=10, verbose=1),
                                  ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1),
                                  TerminateOnNaN()]
                   )

model.save(model_name)

keys = ["loss","Reg_loss","Class_loss","Class_sparse_categorical_accuracy"]
fig,axes = plt.subplots(2,2, figsize=(20,20))
for ax, key in zip(axes.flat, keys):
    ax.plot([i for i in range(len(history.history[key]))],history.history[key],label=key)
    ax.plot([i for i in range(len(history.history[key]))],history.history["val_"+key],label="val_"+key)
    ax.legend()

fig.savefig("plot/history.png")

get_ipython().run_line_magic('matplotlib', 'inline')
predict_val = tf.nn.softmax(model.predict(X_val)[1]) # index 0 is the regression predictio
df = pd.DataFrame()
fpr = {}
tpr = {}
auc1 = {}

plt.figure()
for i, label in enumerate(label_names):

        df[label] = y_val_onehot[:,i]
        df[label + '_pred'] = predict_val[:,i]

        fpr[label], tpr[label], threshold = roc_curve(df[label],df[label+'_pred'])

        auc1[label] = auc(fpr[label], tpr[label])

        plt.plot(fpr[label],tpr[label],label='%s tagger, auc = %.1f%%'%(label,auc1[label]*100.))

plt.ylabel("sig. efficiency")
plt.xlabel("bkg. mistag rate")
plt.ylim(0.000001,1)
plt.grid(True)
plt.legend(loc='lower right')
plt.savefig("plots/auc.png")

predict_reg = model.predict(X_val)[0].flatten()
plt.hist(((predict_reg-y_val_reg)/y_val_reg)*max_mass,50,(-200,200),density=True,histtype="step")
plt.xlabel("Predicted  - True  / True ")
plt.ylabel('Prob. Density (a.u.)', fontsize=15)
plt.savefig("plots/reg.png")

confusion_mat = confusion_matrix(np.argmax(predict_val,axis=1),y_val_class)
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes))
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig("plots/conf.png")


