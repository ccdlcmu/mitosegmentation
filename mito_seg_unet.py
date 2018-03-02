'''unet modified from Ronneberger.et al 2015'''
'''mitochondria segmenation '''

import numpy as np
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import merge 
from keras.layers import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

class myUnet(object):

	def __init__(self, img_rows = 200, img_cols = 200):

		self.img_rows = img_rows
		self.img_cols = img_cols

	def load_data(self):

		# load the data you want to train the Unet on
		# imgs_all = None
		# img_max_thr = imgs_all.max()
		# imgs_all = imgs_all/img_max_thr

		# imgs_mask_all = None
		# imgs_mask_all[imgs_mask_all>0]=1
		# imgs_mask_all[imgs_mask_all<=0]=0

		# imgs_train = imgs_all[:600,:,:,:]
		# imgs_mask_train = imgs_mask_all[:600,:,:,:]
		# imgs_test = imgs_all[600:,:,:,:]
		# imgs_mask_test = imgs_mask_all[600:,:,:,:]

		return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test 

	def get_unet(self):

		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(64, kernel_size=(3,3), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		drop3 = Dropout(0.5)(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop4))
		merge7 = concatenate([conv3,up7], axis = 3) 
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = 3) 
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) 
		merge9 = concatenate([conv1,up9], axis = 3) 
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9) 
		
		model = Model(inputs = inputs, outputs = conv10) 
		model.compile(optimizer = SGD(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
		
		return model

	def run_unet(self):

		print("loading data")
		imgs_train, imgs_mask_train, imgs_test,imgs_mask_test = self.load_data()
		print("loading data done")
		model = self.get_unet()
		print("got unet")

		# training
		model_checkpoint = ModelCheckpoint('./test_result/unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=15, verbose=1, shuffle=True, callbacks=[model_checkpoint])
		
		# test
		# model.load_weights('./test_result/unet.hdf5')
		# print('load weights done')

		test_evaluation = model.evaluate(imgs_test,imgs_mask_test, batch_size=1, verbose=1)
		print test_evaluation
		test_loss, test_acc = test_evaluation
		print("test_loss: ",test_loss)
		print("test_acc: ",test_acc)

		print('predict test data')
		test_predict = model.predict(imgs_test, batch_size=1, verbose=1)
		
		np.save('./test_result/unet_test_predict.npy',test_predict)


if __name__ == '__main__':
	myunet = myUnet()
	myunet.run_unet()








 