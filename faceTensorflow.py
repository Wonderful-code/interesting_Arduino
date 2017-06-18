#faceTensorflow.py
import tensorflow as tf  
import cv2  
import numpy as np  
import os  
from sklearn.model_selection import train_test_split  
import random  
import sys  
   
my_image_path = 'my_faces'  
others_image_path = 'other_people'  
   
image_data = []  
label_data = []  
   
def get_padding_size(image):  
    h, w, _ = image.shape  
    longest_edge = max(h, w)  
    top, bottom, left, right = (0, 0, 0, 0)  
    if h < longest_edge:  
        dh = longest_edge - h  
        top = dh // 2  
        bottom = dh - top  
    elif w < longest_edge:  
        dw = longest_edge - w  
        left = dw // 2  
        right = dw - left  
    else:  
        pass  
    return top, bottom, left, right  
   
def read_data(img_path, image_h=64, image_w=64):  
    for filename in os.listdir(img_path):  
        if filename.endswith('.jpg'):  
            filepath = os.path.join(img_path, filename)  
            image = cv2.imread(filepath)  
   
            top, bottom, left, right = get_padding_size(image)  
            image_pad = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])  
            image = cv2.resize(image_pad, (image_h, image_w))  
   
            image_data.append(image)  
            label_data.append(img_path)  
   
read_data(others_image_path)  
read_data(my_image_path)  
   
image_data = np.array(image_data)  
label_data = np.array([[0,1] if label == 'my_faces' else [1,0] for label in label_data])  
   
train_x, test_x, train_y, test_y = train_test_split(image_data, label_data, test_size=0.05, random_state=random.randint(0, 100))  
   
# image (height=64 width=64 channel=3)  
train_x = train_x.reshape(train_x.shape[0], 64, 64, 3)  
test_x = test_x.reshape(test_x.shape[0], 64, 64, 3)  
   
# nomalize  
train_x = train_x.astype('float32') / 255.0  
test_x = test_x.astype('float32') / 255.0  
   
print(len(train_x), len(train_y))  
print(len(test_x), len(test_y))  
   
#############################################################  
batch_size = 128  
num_batch = len(train_x) // batch_size  
   
X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # 图片大小64x64 channel=3  
Y = tf.placeholder(tf.float32, [None, 2])  
   
keep_prob_5 = tf.placeholder(tf.float32)  
keep_prob_75 = tf.placeholder(tf.float32)  
   
def panda_joke_cnn():  
   
    W_c1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))  
    b_c1 = tf.Variable(tf.random_normal([32]))  
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))  
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv1 = tf.nn.dropout(conv1, keep_prob_5)  
   
    W_c2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))  
    b_c2 = tf.Variable(tf.random_normal([64]))  
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, W_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))  
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv2 = tf.nn.dropout(conv2, keep_prob_5)  
   
    W_c3 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))  
    b_c3 = tf.Variable(tf.random_normal([64]))  
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, W_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))  
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  
    conv3 = tf.nn.dropout(conv3, keep_prob_5)  
   
    # Fully connected layer  
    W_d = tf.Variable(tf.random_normal([8*16*32, 512], stddev=0.01))  
    b_d = tf.Variable(tf.random_normal([512]))  
    dense = tf.reshape(conv3, [-1, W_d.get_shape().as_list()[0]])  
    dense = tf.nn.relu(tf.add(tf.matmul(dense, W_d), b_d))  
    dense = tf.nn.dropout(dense, keep_prob_75)  
   
    W_out = tf.Variable(tf.random_normal([512, 2], stddev=0.01))  
    b_out = tf.Variable(tf.random_normal([2]))  
    out = tf.add(tf.matmul(dense, W_out), b_out)  
    return out  
   
def train_cnn():  
    output = panda_joke_cnn()  
   
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, Y))  
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)  
   
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1)), tf.float32))  
   
    tf.summary.scalar("loss", loss)  
    tf.summary.scalar("accuracy", accuracy)  
    merged_summary_op = tf.summary.merge_all()  
   
    saver = tf.train.Saver()  
    with tf.Session() as sess:  
        sess.run(tf.global_variables_initializer())  
   
        summary_writer = tf.summary.FileWriter('./log', graph=tf.get_default_graph())  
   
        for e in range(50):  
            for i in range(num_batch):  
                batch_x = train_x[i*batch_size : (i+1)*batch_size]  
                batch_y = train_y[i*batch_size : (i+1)*batch_size]  
                _, loss_, summary = sess.run([optimizer, loss, merged_summary_op], feed_dict={X: batch_x, Y: batch_y, keep_prob_5:0.5, keep_prob_75: 0.75})  
   
                summary_writer.add_summary(summary, e*num_batch+i)  
                print(e*num_batch+i, loss_)  
   
                if (e*num_batch+i) % 100 == 0:  
                    acc = accuracy.eval({X: test_x, Y: test_y, keep_prob_5:1.0, keep_prob_75: 1.0})  
                    print(e*num_batch+i, acc)  
                    # save model  
                    if acc > 0.98:  
                        saver.save(sess, "i_am_a_joke.model", global_step=e*num_batch+i)  
                        sys.exit(0)  
   
train_cnn()  



output = panda_joke_cnn()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('.'))  
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={X: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  
   
face_haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
cam = cv2.VideoCapture(0)  
   
while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    faces = face_haar.detectMultiScale(gray_image, 1.3, 5)  
    for face_x,face_y,face_w,face_h in faces:  
        face = img[face_y:face_y+face_h, face_x:face_x+face_w]  
   
        face = cv2.resize(face, (64, 64))  
   
        print(is_my_face(face))  
   
        cv2.imshow('img', face)  
        key = cv2.waitKey(30) & 0xff  
        if key == 27:  
            sys.exit(0)  
   
sess.close()  

'''
资料
sklearn.model_selection.train_test_split随机划分训练集和测试集
官网文档：http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
一般形式：
train_test_split是交叉验证中常用的函数，功能是从样本中随机的按比例选取train data和testdata，形式为：
X_train,X_test, y_train, y_test =
cross_validation.train_test_split(train_data,train_target,test_size=0.4, random_state=0)
参数解释：
train_data：所要划分的样本特征集
train_target：所要划分的样本结果
test_size：样本占比，如果是整数的话就是样本的数量
random_state：是随机数的种子。
随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。
示例
[python] view plain copy
fromsklearn.cross_validation import train_test_split  
train= loan_data.iloc[0: 55596, :]  
test= loan_data.iloc[55596:, :]  
# 避免过拟合，采用交叉验证，验证集占训练集20%，固定随机种子（random_state)  
train_X,test_X, train_y, test_y = train_test_split(train,  
                                                   target,  
                                                   test_size = 0.2,  
                                                   random_state = 0)  
train_y= train_y['label']  
test_y= test_y['label']  
'''