# IMPORT
# ==================================================================================================
import sys

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane')
from Settings import *

sys.path.insert(0, r'C:\Users\OM\Desktop\LOL-Autolane\LOL-Autolane\Data_Set')
import data_generator as dg
import data_labeller as dl
import data_set as ds

win_unicode_console.enable()


# MAIN CLASS
# ==================================================================================================
class Model(object):

    def __init__(self, sess):

        self.sess = sess

        # Specifies image parameters
        self.image_height = 300
        self.image_width = 400
        self.num_parts_x = 20
        self.num_parts_y = 20

        self.images = tf.placeholder(tf.float32,
                                     [None, self.image_height, self.image_width, 3], name='images')
        self.labels = tf.placeholder(tf.float32,
                                     [None, self.num_parts_y, self.num_parts_x, 1], name='labels')

        # Specifies training parameters
        self.num_epoch = 5000
        self.batch_size = 10
        self.learning_rate = 1e-5
        self.drop_prob = 0.25
        self.num_train = 380
        self.num_valid = 20
        self.num_test = 0

        # Creates the model
        self.pred = self.conv_net()

        # Creating the loss function
        eps = 1e-4
        self.loss = -tf.reduce_sum(15 * self.labels * tf.log(self.pred + eps)
                                   + (1 - self.labels) * tf.log(1 - (self.pred) + eps))
        self.saver = tf.train.Saver()

        if (real_time_flag == False) and (train_flag == True) and (test_flag == False):

            print('Loading images...')
            self.images_dataset = ds.load_images_from_dataset()

            print('Loading label matrices...')
            self.label_matrix = dl.load_matrix()

            print('Preparing data...')

            self.shuffled_images, \
            self.shuffled_label_matrix = ds.prepare_data_set(self.images_dataset,
                                                             self.label_matrix,
                                                             self.num_train,
                                                             self.num_valid,
                                                             self.num_test)

            self.shuffled_images[0] = np.array(self.shuffled_images[0]).astype(np.float32)
            self.shuffled_images[1] = np.array(self.shuffled_images[1]).astype(np.float32)
            # noinspection PyInterpreter
            self.shuffled_label_matrix[0] = (
                np.array(self.shuffled_label_matrix[0]).astype(np.float32)).reshape(
                (-1, self.num_parts_y, self.num_parts_x, 1))
            self.shuffled_label_matrix[1] = (
                np.array(self.shuffled_label_matrix[1]).astype(np.float32)).reshape(
                (-1, self.num_parts_y, self.num_parts_x, 1))

    # Create the network architecture (we feed this into the train function)
    def conv_net(self):

        x = self.images

        # kw = filter sizes for x
        # kw = filter sizes for y
        # p = pool size
        # s = strides
        kw = [3, 2, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        kh = [3, 2, 3, 2, 3, 1, 3, 2, 3, 1, 3, 2, 3, 1, 3, 1, 3, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        kw[19] = 3
        kw[20] = 3
        kw[21] = 3
        kw[22] = 3
        kw[23] = 3
        kw[24] = 3
        kw[25] = 2
        p = 0
        s = [1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        # TODO: Add comment for this
        initializer = tf.contrib.layers.xavier_initializer_conv2d()

        # Input is a 300x400x3 size image
        input = tf.reshape(x, shape=[-1, self.image_height, self.image_width, 3])

        conv0 = tf.layers.conv2d(input, 32, (kh[0], kw[0]), s[0], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        mp1 = tf.layers.max_pooling2d(conv0, (kh[1], kw[1]), s[1])

        conv2 = tf.layers.conv2d(mp1, 64, (kh[2], kw[2]), s[2], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        mp3 = tf.layers.max_pooling2d(conv2, (kh[3], kw[3]), s[3])

        conv4 = tf.layers.conv2d(mp3, 128, (kh[4], kw[4]), s[4], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        conv5 = tf.layers.conv2d(conv4, 64, (kh[5], kw[5]), s[5], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        conv6 = tf.layers.conv2d(conv5, 128, (kh[6], kw[6]), s[6], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        mp7 = tf.layers.max_pooling2d(conv6, (kh[7], kw[7]), s[7])

        conv8 = tf.layers.conv2d(mp7, 256, (kh[8], kw[8]), s[8], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        conv9 = tf.layers.conv2d(conv8, 128, (kh[9], kw[9]), s[9], activation=tf.nn.relu,
                                 kernel_initializer=initializer, bias_initializer=initializer)

        conv10 = tf.layers.conv2d(conv9, 256, (kh[10], kw[10]), s[10], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        mp11 = tf.layers.max_pooling2d(conv10, (kh[11], kw[11]), s[11])

        conv12 = tf.layers.conv2d(mp11, 512, (kh[12], kw[12]), s[12], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv13 = tf.layers.conv2d(conv12, 256, (kh[13], kw[13]), s[13], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv14 = tf.layers.conv2d(conv13, 512, (kh[14], kw[14]), s[14], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv15 = tf.layers.conv2d(conv14, 256, (kh[15], kw[15]), s[15], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv16 = tf.layers.conv2d(conv15, 512, (kh[16], kw[16]), s[16], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        mp17 = tf.layers.max_pooling2d(conv16, (kh[17], kw[17]), s[17])

        conv18 = tf.layers.conv2d(mp17, 1024, (kh[18], kw[18]), s[18], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv19 = tf.layers.conv2d(conv18, 512, (kh[19], kw[19]), s[19], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv20 = tf.layers.conv2d(conv19, 1024, (kh[20], kw[20]), s[20], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv21 = tf.layers.conv2d(conv20, 512, (kh[21], kw[21]), s[21], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv22 = tf.layers.conv2d(conv21, 1024, (kh[22], kw[22]), s[22], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv23 = tf.layers.conv2d(conv22, 1024, (kh[23], kw[23]), s[23], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv24 = tf.layers.conv2d(conv23, 1024, (kh[24], kw[24]), s[24], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv25 = tf.layers.conv2d(conv24, 64, (kh[25], kw[25]), s[25], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        conv26 = tf.layers.conv2d(conv25, 1024, (kh[26], kw[26]), s[26], activation=tf.nn.relu,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        output = tf.layers.conv2d(conv26, 1, (kh[27], kw[27]), s[27], activation=tf.sigmoid,
                                  kernel_initializer=initializer, bias_initializer=initializer)

        return output

    def runRealTime(self):

        self.sess.run(tf.global_variables_initializer())

        self.load(checkpoint_dir)

        print("Starting Realtime Running...")

    def train(self):

        # Stochastic gradient descent with the standard backpropagation
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

        counter = 0
        time_ = time.time()

        self.load(checkpoint_dir)

        # Train
        print("Now Start Training...")

        # Run by batch images
        num_batches = self.num_train // self.batch_size
        print('Number mini-batch feeds required: ', num_batches)

        for epoch in range(self.num_epoch):  # Iterating across epochs



            for batch_num in range(0, num_batches):  # Iterating across batches
                batch_images = self.shuffled_images[0][batch_num *
                               self.batch_size: (batch_num + 1) * self.batch_size]
                batch_labels = self.shuffled_label_matrix[0][batch_num *
                               self.batch_size: (batch_num + 1) * self.batch_size]
                counter += 1
                err, __, prd = self.sess.run([self.loss, self.train_op, self.pred],
                                             feed_dict={self.images: batch_images,
                                                        self.labels: batch_labels})

                train_status = "Epoch Remaining: {} Batches/Epoch: {} " \
                               "Batch: {:02} Step: {:02} Time: {} Loss: {}". \
                               format(self.num_epoch-(epoch + 1),num_batches,batch_num, counter,
                               "%.2f" % (time.time() - time_), "%.2f" % err)
                train_status = train_status.encode("utf-8").decode("ascii")
                print(train_status)

            if counter % 500 == 0:
                self.save(checkpoint_dir, counter)
                print("Saved checkpoint.")

    def checkpoint_dir():

        if real_time_flag == False:
            return os.path.join('./{}'.format(checkpoint_dir), "train.h5")

    def save(self, checkpoint_dir, step):

        model_name = "CNN.model"
        model_dir = "%s" % ("cnn")
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):

        print("\nReading Checkpoints.....\n\n")
        model_dir = "%s" % ("cnn")  # give the model name by label_size
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)

        # Check the checkpoint is exist
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_path = str(checkpoint.model_checkpoint_path)  # convert the unicode to string
            self.saver.restore(self.sess, os.path.join(os.getcwd(), checkpoint_path))
            print("\n Checkpoint Loading Success! %s\n\n" % checkpoint_path)
        else:
            print("\n! Checkpoint Loading Failed \n\n")

