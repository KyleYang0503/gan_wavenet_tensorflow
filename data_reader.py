# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 20:07:47 2021

@author: Kyle
"""
import os
import threading
import numpy as np
import tensorflow as tf

def find_files(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def _read_text(filename):
  with open(filename, "r") as f:
    return list(f.read().split(','))

def load_generic_text(directory):
    files = find_files(directory)
    for filename in files:
        text = _read_text(filename)        
    
        for index, item in enumerate(text):
            text[index] = text[index]
        text = np.array(text, dtype='int8')
        text = text.reshape(-1, 1)
        yield text, filename

class TextReader(object):
    def __init__(self,
                 text_dir,
                 coord,
                 sample_size=None,
                 queue_size=256):
        
        self.text_dir = text_dir
        self.coord = coord
        self.sample_size = sample_size
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.int8, shape=None)
        self.queue = tf.queue.PaddingFIFOQueue(queue_size,
                                         ['int8'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output
    
    def thread_main(self, sess):
        buffer_ = np.array([],dtype='int8')
        stop = False
        while not stop:
            iterator = load_generic_text(self.text_dir)
            for text, filename in iterator:
                if self.coord.should_stop():
                    self.stop_threads()
                    stop = True
                    break
                # Cut samples into fixed size pieces
                buffer_ = np.append(buffer_, text)
                while len(buffer_) > self.sample_size:
                    piece = buffer_[:self.sample_size]
                    feed_dict={self.sample_placeholder: piece}
                    print(feed_dict)
                    sess.run(self.enqueue,
                             feed_dict)
                    buffer_ = buffer_[self.sample_size:]
                    
    def stop_threads(self):
        for t in self.threads:
            t.requese_stop()

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  
            thread.start()
            self.threads.append(thread)
        return self.threads
