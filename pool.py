import random
import tensorflow as tf

class Pool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_items = 0
            self.items = []
    
    def query(self, new_items):
        if self.pool_size == 0:
            return new_items
        return_items = []
        for i in range(new_items.shape[0]):
            item = new_items[i,:,:,:]
            if self.num_items < self.pool_size:
                self.num_items += 1
                self.items.append(item)
                return_items.append(item)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = tf.identity(self.items[random_id])
                    self.items[random_id] = item
                    return_items.append(tmp)
                else:
                    return_items.append(item)
        return tf.stack(return_items)
        
    