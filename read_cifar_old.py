import numpy as np
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as file:
        data_dict = pickle.load(file, encoding='bytes')
    images = data_dict[b'data']
    labels = data_dict[b'labels']
    return images, labels

# def load_cifar10_data(folder_path):
#     images = []
#     labels = []
#     for i in range(1, 6):
#         file_path = folder_path + '/data_batch_' + str(i) + '.bin'
#         batch_images, batch_labels = load_cifar10_batch(file_path)
#         images.append(batch_images)
#         labels.extend(batch_labels)
#     images = np.concatenate(images, axis=0)
#     return images, labels

train_images, train_labels = load_cifar10_batch('./cifar/data_batch_1')
train_images = train_images.reshape((10000, 3, 32, 32))#, order='A')
# test_images, test_labels = load_cifar10_batch('path_to_folder/test_batch.bin')


label_dict = unpickle("./cifar/batches.meta")
# print(type(label_dict))
# print("train_images:", type(train_images), train_images.shape)
# print("train_labels:", type(train_labels), len(train_labels))
# print("label_dict:", label_dict)


i = 101
# print(train_images[0].shape)
# print(train_labels[i])
plt.imshow(np.transpose(train_images[i],(1,2,0)))
# plt.show()

def segment(image, N=2):
    # with batches
    '''
    Args:
        image: ndarray with shape (batch_size*3*32*32)
        N: to segment the image to N*N blocks
    '''
    assert 32 % N == 0, "The target size is illegal."

    batch_size, channels, height, width = image.shape
    assert channels == 3, "Number of channels is not 3."
    assert height == 32, "Height is not 32."
    assert width == 32, "Width is not 32."

    block_height = 32 // N
    block_width = 32 // N
    blocks_sorted = image.reshape(batch_size, channels, N, block_height, N, block_width)
    # blocks_sorted = blocks_sorted.transpose(0,1,2,4,3,5)    # batch_size * 3 * N * N * (32//N) * (32//N)
    blocks_sorted = blocks_sorted.transpose(0,2,4,1,3,5)    # batch_size * N * N * 3 * (32//N) * (32//N)
    
    blocks_sorted = blocks_sorted.reshape(batch_size,N*N,3,32//N,32//N)

    blocks_shuffled = np.zeros((batch_size,N*N,3,32//N,32//N), dtype=np.uint8)
    labels = np.zeros((batch_size, N*N, N*N), dtype=int)
    for img_index in range(batch_size):
        shuffle_indices = np.random.permutation(N*N)
        blocks_shuffled[img_index] = blocks_sorted[img_index, shuffle_indices]
        for i, j in enumerate(shuffle_indices):
            labels[img_index, i, j] = 1

    # print("blocks_shuffled.shape:", blocks_shuffled.shape)

    img_to_show = 101
    print(labels[img_to_show])
    i = 0
    for pieces in range(N*N):
        plt.subplot(N, N, i+1)
        plt.imshow(blocks_shuffled[img_to_show, pieces, :, :, :].transpose(1, 2, 0))
        plt.axis('off')
        i += 1
    plt.show()

segment(train_images, 4)

# for i in range(train_images.shape[0]):
#     segment(train_images[i][np.newaxis,:,:,:])
