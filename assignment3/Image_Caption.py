'''
Description: Copyright © 1999 - 2020 Winter. All Rights Reserved. 

             Finished RNN_Caption.ipynb and LSTM_Caption.ipynb here.
             Implement image caption with RNN and LSTM here.
             
Author: Winter
Email: 837950571@qq.com
Date: 2020-12-14 16:05:04
LastEditTime: 2020-12-20 16:37:54
'''
import numpy as np
import matplotlib.pyplot as plt
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url
from rnn import CaptioningRNN
from captioning_solver import CaptioningSolver

if __name__ == '__main__':
    data = load_coco_data(pca_features=True)
    # np.random.seed(231)

    # small_data = load_coco_data(max_train=50)

    # small_rnn_model = CaptioningRNN(
    #     cell_type='rnn',
    #     word_to_idx=data['word_to_idx'],
    #     input_dim=data['train_features'].shape[1],
    #     hidden_dim=512,
    #     wordvec_dim=256,
    # )

    # small_rnn_solver = CaptioningSolver(
    #     small_rnn_model,
    #     small_data,
    #     update_rule='adam',
    #     num_epochs=50,
    #     batch_size=25,
    #     optim_config={
    #         'learning_rate': 5e-3,
    #     },
    #     lr_decay=0.95,
    #     verbose=True,
    #     print_every=10,
    # )

    # small_rnn_solver.train()

    # # Plot the training losses
    # plt.plot(small_rnn_solver.loss_history)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Training loss history')
    # plt.show()

    # for split in ['train', 'val']:
    #     minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
    #     gt_captions, features, urls = minibatch
    #     gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    #     sample_captions = small_rnn_model.sample(features)
    #     sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    # for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
    #     plt.imshow(image_from_url(url))
    #     plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
    #     plt.axis('off')
    #     plt.show()
    np.random.seed(231)

    small_data = load_coco_data(max_train=50)

    small_lstm_model = CaptioningRNN(
            cell_type='lstm',
            word_to_idx=data['word_to_idx'],
            input_dim=data['train_features'].shape[1],
            hidden_dim=512,
            wordvec_dim=256,
            dtype=np.float32,
            )

    small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,
            update_rule='adam',
            num_epochs=50,
            batch_size=25,
            optim_config={
                'learning_rate': 5e-3,
            },
            lr_decay=0.995,
            verbose=True, print_every=10,
            )

    small_lstm_solver.train()

    # Plot the training losses
    plt.plot(small_lstm_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.show()
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = small_lstm_model.sample(features)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()