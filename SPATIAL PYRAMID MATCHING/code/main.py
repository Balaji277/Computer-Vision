from os.path import join

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import util
import visual_words
import visual_recog
import imageio
from opts import get_opts


def main():
    opts = get_opts()

    # Q1.1
    img_path = join(opts.data_dir,"C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\aquarium\\sun_aztvjgubyrgvirup.jpg")
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32) / 255
    filter_responses = visual_words.extract_filter_responses(opts, img)
    util.display_filter_responses(opts, filter_responses) 

    # Q1.2
    n_cpu = util.get_num_CPU()
    visual_words.compute_dictionary(opts, n_worker=n_cpu)

    # Q1.3
    img_path = join(opts.data_dir, "C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\waterfall\\sun_adtiwvbnsxsyohuw.jpg")
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    dictionary = np.load(join(opts.out_dir, 'C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\Dictionary\\dictionary.npy'))
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    util.visualize_wordmap(wordmap)

    # Q2.1-2.4
    n_cpu = util.get_num_CPU()
    visual_recog.build_recognition_system(opts, n_worker=n_cpu)

    # Q2.5
    n_cpu = util.get_num_CPU()
    conf, accuracy ,test, predict, miss= visual_recog.evaluate_recognition_system(opts, n_worker=n_cpu)
    key=list(miss.keys()) 
    img1=imageio.imread(miss[key[0]])
    plt.imshow(img1)
    img2=imageio.imread(miss[key[1]])
    plt.imshow(img2)
    img3=imageio.imread(miss[key[2]])
    plt.imshow(img3)
    np.savetxt(join(opts.out_dir, 'C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\confmat.csv'), conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'C:\\Users\\balub\\Downloads\\hw1-5 (2)\\hw1\\data\\accuracy.txt'), [accuracy], fmt='%g')


if __name__ == '__main__':
    main()
