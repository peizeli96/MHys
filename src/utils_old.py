import os
import torch
from dictionary import Dictionary
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def tfidf_loading(use_tfidf, w_emb, dataroot):
    tfidf = None
    weights = None

    if use_tfidf:
        dict = Dictionary.load_from_file('%s/dictionary.pkl' % dataroot)
        # load extracted tfidf and weights from file for saving loading time
        if os.path.isfile('%s/embed_tfidf_weights.pkl' % dataroot) == True:
            print("Loading embedding tfidf and weights from file")
            with open('%s/embed_tfidf_weights.pkl' % dataroot, 'rb') as f:
                # tfidf, weights = pickle.load(f)
                w_emb = torch.load(f, map_location=torch.device("cuda"))
            # tfidf = utils.to_sparse(tfidf)
            print("Load embedding tfidf and weights from file successfully")
        else:
            print("Embedding tfidf and weights haven't been saving before")
            # tfidf, weights = dataset.tfidf_from_questions(['train', 'valid', 'testdev'], dict, 'data/dataset')
            w_emb.init_embedding('%s/glove6b_init_300d.npy' % dataroot, tfidf, weights)
            with open('%s/embed_tfidf_weights.pkl' % dataroot, 'wb') as f:
                # pickle.dump((tfidf.to_dense(), weights), f)
                torch.save(w_emb, f)
            print("Saving embedding with tfidf and weights successfully")
    else:
        w_emb.init_embedding('%s/glove6b_init_300d.npy' % dataroot, tfidf, weights)

    return w_emb