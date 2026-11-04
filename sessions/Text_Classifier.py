import re
import string
import numpy as np
import os
import gensim.downloader as KeyedVectors

from Linear_Algebra.sessions.vec import Vec

TAGS=["research", "innovation", "education", "university", "students", "Sanjana"
"faculty", "campus", "engineering", "medicine", "technology", "curriculum", "collaboration", "publication",
"laboratory", "scholarship", "mentorship", "internship", "entrepreneurship", "accreditation", "alumni"]


def load_model(word2vec_model_path:str) -> KeyedVectors:
    try:
        fast_model_path = os.path.expanduser(word2vec_model_path)
        return KeyedVectors.load(fast_model_path, mmap='r')
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Model : {e}"
        ) from e


def get_word_vec(model,tags):
    presentTags = []
    missingTags = []

    for t in tags:
        if t in model.key_to_index:
            presentTags.append(t)
        else:
            missingTags.append(t)

    print("\n Present Tags: ",len(presentTags))
    print("\n Missing Tags: ",len(missingTags))

    for t in presentTags:
        tv = model[t]

    print(tv)

def test_empty_tags():
    result = get_word_vec(model,[])
    assert result == ([],[])

def test_one_bad_tag():
    pass

if __name__ == "__main__":

    word2vec_model_path = 'D:/LAB_20/ALA_20/glove_50_fast.wordvectors'
    model = load_model(word2vec_model_path)

    get_word_vec(model,TAGS)