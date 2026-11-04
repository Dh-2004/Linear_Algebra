import numpy as np

def attention(x, w):
    wq, bq = w
    return wq @ x + bq

def testattention():
    rng = np.random.default_rng(42)

    # Random input: 3 features, 4 tokens
    x = rng.standard_normal((3, 4))

    # Random weights and bias
    wq = rng.standard_normal((2, 3))
    bq = rng.standard_normal((2, 1))

    q = attention(x, (wq, bq))

    # Assertions
    assert wq.shape == (2, 3)
    assert bq.shape == (2, 1)
    assert q.shape == (2, 4)
    np.testing.assert_allclose(q, wq @ x + bq)

    print("X:\n", x)
    print("Wq:\n", wq)
    print("bq:\n", bq)
    print("Q:\n", q)
    print("All tests passed!")

testattention()