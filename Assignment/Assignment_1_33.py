
from typing import Self


class Vec:

    def __init__(self, src=None) -> Self:
        if src is None:
            self.elements = ()
        else:
            self.elements = tuple(src)

    def scalar_mul(self, alpha):
        result = Vec(x * alpha for x in self.elements)
        return result

    def mean(self):
       
        if len(self.elements) == 0:
            raise ValueError("Cannot calculate mean of an empty vector")

        return sum(self.elements) / len(self.elements)

    def demean(self):
        
        mean_value = self.mean()

        return Vec(x - mean_value for x in self.elements)

    def std(self):
        
        demeaned = self.demean()

        squared_deviations = [
            x * x for x in demeaned.elements
        ]

        return (sum(squared_deviations) / len(squared_deviations)) ** 0.5

    def __repr__(self):
        return repr(self.elements)


if __name__ == "__main__":

    v1 = Vec((10, 20, 1.03))

    print("Vector:", v1)

    v2 = v1.scalar_mul(2.2)
    print("Scalar multiplication:", v2)

    print("Mean:", v1.mean())

    print("Demeaned vector:", v1.demean())

    print("Standard deviation:", v1.std())