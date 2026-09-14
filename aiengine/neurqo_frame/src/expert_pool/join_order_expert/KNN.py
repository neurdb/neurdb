import random


class KNN:
    def __init__(
        self,
        k_neighbours,
        method="brute",
        max_distance=0.05**2 + 0.05**2 + 0.05**2 + 10,
        max_v=4,
    ):
        """[summary]

        Args:
            k_neighbours ([int]): [number of neighbours]
            method (str, optional): [description]. Defaults to 'brute'.
        """
        self.k_neighbours = k_neighbours
        self.max_distance = max_distance
        self.max_v = max_v
        self.method = method
        if method == "brute":
            self.kvs = []
        if method == "kd-tree":
            pass

    def distance(self, v1, v2):
        # print(zip(v1,v2))
        return sum([abs(v[0] - v[1]) for v in zip(v1[1:2], v2[1:2])])

    def insertValues(self, data):
        """[summary]

        Args:
            data ([list]): [description]
        """
        if self.method == "brute":
            self.kvs.extend(data)

    def insertAValue(self, data):
        """[summary]

        Args:
            data ([list]): [description]
        """
        data = (data[0], abs(data[1]))
        self.kvs.append(data)

    def kNeighbours(self, v, k_neighbours=0):
        if k_neighbours == 0:
            k_neighbours = self.k_neighbours
        chosen_data = sorted(
            [(self.distance(v, x[0]), x[1]) for x in self.kvs], key=lambda x: x[0]
        )[:k_neighbours]
        if len(self.kvs) < k_neighbours:
            return []
        return chosen_data

    def kNeightboursSample(self, v, k_neighbours=0):
        k_neighbours = self.kNeighbours(v, k_neighbours)
        if len(k_neighbours) == 0 or k_neighbours[-1][0] > self.max_distance:
            return self.max_v
        return random.sample(k_neighbours, 1)[0][1]
