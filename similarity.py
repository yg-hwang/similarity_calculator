import numpy as np
from tqdm import tqdm
from typing import Union, Optional, Any, List, Dict, Tuple, Set
from scipy.spatial import distance
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


class SimilarityFunctions:
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _compute_similarity(
        self, a: Union[list, dict, np.ndarray], b: Union[list, dict, np.ndarray]
    ):
        return

    def create_similarity_dict(self, data: dict, top_n: Optional[int] = None):
        return


class JacarrdSimilarity(SimilarityFunctions):
    def __init__(self, data: dict):
        super().__init__(data)
        self.data = data

    def _compute_similarity(
        self, a: Union[list, np.ndarray], b: Union[list, np.ndarray]
    ) -> Union[int, float]:
        """
        두 집합의 자카드 유사도 계산
        """
        if (len(a) > 0) | (len(b) > 0):
            set_a, set_b = set(a), set(b)
            intersection = set_a & set_b
            if len(intersection) > 0:
                union = set_a | set_b
                return round(len(intersection) / len(union), 5)
            else:
                return 0
        else:
            raise ValueError("a and b is empty list.")

    def create_similarity_dict(
        self, top_n: Optional[int] = None, progressbar: bool = True
    ) -> dict:
        """
        전체 document 자카드 유사도 계산
        """
        result = {}
        for k1, v1 in tqdm(
            self.data.items(),
            total=len(self.data),
            desc="creating jaccard similarity...",
            disable=False if progressbar else True,
        ):
            similarities = {}
            for k2, v2 in self.data.items():
                if k1 == k2:
                    continue
                else:
                    similarity = self._compute_similarity(a=v1, b=v2)
                    if similarity > 0:
                        similarities[k2] = similarity
                    else:
                        continue
            if top_n is None:
                result[k1] = similarities
            else:
                similarities = dict(
                    sorted(similarities.items(), key=lambda t: t[::-1], reverse=True)[
                        :top_n
                    ]
                )
                result[k1] = similarities
        return result


class EuclideanDistance(SimilarityFunctions):
    def __init__(self, data: dict):
        super().__init__(data)
        self.data = data

    def _compute_similarity(self, a: dict, b: dict) -> Union[int, float]:
        """
        두 집합의 유클리디언 거리 계산
        (두 집합의 원소가 다를 경우, 각 원소의 값을 0으로 할당하여 계산)
        """
        a_keys, a_values = a.keys(), a.values()
        b_keys, b_values = b.keys(), b.values()

        intersection = set(a_keys) & set(b_keys)
        a_only = set(a_keys) - intersection
        b_only = set(b_keys) - intersection

        if intersection:
            A, B = [], []
            for element in intersection:
                A.append(a[element])
                B.append(b[element])
            a_only_cnt, b_only_cnt = len(a_only), len(b_only)
            if a_only_cnt > 0:
                for element in a_only:
                    A.append(a[element])
            for _ in range(b_only_cnt):  # a에 b만큼 0 추가
                A.append(0)

            for _ in range(a_only_cnt):  # b에 a만큼 0 미리 추가
                B.append(0)
            if b_only_cnt > 0:
                for element in b_only:
                    B.append(b[element])
            return distance.euclidean(A, B)
        else:
            return 0

    def create_similarity_dict(
        self, top_n: Optional[int] = None, progressbar: bool = True
    ) -> Dict[Union[int, str], Union[int, float]]:
        """
        전체 document 유클리디언 거리 계산
        """
        result = {}
        for k1, v1 in tqdm(
            self.data.items(),
            total=len(self.data),
            desc="creating euclidean distance...",
            disable=False if progressbar else True,
        ):
            similarities = {}
            for k2, v2 in self.data.items():
                if k1 == k2:
                    continue
                else:
                    similarity = self._compute_similarity(a=v1, b=v2)
                    if similarity > 0:
                        similarities[k2] = similarity
                    else:
                        continue
            if top_n is None:
                result[k1] = similarities
            else:
                similarities = dict(
                    sorted(similarities.items(), key=lambda t: t[::-1], reverse=True)[
                        :top_n
                    ]
                )
                result[k1] = similarities
        return result


class CosineSimilarity(SimilarityFunctions):
    def __init__(self, data: dict):
        super().__init__(data)
        self.data = data

    def _compute_similarity(
        self, a: Union[list, np.ndarray], b: Union[list, np.ndarray]
    ):
        # origin return: array([[x1, x2, x3...]]) --> array([[x1, x2, x3...]])[0]
        return cosine_similarity(X=a, Y=b)[0]

    def create_similarity_dict(
        self, top_n: Optional[int] = None, progressbar: bool = True
    ) -> dict[Any, dict[Any, Any]]:
        """
        전체 document 코사인 유사도 계산
        """
        result = {}
        data_keys = list(self.data.keys())
        data_values = list(self.data.values())
        for k, v in tqdm(
            self.data.items(),
            total=len(self.data),
            disable=False if progressbar else True,
        ):
            k_idx = data_keys.index(k)
            data_keys.pop(k_idx)
            data_values.pop(k_idx)
            similarities = self._compute_similarity(a=[v], b=data_values)
            if top_n is None:
                result[k] = dict(zip(data_keys, similarities))
            else:
                similarities = dict(zip(data_keys, similarities))
                similarities = dict(
                    sorted(similarities.items(), key=lambda t: t[::-1], reverse=True)[
                        :top_n
                    ]
                )
                result[k] = similarities
            data_keys.append(k)
            data_values.append(v)
        return result

    @staticmethod
    def create_tfidf_matrix(data: Union[list, np.ndarray]):
        """
        TF-IDF 행렬 생성
        """
        tfidf = TfidfVectorizer()
        return tfidf.fit_transform(data)

    def create_tfidf_cosine_similarity_dict(
        self, data: dict, top_n: Optional[int] = None, progressbar: bool = True
    ):
        """
        TF-IDF 행렬에 의한 코사인 유사도 계산
        """
        for k, v in data.items():
            data[k] = " ".join(v)

        # tf-idf 행렬 생성
        tfidf_matrix = self.create_tfidf_matrix(data=list(data.values()))

        # cosine 유사도 계산
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # cosine 유사도를 key:value로 맵핑 e.g. {0: {1: 0.123, 2: 0.234} ..}
        similarities_dict = {}
        for idx, similarity in tqdm(
            enumerate(cosine_sim),
            total=len(cosine_sim),
            desc="creating cosine similarity...",
            disable=False if progressbar else True,
        ):
            similarities = list(enumerate(similarity))
            similarities = [x for x in similarities if x[1] > 0]  # 유사도 0 초과인 것만
            if top_n is not None:
                similarities = sorted(
                    similarities, key=lambda x: x[1], reverse=True
                )  # 내림차순
                similarities = similarities[:top_n]  # 상위 N개
            else:
                pass
            for i, sim in enumerate(similarities):
                if idx == sim[0]:
                    del similarities[i]  # 자기 유사도 제거

            # [(0, 0.123124), (1, 0.5124123), ...]로 되어있는 유사도를 key:value로 변경 {0: 0.123124, 1:0.5124123, ...}
            similarities_dict_tmp = {}
            for i, sim in similarities:
                similarities_dict_tmp[i] = sim
            similarities_dict[idx] = similarities_dict_tmp

        # 정수 인덱스를 원래의 key로 변경
        result = {}
        origin_keys = list(data.keys())
        for k1, v1 in similarities_dict.items():
            tmp_dict = {}
            for k2, v2 in v1.items():
                new_key2 = origin_keys[k2]
                tmp_dict[new_key2] = v2
            new_key1 = origin_keys[k1]
            result[new_key1] = tmp_dict
        return result


class SimilarityCalculator(JacarrdSimilarity, EuclideanDistance, CosineSimilarity):
    """
    s = SimilarityCalculator()

    x = {"a": [1, 2], "b": [2]}
    s.calculate(method="jaccard", data=x)
    >>> {'a': {'b': 0.5}, 'b': {'a': 0.5}}

    x = {"a": {"x": 1, "y": 3}, "b": {"x": 2}}
    s.calculate(method="euclidean", data=x)
    >>> {'a': {'b': 3.1622776601683795}, 'b': {'a': 3.1622776601683795}}

    x = {"a": [1, 2], "b": [2, 2]}
    s.calculate(method="cosine", data=x)
    >>> {'a': {'b': 0.9486832980505137}, 'b': {'a': 0.9486832980505137}}
    """

    def __init__(self, data: dict = None):
        super().__init__(data)
        self.method = None
        self.__calculator = None

    def _create_calculator(self):
        if self.method == "jaccard":
            return JacarrdSimilarity(self.data)
        elif self.method == "euclidean":
            return EuclideanDistance(self.data)
        elif self.method == "cosine":
            return CosineSimilarity(self.data)
        else:
            raise ValueError(
                "Invalid `method`. Options: ['jaccard', 'euclidean', 'cosine']"
            )

    def calculate(self, method: str, data: dict) -> dict:
        self.method = method
        self.data = data
        self.__calculator = self._create_calculator()
        return self.__calculator.create_similarity_dict()
