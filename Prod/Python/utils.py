from typing import Tuple


def extract_first_second_element(data)-> Tuple[int,int]:
    if isinstance(data, tuple):
        data = list(data)
    if isinstance(data, list):
        if len(data) == 0:
            raise Exception(f"must contains at least 1 element")
        if len(data) == 1:
            return (data[0], data[0])
        return (data[0], data[1])
    if isinstance(data, int) or isinstance(data, float):
        return [data, data]
    raise Exception(f"not managed type {type(data)} {data}")




def count_integers(data) -> int:
    if isinstance(data, int):
        return 1
    if isinstance(data, tuple):
        return len(data)
    if isinstance(data, list):
        return len(data)
    