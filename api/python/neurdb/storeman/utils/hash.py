import hashlib
from typing import List


class Hash:
    @staticmethod
    def md5_list(str_list: List[str]) -> str:
        """
        Get the MD5 hash of a list of strings
        :param str_list: List of strings
        :return: MD5 hash of the list of strings
        """
        return hashlib.md5(",".join(str_list).encode()).hexdigest()

    @staticmethod
    def md5_str(string: str) -> str:
        """
        Get the MD5 hash of a string
        :param string: The string to hash
        :return: MD5 hash of the string
        """
        return hashlib.md5(string.encode()).hexdigest()
