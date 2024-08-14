#ifndef MD5_H
#define MD5_H


/**
 * Calculate the md5 hash of a string
 * @param string The string to be hashed
 * @return The md5 hash of the string
 */
char *nr_md5_str(const char *string);


/**
 * Calculate the md5 hash of a list of strings
 * @param str_list The list of strings to be concatenated and hashed
 * @param list_size The size of the list
 * @return
 */
char *nr_md5_list(char **str_list, int list_size);

#endif //MD5_H
