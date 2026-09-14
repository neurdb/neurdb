import os


def list_files(filepath, suffix=None, prefix=None, isdepth=True):
    files = []
    for fpathe, dirs, fs in os.walk(filepath):
        for f in fs:
            if suffix is None and prefix is None:
                files.append(os.path.join(fpathe, f))
            elif prefix is None:
                if f.endswith(suffix):
                    files.append(os.path.join(fpathe, f))
            elif suffix is None:
                if f.startswith(prefix):
                    files.append(os.path.join(fpathe, f))
            else:
                if f.startswith(prefix) and f.endswith(suffix):
                    files.append(os.path.join(fpathe, f))

        if isdepth == False:
            break
    return files


def list_filenames(filepath, suffix=None, prefix=None):
    filenames = []
    for fpathe, dirs, fs in os.walk(filepath):
        for f in fs:
            if suffix is None and prefix is None:
                filenames.append(f)
            elif prefix is None:
                if f.endswith(suffix):
                    filenames.append(f)
            elif suffix is None:
                if f.startswith(prefix):
                    filenames.append(f)
            else:
                if f.startswith(prefix) and f.endswith(suffix):
                    filenames.append(f)
        break
    return filenames


def detect_and_create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def all_files_exist(paths):
    flag = True
    for path in paths:
        if not os.path.exists(path):
            flag = False
            break

    return flag


def load_str_str_map(filepath, delim="||", key_pos=0, val_pos=1):
    m = {}
    with open(filepath, "r") as reader:
        lines = reader.readlines()
        for line in lines:
            terms = line.strip().split(delim)
            key = terms[key_pos]
            if val_pos < 0:
                value = line
            else:
                value = terms[val_pos]
            m[key] = value
    return m


def read_all_lines(path):
    with open(path, "r") as reader:
        lines = reader.readlines()
    return lines


def read_first_line(path):
    with open(path, "r") as reader:
        line = reader.readline()
    return line


def read_all(path):
    with open(path, "r") as reader:
        s = reader.read()
    return s


def write_all(path, s):
    with open(path, "w") as writer:
        writer.write(s)


def write_all_lines(path, lines):
    with open(path, "w") as writer:
        writer.writelines(lines)


def find_start_line(lines):
    for i, line in enumerate(lines):
        if line.startswith("--"):
            return i + 1
    return 0
