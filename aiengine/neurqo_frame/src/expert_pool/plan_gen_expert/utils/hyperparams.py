import copy
import re
from typing import Any, Dict, Iterator, Tuple, List, Optional, Union
import numpy as np
import ast
import sys
import enum


def to_flattened_text_dict(params: Any, quote_all: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Flattens entries in 'params' (dict or Params) into a textual format.

    Args:
        params: A Params object or dict to flatten.
        quote_all: If True, quote all values as strings; otherwise, use native string representation.

    Returns:
        A tuple of (key-value dict, types dict) where keys are parameter names (with dot notation for nesting)
        and values are string representations.
    """
    kv: Dict[str, str] = {}
    types: Dict[str, str] = {}

    def get_repr(val: Any) -> Any:
        """Get the string representation of a value."""
        if isinstance(val, Params):
            return {k: get_repr(v) for k, v in val.iter_params()}
        if isinstance(val, dict):
            return {k: get_repr(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return type(val)([get_repr(v) for v in val])
        if isinstance(val, (int, float, bool, str)):
            return val
        if np.isscalar(val):
            return float(val.item())
        if isinstance(val, type):
            return f"type/{val.__module__}/{val.__name__}"
        return type(val).__name__

    def traverse(param: Any, prefix: str, kv: Dict[str, str]) -> None:
        """Traverses 'param' and inserts key-value pairs to 'kv'."""
        if isinstance(param, dict):
            for key, val in param.items():
                traverse(val, f"{prefix}.{key}" if prefix else key, kv)
        elif isinstance(param, Params):
            for key, val in param.iter_params():
                traverse(val, f"{prefix}.{key}" if prefix else key, kv)
        elif isinstance(param, (list, tuple)) and all(isinstance(x, Params) for x in param):
            for i, val in enumerate(param):
                traverse(val, f"{prefix}[{i}]", kv)
        elif isinstance(param, str):
            kv[prefix] = quote_string(param) if quote_all else param
            types[prefix[1:] if prefix else prefix] = 'str'
        else:
            kv[prefix] = str(get_repr(param)) if quote_all else str(get_repr(param))
            types[prefix[1:] if prefix else prefix] = type(param).__name__

    traverse(params, '', kv)
    return kv, types


def quote_string(s: str) -> str:
    """Quotes a string with appropriate quotes and escaping.

    Chooses single or double quotes to minimize escaping and escapes those quotes and backslashes.
    Does not escape newlines; they are output verbatim.

    Args:
        s: String to quote.

    Returns:
        Quoted string (possibly multiline).
    """
    single_quote_count = s.count("'")
    double_quote_count = s.count('"')
    quote_delim = "'" if single_quote_count <= double_quote_count else '"'
    encoded = re.sub(r'([%s\\])' % quote_delim, r'\\\1', s)
    return quote_delim + encoded + quote_delim


def unquote_string(quoted: str) -> str:
    """Unquotes a string, removing quotes and handling escaped characters.

    Supports only the escaping produced by quote_string.

    Args:
        quoted: String to unquote.

    Returns:
        Unquoted string.
    """
    if quoted and quoted[0] in ['"', "'"]:
        contents = quoted.strip(quoted[0])
        return re.sub(r"""\\([\\'"])""", r'\1', contents)
    return quoted


def ends_with_terminal_quote(s: str, quote_char: str) -> bool:
    """Returns whether a string ends with a valid terminal quote.

    Args:
        s: String to check.
        quote_char: Quote character (' or ").

    Returns:
        True if the string ends with an unescaped quote, False otherwise.
    """
    endm = re.search(r'(\\*)%s$' % quote_char, s)
    if not endm:
        return False
    backslashes = endm.group(1)
    return len(backslashes) % 2 == 0


class _Param:
    """Stores data for a single parameter."""

    def __init__(self, name: str, default_value: Any, description: str) -> None:
        self._name = name
        self._value = default_value
        self._description = description

    def __eq__(self, other: '_Param') -> bool:
        return self._name == other._name and self._value == other._value

    def __deepcopy__(self, memo: Dict) -> '_Param':
        value = copy.deepcopy(self._value, memo)
        p = _Param(self._name, value, self._description)
        memo[id(self)] = p
        return p

    def to_string(self, nested_depth: int) -> str:
        """Prints the parameter as a string with proper indentation.

        Args:
            nested_depth: The level of indentation for nested parameters.

        Returns:
            A string representation of the parameter.
        """
        def get_repr(val: Any) -> Any:
            if isinstance(val, Params):
                return {k: get_repr(v) for k, v in val.iter_params()}
            if isinstance(val, dict):
                return {k: get_repr(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return type(val)([get_repr(v) for v in val])
            if hasattr(val, 'Repr'):
                return val.Repr()
            return val

        nested_indent = '  ' * nested_depth
        if isinstance(self._value, Params):
            value_str = self._value._to_string(nested_depth)  # Call _to_string directly
        elif isinstance(self._value, str):
            return f'{nested_indent}{self._name}: "{self._value}"'
        else:
            value_str = str(get_repr(self._value))
        return f'{nested_indent}{self._name}: {value_str}'

    def set(self, value: Any) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value


def copy_fields_to(from_params: 'Params', to_params: 'Params', skip: Optional[List[str]] = None) -> 'Params':
    """Copy fields from one Params to another, with optional skipped params.

    Args:
        from_params: Source Params to copy from.
        to_params: Destination Params to copy to.
        skip: List of parameter names to skip.

    Returns:
        The updated to_params.
    """
    skip = skip or []
    for name, value in from_params.iter_params():
        if name in skip:
            continue
        if isinstance(value, Params):
            to_params.set(**{name: value.copy()})
        else:
            to_params.set(**{name: value})
    return to_params


class Params:
    """Stores data for a set of parameters.

    Provides attribute-based API, e.g. "params.foo = 5".
    Uses internal {'name': _Param} dict for storing parameter data.
    """

    def __init__(self) -> None:
        self.__dict__['_immutable'] = False
        self._params: Dict[str, '_Param'] = {}  # name => _Param

    def __setattr__(self, name: str, value: Any) -> None:
        """Sets an attribute or parameter value."""
        if self._immutable:
            raise TypeError('This Params instance is immutable.')
        if name in ('_params', '_immutable'):
            self.__dict__[name] = value
        else:
            try:
                self._params[name].set(value)
            except KeyError:
                raise AttributeError(self._key_error_string(name))

    def __getattr__(self, name: str) -> Any:
        """Gets an attribute or parameter value."""
        if name in ('_params', '_immutable'):
            return self.__dict__[name]
        try:
            return self._params[name].get()
        except KeyError:
            raise AttributeError(self._key_error_string(name))

    def __dir__(self) -> List[str]:
        """Returns a sorted list of parameter names."""
        return sorted(self._params.keys())

    def __contains__(self, name: str) -> bool:
        """Checks if a parameter exists."""
        return name in self._params

    def __len__(self) -> int:
        """Returns the number of parameters."""
        return len(self._params)

    def __eq__(self, other: Any) -> bool:
        """Checks if two Params instances are equal."""
        return isinstance(other, Params) and self._params == other._params

    def __ne__(self, other: Any) -> bool:
        """Checks if two Params instances are not equal."""
        return not self == other

    def __str__(self) -> str:
        """Returns a string representation of the Params instance."""
        return self._to_string(0)

    def _to_string(self, nested_depth: int) -> str:
        """Helper method to generate a string representation with indentation."""
        sorted_param_strs = [
            v.to_string(nested_depth + 1) for (_, v) in sorted(self._params.items())
        ]
        nested_indent = '  ' * nested_depth
        return '{\n%s\n%s}' % ('\n'.join(sorted_param_strs), nested_indent)

    def __deepcopy__(self, unused_memo: Dict) -> 'Params':
        """Creates a deep copy of the Params instance."""
        return self.copy()

    def _similar_keys(self, name: str) -> List[str]:
        """Returns a list of parameter keys similar to the given name."""

        def _overlaps(name: str, key: str) -> float:
            """Calculates the fraction of 3-char substrings in name that appear in key."""
            matches = 0
            trials = 0
            for i in range(len(name) - 3):
                trials += 1
                if name[i:i + 3] in key:
                    matches += 1
            return float(matches) / trials if trials else 0

        if '_params' in self.__dict__:
            return [key for key in self._params if _overlaps(name, key) > 0.5]
        return []

    def _key_error_string(self, name: str) -> str:
        """Generates an error message with suggestions for similar parameter names."""
        similar = self._similar_keys(name)
        if similar:
            return f"{name} (did you mean: [{','.join(sorted(similar))}]"
        return name

    def copy(self) -> 'Params':
        """Creates a deep copy of the Params instance."""
        return self.copy_to(type(self)())

    def copy_to(self, res: 'Params') -> 'Params':
        """Copies the current Params instance to another instance."""
        res._params = copy.deepcopy(self._params)
        res._immutable = self._immutable
        return res

    def define(self, name: str, default_value: Any, description: str) -> None:
        """Defines a new parameter.

        Args:
            name: The parameter name. Must only contain lowercase letters, numbers,
                and underscores. Must start with a lowercase letter.
            default_value: Default value for this parameter. May be None.
            description: String description of this parameter.

        Raises:
            AttributeError: If parameter 'name' is already defined.
        """
        if self._immutable:
            raise TypeError('This Params instance is immutable.')
        assert name is not None and isinstance(name, str) and re.match('^[a-z][a-z0-9_]*$', name)
        if name in self._params:
            raise AttributeError(f'Parameter {name} is already defined')
        self._params[name] = _Param(name, default_value, description)

    def freeze(self) -> None:
        """Marks this Params instance as immutable."""
        self._immutable = True

    def is_immutable(self) -> bool:
        """Returns whether this Params instance is immutable."""
        return self._immutable

    def _get_nested(self, name: str) -> Tuple['Params', str]:
        """Returns the nested Params object and key for a dotted name."""
        parts = name.split('.')
        curr = self
        for i, part in enumerate(parts[:-1]):
            try:
                is_list = re.match(r'^(.+)\[(.+)\]$', part)
                if is_list:
                    part = is_list.group(1)
                    list_index = int(is_list.group(2))
                curr = curr._params[part].get()
                if is_list:
                    curr = curr[list_index]
            except KeyError:
                raise AttributeError('.'.join(parts[:i + 1]))
            assert isinstance(curr, Params), (
                f'Cannot introspect {type(curr)} for {",".join(parts[:i + 1])}')
        return curr, parts[-1]

    def set(self, **kwargs: Any) -> 'Params':
        """Sets multiple parameters using keyword arguments.

        Args:
            **kwargs: Name-value pairs to set. Dots in names indicate navigation
                into nested Params objects.
        """
        if self._immutable:
            raise TypeError(f'This Params instance is immutable: {self}')
        for name, value in kwargs.items():
            param, key = self._get_nested(name)
            try:
                param._params[key].set(value)
            except KeyError:
                raise AttributeError(self._key_error_string(name))
        return self

    def get(self, name: str) -> Any:
        """Gets a parameter value by name.

        Args:
            name: The parameter name. Dots indicate navigation into nested Params.

        Returns:
            The parameter value.

        Raises:
            AttributeError: If the parameter is not found.
        """
        param, key = self._get_nested(name)
        try:
            return param._params[key].get()
        except KeyError:
            raise AttributeError(self._key_error_string(name))

    def delete(self, *args: str) -> 'Params':
        """Deletes multiple parameters.

        Args:
            *args: List of parameter names to delete. Dots indicate navigation
                into nested Params.
        """
        if self._immutable:
            raise TypeError('This Params instance is immutable.')
        for name in args:
            param, key = self._get_nested(name)
            try:
                del param._params[key]
            except KeyError:
                raise AttributeError(self._key_error_string(name))
        return self

    def iter_params(self) -> Iterator[Tuple[str, Any]]:
        """Yields parameter names and values for iteration."""
        for name, param in self._params.items():
            yield (name, param.get())

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """Allows treating this class as a Python dict."""
        return self.iter_params()

    def to_text(self, include_types: bool = False) -> Union[str, Tuple[str, Dict[str, str]]]:
        """Encodes parameters into a simple text format.

        Args:
            include_types: If True, returns a tuple with the text and a dictionary
                of parameter types.

        Returns:
            The encoded text or a tuple of (text, types dict) if include_types is True.
        """
        kv, types = to_flattened_text_dict(self)
        ret = ''
        for (k, v) in sorted(kv.items()):
            ret += k + ' : ' + v + '\n'
        return (ret, types) if include_types else ret

    def from_text(self, text: str, type_overrides: Optional[Dict[str, str]] = None) -> None:
        """Merges parameters from a text representation into this instance.

        Args:
            text: A text representation of parameters.
            type_overrides: Optional dictionary to override parameter types.

        Raises:
            AttributeError: If the text contains an invalid parameter key.
            ValueError: If the text contains an invalid parameter value or format.
        """
        if self._immutable:
            raise TypeError('This Params instance is immutable.')
        kv: Dict[str, str] = {}
        type_overrides = type_overrides or {}
        string_continue = None  # None or (key, quote, value)
        for line in text.split('\n'):
            if string_continue:
                value_stripped = line.rstrip()
                if not ends_with_terminal_quote(value_stripped, string_continue[1]):
                    string_continue = (string_continue[0], string_continue[1],
                                       string_continue[2] + '\n' + line)
                    continue
                kv[string_continue[0]] = string_continue[2] + '\n' + value_stripped
                string_continue = None
                continue

            line = line.strip()
            if not line or line[0] == '#':
                continue
            pair = line.split(':', 1)
            if len(pair) == 2:
                key = pair[0].strip()
                value = pair[1].lstrip()
                value_stripped = value.rstrip()
                if value and value[0] in ['"', '\'']:
                    quote_char = value[0]
                    if not ends_with_terminal_quote(value[1:], quote_char):
                        string_continue = (key, quote_char, value)
                        continue
                kv[key] = value_stripped
            else:
                raise ValueError(f'Line {line} is not in <key>:<value> format')

        def _value_from_text(key: str, old_val: Any, val: str) -> Any:
            """Converts a text value to the appropriate type based on the existing value."""
            val_type = type(old_val).__name__
            if isinstance(old_val, str):
                val_type = 'str'
            if key in type_overrides:
                val_type = type_overrides[key]
            if val_type == 'bool':
                return val and val not in ('False', 'false')
            elif val_type == 'int':
                return int(val)
            elif val_type == 'float':
                return float(val)
            elif val_type in ['list', 'tuple']:
                return ast.literal_eval(val)
            elif val_type == 'dict':
                return ast.literal_eval(val) if val != 'dict' else {}
            elif val_type == 'str':
                val = unquote_string(val)
                if val.startswith('[') and val.endswith(']'):
                    try:
                        return ast.literal_eval(val)
                    except ValueError:
                        pass
                return val
            elif isinstance(old_val, enum.Enum):
                cls, _, name = val.rpartition('.')
                if val_type != cls:
                    raise ValueError(f'Expected enum of class {val_type} but got {cls}')
                return type(old_val)[name]
            elif isinstance(old_val, type) or old_val is None:
                if val == 'NoneType':
                    return None
                elif old_val is None and val in ('False', 'false'):
                    return False
                elif old_val is None and val in ('True', 'true'):
                    return True
                else:
                    try:
                        val_type, pkg, cls = val.split('/', 2)
                        if val_type == 'type':
                            return getattr(sys.modules[pkg], cls)
                    except ValueError as e:
                        raise ValueError(f'Error processing {key!r} : {val!r} with {e!r}')
            else:
                raise ValueError(f'Failed to read a parameter: {key!r} : {val!r}')

        for key, val in kv.items():
            old_val = self.get(key)
            new_val = _value_from_text(key, old_val, val)
            self.set(**{key: new_val})

    def to_text_with_types(self) -> str:
        """Encodes parameters and their types into a text format."""
        text, types = self.to_text(include_types=True)
        text += '\n\n'
        for (k, v) in sorted(types.items()):
            text += k + ' : ' + v + '\n'
        return text

    def from_text_with_types(self, text: str) -> None:
        """Merges parameters and types from a text representation."""
        text, types_str = text.split('\n\n')
        types: Dict[str, str] = {}
        for row in types_str.split('\n'):
            if not row:
                continue
            k, v = row.split(':')
            types[k.strip()] = v.strip()
        self.from_text(text, type_overrides=types)

    def text_diff(self, other: 'Params') -> str:
        """Returns a string describing differences between this and another Params instance.

        Args:
            other: The other Params object to compare with.

        Returns:
            A string of differences.
        """

        def text_diff_helper(a: 'Params', b: 'Params', spaces: str) -> str:
            """Helper to compute differences between two Params instances."""
            a_keys = set([key for key, _ in a.iter_params()])
            b_keys = set([key for key, _ in b.iter_params()])
            all_keys = a_keys.union(b_keys)
            diff = ''
            for key in sorted(all_keys):
                if key in a_keys and key not in b_keys:
                    diff += '>' + spaces + key + ': ' + str(a.get(key)) + '\n'
                elif key in b_keys and key not in a_keys:
                    diff += '<' + spaces + key + ': ' + str(b.get(key)) + '\n'
                elif a.get(key) != b.get(key):
                    if isinstance(a.get(key), Params):
                        diff += '?' + spaces + key + ':\n'
                        diff += text_diff_helper(a.get(key), b.get(key), spaces + '  ')
                    else:
                        diff += '>' + spaces + key + ': ' + str(a.get(key)) + '\n'
                        diff += '<' + spaces + key + ': ' + str(b.get(key)) + '\n'
            return diff

        return text_diff_helper(self, other, spaces=' ')


class InstantiableParams(Params):
    """Params which can be instantiated.

    When using InstantiableParams, callers must provide a class which supports
    initialization using a Params instance.

    This covers a common use case of Params to hold a configuration for a given
    class.
    """

    def __init__(self, cls=None):
        super().__init__()
        self.define('cls', cls, 'Cls that this param object is associated with.')

    def instantiate(self, **args):
        """Instantiate an instance that this Params is configured for.

        Example:
          params = InstantiableParams(cls=MyObject)
          params.Define('weight', 0.2, 'Training weight.')
          params.weight = 0.9
          obj = params.Instantiate()

        It's common for classes to have a classmethod called Params that returns
        a pre-made InstantiableParams, like this:

          params = MyObject.Params()
          params.weight = 0.9
          obj = params.Instantiate()

        By convention, anything that parameterizes the behavior of your class
        should be stored in this Params object. However, your class may also use
        shared state objects which aren't really parameters, like a shared lock.
        These can be passed as extra arguments to Instantiate.

        Example:
          lock = threading.Lock()
          params = MyObject.Params()
          obj_a = params.Instantiate(lock=lock)
          obj_b = params.Instantiate(lock=lock)

        Args:
          **args: Additional keyword arguments to pass to the constructor in
            addition to this Params object.

        Returns:
          A constructed object where type(object) == cls.
        """
        assert self.cls is not None

        # The class initializer is expected to support initialization using Params.
        return self.cls(self, **args)

    def copy(self):
        """See base class."""
        return self.copy_to(type(self)(self.cls))


def sanitize_to_text(d):
    """Sanitizes result/config dicts for W&B textual logging."""
    # For configs: flattens all so w&b can visualize nicely.
    # Don't quote all because otherwise scalars are converted to strings.
    ret, _ = to_flattened_text_dict(d, quote_all=False)
    return ret
