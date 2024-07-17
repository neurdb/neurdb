# NeurDB Python API

## Installation

### Editable (for Development)

```sh
pip install -e . --config-settings editable_mode=compat
```

### Normal (for Distribution)

Changes to the source files are not directly reflected in the `neurdb` package. You should uninstall and reinstall the package to take effect.

```sh
pip install .
```

Reinstall using

```sh
pip uninstall neurdb
pip install .
```