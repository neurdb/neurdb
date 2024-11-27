# NeurDB Kernel

This is the kernel of NeurDB that extends the original implementation of PostgreSQL, including planner, executor, etc.

## Installation

First, build the kernel object:

```bash
make && make install
```

Then, set the kernel object to preloaded in `postgresql.conf`:

```ini
# in postgresql.conf
shared_preload_libraries = 'nr_kernel'
```

## Development

### Debug build

To debug with breakpoints, build the kernel object with debug symbols:

```bash
make debug && make install
```
