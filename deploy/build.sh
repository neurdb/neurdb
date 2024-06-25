
# Exit immediately if a command exits with a non-zero status
set -e

# Print each command before executing it
set -x

docker rm -f neurdb_dev
docker build -t neurdbimg .
docker run -d --name neurdb_dev \
    -v ~/neurdb-dev:/code/neurdb-dev \
    neurdbimg

docker logs -f neurdb_dev