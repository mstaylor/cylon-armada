# soci-index-generator-lambda bootstrap

`bootstrap` is a static linux/amd64 build of AWS's SOCI index generator Lambda
(https://github.com/awslabs/cfn-ecr-aws-soci-index-builder), pinned to the commit in `VERSION`.
This directory intentionally does not vendor the upstream source — `Dockerfile.build` plus the
pinned commit in `VERSION` is the reproducible record.

Build on a host with Docker (this generator needs CGO — build it on an actual linux/amd64 Docker
host, not by cross-compiling):

```
docker build --platform=linux/amd64 --build-arg SOCI_COMMIT=$(cat VERSION) \
  -f Dockerfile.build -t soci-index-generator-build .
```

Extract the binary:

```
docker cp "$(docker create soci-index-generator-build)":/bootstrap ./bootstrap
```

`docker cp` leaves the temporary container stopped on disk; `docker ps -a` and `docker rm` it
afterward if you want it gone.

To rebuild against a newer upstream commit, update `VERSION` and pass the new SHA as
`SOCI_COMMIT`.