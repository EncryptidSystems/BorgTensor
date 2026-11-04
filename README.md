# BorgTensor

BorgTensor shapes any safetensors checkpoint onto a model's geometry. Tensor ids stay. Layer count and axis lengths become the target model's.

Each tensor is read from the source shard that holds it, resampled, and written to the output shard with that same index.

```
tensor → source shard → output shard → shaped tensor
```

A matrix is resampled by `splineGrid` and `sampleGrid`, with `linCoords` on each axis. A vector, or one axis of a wider tensor, is that same path with a single column.

## Build

Go 1.24.

```
go build -o borgtensor.exe .
go test .
```

## Run

```
borgtensor -src SRC -model MODEL -out OUT
```

`-src` is the source checkpoint, a safetensors file or a directory. Repeat it for more than one shard. `-model` is the target geometry, a safetensors file or a directory. `-out` is the output file or directory. All three are required.

A directory is read from `model.safetensors.index.json` when that file is present. Otherwise the sharded `*-of-*` files, otherwise `model.safetensors`, otherwise every `.safetensors` file in the directory.

One source shard and an `-out` path that ends in `.safetensors` writes that file. Several source shards write `model-00001-of-N.safetensors` in the output directory, plus `model.safetensors.index.json`. Weights are BF16.
