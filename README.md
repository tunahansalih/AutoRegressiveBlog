# Autoregressive Image Generation

This is a simple implementation of autoregressive image generation.

The code is based on the blog post [Generating Pixels One by One: Your First Autoregressive Image Generation Model](https://tunahansalih.github.io/posts/autoregressive-image-generation-part-1/).

<video alt="test" muted autoplay loop height="320">
    <source src="autoregressive_mlp.mov" type="video/mp4">
</video>

To run the code, you can use `uv`

To install `uv` run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install the dependencies, run:

```bash
uv sync
```

You can then use the jupyter notebook to run the code.

```bash
uv run jupyter notebook
```
