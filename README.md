# LoRA-Torch

This codebase reimplementes [LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)](https://openreview.net/forum?id=nZeVKeeFYf9) and is reconstructed based on [loralib](https://github.com/microsoft/LoRA). 



## Features

**The implementations of ``loratorch`` and ``loralib`` are very different.** We take the ``nn.Linear`` as an example as follows.

1. For ``loralib``,
   $h = x W_0^\top + \frac{\alpha}{r} x(BA)^\top,$

where $x\in\mathbb{R}^{k\times n}$ is the input matrix, $W_0\in\mathbb{R}^{m\times n}$ is the pre-trained weight matrix, $r$ is the predefined LoRA rank, $B\in\mathbb{R}^{m\times r}$ and $A\in \mathbb{R}^{r\times n}$ are the LoRA matrixes, and $\alpha$ is a hyper-parameter.

2. For ``loratorch``,
   
   $h = x (W_0 + \frac{\alpha}{r} BA)^\top.$
   
   

``loralib`` computes $xW_0^\top$ and $x(BA)^\top$ respectively and then merges the results. While ``loratorch`` merges pre-trained weight $W_0$ and its LoRA weight $BA$ and then computes the results by simply using ``nn.Linear.forward()``. There is no difference between ``loralib`` and ``loratorch`` in the linear layers. But in some no-linear or complex layers, we are no sure whether this layer satisfies $L(x, W_0)+L(x, BA) = L(x, W_0+BA)$. Hence, it is difficult to extend LoRA to some complex layers by using ``loralib``. On the contrary, the idea of merging weights first in ``loratorch`` is more general and extensible. You just call ``merge_lora_param()`` in ``loratorch`` to merge weights and then call ``forward()`` in the original layer to compute the results. With the help of ``loratorch``, you can easily implement LoRA to any type of layer of ``torch.nn``.

 

## Quick Start

**The usage of ``loratorch`` is the same as ``loralib``.**

1. Install ``loratorch``.
   
   ```bash
   pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
   # Alternatively for developers
   # git clone https://github.com/Baijiong-Lin/LoRA-Torch
   # cd LoRA-Torch
   # pip install -e .
   ```

2. Replace the layers where you would like to use LoRA by using ``loratorch``.
   
   ```python
   # ===== Before =====
   # layer = nn.Linear(in_features, out_features)
   
   # ===== After ======
   import loratorch as lora
   # Add a pair of low-rank adaptation matrices with rank r=16 and alpha=32
   layer = lora.Linear(in_features, out_features, r=16, lora_alpha=32)
   ```

3. Mark only LoRA parameters as trainable before the training loop.
   
   ```python
   model = Model()
   # This sets requires_grad to False for all parameters without the string "lora_" in their names
   lora.mark_only_lora_as_trainable(model)
   # Training loop
   for batch in dataloader:
       model.train()
       ...
   ```

4. Save LoRA model (only the LoRA matrixes will be saved).
   
   ```python
   # ===== Before =====
   # torch.save(model.state_dict(), checkpoint_path)
   # ===== After =====
   torch.save(lora.lora_state_dict(model), checkpoint_path)
   ```
5. Load LoRA model (need to load the pre-trained model first).
   
   ```python
   # Load the pre-trained checkpoint first
   model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
   # Then load the LoRA checkpoint
   model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
   ```

## Contributor

``loratorch`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io).

## Contact Us

If you have any question or suggestion, please feel free to contact us by [raising an issue](https://github.com/Baijiong-Lin/LoRA-Torch/issues) or sending an email to ``bj.lin.email@gmail.com``.

## Acknowledgements

``loratorch`` is heavily based on ``loralib``. We thank its authors for their wonderful and open-source codebase.

## Citation

If you find ``loratorch`` useful for your research or development, please cite the following:

```BibTeX
@inproceedings{hu2022lora,
title={Lo{RA}: Low-Rank Adaptation of Large Language Models},
author={Edward J Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
booktitle={International Conference on Learning Representations},
year={2022},
}

@software{lin2023loratorch,
  author = {Baijiong Lin},
  title = {{LoRA-Torch}: {PyTorch} Reimplementation of {LoRA}},
  url = {https://github.com/Baijiong-Lin/LoRA-Torch},
  year = {2023}
}
```

## License

``loratorch`` is released under the [MIT](./LICENSE) license.
