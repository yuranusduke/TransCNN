#  ***Transformer in Convolutional Neural Networks***

Unofficial toy implementation for ***Transformer in Convolutional Neural Networks***.
There are many ambiguous points in the paper, so implementation may differ
a lot in the original paper and I only get models 
for 256 x 256 inputs.

Paper: https://arxiv.org/abs/2106.03180

## Import
```python
from model import TransCNN_Model

num_classes = 1000
in_channels = 3
input_size = 256 

small_model = TransCNN_Model(num_classes, in_channels = in_channels, type = 'small').cuda()
```


## Number of Params

<table>
    <tr>
        <td>Model</td>
        <td># of Params</td>
    </tr>
    <tr>
        <td rowspan="1">TransCNN-Small</td>
        <td>13.91M</td>
    </tr>
    <tr>
        <td rowspan="1">TransCNN-Base</td>
        <td>25.94M</td>
    </tr>
</table>

## Citation

    @misc{liu2021transformer,
      title={Transformer in Convolutional Neural Networks}, 
      author={Yun Liu and Guolei Sun and Yu Qiu and Le Zhang and Ajad Chhatkuli and Luc Van Gool},
      year={2021},
      eprint={2106.03180},
      archivePrefix={arXiv},
      primaryClass={cs.CV}}


***<center>Veni，vidi，vici --Caesar</center>***
