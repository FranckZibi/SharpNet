import pytest
import sys, os

from pytorch_to_sharpnet import pytorch_to_sharpnet


def test_normalize_parameter_name():
    result = pytorch_to_sharpnet.normalize_parameter_name("/Multi_head_attention.0.1.out_proj.weight")
    expected_output = "/multi_head_attention_0_1.out_proj.weight"
    assert result == expected_output, f"Expected {expected_output} but got {result}"    

    result = pytorch_to_sharpnet.normalize_parameter_name("/Multi_head_attention.0.1.in_proj_weight")
    expected_output = "/multi_head_attention_0_1.in_proj_weight"
    assert result == expected_output, f"Expected {expected_output} but got {result}"    


