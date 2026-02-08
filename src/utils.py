from nn import LCBHAM
from torch import nn
from typing import Mapping, Callable, Optional
import itertools
from ultralytics import YOLO

def inject_custom_layer(model, m_: nn.Module, i, f=-1, device = "cuda"):
	m = type(m_)
	t = str(m)[8:-2].replace("__main__.", "")  # module type
	m_.np = sum(x.numel() for x in m_.parameters())  # number params
	m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
	model.model.model[i] = m_.to(device)  # replace layer on model and move to device
	#save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
	#layers.append(m_)
	#if i == 0:
	#    ch = []
	#ch.append(c2)

def init_dsp_yolo(
		multi_lcam,
		multi_ldsam,
		shared,
		inject,
	) -> tuple[YOLO, Optional[list[int]], int]:
	model = YOLO('yolov8n.pt', task='detect', verbose=True)
	train_layers_list = None
	if inject >= 1:
		inject_custom_layer(model, LCBHAM(64, multi_lcam=multi_lcam, multi_ldsam=multi_ldsam, shared=shared), i=16)  # 64 fchannels for nano, inject in place of first cbs
		train_layers_list = [16]  # Only train the injected layer(s)
	if inject >= 2:
		inject_custom_layer(model, LCBHAM(128, multi_lcam=multi_lcam, multi_ldsam=multi_ldsam, shared=shared), i=19)  # 128 fchannels for nano, inject in place of second cbs
		train_layers_list.append(19) # Only train the injected layer(s)
	return model, train_layers_list


def product_dict(
	valid_mapping: Mapping,
	condition: Callable = lambda **_: True,

):
	"""
	Given a `valid_mapping` of key value pairs as (name, iterable_of_valid_values),
	generates a mapping that satisfies `condition` for each iteration as
	(name, possible_value) using cartesian product.

	Usage:
```
for combination in product_dict(valid_mapping, condition):
	# Do sth with the combination
	...
```
	"""
	keys = valid_mapping.keys()
	iter_of_valid_vals = valid_mapping.values()
	for val_comb in itertools.product(*iter_of_valid_vals):
		comb = dict(zip(keys, val_comb))
		if condition(**comb):
			yield comb
			
def stringify_map(_map: Mapping, delim="_"):
	return delim.join([f"{k}:{v}" for k, v in _map.items()])