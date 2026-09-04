
import os
import argparse
import copy
import datetime
import json
import time
import math
from collections import defaultdict
from typing import Tuple
from dataclasses import dataclass, field


import torch


import numpy as np
from tqdm import tqdm
import wandb
from rich import print
from prettytable import PrettyTable


from data import get_dataloaders
from src.epoch import evaluate_synset
from src.networks import CLIPModel_full
from src.utils import get_time
from src.vl_distill_utils import get_images_texts

def print_memory_usage(step: int):

	allocated_memory = torch.cuda.memory_allocated()


	reserved_memory = torch.cuda.memory_reserved()


	peak_allocated_memory = torch.cuda.max_memory_allocated()

	print(f"Allocated GPU Memory: {allocated_memory / (1024**2):.2f} MB | Reserved GPU Memory: {reserved_memory / (1024**2):.2f} MB | Peak Allocated GPU Memory: {peak_allocated_memory / (1024**2):.2f} MB")

	wandb.log(
		{
			"GPUMemory/Allocated": allocated_memory / (1024**2),
			"GPUMemory/Reserved": reserved_memory / (1024**2),
			"GPUMemory/PeakAllocated": peak_allocated_memory / (1024**2),
		},
		step=step,
	)

	torch.cuda.reset_peak_memory_stats()

@dataclass
class WallClockTracker:
	metric_name: str = "r_mean"
	run_start_s: float = field(default_factory=time.perf_counter)
	best_val: float = float("-inf")
	best_time_s: float | None = None
	target_value: float | None = None
	target_time_s: float | None = None
	epsilon: float = 1e-9
	history: list = field(default_factory=list)

	def stamp_eval(self, it: int, value: float):

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		now = time.perf_counter()
		elapsed = now - self.run_start_s
		self.history.append({"it": int(it), "t_s": float(elapsed), "metric": float(value)})

		is_new_best = (value > self.best_val + self.epsilon)
		if is_new_best:
			self.best_val = float(value)
			self.best_time_s = float(elapsed)

		if (self.target_value is not None) and (self.target_time_s is None) and (value >= self.target_value):
			self.target_time_s = float(elapsed)

		return {
			"new_best": bool(is_new_best),
			"elapsed_s": float(elapsed),
			"best_time_s": float(self.best_time_s) if self.best_time_s is not None else None,
			"best_val": float(self.best_val),
		}

	def finalize(self, out_dir: str):
		path = os.path.join(out_dir, "wallclock_times.json")
		with open(path, "w") as f:
			json.dump({
				"metric": self.metric_name,
				"best_val": self.best_val,
				"time_to_best_s": self.best_time_s,
				"target_value": self.target_value,
				"time_to_target_s": self.target_time_s,
				"history": self.history,
			}, f, indent=2)
		return path

@torch.no_grad()
def print_results(multi_eval_aggr_result, title='image-text retrieval results'):
	mean_result, std_result = defaultdict(), defaultdict()
	for k, v in multi_eval_aggr_result.items(): mean_result[k], std_result[k] = round(np.mean(v), 2), round(np.std(v), 2)


	import prettytable
	results_table = prettytable.PrettyTable()
	results_table.float_format = ".2f"
	results_table.align = 'c'
	results_table.border = True
	results_table.field_names = ['Img R@1', 'Img R@5', 'Img R@10', 'Img R_Mean', 
								 'Txt R@1', 'Txt R@5', 'Txt R@10', 'Txt R_Mean', 
								 'R_Mean']
	results_table.title = title

	if prettytable.__version__ >= '3.16.0':
		results_table.add_divider()
	else:
		results_table.add_row(['-'*10]*9)         


	results_table.add_row(
		[f"{mean_result['img_r1']:.2f}", f"{mean_result['img_r5']:.2f}", f"{mean_result['img_r10']:.2f}", f"{mean_result['img_r_mean']:.2f}", 
		 f"{mean_result['txt_r1']:.2f}", f"{mean_result['txt_r5']:.2f}", f"{mean_result['txt_r10']:.2f}", f"{mean_result['txt_r_mean']:.2f}", 
		 f"{mean_result['r_mean']:.2f}"]
		)

	results_table.add_row(

		[f"± {std_result['img_r1']:.2f}", f"± {std_result['img_r5']:.2f}", f"± {std_result['img_r10']:.2f}", f"± {std_result['img_r_mean']:.2f}", 
		f"± {std_result['txt_r1']:.2f}", f"± {std_result['txt_r5']:.2f}", f"± {std_result['txt_r10']:.2f}", f"± {std_result['txt_r_mean']:.2f}", 
		f"± {std_result['r_mean']:.2f}"]
		)
	
	return results_table

class AverageMeter:
	"""Computes and stores the average and current value. Tracks only inputted values (no initial zero)."""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = None
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		if self.count > 0:
			self.avg = self.sum / self.count


def save_range_rank_history(
	records: list[dict],
	out_dir: str,
	*,
	prefix: str = "range_rank_k",
) -> tuple[str, str | None]:
	"""Write range/residual SVD rank k per global iteration to JSON and a line plot."""
	os.makedirs(out_dir, exist_ok=True)
	json_path = os.path.join(out_dir, f"{prefix}.json")
	with open(json_path, "w") as f:
		json.dump(records, f, indent=2)
	its = [r["it"] for r in records if r.get("k") is not None]
	ks = [int(r["k"]) for r in records if r.get("k") is not None]
	png_path: str | None = None
	if its:
		import matplotlib
		matplotlib.use("Agg")
		import matplotlib.pyplot as plt

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.plot(its, ks, color="tab:blue", linewidth=1.2, marker="o", markersize=2, alpha=0.85)
		ax.set_xlabel("Global iteration")
		ax.set_ylabel("Range rank k (energy SVD)")
		ax.set_title("Range rank k over distillation iterations")
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		png_path = os.path.join(out_dir, f"{prefix}.png")
		fig.savefig(png_path, dpi=150, bbox_inches="tight")
		plt.close(fig)
	return json_path, png_path






def _hyp_center(x: torch.Tensor) -> torch.Tensor:
	return x - x.mean(dim=0, keepdim=True)


def _hyp_cross_cov(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
	b = min(x.shape[0], y.shape[0])
	if b <= 1:
		return x.new_zeros((x.shape[1], y.shape[1]))
	xc = _hyp_center(x[:b])
	yc = _hyp_center(y[:b])
	return (xc.T @ yc) / (b - 1 + eps)


def _hyp_svd_jittered(
	m: torch.Tensor,
	*,
	full_matrices: bool = False,
	base_eps: float = 1e-6,
	max_tries: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	orig_dtype = m.dtype
	work_dtype = torch.float64 if m.dtype in (torch.float16, torch.bfloat16, torch.float32) else m.dtype
	mw = m.to(work_dtype)
	r, c = mw.shape
	eye_rect = torch.eye(r, c, device=mw.device, dtype=mw.dtype)
	scale = mw.norm(p="fro").detach() / float(max(min(r, c), 1))
	jitter = base_eps * (scale + mw.new_tensor(1.0))
	for _ in range(max_tries):
		try:
			u, s, vh = torch.linalg.svd(mw + jitter * eye_rect, full_matrices=full_matrices)
			return u.to(orig_dtype), s.to(orig_dtype), vh.to(orig_dtype)
		except RuntimeError:
			jitter = jitter * 10.0
	gx = mw @ mw.T
	gy = mw.T @ mw
	gx = 0.5 * (gx + gx.T)
	gy = 0.5 * (gy + gy.T)
	ex, u = torch.linalg.eigh(gx)
	ey, v = torch.linalg.eigh(gy)
	ex = ex.clamp_min(0.0)
	ey = ey.clamp_min(0.0)
	idx_x = torch.argsort(ex, descending=True)
	idx_y = torch.argsort(ey, descending=True)
	u = u[:, idx_x]
	v = v[:, idx_y]
	k = min(r, c)
	s = torch.sqrt(0.5 * (ex[:k] + ey[:k]).clamp_min(0.0))
	if full_matrices:
		vh = v.T
	else:
		u = u[:, :k]
		vh = v[:, :k].T
	return u.to(orig_dtype), s.to(orig_dtype), vh.to(orig_dtype)


def _exp_map0_space(x: torch.Tensor, c: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
	"""Exponential map at origin (Lorentz): tangent -> hyperboloid space components."""
	c_t = x.new_tensor(c)
	rc_xnorm = torch.sqrt(c_t) * torch.norm(x, dim=-1, keepdim=True)
	rc_xnorm = torch.clamp(rc_xnorm, min=eps, max=math.asinh(2**15))
	return torch.sinh(rc_xnorm) * x / torch.clamp(rc_xnorm, min=eps)


def _log_map0_space(x_space: torch.Tensor, c: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
	"""Log map at origin: hyperboloid space -> tangent."""
	c_t = x_space.new_tensor(c)
	rc_x_time = torch.sqrt(1.0 + c_t * (x_space**2).sum(dim=-1, keepdim=True))
	rc_xnorm = torch.sqrt(c_t) * torch.norm(x_space, dim=-1, keepdim=True)
	d0 = torch.acosh(torch.clamp(rc_x_time, min=1.0 + eps))
	return d0 * x_space / torch.clamp(rc_xnorm, min=eps)


def lift_euclid_to_lorentz(
	feat: torch.Tensor,
	*,
	c: float = 1.0,
	scale: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Lift Euclidean features to Lorentz hyperboloid (space components)."""
	work = feat.float() if feat.dtype in (torch.float16, torch.bfloat16) else feat
	feat_tan = work * scale
	hyp_space = _exp_map0_space(feat_tan, c=c, eps=eps)
	return hyp_space if hyp_space.dtype == feat.dtype else hyp_space.to(feat.dtype)


def to_tangent_features(
	feat: torch.Tensor,
	*,
	c: float = 1.0,
	scale: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Euclidean -> Lorentz (exp-map) -> tangent at origin (log-map)."""
	x_space = lift_euclid_to_lorentz(feat, c=c, scale=scale, eps=eps)
	x_tan = _log_map0_space(x_space.float(), c=c, eps=eps)
	return x_tan if x_tan.dtype == feat.dtype else x_tan.to(feat.dtype)


def _pairwise_dist_space(
	x_space: torch.Tensor,
	y_space: torch.Tensor,
	c: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Pairwise Lorentz geodesic distance (space components on hyperboloid)."""
	c_t = x_space.new_tensor(c)
	x_time = torch.sqrt(1.0 / c_t + (x_space**2).sum(dim=-1, keepdim=True))
	y_time = torch.sqrt(1.0 / c_t + (y_space**2).sum(dim=-1, keepdim=True))
	xyl = x_space @ y_space.T - x_time @ y_time.T
	c_xyl = -c_t * xyl
	dist = torch.acosh(torch.clamp(c_xyl, min=1.0 + eps))
	return dist / torch.sqrt(c_t)


def _geodesic_infonce_from_space(
	img_h: torch.Tensor,
	txt_h: torch.Tensor,
	temperature: float = 0.07,
	c: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Bidirectional InfoNCE in Lorentz space (geodesic distances as logits)."""
	if img_h.shape[0] != txt_h.shape[0]:
		bsz = min(img_h.shape[0], txt_h.shape[0])
		img_h, txt_h = img_h[:bsz], txt_h[:bsz]
	if img_h.shape[0] <= 1:
		return img_h.new_tensor(0.0)
	scale = 1.0 / max(temperature, eps)
	img_h_f = img_h.float()
	txt_h_f = txt_h.float()
	image_logits = -_pairwise_dist_space(img_h_f, txt_h_f, c=c, eps=eps) * scale
	text_logits = -_pairwise_dist_space(txt_h_f, img_h_f, c=c, eps=eps) * scale
	labels = torch.arange(img_h.shape[0], device=img_h.device)
	loss_i2t = torch.nn.functional.cross_entropy(image_logits, labels)
	loss_t2i = torch.nn.functional.cross_entropy(text_logits, labels)
	return 0.5 * (loss_i2t + loss_t2i)


def hyperbolic_infonce_loss(
	img_feat_syn: torch.Tensor,
	txt_feat_syn: torch.Tensor,
	temperature: float = 0.07,
	c: float = 1.0,
	scale: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Base hyperbolic bidirectional InfoNCE on synthetic image-text pairs."""
	img_syn_h = lift_euclid_to_lorentz(img_feat_syn.float(), c=c, scale=scale, eps=eps)
	txt_syn_h = lift_euclid_to_lorentz(txt_feat_syn.float(), c=c, scale=scale, eps=eps)
	return _geodesic_infonce_from_space(img_syn_h, txt_syn_h, temperature=temperature, c=c, eps=eps)


def hyperbolic_wbce_loss(
	img_feat_syn: torch.Tensor,
	txt_feat_syn: torch.Tensor,
	temperature: float = 0.07,
	c: float = 1.0,
	scale: float = 1.0,
	eps: float = 1e-8,
) -> torch.Tensor:
	"""Bidirectional weighted BCE on geodesic logits (same logits as InfoNCE, sigmoid targets)."""
	img_syn_h = lift_euclid_to_lorentz(img_feat_syn.float(), c=c, scale=scale, eps=eps)
	txt_syn_h = lift_euclid_to_lorentz(txt_feat_syn.float(), c=c, scale=scale, eps=eps)
	if img_syn_h.shape[0] != txt_syn_h.shape[0]:
		bsz = min(img_syn_h.shape[0], txt_syn_h.shape[0])
		img_syn_h, txt_syn_h = img_syn_h[:bsz], txt_syn_h[:bsz]
	n = img_syn_h.shape[0]
	if n <= 1:
		return img_feat_syn.new_tensor(0.0)
	scale_t = 1.0 / max(temperature, eps)
	img_f = img_syn_h.float()
	txt_f = txt_syn_h.float()
	logits_i2t = -_pairwise_dist_space(img_f, txt_f, c=c, eps=eps) * scale_t
	logits_t2i = -_pairwise_dist_space(txt_f, img_f, c=c, eps=eps) * scale_t
	labels = torch.eye(n, device=img_feat_syn.device, dtype=logits_i2t.dtype)
	pos_weight = img_feat_syn.new_tensor(float(max(n - 1, 1)))
	loss_i2t = torch.nn.functional.binary_cross_entropy_with_logits(
		logits_i2t, labels, pos_weight=pos_weight
	)
	loss_t2i = torch.nn.functional.binary_cross_entropy_with_logits(
		logits_t2i, labels, pos_weight=pos_weight
	)
	return 0.5 * (loss_i2t + loss_t2i)


def range_residual_bases(
	c_real: torch.Tensor,
	energy: float = 0.95,
	max_rank: int | None = None,
	eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
	"""Range basis from real cross-covariance SVD. Returns (U_r, V_r, rank_used)."""
	u, s, vh = _hyp_svd_jittered(c_real, full_matrices=False, base_eps=eps)
	v = vh.T
	max_k = int(s.numel())
	if max_k == 0:
		return u[:, :0], v[:, :0], 0
	s2 = s.pow(2)
	denom = s2.sum().clamp_min(eps)
	cum = torch.cumsum(s2, dim=0) / denom
	k = int((cum < float(energy)).sum().item() + 1)
	if max_rank is not None and max_rank > 0:
		k = min(k, int(max_rank))
	k = max(1, min(k, max_k))
	return u[:, :k], v[:, :k], k


def _project_range_features(x: torch.Tensor, u_r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
	x_r_coord = x @ u_r
	x_r = x_r_coord @ u_r.T
	return x_r_coord, x_r


def _project_residual_features(x: torch.Tensor, u_r: torch.Tensor) -> torch.Tensor:
	_, x_r = _project_range_features(x, u_r)
	return x - x_r


def _project_residual_crosscov(c_xy: torch.Tensor, u_r: torch.Tensor, v_r: torch.Tensor) -> torch.Tensor:
	term_x = u_r @ (u_r.T @ c_xy)
	term_y = (c_xy @ v_r) @ v_r.T
	term_xy = u_r @ (u_r.T @ c_xy @ v_r) @ v_r.T
	return c_xy - term_x - term_y + term_xy


def _pairwise_row_kl_cost(
	q_syn: torch.Tensor,
	logq_syn: torch.Tensor,
	p_real: torch.Tensor,
	eps: float = 1e-8,
) -> torch.Tensor:
	logp_real = torch.log(p_real.clamp_min(eps))
	self_term = (q_syn * logq_syn).sum(dim=1, keepdim=True)
	cross_term = q_syn @ logp_real.T
	return (self_term - cross_term).clamp_min(0.0)


@torch.no_grad()
def _sinkhorn_uniform_transport(
	cost: torch.Tensor,
	reg: float = 0.05,
	n_iter: int = 20,
	eps: float = 1e-8,
) -> torch.Tensor:
	n, m = cost.shape
	a = torch.full((n,), 1.0 / max(n, 1), device=cost.device, dtype=cost.dtype)
	b = torch.full((m,), 1.0 / max(m, 1), device=cost.device, dtype=cost.dtype)
	cost_norm = cost / cost.mean().clamp_min(eps)
	k = torch.exp(-cost_norm / max(reg, eps)).clamp_min(eps)
	u, v = torch.ones_like(a), torch.ones_like(b)
	for _ in range(n_iter):
		u = a / (k @ v).clamp_min(eps)
		v = b / (k.T @ u).clamp_min(eps)
	t = (u.unsqueeze(1) * k) * v.unsqueeze(0)
	return t / t.sum().clamp_min(eps)


def _transport_row_dist_loss(
	logits_real: torch.Tensor,
	logits_syn: torch.Tensor,
	teacher_temp: float = 0.07,
	ot_reg: float = 0.05,
	ot_iters: int = 20,
	eps: float = 1e-8,
) -> torch.Tensor:
	if logits_real.shape[0] <= 1 or logits_real.shape[1] <= 1:
		return logits_real.new_tensor(0.0)
	p_real = torch.nn.functional.softmax(logits_real.detach() / max(teacher_temp, eps), dim=1)
	q_syn = torch.nn.functional.softmax(logits_syn / max(teacher_temp, eps), dim=1)
	logq_syn = torch.log(q_syn.clamp_min(eps))
	cost = _pairwise_row_kl_cost(q_syn, logq_syn, p_real, eps=eps)
	t = _sinkhorn_uniform_transport(cost.detach(), reg=ot_reg, n_iter=ot_iters, eps=eps)
	return (t * cost).sum() * (teacher_temp * teacher_temp)


def _bidirectional_transport_loss(
	logits_real: torch.Tensor,
	logits_syn: torch.Tensor,
	teacher_temp: float = 0.07,
	ot_reg: float = 0.05,
	ot_iters: int = 20,
	eps: float = 1e-8,
) -> torch.Tensor:
	i2t = _transport_row_dist_loss(logits_real, logits_syn, teacher_temp=teacher_temp, ot_reg=ot_reg, ot_iters=ot_iters, eps=eps)
	t2i = _transport_row_dist_loss(logits_real.T, logits_syn.T, teacher_temp=teacher_temp, ot_reg=ot_reg, ot_iters=ot_iters, eps=eps)
	return 0.5 * (i2t + t2i)


def hyperbolic_relevance_matching_loss(
	img_feat_syn: torch.Tensor,
	txt_feat_syn: torch.Tensor,
	img_feat_real: torch.Tensor,
	txt_feat_real: torch.Tensor,
	c: float = 1.0,
	scale: float = 1.0,
	relevance_temp: float = 0.07,
	rn_energy: float = 0.95,
	rn_max_rank: int | None = None,
	rn_ot_reg: float = 0.05,
	rn_ot_iters: int = 20,
	w_range_xcov: float = 1.0,
	w_residual_xcov: float = 1.0,
	w_residual_compress: float = 1.0,
	eps: float = 1e-8,
) -> dict:
	"""Range/residual subspace matching in tangent space: real->synthetic relevance + residual compress."""
	b = min(img_feat_syn.shape[0], txt_feat_syn.shape[0], img_feat_real.shape[0], txt_feat_real.shape[0])
	if b <= 1:
		z = img_feat_syn.new_tensor(0.0)
		return {
			"range_dist": z, "residual_dist": z, "range_expand": z, "residual_compress": z,
			"residual_ratio": z, "range_energy_real": z, "range_energy_syn": z, "residual_energy_syn": z,
			"total": z, "range_rank": 0,
		}
	x_s = img_feat_syn[:b].float()
	y_s = txt_feat_syn[:b].float()
	x_r = img_feat_real[:b].float().detach()
	y_r = txt_feat_real[:b].float().detach()
	x_r_tan = to_tangent_features(x_r, c=c, scale=scale, eps=eps)
	y_r_tan = to_tangent_features(y_r, c=c, scale=scale, eps=eps)
	x_s_tan = to_tangent_features(x_s, c=c, scale=scale, eps=eps)
	y_s_tan = to_tangent_features(y_s, c=c, scale=scale, eps=eps)
	c_real = _hyp_cross_cov(x_r_tan, y_r_tan, eps=eps)
	c_syn = _hyp_cross_cov(x_s_tan, y_s_tan, eps=eps)
	u_r, v_r, rank_used = range_residual_bases(c_real, energy=rn_energy, max_rank=rn_max_rank, eps=eps)
	x_r_range_coord, _ = _project_range_features(x_r_tan, u_r)
	y_r_range_coord, _ = _project_range_features(y_r_tan, v_r)
	x_s_range_coord, _ = _project_range_features(x_s_tan, u_r)
	y_s_range_coord, _ = _project_range_features(y_s_tan, v_r)
	x_r_residual = _project_residual_features(x_r_tan, u_r)
	y_r_residual = _project_residual_features(y_r_tan, v_r)
	x_s_residual = _project_residual_features(x_s_tan, u_r)
	y_s_residual = _project_residual_features(y_s_tan, v_r)
	rel_temp = max(relevance_temp, eps)
	logits_range_real = (x_r_range_coord @ y_r_range_coord.T) / rel_temp if rank_used > 0 else c_real.new_zeros((b, b))
	logits_range_syn = (x_s_range_coord @ y_s_range_coord.T) / rel_temp if rank_used > 0 else c_real.new_zeros((b, b))
	logits_residual_real = (x_r_residual @ y_r_residual.T) / rel_temp
	logits_residual_syn = (x_s_residual @ y_s_residual.T) / rel_temp
 

	import time

	start_time_range = time.time()
	range_dist = _bidirectional_transport_loss(
		logits_range_real, logits_range_syn, teacher_temp=rel_temp, ot_reg=rn_ot_reg, ot_iters=rn_ot_iters, eps=eps
	) if rank_used > 0 else c_real.new_tensor(0.0)
	end_time_range = time.time()
	print(f">> range_dist computation time: {end_time_range - start_time_range:.6f} seconds")

	start_time_residual = time.time()
	residual_dist = _bidirectional_transport_loss(
		logits_residual_real, logits_residual_syn, teacher_temp=rel_temp, ot_reg=rn_ot_reg, ot_iters=rn_ot_iters, eps=eps
	)
	end_time_residual = time.time()
	print(f">> residual_dist computation time: {end_time_residual - start_time_residual:.6f} seconds")



	
	crr_real = (u_r.T @ c_real @ v_r) if rank_used > 0 else c_real.new_zeros((0, 0))
	crr_syn = (u_r.T @ c_syn @ v_r) if rank_used > 0 else c_real.new_zeros((0, 0))
	range_energy_real = crr_real.pow(2).mean().detach() if crr_real.numel() > 0 else c_real.pow(2).mean().detach().clamp_min(eps)
	range_energy_syn = crr_syn.pow(2).mean() if crr_syn.numel() > 0 else c_syn.pow(2).mean().clamp_min(eps)
	range_expand = torch.nn.functional.relu(range_energy_real - range_energy_syn) / range_energy_real.clamp_min(eps)
	cnn_syn = _project_residual_crosscov(c_syn, u_r, v_r)
	residual_energy_syn = cnn_syn.pow(2).mean()
	residual_ratio = residual_energy_syn / range_energy_syn.clamp_min(eps)
	residual_compress = residual_ratio + torch.nn.functional.relu(residual_ratio - 1.0)
	total = (

		float(w_range_xcov) * (range_dist + range_expand)
		+ float(w_residual_xcov) * residual_dist + float(w_residual_compress) * residual_compress
  



  



  



  
  



		
	)
	return {
		"range_dist": range_dist, "residual_dist": residual_dist, "range_expand": range_expand,
		"residual_compress": residual_compress, "residual_ratio": residual_ratio,
		"range_energy_real": range_energy_real, "range_energy_syn": range_energy_syn,
		"residual_energy_syn": residual_energy_syn, "total": total, "range_rank": int(rank_used),
	}


def range_residual_itc_loss(
	img_feat_real: torch.Tensor,
	txt_feat_real: torch.Tensor,
	img_feat_syn: torch.Tensor,
	txt_feat_syn: torch.Tensor,
	c: float = 1.0,
	scale: float = 1.0,
	temperature: float = 0.07,
	relevance_temp: float = 0.07,
	rn_energy: float = 0.95,
	rn_max_rank: int | None = None,
	rn_ot_reg: float = 0.05,
	rn_ot_iters: int = 20,
	w_hyp_nce: float = 1.0,
	w_range_xcov: float = 0.6,
	w_residual_xcov: float = 0.25,
	w_residual_compress: float = 0.1,
	eps: float = 1e-8,
	use_wbce: bool = False,
) -> dict:
	"""Combined: base hyperbolic InfoNCE + range/residual real-synthetic matching."""
	if use_wbce:
		l_hyp_nce = hyperbolic_wbce_loss(
			img_feat_syn, txt_feat_syn, temperature=temperature, c=c, scale=scale, eps=eps
		)
	else:
		l_hyp_nce = hyperbolic_infonce_loss(
			img_feat_syn, txt_feat_syn, temperature=temperature, c=c, scale=scale, eps=eps
		)
	rel = hyperbolic_relevance_matching_loss(
		img_feat_syn, txt_feat_syn, img_feat_real, txt_feat_real,
		c=c, scale=scale, relevance_temp=relevance_temp, rn_energy=rn_energy,
		rn_max_rank=rn_max_rank, rn_ot_reg=rn_ot_reg, rn_ot_iters=rn_ot_iters,
		w_range_xcov=w_range_xcov, w_residual_xcov=w_residual_xcov, w_residual_compress=w_residual_compress, eps=eps,
	)

	total = l_hyp_nce
	total += rel["total"]
	total *= float(w_hyp_nce)

	return {
		"hyp_nce": l_hyp_nce,
		"range_dist": rel["range_dist"], "residual_dist": rel["residual_dist"],
		"range_expand": rel["range_expand"], "residual_compress": rel["residual_compress"],
		"residual_ratio": rel["residual_ratio"],
		"range_energy_real": rel["range_energy_real"], "range_energy_syn": rel["range_energy_syn"],
		"residual_energy_syn": rel["residual_energy_syn"],
		"relevance_total": rel["total"], "total": total,
		"range_rank": rel["range_rank"],
	}


def make_timestamp(prefix: str="", suffix: str="") -> str:
	KST_TIMEZONE = 9
	tmstamp = datetime.datetime.now() + datetime.timedelta(hours=KST_TIMEZONE)
	tmstamp = '{:%m_%d_%Y_%H%M%S}'.format(tmstamp)
	return tmstamp
	
def set_seed(seed):	
	import random
	from lightning.fabric import seed_everything
	seed_everything(seed)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.cuda.empty_cache()
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	os.environ['PYTHONHASHSEED'] = str(seed)
	

def clean_cache():
	import gc
	gc.collect()
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()
	torch.cuda.memory_summary(device=None, abbreviated=True)

def main(args):
	set_seed(args.seed)


	trainloader, testloader, train_dataset, test_dataset = get_dataloaders(args)
	train_iter = iter(trainloader)

	def get_batch_real(net, train_iter):
		try:
			images, text, _ = next(train_iter)
		except StopIteration:
			train_iter = iter(trainloader)
			images, text, _ = next(train_iter)
		
		if args.text_encoder in ['bert', 'distilbert']:
			encoding = net.text_encoder.tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, truncation=True)
			input_ids = encoding['input_ids'].to(args.device)
			attention_mask = encoding['attention_mask'].to(args.device)

			text_emb = net.text_encoder.model.embeddings(
				input_ids=input_ids,
			)
			return images, text_emb, attention_mask, train_iter
		elif args.text_encoder == 'clip':
			tokens = clip.tokenize(text).cuda()
			text_emb = net.text_encoder.model.token_embedding(tokens)
			return images, text_emb, tokens, train_iter


	realloader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=args.batch_size_train,
		num_workers=2,
		pin_memory=True,
		sampler=None,
		shuffle=True,
		collate_fn=None,
		drop_last=True,
	)

	real_iter = iter(realloader)


	print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))
	print('Hyper-parameters: \n', args.__dict__)



	args.name = 'HGA' if args.name == '' else args.name
	args.log_dir = f'{args.log_dir}/{args.name}'
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir, exist_ok=True)
	if args.wandb:
		wandb.init(
			entity='andyj1',
			project="ECCV2026_HGA_Submission",
			name=f"{args.name}_{args.dataset}_N{args.num_queries}_seed{args.seed}_{make_timestamp()}",
			config=args
		)
	else:
		wandb.init(mode = 'disabled')

	use_cuda_timing = torch.cuda.is_available()
	if use_cuda_timing:
		start_model_init = torch.cuda.Event(enable_timing=True)
		end_model_init = torch.cuda.Event(enable_timing=True)
		start_distill_iter = torch.cuda.Event(enable_timing=True)
		end_distill_iter = torch.cuda.Event(enable_timing=True)
		start_distillation = torch.cuda.Event(enable_timing=True)
		end_distillation = torch.cuda.Event(enable_timing=True)


	student_net = CLIPModel_full(args).to('cuda')
	student_net.eval()

	image_encoder_weights = copy.deepcopy(student_net.image_encoder.state_dict())
	text_encoder_weights = copy.deepcopy(student_net.text_encoder.state_dict())

	data_init_start = time.time()
	image_syn, text_syn, mask_syn = get_images_texts(
		args.num_queries,
		train_dataset,
		args,
		student_net.text_encoder,
		seed=args.seed,
	)
	data_init_time = time.time() - data_init_start
	print(f"Data init time: {data_init_time:.2f} seconds")

	del student_net


	image_syn = image_syn.detach().to(args.device).requires_grad_(True)
	text_syn = text_syn.detach().to(args.device).requires_grad_(True)

	optimizer = torch.optim.SGD(
		[
			{"params": [image_syn], "lr": args.lr_img, "momentum": args.momentum_syn},
			{"params": [text_syn], "lr": args.lr_txt, "momentum": args.momentum_syn},
		],
		lr=0,
	)
	optimizer.zero_grad()

	total_model_init_time = 0.0
	total_distill_time = 0.0
	model_init_time_count = 0
	distill_time_count = 0
	best_r_mean = -float("inf")
	best_eval_iter = -1
	best_eval_results = None
	early_stop_patience = 100
	model_init_times = []
	data_init_times = []
	distillation_times = []
	wall_clock_tracker = WallClockTracker()
	range_rank_history: list[dict] = []

	for it in tqdm(range(1, args.Iteration + 1), ncols=100):
		clean_cache()


		if it > 0 and it % args.eval_it == 0:
			print(
				"Evaluation\nimage_model_train = %s, text_model_train = %s, iteration = %d"
				% (args.image_encoder, args.text_encoder, it)
			)

			multi_eval_aggr_result = defaultdict(list)

			for it_eval in range(args.num_eval):
				net_eval = CLIPModel_full(args)

				net_eval.image_encoder.load_state_dict(image_encoder_weights)
				net_eval.text_encoder.load_state_dict(text_encoder_weights)

				(
					image_syn_eval,
					text_syn_eval,
					mask_syn_eval,
				) = (
					copy.deepcopy(image_syn.detach()),
					copy.deepcopy(text_syn.detach()),
					copy.deepcopy(mask_syn.detach()),
				)

				_, _, best_val_result = evaluate_synset(
					it_eval,
					net_eval,
					image_syn_eval,
					text_syn_eval,
					mask_syn_eval,
					testloader,
					test_dataset,
					args,
				)

				for k, v in best_val_result.items():
					multi_eval_aggr_result[k].append(v)

			for key, values in multi_eval_aggr_result.items():
				print(f"{key}: {np.mean(values):.2f} ({np.std(values):.2f})")

			table = PrettyTable()
			table.field_names = ["Metric", "Mean", "Std"]
			metric_order = [
				"img_r1",
				"img_r5",
				"img_r10",
				"img_r_mean",
				"txt_r1",
				"txt_r5",
				"txt_r10",
				"txt_r_mean",
				"r_mean",
			]
			for key in metric_order:
				if key in multi_eval_aggr_result:
					values = multi_eval_aggr_result[key]
					table.add_row(
						[key, f"{np.mean(values):.4f}", f"{np.std(values):.4f}"]
					)
			print(table)

			current_r_mean = None
			if "r_mean" in multi_eval_aggr_result:
				current_r_mean = float(np.mean(multi_eval_aggr_result["r_mean"]))
				if current_r_mean > best_r_mean:
					best_r_mean = current_r_mean
					best_eval_iter = it
					best_eval_results = copy.deepcopy(multi_eval_aggr_result)

				wc = wall_clock_tracker.stamp_eval(it=it, value=current_r_mean)


				wandb.log(
					{
						"WallClock/current_r_mean": current_r_mean,
						"WallClock/elapsed_s": wc["elapsed_s"],
					},
					step=it,
				)


				if wc["new_best"] and best_eval_results is not None:
					for key, values in best_eval_results.items():
						mean_val_b = float(np.mean(values))
						std_val_b = float(np.std(values))
						wandb.log(
							{
								f"BestEval/Mean/{key}": mean_val_b,
								f"BestEval/Std/{key}": std_val_b,
							},
							step=it,
						)

					best_path = os.path.join(args.log_dir, f"distilled_pairs_best_iter{it}.pt")
					torch.save(
						{
							"image": image_syn.detach().cpu(),
							"text": text_syn.detach().cpu(),
							"mask": mask_syn.detach().cpu(),
							"iter": it,
						},
						best_path,
					)
					print(f"Best results at iteration {it} saved to {best_path}")

					results_table = print_results(
						best_eval_results,
						title=(
							f"<best> image-text retrieval results "
							f"(avg {args.num_eval} evals) for {args.dataset} "
							f"at iteration {it}"
						),
					)
					print(results_table)

					msg = (
						f"[BEST] it={it} | r_mean={wc['best_val']:.4f} | "
						f"time_to_best={wc['best_time_s']:.2f} seconds"
					)
					print(msg)
					wandb.log(
						{
							"BestEval/iter": best_eval_iter,
							"WallClock/time_to_best_s": wc["best_time_s"],
							"WallClock/best_r_mean": wc["best_val"],
						},
						step=it,
					)

			eval_log_dict = {}
			for key, values in multi_eval_aggr_result.items():
				mean_val = float(np.mean(values))
				std_val = float(np.std(values))
				if key in ["img_r_mean", "txt_r_mean"]:
					continue
				wandb.log(
					{
						"Eval/Mean/{}".format(key): mean_val,
						"Eval/Std/{}".format(key): std_val,
					},
					step=it,
				)
				eval_log_dict[f"Eval/Mean/{key}"] = mean_val
				eval_log_dict[f"Eval/Std/{key}"] = std_val

			if current_r_mean is not None:
				eval_log_dict["Eval/Mean/r_mean"] = current_r_mean
				eval_log_dict["BestEval/Mean/r_mean"] = best_r_mean
				eval_log_dict["BestEval/iter"] = best_eval_iter

			if eval_log_dict:
				wandb.log(eval_log_dict, step=it)



			clean_cache()
			with torch.no_grad():
				save_dir = args.log_dir
				print("Saving to {}".format(save_dir))
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)

				image_save = image_syn.detach().cpu()
				text_save = text_syn.detach().cpu()
				mask_save = mask_syn.detach().cpu()

				torch.save({
					"image": image_save,
					"text": text_save,
					"mask": mask_save,
				}, os.path.join(save_dir, "distilled_{}.pt".format(it)) )

			if best_eval_iter >= 0 and (it - best_eval_iter) >= early_stop_patience:
				print(
					"[EarlyStopping] Stopping at iteration %d "
					"(no improvement in r_mean for %d iterations; "
					"best r_mean = %.4f at iteration %d)"
					% (it, early_stop_patience, best_r_mean, best_eval_iter)
				)
				break


		clean_cache()

		if use_cuda_timing:
			start_model_init.record()
		else:
			model_init_start_time = time.time()
		student_net = CLIPModel_full(args).to('cuda')
		student_net.eval()

		student_net.image_encoder.load_state_dict(image_encoder_weights)
		student_net.text_encoder.load_state_dict(text_encoder_weights)

		optimizer_net = torch.optim.SGD([
			{'params': student_net.image_encoder.parameters(), 'lr': args.lr_encoder_img},
			{'params': student_net.image_projection.parameters(), 'lr': args.lr_proj_img},
			{'params': student_net.text_encoder.parameters(), 'lr': args.lr_encoder_txt},
			{'params': student_net.text_projection.parameters(), 'lr': args.lr_proj_txt},
		], lr=0, momentum=0.9, weight_decay=0.0005)

		if use_cuda_timing:
			end_model_init.record()
			torch.cuda.synchronize()
			model_init_time = start_model_init.elapsed_time(end_model_init) / 1000.0
		else:
			model_init_time = time.time() - model_init_start_time
		model_init_times.append(model_init_time)
		if float(model_init_time) > 0.0:
			total_model_init_time += float(model_init_time)
			model_init_time_count += 1
		data_init_times.append(data_init_time)

		if use_cuda_timing:
			start_distill_iter.record()
		else:
			distill_start_time = time.time()

		hyp_losses: dict = {}
		for ol in range(args.outer_loop):
			if use_cuda_timing:
				start_distillation.record()
			else:
				start_ol = time.time()
			student_net.eval()


			with torch.no_grad():
				image_real, text_real, mask_real, train_iter = get_batch_real(student_net, train_iter)

				image_real = image_real.to(args.device).detach()
				text_real = text_real.to(args.device).detach()

				img_embed_real = student_net.image_encoder(image_real)
				img_embed_real = img_embed_real.float()
				img_feat_real = student_net.image_projection(img_embed_real)

				txt_embed_real = student_net.text_encoder(text_real, mask_real)
				txt_embed_real = txt_embed_real.float()
				txt_feat_real = student_net.text_projection(txt_embed_real)

			if args.num_queries > args.batch_syn:
				idx_batch = np.random.permutation(args.num_queries)[:args.batch_syn]
				image_syn_batch = image_syn[idx_batch]
				text_syn_batch = text_syn[idx_batch]
				mask_syn_batch = mask_syn[idx_batch]
			else:
				image_syn_batch = image_syn
				text_syn_batch = text_syn
				mask_syn_batch = mask_syn

			img_embed_syn = student_net.image_encoder(image_syn_batch)
			img_embed_syn = img_embed_syn.float()
			img_feat_syn = student_net.image_projection(img_embed_syn)

			txt_embed_syn = student_net.text_encoder(text_syn_batch, mask_syn_batch)
			txt_embed_syn = txt_embed_syn.float()
			txt_feat_syn = student_net.text_projection(txt_embed_syn)

			loss_cov = torch.tensor(0.0, device=args.device)





			loss_img_feat = torch.tensor(0.0, device=args.device)
			loss_txt_feat = torch.tensor(0.0, device=args.device)


			if getattr(args, "w_hyp", 0.0) > 0.0:
				hyp_losses = range_residual_itc_loss(
					img_feat_real,
					txt_feat_real,
					img_feat_syn,
					txt_feat_syn,
					c=max(float(getattr(args, "hyperbolic_c", 1.0)), 1e-8),
					scale=float(getattr(args, "hyp_scale", 1.0)),
					temperature=float(getattr(args, "hyp_temperature", 0.07)),
					relevance_temp=float(getattr(args, "relevance_temp", 0.07)),
					rn_energy=float(getattr(args, "rn_energy", 0.95)),
					rn_max_rank=None if getattr(args, "rn_max_rank", -1) <= 0 else args.rn_max_rank,
					rn_ot_reg=float(getattr(args, "rn_ot_reg", 0.05)),
					rn_ot_iters=int(getattr(args, "rn_ot_iters", 20)),
					w_hyp_nce=float(getattr(args, "w_hyp_nce", 1.0)),
					w_range_xcov=float(getattr(args, "w_range_xcov", 0.6)),
					w_residual_xcov=float(getattr(args, "w_residual_xcov", 0.25)),
					w_residual_compress=float(getattr(args, "w_residual_compress", 0.1)),
					use_wbce=bool(getattr(args, "use_wbce", False)),
				)
				loss_hyp_total = hyp_losses["total"]
			else:
				loss_hyp_total = torch.tensor(0.0, device=args.device)
				hyp_losses = {}

			total_loss = getattr(args, "w_hyp", 0.0) * loss_hyp_total
			
			optimizer.zero_grad()
			total_loss.backward()
			optimizer.step()
			if use_cuda_timing:
				end_distillation.record()
				torch.cuda.synchronize()
				distillation_times.append(
					start_distillation.elapsed_time(end_distillation) / 1000.0
				)
			else:
				distillation_times.append(time.time() - start_ol)

			norm_img = torch.linalg.norm(image_syn.view(image_syn.shape[0], -1), dim=1)
			norm_img = torch.mean(norm_img)

			norm_txt = torch.linalg.norm(text_syn, dim=1)
			norm_txt = torch.mean(norm_txt)


			log_dict = {
				"Loss/total_loss": total_loss.item(),



				"Norm/norm_img": norm_img.item(),
				"Norm/norm_txt": norm_txt.item(),
			}
			if hyp_losses:
				log_dict["Loss/hyp_total"] = loss_hyp_total.item()
				log_dict["Loss/hyp_nce"] = hyp_losses["hyp_nce"].item()
				log_dict["Loss/range_dist"] = hyp_losses["range_dist"].item()
				log_dict["Loss/residual_dist"] = hyp_losses["residual_dist"].item()
				log_dict["Loss/range_expand"] = hyp_losses["range_expand"].item()
				log_dict["Loss/residual_compress"] = hyp_losses["residual_compress"].item()
				log_dict["RangeResidual/range_rank"] = int(hyp_losses["range_rank"])
			wandb.log(log_dict, step=it)

			if ol == args.outer_loop - 1:
				break


			student_net.train()
			loss_train, acc_train, num_exp = 0, 0, 0

			for i in range(args.inner_loop):
				try:
					image, text_raw, _ = next(real_iter)
				except StopIteration:
					real_iter = iter(realloader)

				with torch.no_grad():
					encoding = student_net.text_encoder.tokenizer.batch_encode_plus(text_raw, return_tensors='pt', padding=True, truncation=True)
					input_ids = encoding['input_ids'].to(args.device)
					mask = encoding['attention_mask'].to(args.device)

					text = student_net.text_encoder.model.embeddings(
						input_ids=input_ids,
					)

				image = image.to(args.device)
				n_b = image.shape[0]

				loss, acc = student_net(image, text, mask)

				loss_train += loss.item() * n_b
				acc_train += acc
				num_exp += n_b

				optimizer_net.zero_grad()
				loss.backward()
				optimizer_net.step()

			if num_exp > 0:
				loss_train /= num_exp
				acc_train /= num_exp
			else:
				loss_train = 0.0
				acc_train = 0.0

			torch.cuda.empty_cache()

		if hyp_losses and "range_rank" in hyp_losses:
			range_rank_history.append(
				{"it": int(it), "k": int(hyp_losses["range_rank"])}
			)
		else:
			range_rank_history.append({"it": int(it), "k": None})

		if use_cuda_timing:
			end_distill_iter.record()
			torch.cuda.synchronize()
			distill_time = start_distill_iter.elapsed_time(end_distill_iter) / 1000.0
		else:
			distill_time = time.time() - distill_start_time
		if float(distill_time) > 0.0:
			total_distill_time += float(distill_time)
			distill_time_count += 1

		avg_model_init_time = total_model_init_time / max(model_init_time_count, 1)
		avg_distill_time = total_distill_time / max(distill_time_count, 1)

		print(
			"\n[Wallclock] iter = %04d | model_init = %.3fs (avg %.3fs) | "
			"distill = %.3fs (avg %.3fs)"
			% (
				it,
				model_init_time,
				avg_model_init_time,
				distill_time,
				avg_distill_time,
			)
		)

		wandb.log(
			{
				"Time/model_init": model_init_time,
				"Time/model_init_avg": avg_model_init_time,
				"Time/distill": distill_time,
				"Time/distill_avg": avg_distill_time,
			},
			step=it,
		)

		if it > 0 and it % args.log_freq == 0:
			print('%s iter = %04d, total_loss = %.4f, loss_img_feat = %.4f, loss_txt_feat = %.4f, loss_cov = %.4f, norm_img = %.4f, norm_txt = %.4f' % (get_time(), it, total_loss.item(), loss_img_feat.item(), loss_txt_feat.item(), loss_cov.item(), norm_img.item(), norm_txt.item()))
		
		if it > 0 and it % args.log_freq == 0:
			print_memory_usage(step=it)
			print(f'---------------[yellow]{it}th iteration[/yellow]-----------------')
			if data_init_times:
				print(f'time taken to get data: {np.mean(data_init_times):.2f} seconds')
			if model_init_times:
				mi_nonzero = [t for t in model_init_times if float(t) > 0.0]
				print(f'average time taken to initialize model: {(np.mean(mi_nonzero) if mi_nonzero else 0.0):.2f} seconds')
			if distillation_times:
				di_nonzero = [t for t in distillation_times if float(t) > 0.0]
				print(f'average time taken to distillation: {(np.mean(di_nonzero) if di_nonzero else 0.0):.2f} seconds')
			print(f'---------------[yellow]End of {it}th iteration[/yellow]-----------------')

		del student_net

	save_dir = args.log_dir
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	json_path = wall_clock_tracker.finalize(save_dir)
	time_to_best_s = f"{wall_clock_tracker.best_time_s:.2f}s" if wall_clock_tracker.best_time_s is not None else "N/A"
	time_to_target_s = f"{wall_clock_tracker.target_time_s:.2f}s" if wall_clock_tracker.target_time_s is not None else "N/A"
	print(
		f"[WallClock] best {wall_clock_tracker.metric_name}={wall_clock_tracker.best_val:.4f} | "
		f"time_to_best={time_to_best_s} | time_to_target={time_to_target_s}"
	)
	print(f"[WallClock] history saved -> {json_path}")

	if range_rank_history:
		rn_json, rn_png = save_range_rank_history(range_rank_history, save_dir)
		print(f"[RangeResidual] rank k history saved -> {rn_json}")
		if rn_png:
			print(f"[RangeResidual] rank k plot saved -> {rn_png}")

	wandb.finish()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Parameter Processing')


	
	parser.add_argument('--name', type=str, default='')
	parser.add_argument('--dataset', type=str, default='flickr', help='dataset')
	parser.add_argument('--num_queries', type=int, default=100, help='number of queries')
	parser.add_argument('--rho', type=float, default=1.0, help='scaling factor for real cross-covariance')
	parser.add_argument('--lamda', type=float, default=1.0, help='weight for feature matching loss')
	parser.add_argument('--w_cov', type=float, default=1.0, help='weight for covariance loss (0 to disable)')


	parser.add_argument('--w_hyp', type=float, default=1.0, help='weight for hyperbolic block (0 to disable)')
	parser.add_argument('--hyperbolic_c', type=float, default=1.0, help='Lorentz curvature')
	parser.add_argument('--hyp_scale', type=float, default=1.0, help='scale before exp-map (Euclidean -> Lorentz)')
	parser.add_argument('--hyp_temperature', type=float, default=0.07, help='temperature for hyperbolic InfoNCE')
	parser.add_argument('--use_wbce', default=False, type=bool, help='use hyperbolic weighted BCE instead of InfoNCE for base hyp term')
	parser.add_argument('--w_hyp_nce', type=float, default=1.0, help='weight: base hyperbolic bidirectional InfoNCE')
	parser.add_argument('--w_range_xcov', type=float, default=0.6, help='weight: range subspace real-synthetic matching')
	parser.add_argument('--w_residual_xcov', dest='w_residual_xcov', type=float, default=0.25, help='weight: residual subspace real-synthetic matching')
	parser.add_argument('--w_residual_compress', dest='w_residual_compress', type=float, default=0.1, help='weight: residual/range ratio regularization')
	parser.add_argument('--relevance_temp', type=float, default=0.07, help='temperature for range/residual relevance logits')
	parser.add_argument('--rn_energy', type=float, default=0.95, help='energy ratio for range rank selection')
	parser.add_argument('--rn_max_rank', type=int, default=-1, help='max range rank (-1 = no cap)')
	parser.add_argument('--rn_ot_reg', type=float, default=0.05, help='entropic OT reg for range/residual transport')
	parser.add_argument('--rn_ot_iters', type=int, default=20, help='Sinkhorn iters for range/residual transport')


	parser.add_argument('--image_encoder', type=str, default='nfnet',  help='image encoder')
	parser.add_argument('--text_encoder', type=str, default='bert', help='text encoder')
	parser.add_argument('--image_pretrained', type=bool, default=True, help='image_pretrained')
	parser.add_argument('--text_pretrained', type=bool, default=True, help='text_pretrained')
	parser.add_argument('--image_trainable', type=bool, default=True, help='image_trainable')
	parser.add_argument('--text_trainable', type=bool, default=True, help='text_trainable')
	parser.add_argument('--proj_dim', type=int, default=2304, help='projection dimension')


	parser.add_argument('--image_size', type=int, default=224, help='image_size')
	parser.add_argument('--ann_root', type=str, default='./data/Flickr30k_ann/', help='location of ann root')
	parser.add_argument('--image_root', type=str, default='distill_utils/data/Flickr30k/', help='location of image root')
	parser.add_argument('--coyo_max_samples', type=int, default=None, help='optional COYO train split cap')
	parser.add_argument('--coyo_eval_max_samples', type=int, default=10000, help='optional COYO retrieval eval split cap')


	parser.add_argument('--Iteration', type=int, default=3000, help='how many distillation steps to perform')
	parser.add_argument('--outer_loop', type=int, default=50, help='number of online model update before initialization')
	parser.add_argument('--inner_loop', type=int, default=1, help='number of training steps for one online model update')
	parser.add_argument('--batch_size_train', type=int, default=128, help='batch_size_train (for real)')
	parser.add_argument('--batch_syn', type=int, default=64, help='batch_syn')
	parser.add_argument('--lr_img', type=float, default=1, help='learning rate for updating synthetic images')
	parser.add_argument('--lr_txt', type=float, default=1, help='learning rate for updating synthetic texts')
	parser.add_argument('--momentum_syn', type=float, default=0.5)


	parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')
	parser.add_argument('--eval_eval_it', type=int, default=100, help='how often to evaluate the evaluation model')
	parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
	parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
	parser.add_argument('--batch_size_test', type=int, default=256, help='batch_size_test')
	parser.add_argument('--lr_encoder_img', type=float, default=0.01, help='learning rate for updating network parameters')
	parser.add_argument('--lr_encoder_txt', type=float, default=0.01, help='learning rate for updating network parameters')
	parser.add_argument('--lr_proj_img', type=float, default=0.1, help='learning rate for updating network parameters')
	parser.add_argument('--lr_proj_txt', type=float, default=0.1, help='learning rate for updating network parameters')


	parser.add_argument('--wandb', action="store_true", help='wandb')
	parser.add_argument('--save', action="store_true", help='save')
	parser.add_argument('--device', type=str, default='cuda', help='device')
	parser.add_argument('--seed', type=int, default=0, help='seed')
	parser.add_argument('--log_dir', type=str, default='results', help='path to save synthetic dataset')
	parser.add_argument('--log_freq', type=int, default=10, help='iteration frequency for detailed timing and memory logs')
	args = parser.parse_args()

	args.log_dir = os.path.join(
		args.log_dir,
		args.dataset,
		str(args.num_queries),
		make_timestamp(),
	)
	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir)

	BASE_DIR = '/mnt/sdc/mdd_datasets'
	if args.dataset == 'flickr':
		args.image_root = os.path.join(BASE_DIR, 'flickr30k')
	elif args.dataset == 'flickr8k':
		args.image_root = os.path.join(BASE_DIR, 'flickr8k')
	elif args.dataset == 'coco':
		args.image_root = os.path.join(BASE_DIR, 'coco2014')
	elif args.dataset == 'cc3m_595k_llava':
		args.image_root = os.path.join(BASE_DIR, 'cc3m_595k_llava')
	elif args.dataset == 'coyo':
		args.image_root = '/mnt/sdf/mdd_datasets/coyo_700m'
	elif args.dataset == 'mmcelebahq':
		args.image_root = '/mnt/sdk/mdd_datasets/mmcelebahq/multi-modal-celeba'
	
	args.ann_root = os.path.join(BASE_DIR, 'annotations_retrieval')

	main(args)
