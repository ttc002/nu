from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np


# Координаты фазового вектора:
# [x, px, y, py, z, delta]
# x, y, z [м], px=dx/ds, py=dy/ds [рад], delta = dP/P0


def _focus_block(k: float, L: float) -> np.ndarray:
	"""2x2 блок линейной оптики для уравнения u'' + k u = 0."""
	if abs(k) < 1e-14:
		return np.array([[1.0, L], [0.0, 1.0]], dtype=float)

	if k > 0.0:
		root = np.sqrt(k)
		phi = root * L
		c = np.cos(phi)
		s = np.sin(phi)
		return np.array([[c, s / root], [-root * s, c]], dtype=float)

	root = np.sqrt(-k)
	phi = root * L
	ch = np.cosh(phi)
	sh = np.sinh(phi)
	return np.array([[ch, sh / root], [root * sh, ch]], dtype=float)


def twiss_to_sigma(beta: float, alpha: float, emit: float) -> np.ndarray:
	gamma = (1.0 + alpha**2) / beta
	return emit * np.array([[beta, -alpha], [-alpha, gamma]], dtype=float)


def sigma_to_twiss(s2: np.ndarray) -> Tuple[float, float, float]:
	det = np.linalg.det(s2)
	emit = np.sqrt(max(det, 0.0))
	if emit < 1e-20:
		return np.nan, np.nan, 0.0
	beta = s2[0, 0] / emit
	alpha = -s2[0, 1] / emit
	return beta, alpha, emit


def build_sigma6(
	beta_x: float,
	alpha_x: float,
	emit_x: float,
	beta_y: float,
	alpha_y: float,
	emit_y: float,
	sigma_z: float,
	sigma_delta: float,
	Dx: float = 0.0,
	Dpx: float = 0.0,
	Dy: float = 0.0,
	Dpy: float = 0.0,
) -> np.ndarray:
	s = np.zeros((6, 6), dtype=float)

	sx = twiss_to_sigma(beta_x, alpha_x, emit_x)
	sy = twiss_to_sigma(beta_y, alpha_y, emit_y)

	s[0:2, 0:2] = sx
	s[2:4, 2:4] = sy
	s[4, 4] = sigma_z**2
	s[5, 5] = sigma_delta**2

	# Добавка от дисперсии
	s[0, 0] += (Dx * sigma_delta) ** 2
	s[1, 1] += (Dpx * sigma_delta) ** 2
	s[2, 2] += (Dy * sigma_delta) ** 2
	s[3, 3] += (Dpy * sigma_delta) ** 2

	s[0, 1] += Dx * Dpx * sigma_delta**2
	s[1, 0] = s[0, 1]
	s[2, 3] += Dy * Dpy * sigma_delta**2
	s[3, 2] = s[2, 3]

	s[0, 5] = Dx * sigma_delta**2
	s[5, 0] = s[0, 5]
	s[1, 5] = Dpx * sigma_delta**2
	s[5, 1] = s[1, 5]
	s[2, 5] = Dy * sigma_delta**2
	s[5, 2] = s[2, 5]
	s[3, 5] = Dpy * sigma_delta**2
	s[5, 3] = s[3, 5]

	return s


def sample_gaussian_beam(n_particles: int, sigma6: np.ndarray, seed: int = 1) -> np.ndarray:
	"""
	Генерация гауссова пучка с максимально точным соответствием заданной Sigma-матрице
	в выборочной оценке (для устойчивой проверки дисперсии).
	"""
	rng = np.random.default_rng(seed)
	if n_particles < 8:
		mean = np.zeros(6, dtype=float)
		return rng.multivariate_normal(mean=mean, cov=sigma6, size=n_particles)

	# 1) Стандартные нормальные координаты
	z = rng.standard_normal(size=(n_particles, 6))
	z -= np.mean(z, axis=0, keepdims=True)

	# 2) Whiten: приводим выборочную ковариацию z к единичной
	cz = (z.T @ z) / (n_particles - 1)
	Lz = np.linalg.cholesky(cz)
	z_white = z @ np.linalg.inv(Lz.T)

	# 3) Накладываем целевую ковариацию
	Ls = np.linalg.cholesky(sigma6)
	p = z_white @ Ls.T

	# Численно центрируем
	p -= np.mean(p, axis=0, keepdims=True)
	return p


@dataclass
class Element:
	name: str
	L: float

	def linear_matrix(self) -> np.ndarray:
		raise NotImplementedError

	def track_particles(self, particles: np.ndarray) -> np.ndarray:
		"""Нелинейный трекинг (по умолчанию линейный матричный)."""
		R = self.linear_matrix()
		return particles @ R.T


@dataclass
class Drift(Element):
	def linear_matrix(self) -> np.ndarray:
		R = np.eye(6, dtype=float)
		R[0, 1] = self.L
		R[2, 3] = self.L
		return R


@dataclass
class Quadrupole(Element):
	k1: float
	n_slices: int = 20

	def linear_matrix(self) -> np.ndarray:
		R = np.eye(6, dtype=float)
		mx = _focus_block(self.k1, self.L)
		my = _focus_block(-self.k1, self.L)
		R[0:2, 0:2] = mx
		R[2:4, 2:4] = my
		return R

	def track_particles(self, particles: np.ndarray) -> np.ndarray:
		# Симплектический split-operator, чтобы легко расширять нелинейности
		p = particles.copy()
		ds = self.L / self.n_slices
		half = 0.5 * ds

		for _ in range(self.n_slices):
			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

			# kick
			p[:, 1] += -self.k1 * p[:, 0] * ds
			p[:, 3] += self.k1 * p[:, 2] * ds

			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

		return p


@dataclass
class SkewQuadrupole(Element):
	k1s: float
	n_slices: int = 20

	def linear_matrix(self) -> np.ndarray:
		# Тонкая линза + дрейфы по краям (достаточно для инженерной модели)
		d = Drift(name=f"{self.name}_d", L=0.5 * self.L).linear_matrix()
		kL = self.k1s * self.L
		kick = np.eye(6, dtype=float)
		kick[1, 2] = -kL
		kick[3, 0] = -kL
		return d @ kick @ d

	def track_particles(self, particles: np.ndarray) -> np.ndarray:
		p = particles.copy()
		ds = self.L / self.n_slices
		half = 0.5 * ds

		for _ in range(self.n_slices):
			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

			# скрю-квадруполь: x-y coupling
			p[:, 1] += -self.k1s * p[:, 2] * ds
			p[:, 3] += -self.k1s * p[:, 0] * ds

			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

		return p


@dataclass
class Sextupole(Element):
	k2: float
	n_slices: int = 24

	def linear_matrix(self) -> np.ndarray:
		# В линейной модели sextupole не даёт первого порядка около орбиты x=y=0.
		return Drift(name=f"{self.name}_lin", L=self.L).linear_matrix()

	def track_particles(self, particles: np.ndarray) -> np.ndarray:
		# Симплектический drift-kick-drift для нелинейного поля sextupole
		p = particles.copy()
		ds = self.L / self.n_slices
		half = 0.5 * ds

		for _ in range(self.n_slices):
			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

			x = p[:, 0]
			y = p[:, 2]
			# x' += -(k2/2)(x^2-y^2) ds ; y' += k2*x*y ds
			p[:, 1] += -0.5 * self.k2 * (x * x - y * y) * ds
			p[:, 3] += self.k2 * x * y * ds

			p[:, 0] += half * p[:, 1]
			p[:, 2] += half * p[:, 3]

		return p


@dataclass
class SectorDipole(Element):
	angle: float
	e1: float = 0.0
	e2: float = 0.0
	k1: float = 0.0
	n_slices: int = 40

	@property
	def rho(self) -> float:
		if abs(self.angle) < 1e-14:
			return np.inf
		return self.L / self.angle

	def linear_matrix(self) -> np.ndarray:
		def edge_matrix(psi: float) -> np.ndarray:
			E = np.eye(6, dtype=float)
			if abs(self.angle) < 1e-14:
				return E
			h = 1.0 / self.rho
			t = np.tan(psi)
			# Стандартная тонкая модель краевой фокусировки
			# x' += h*tan(psi)*x, y' += -h*tan(psi)*y
			E[1, 0] = h * t
			E[3, 2] = -h * t
			return E

		R = np.eye(6, dtype=float)
		if abs(self.angle) < 1e-14:
			R[0, 1] = self.L
			R[2, 3] = self.L
			return R

		h = 1.0 / self.rho
		kx = h * h + self.k1
		ky = -self.k1

		mx = _focus_block(kx, self.L)
		my = _focus_block(ky, self.L)
		R[0:2, 0:2] = mx
		R[2:4, 2:4] = my

		# Дисперсионный отклик от кривизны в combined-function dipole
		if abs(kx) < 1e-14:
			R[0, 5] = 0.5 * h * self.L * self.L
			R[1, 5] = h * self.L
		elif kx > 0.0:
			root = np.sqrt(kx)
			phi = root * self.L
			R[0, 5] = (h / kx) * (1.0 - np.cos(phi))
			R[1, 5] = (h / root) * np.sin(phi)
		else:
			root = np.sqrt(-kx)
			phi = root * self.L
			R[0, 5] = (h / (-kx)) * (np.cosh(phi) - 1.0)
			R[1, 5] = (h / root) * np.sinh(phi)

		E1 = edge_matrix(self.e1)
		E2 = edge_matrix(self.e2)
		return E2 @ R @ E1

	def track_particles(self, particles: np.ndarray) -> np.ndarray:
		# Нелинейный шаговый трекинг в кривизне орбиты
		if abs(self.angle) < 1e-14:
			return Drift(name=f"{self.name}_dr", L=self.L).track_particles(particles)

		def apply_edge_kick(p: np.ndarray, psi: float) -> None:
			h = 1.0 / self.rho
			t = np.tan(psi)
			p[:, 1] += h * t * p[:, 0]
			p[:, 3] += -h * t * p[:, 2]

		p = particles.copy()
		ds = self.L / self.n_slices
		h = 1.0 / self.rho

		apply_edge_kick(p, self.e1)

		for _ in range(self.n_slices):
			# Drift-like advance
			p[:, 0] += p[:, 1] * ds
			p[:, 2] += p[:, 3] * ds

			# Curvature kick (слабонелинейная модель)
			# x'' + (h^2 + k1)x = h * delta / (1+delta)
			# y'' - k1 y = 0
			denom = 1.0 + p[:, 5]
			denom = np.where(np.abs(denom) < 1e-8, np.sign(denom) * 1e-8, denom)
			p[:, 1] += (-(h * h + self.k1) * p[:, 0] + h * p[:, 5] / denom) * ds
			p[:, 3] += (self.k1 * p[:, 2]) * ds

		apply_edge_kick(p, self.e2)

		return p


@dataclass
class Beamline:
	elements: List[Element] = field(default_factory=list)

	def add(self, element: Element) -> None:
		self.elements.append(element)

	def total_matrix(self) -> np.ndarray:
		R = np.eye(6, dtype=float)
		for elem in self.elements:
			R = elem.linear_matrix() @ R
		return R

	def track_linear(self, particles: np.ndarray) -> np.ndarray:
		p = particles.copy()
		for elem in self.elements:
			p = p @ elem.linear_matrix().T
		return p

	def track_nonlinear(self, particles: np.ndarray) -> np.ndarray:
		p = particles.copy()
		for elem in self.elements:
			p = elem.track_particles(p)
		return p

	def propagate_sigma_linear(self, sigma0: np.ndarray) -> np.ndarray:
		R = self.total_matrix()
		return R @ sigma0 @ R.T


def estimate_phase_ellipse(particles: np.ndarray) -> dict:
	cov = np.cov(particles.T, bias=False)
	bx, ax, ex = sigma_to_twiss(cov[0:2, 0:2])
	by, ay, ey = sigma_to_twiss(cov[2:4, 2:4])

	# Оценка дисперсии из ковариаций с delta
	sdd = cov[5, 5]
	Dx = cov[0, 5] / sdd if sdd > 0 else np.nan
	Dpx = cov[1, 5] / sdd if sdd > 0 else np.nan
	Dy = cov[2, 5] / sdd if sdd > 0 else np.nan
	Dpy = cov[3, 5] / sdd if sdd > 0 else np.nan

	return {
		"beta_x": bx,
		"alpha_x": ax,
		"emit_x": ex,
		"beta_y": by,
		"alpha_y": ay,
		"emit_y": ey,
		"Dx": Dx,
		"Dpx": Dpx,
		"Dy": Dy,
		"Dpy": Dpy,
	}


def match_zero_dispersion(
	beamline: Beamline,
	quad_refs: List[Quadrupole],
	max_iter: int = 60,
	tol: float = 1e-10,
) -> Tuple[float, float]:
	"""
	Подстройка 4 квадруполей, чтобы на выходе из канала получить Dx≈0 и Dpx≈0.
	Используется coordinate search без внешних библиотек.
	"""

	def objective() -> Tuple[float, float, float]:
		R = beamline.total_matrix()
		Dx = R[0, 5]
		Dpx = R[1, 5]
		return Dx, Dpx, Dx * Dx + Dpx * Dpx

	damping = 0.8
	eps = 1e-4

	for _ in range(max_iter):
		Dx, Dpx, f0 = objective()
		if f0 < tol:
			break

		F = np.array([Dx, Dpx], dtype=float)

		# Jacobian J_ij = dFi/dk_j, i in [Dx, Dpx], j по квадруполям
		J = np.zeros((2, len(quad_refs)), dtype=float)
		for j, q in enumerate(quad_refs):
			k0 = q.k1
			q.k1 = k0 + eps
			Dx1, Dpx1, _ = objective()
			q.k1 = k0 - eps
			Dx2, Dpx2, _ = objective()
			q.k1 = k0

			J[0, j] = (Dx1 - Dx2) / (2.0 * eps)
			J[1, j] = (Dpx1 - Dpx2) / (2.0 * eps)

		# Минимизируем ||F + J*dk|| в МНК смысле
		dk = -np.linalg.pinv(J) @ F

		# Ограничим шаг, чтобы не улететь
		max_step = 1.5
		norm_inf = float(np.max(np.abs(dk))) if dk.size else 0.0
		if norm_inf > max_step:
			dk *= max_step / norm_inf

		# backtracking по функционалу
		k_old = [q.k1 for q in quad_refs]
		accepted = False
		for _ls in range(8):
			for q, k_base, dki in zip(quad_refs, k_old, dk):
				q.k1 = k_base + damping * float(dki)

			_, _, f_try = objective()
			if f_try < f0:
				accepted = True
				break
			damping *= 0.6

		if not accepted:
			# если не улучшили - мягко дёргаем один канал
			for q, k_base in zip(quad_refs, k_old):
				q.k1 = k_base
			break

		damping = min(0.9, damping * 1.15)

	Dx, Dpx, _ = objective()
	return Dx, Dpx


def match_quads_to_targets(
	beamline: Beamline,
	quad_refs: List[Quadrupole],
	sigma_in: np.ndarray,
	target_out: dict,
	weights: dict | None = None,
	keys: List[str] | None = None,
	scales: dict | None = None,
	max_iter: int = 80,
	tol: float = 1e-10,
) -> dict:
	"""
	Подстройка квадруполей по многокритериальной функции:
	- beta_x, alpha_x, beta_y, alpha_y
	- Dx, Dpx, Dy, Dpy
	"""
	if weights is None:
		weights = {
			"beta_x": 1.0,
			"alpha_x": 1.0,
			"beta_y": 1.0,
			"alpha_y": 1.0,
			"Dx": 30.0,
			"Dpx": 30.0,
			"Dy": 8.0,
			"Dpy": 8.0,
		}

	if keys is None:
		# 4 квадруполя -> разумно 4-6 целевых компонент в МНК,
		# приоритетом оставляем дисперсию.
		keys = ["beta_x", "alpha_x", "beta_y", "alpha_y", "Dx", "Dpx"]

	if scales is None:
		scales = {
			"beta_x": max(abs(float(target_out["beta_x"])), 1.0),
			"alpha_x": 1.0,
			"beta_y": max(abs(float(target_out["beta_y"])), 1.0),
			"alpha_y": 1.0,
			"Dx": 1e-3,
			"Dpx": 1e-3,
			"Dy": 1e-3,
			"Dpy": 1e-3,
		}

	def model_params() -> dict:
		s = beamline.propagate_sigma_linear(sigma_in)
		bx, ax, _ = sigma_to_twiss(s[0:2, 0:2])
		by, ay, _ = sigma_to_twiss(s[2:4, 2:4])
		sdd = s[5, 5]
		Dx = s[0, 5] / sdd if sdd > 0 else np.nan
		Dpx = s[1, 5] / sdd if sdd > 0 else np.nan
		Dy = s[2, 5] / sdd if sdd > 0 else np.nan
		Dpy = s[3, 5] / sdd if sdd > 0 else np.nan
		return {
			"beta_x": bx,
			"alpha_x": ax,
			"beta_y": by,
			"alpha_y": ay,
			"Dx": Dx,
			"Dpx": Dpx,
			"Dy": Dy,
			"Dpy": Dpy,
		}

	def residual_vec() -> np.ndarray:
		p = model_params()
		r = []
		for k in keys:
			scale = float(scales.get(k, 1.0))
			wk = float(weights.get(k, 1.0))
			r.append(wk * (float(p[k]) - float(target_out[k])) / scale)
		return np.array(r, dtype=float)

	eps = 1e-5
	lmbd = 1e-2

	for _ in range(max_iter):
		r0 = residual_vec()
		f0 = float(r0 @ r0)
		if f0 < tol:
			break

		J = np.zeros((len(r0), len(quad_refs)), dtype=float)
		for j, q in enumerate(quad_refs):
			k0 = q.k1
			q.k1 = k0 + eps
			r_plus = residual_vec()
			q.k1 = k0 - eps
			r_minus = residual_vec()
			q.k1 = k0
			J[:, j] = (r_plus - r_minus) / (2.0 * eps)

		A = J.T @ J + lmbd * np.eye(len(quad_refs), dtype=float)
		b = -(J.T @ r0)
		dk = np.linalg.solve(A, b)
		norm_inf = float(np.max(np.abs(dk))) if dk.size else 0.0
		if norm_inf > 0.4:
			dk *= 0.4 / norm_inf

		k_old = [q.k1 for q in quad_refs]
		accepted = False
		for a in (1.0, 0.5, 0.25, 0.1, 0.05):
			for q, kb, dki in zip(quad_refs, k_old, dk):
				q.k1 = kb + a * float(dki)
				q.k1 = float(np.clip(q.k1, -15.0, 15.0))

			r_try = residual_vec()
			f_try = float(r_try @ r_try)
			if f_try < f0:
				accepted = True
				break

		if not accepted:
			for q, kb in zip(quad_refs, k_old):
				q.k1 = kb
			lmbd *= 5.0
			if lmbd > 1e5:
				break
			continue

		lmbd = max(1e-6, lmbd * 0.7)

	res = model_params()
	res["objective"] = float(residual_vec() @ residual_vec())
	return res


@dataclass
class Knob:
	name: str
	get_value: Callable[[], float]
	set_value: Callable[[float], None]
	vmin: float
	vmax: float

	def get(self) -> float:
		return float(self.get_value())

	def set(self, value: float) -> None:
		self.set_value(float(np.clip(value, self.vmin, self.vmax)))


def compute_linear_exit_params(beamline: Beamline, sigma_in: np.ndarray) -> dict:
	s = beamline.propagate_sigma_linear(sigma_in)
	bx, ax, ex = sigma_to_twiss(s[0:2, 0:2])
	by, ay, ey = sigma_to_twiss(s[2:4, 2:4])
	sdd = s[5, 5]
	Dx = s[0, 5] / sdd if sdd > 0 else np.nan
	Dpx = s[1, 5] / sdd if sdd > 0 else np.nan
	Dy = s[2, 5] / sdd if sdd > 0 else np.nan
	Dpy = s[3, 5] / sdd if sdd > 0 else np.nan
	return {
		"beta_x": bx,
		"alpha_x": ax,
		"emit_x": ex,
		"beta_y": by,
		"alpha_y": ay,
		"emit_y": ey,
		"Dx": Dx,
		"Dpx": Dpx,
		"Dy": Dy,
		"Dpy": Dpy,
	}


def match_realistic_lattice(
	beamline: Beamline,
	knobs: List[Knob],
	sigma_in: np.ndarray,
	target_out: dict,
	weights: dict | None = None,
	scales: dict | None = None,
	keys: List[str] | None = None,
	max_iter: int = 100,
	tol: float = 1e-12,
) -> dict:
	"""
	Реалистичный matching: response-matrix + SVD/LM с ограничениями ручек.
	По стилю близко к LOCO/оптическому matching в реальных машинах.
	"""
	if weights is None:
		weights = {
			"beta_x": 1.0,
			"alpha_x": 1.0,
			"beta_y": 1.0,
			"alpha_y": 1.0,
			"Dx": 60.0,
			"Dpx": 60.0,
			"Dy": 8.0,
			"Dpy": 8.0,
		}

	if scales is None:
		scales = {
			"beta_x": max(abs(float(target_out["beta_x"])), 1.0),
			"alpha_x": 1.0,
			"beta_y": max(abs(float(target_out["beta_y"])), 1.0),
			"alpha_y": 1.0,
			"Dx": 1e-4,
			"Dpx": 1e-4,
			"Dy": 1e-4,
			"Dpy": 1e-4,
		}

	if keys is None:
		keys = ["beta_x", "alpha_x", "beta_y", "alpha_y", "Dx", "Dpx"]

	def residual_vec() -> np.ndarray:
		p = compute_linear_exit_params(beamline, sigma_in)
		r = []
		for k in keys:
			wk = float(weights.get(k, 1.0))
			s = float(scales.get(k, 1.0))
			r.append(wk * (float(p[k]) - float(target_out[k])) / s)
		return np.array(r, dtype=float)

	mu = 1e-3
	eps = 1e-5

	for _ in range(max_iter):
		r0 = residual_vec()
		f0 = float(r0 @ r0)
		if f0 < tol:
			break

		J = np.zeros((len(r0), len(knobs)), dtype=float)
		for j, kb in enumerate(knobs):
			v0 = kb.get()
			h = eps * max(1.0, abs(v0))

			kb.set(v0 + h)
			rp = residual_vec()
			kb.set(v0 - h)
			rm = residual_vec()
			kb.set(v0)

			J[:, j] = (rp - rm) / (2.0 * h)

		A = J.T @ J + mu * np.eye(len(knobs), dtype=float)
		b = -(J.T @ r0)
		try:
			dv = np.linalg.solve(A, b)
		except np.linalg.LinAlgError:
			dv = np.linalg.pinv(A) @ b

		# trust-region
		dvmax = float(np.max(np.abs(dv))) if dv.size else 0.0
		if dvmax > 0.5:
			dv *= 0.5 / dvmax

		v_old = [kb.get() for kb in knobs]
		accepted = False
		for a in (1.0, 0.5, 0.25, 0.1, 0.05):
			for kb, v0, d in zip(knobs, v_old, dv):
				kb.set(v0 + a * float(d))

			r1 = residual_vec()
			f1 = float(r1 @ r1)
			if f1 < f0:
				accepted = True
				break

		if not accepted:
			for kb, v0 in zip(knobs, v_old):
				kb.set(v0)
			mu *= 5.0
			if mu > 1e7:
				break
			continue

		mu = max(1e-8, mu * 0.6)

	out = compute_linear_exit_params(beamline, sigma_in)
	out["objective"] = float(residual_vec() @ residual_vec())
	out["knobs"] = {kb.name: kb.get() for kb in knobs}
	return out


def multistart_match_realistic(
	beamline: Beamline,
	knobs: List[Knob],
	sigma_in: np.ndarray,
	target_out: dict,
	weights: dict,
	keys: List[str],
	n_starts: int = 24,
	seed: int = 7,
) -> dict:
	"""Глобальный поиск: несколько random starts + локальный LM matching."""
	rng = np.random.default_rng(seed)

	best = match_realistic_lattice(
		beamline=beamline,
		knobs=knobs,
		sigma_in=sigma_in,
		target_out=target_out,
		weights=weights,
		keys=keys,
		max_iter=120,
	)
	best_state = [kb.get() for kb in knobs]
	best_obj = float(best["objective"])

	for _ in range(n_starts):
		# Случайная инициализация ручек в допустимых пределах
		for kb in knobs:
			kb.set(float(rng.uniform(kb.vmin, kb.vmax)))

		out = match_realistic_lattice(
			beamline=beamline,
			knobs=knobs,
			sigma_in=sigma_in,
			target_out=target_out,
			weights=weights,
			keys=keys,
			max_iter=90,
		)
		obj = float(out["objective"])
		if obj < best_obj:
			best_obj = obj
			best = out
			best_state = [kb.get() for kb in knobs]

	# Вернём лучшее найденное состояние
	for kb, v in zip(knobs, best_state):
		kb.set(v)

	return best


def compute_nonlinear_exit_params(beamline: Beamline, particles0: np.ndarray) -> dict:
	p = beamline.track_nonlinear(particles0)
	return estimate_phase_ellipse(p)


def fine_tune_nonlinear(
	beamline: Beamline,
	knobs: List[Knob],
	sigma_in: np.ndarray,
	particles0: np.ndarray,
	target_out: dict,
	weights: dict | None = None,
	scales: dict | None = None,
	keys: List[str] | None = None,
	max_iter: int = 16,
	local_span_frac: float = 0.12,
	local_span_min: float = 0.002,
	linear_guard: float = 0.6,
	tol: float = 1e-10,
) -> dict:
	"""
	Тонкая подстройка по нелинейному трекингу в локальном диапазоне вокруг
	результата грубого линейного matching.
	"""
	if weights is None:
		weights = {
			"beta_x": 2.0,
			"alpha_x": 2.0,
			"beta_y": 2.0,
			"alpha_y": 2.0,
			"Dx": 160.0,
			"Dpx": 160.0,
			"Dy": 160.0,
			"Dpy": 160.0,
		}

	if scales is None:
		scales = {
			"beta_x": max(abs(float(target_out["beta_x"])), 1.0),
			"alpha_x": 1.0,
			"beta_y": max(abs(float(target_out["beta_y"])), 1.0),
			"alpha_y": 1.0,
			"Dx": 2e-4,
			"Dpx": 2e-4,
			"Dy": 2e-4,
			"Dpy": 2e-4,
		}

	if keys is None:
		keys = ["beta_x", "alpha_x", "beta_y", "alpha_y", "Dx", "Dpx", "Dy", "Dpy"]

	# Локальные диапазоны вокруг текущих значений ручек
	centers = [kb.get() for kb in knobs]
	local_bounds: list[tuple[float, float]] = []
	for kb, c in zip(knobs, centers):
		span = max(local_span_min * (kb.vmax - kb.vmin), local_span_frac * max(1.0, abs(c)))
		vmin = max(kb.vmin, c - span)
		vmax = min(kb.vmax, c + span)
		local_bounds.append((vmin, vmax))

	def set_local_value(i: int, value: float) -> None:
		vmin, vmax = local_bounds[i]
		knobs[i].set(float(np.clip(value, vmin, vmax)))

	def residual_vec() -> np.ndarray:
		p = compute_nonlinear_exit_params(beamline, particles0)
		p_lin = compute_linear_exit_params(beamline, sigma_in)
		r = []
		for k in keys:
			wk = float(weights.get(k, 1.0))
			s = float(scales.get(k, 1.0))
			r.append(wk * (float(p[k]) - float(target_out[k])) / s)
			if linear_guard > 0.0:
				r.append(linear_guard * wk * (float(p_lin[k]) - float(target_out[k])) / s)
		return np.array(r, dtype=float)

	mu = 1e-2

	for _ in range(max_iter):
		r0 = residual_vec()
		f0 = float(r0 @ r0)
		if f0 < tol:
			break

		J = np.zeros((len(r0), len(knobs)), dtype=float)
		for j, kb in enumerate(knobs):
			v0 = kb.get()
			vmin, vmax = local_bounds[j]
			span = max(vmax - vmin, 1e-9)
			h = 0.03 * span

			set_local_value(j, v0 + h)
			rp = residual_vec()
			set_local_value(j, v0 - h)
			rm = residual_vec()
			set_local_value(j, v0)

			den = max(2.0 * h, 1e-12)
			J[:, j] = (rp - rm) / den

		A = J.T @ J + mu * np.eye(len(knobs), dtype=float)
		b = -(J.T @ r0)
		try:
			dv = np.linalg.solve(A, b)
		except np.linalg.LinAlgError:
			dv = np.linalg.pinv(A) @ b

		# ограничим шаг тонкого этапа
		dvmax = float(np.max(np.abs(dv))) if dv.size else 0.0
		if dvmax > 0.03:
			dv *= 0.03 / dvmax

		v_old = [kb.get() for kb in knobs]
		accepted = False
		for a in (1.0, 0.5, 0.25, 0.1):
			for j, (kb, v0, d) in enumerate(zip(knobs, v_old, dv)):
				set_local_value(j, v0 + a * float(d))

			r1 = residual_vec()
			f1 = float(r1 @ r1)
			if f1 < f0:
				accepted = True
				break

		if not accepted:
			for j, v0 in enumerate(v_old):
				set_local_value(j, v0)
			mu *= 4.0
			if mu > 1e6:
				break
			continue

		mu = max(1e-8, mu * 0.7)

	out = compute_nonlinear_exit_params(beamline, particles0)
	out["objective"] = float(residual_vec() @ residual_vec())
	out["knobs"] = {kb.name: kb.get() for kb in knobs}
	return out


def build_example_beamline() -> tuple[
	Beamline,
	List[Quadrupole],
	List[Quadrupole],
	List[SkewQuadrupole],
	List[SectorDipole],
	List[Sextupole],
	Drift,
]:
	"""
	Пример канала:
	- 4 основных квадруполя (arc optics)
	- 2 matching-квадруполя (trim section, как в реальных ускорителях)
	- 2 скрю-квадруполя (x-y coupling)
	- 3 диполя по 30° (итого 90°)
	- 2 sextupole (нелинейная/хроматическая коррекция)
	"""
	bl = Beamline()

	q1 = Quadrupole(name="Q1", L=0.25, k1=1.10)
	q2 = Quadrupole(name="Q2", L=0.25, k1=-1.00)
	q3 = Quadrupole(name="Q3", L=0.25, k1=0.90)
	q4 = Quadrupole(name="Q4", L=0.25, k1=-0.85)
	mq1 = Quadrupole(name="MQ1", L=0.20, k1=0.35)
	mq2 = Quadrupole(name="MQ2", L=0.20, k1=-0.30)
	mq3 = Quadrupole(name="MQ3", L=0.20, k1=0.25)
	mq4 = Quadrupole(name="MQ4", L=0.20, k1=-0.20)
	mq5 = Quadrupole(name="MQ5", L=0.20, k1=0.15)
	mq6 = Quadrupole(name="MQ6", L=0.20, k1=-0.10)

	sq1 = SkewQuadrupole(name="SQ1", L=0.18, k1s=0.0)
	sq2 = SkewQuadrupole(name="SQ2", L=0.18, k1s=0.0)
	dout = Drift(name="DOUT", L=0.60)

	# В реальных машинах полюсные края дают заметную вертикальную фокусировку
	edge = np.deg2rad(9.0)
	b1 = SectorDipole(name="B1", L=0.70, angle=np.deg2rad(30.0), e1=edge, e2=edge)
	b2 = SectorDipole(name="B2", L=0.70, angle=np.deg2rad(30.0), e1=edge, e2=edge)
	b3 = SectorDipole(name="B3", L=0.70, angle=np.deg2rad(30.0), e1=edge, e2=edge)

	# Sextupole-семейства для нелинейной/хроматической коррекции
	sx1 = Sextupole(name="SX1", L=0.12, k2=12.0)
	sx2 = Sextupole(name="SX2", L=0.12, k2=-10.0)

	# Схема — легко расширяется через bl.add(...)
	bl.add(q1)
	bl.add(Drift(name="D01", L=0.25))
	bl.add(b1)
	bl.add(Drift(name="D02", L=0.22))
	bl.add(sq1)
	bl.add(sx1)
	bl.add(Drift(name="D03", L=0.20))

	bl.add(q2)
	bl.add(Drift(name="D04", L=0.20))
	bl.add(b2)
	bl.add(Drift(name="D05", L=0.22))
	bl.add(sq2)
	bl.add(sx2)
	bl.add(Drift(name="D06", L=0.20))

	bl.add(q3)
	bl.add(Drift(name="D07", L=0.20))
	bl.add(b3)
	bl.add(Drift(name="D08", L=0.25))
	bl.add(q4)
	bl.add(Drift(name="D09", L=0.25))
	bl.add(mq1)
	bl.add(Drift(name="D10", L=0.25))
	bl.add(mq2)
	bl.add(Drift(name="D11", L=0.25))
	bl.add(mq3)
	bl.add(Drift(name="D12", L=0.25))
	bl.add(mq4)
	bl.add(Drift(name="D13", L=0.25))
	bl.add(mq5)
	bl.add(Drift(name="D14", L=0.25))
	bl.add(mq6)
	bl.add(dout)

	return bl, [q1, q2, q3, q4], [mq1, mq2, mq3, mq4, mq5, mq6], [sq1, sq2], [b1, b2, b3], [sx1, sx2], dout


def pretty_print_params(title: str, p: dict) -> None:
	print(f"\n{title}")
	print(
		"  "
		f"beta_x={p['beta_x']:.5g}, alpha_x={p['alpha_x']:.5g}, emit_x={p['emit_x']:.5g} m*rad, "
		f"Dx={p['Dx']:.5g}, Dpx={p['Dpx']:.5g}"
	)
	print(
		"  "
		f"beta_y={p['beta_y']:.5g}, alpha_y={p['alpha_y']:.5g}, emit_y={p['emit_y']:.5g} m*rad, "
		f"Dy={p['Dy']:.5g}, Dpy={p['Dpy']:.5g}"
	)


def _ellipse_curve_from_twiss(beta: float, alpha: float, emit: float, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
	t = np.linspace(0.0, 2.0 * np.pi, n)
	a = np.sqrt(max(beta * emit, 0.0))
	b = np.sqrt(max(emit / beta, 0.0)) if beta > 1e-16 else 0.0
	u = a * np.cos(t)
	up = -alpha * b * np.cos(t) - b * np.sin(t)
	return u, up


def _track_history_by_names(beamline: Beamline, particles0: np.ndarray, watch_names: List[str]) -> dict[str, np.ndarray]:
	history: dict[str, np.ndarray] = {}
	p = particles0.copy()
	if "START" in watch_names:
		history["START"] = p.copy()

	for elem in beamline.elements:
		p = elem.track_particles(p)
		if elem.name in watch_names:
			history[elem.name] = p.copy()

	if "END" in watch_names:
		history["END"] = p.copy()

	return history


def _core_mask_2d(data2: np.ndarray, keep_fraction: float = 0.98) -> np.ndarray:
	"""Маска центрального ядра по Mahalanobis distance."""
	if data2.shape[0] < 20:
		return np.ones(data2.shape[0], dtype=bool)

	mu = np.mean(data2, axis=0)
	centered = data2 - mu
	C = np.cov(centered.T, bias=False)
	C = C + 1e-24 * np.eye(2, dtype=float)
	try:
		Ci = np.linalg.inv(C)
	except np.linalg.LinAlgError:
		Ci = np.linalg.pinv(C)

	md2 = np.einsum("ni,ij,nj->n", centered, Ci, centered)
	thr = np.quantile(md2, keep_fraction)
	return md2 <= thr


def plot_phase_ellipses_along_beamline(
	beamline: Beamline,
	particles0: np.ndarray,
	target_out: dict,
	watch_names: List[str] | None = None,
	output_dir: str = "plots",
	max_scatter_points: int = 2500,
) -> None:
	"""Рисует фазовые эллипсы X/X' и Y/Y' на выбранных элементах канала.
	На конечной позиции накладывается требуемый (target) эллипс.
	"""
	try:
		import matplotlib.pyplot as plt
	except Exception as ex:  # pragma: no cover
		print(f"[plot] matplotlib недоступен: {ex}")
		return

	if watch_names is None:
		watch_names = ["START", "B1", "B2", "B3", "DOUT", "END"]

	history = _track_history_by_names(beamline, particles0, watch_names)
	ordered_names = [name for name in watch_names if name in history]
	if not ordered_names:
		print("[plot] Не найдено точек для визуализации")
		return

	os.makedirs(output_dir, exist_ok=True)

	# Подвыборка для быстрого scatter
	idx = np.arange(particles0.shape[0])
	if particles0.shape[0] > max_scatter_points:
		step = max(1, particles0.shape[0] // max_scatter_points)
		idx = idx[::step]

	# Рисуем 2 строки: X-плоскость и Y-плоскость, столбцы = элементы
	fig, axes = plt.subplots(2, len(ordered_names), figsize=(4.4 * len(ordered_names), 8.2), squeeze=False)

	for col, name in enumerate(ordered_names):
		p = history[name]

		# ---------- X-plane ----------
		ax = axes[0, col]
		x2 = p[:, [0, 1]]
		mx = _core_mask_2d(x2, keep_fraction=0.98)
		ax.scatter(p[idx, 0], p[idx, 1], s=3, alpha=0.12, color="#9ecae1", label="particles (all)")
		ax.scatter(p[mx, 0], p[mx, 1], s=3, alpha=0.20, color="#1f77b4", label="particles (core)")
		bx, axx, ex = sigma_to_twiss(np.cov(x2[mx].T, bias=False))
		ex_u, ex_up = _ellipse_curve_from_twiss(bx, axx, ex)
		ax.plot(ex_u, ex_up, color="#0b3d91", lw=2.0, label="fitted ellipse")

		if name == "END":
			tu, tup = _ellipse_curve_from_twiss(
				beta=target_out["beta_x"], alpha=target_out["alpha_x"], emit=target_out["emit_x"]
			)
			ax.plot(tu, tup, "--", color="#d62728", lw=2.2, label="target ellipse")

			# Zoom around physically relevant scale (core + target), чтобы target был видим
			xc = x2[mx, 0] - np.median(x2[mx, 0])
			xpc = x2[mx, 1] - np.median(x2[mx, 1])
			x_lim = 1.25 * max(np.quantile(np.abs(xc), 0.995), np.max(np.abs(tu)), 1e-12)
			xp_lim = 1.25 * max(np.quantile(np.abs(xpc), 0.995), np.max(np.abs(tup)), 1e-12)
			ax.set_xlim(np.median(x2[mx, 0]) - x_lim, np.median(x2[mx, 0]) + x_lim)
			ax.set_ylim(np.median(x2[mx, 1]) - xp_lim, np.median(x2[mx, 1]) + xp_lim)

		ax.set_title(f"{name} : X-X'", fontsize=11)
		ax.set_xlabel("x [m]")
		ax.set_ylabel("x' [rad]")
		ax.grid(alpha=0.25)
		if col == len(ordered_names) - 1:
			ax.legend(loc="best", fontsize=8)

		# ---------- Y-plane ----------
		ay = axes[1, col]
		y2 = p[:, [2, 3]]
		my = _core_mask_2d(y2, keep_fraction=0.98)
		ay.scatter(p[idx, 2], p[idx, 3], s=3, alpha=0.12, color="#a1d99b", label="particles (all)")
		ay.scatter(p[my, 2], p[my, 3], s=3, alpha=0.20, color="#2ca02c", label="particles (core)")
		by, ayy, ey = sigma_to_twiss(np.cov(y2[my].T, bias=False))
		ey_u, ey_up = _ellipse_curve_from_twiss(by, ayy, ey)
		ay.plot(ey_u, ey_up, color="#146c2e", lw=2.0, label="fitted ellipse")

		if name == "END":
			tu, tup = _ellipse_curve_from_twiss(
				beta=target_out["beta_y"], alpha=target_out["alpha_y"], emit=target_out["emit_y"]
			)
			ay.plot(tu, tup, "--", color="#d62728", lw=2.2, label="target ellipse")

			yc = y2[my, 0] - np.median(y2[my, 0])
			ypc = y2[my, 1] - np.median(y2[my, 1])
			y_lim = 1.25 * max(np.quantile(np.abs(yc), 0.995), np.max(np.abs(tu)), 1e-12)
			yp_lim = 1.25 * max(np.quantile(np.abs(ypc), 0.995), np.max(np.abs(tup)), 1e-12)
			ay.set_xlim(np.median(y2[my, 0]) - y_lim, np.median(y2[my, 0]) + y_lim)
			ay.set_ylim(np.median(y2[my, 1]) - yp_lim, np.median(y2[my, 1]) + yp_lim)

		ay.set_title(f"{name} : Y-Y'", fontsize=11)
		ay.set_xlabel("y [m]")
		ay.set_ylabel("y' [rad]")
		ay.grid(alpha=0.25)
		if col == len(ordered_names) - 1:
			ay.legend(loc="best", fontsize=8)

	fig.suptitle("Phase ellipses along beamline (nonlinear tracking)", fontsize=14)
	fig.tight_layout(rect=[0, 0, 1, 0.95])

	out_path = os.path.join(output_dir, "phase_ellipses_along_beamline.png")
	fig.savefig(out_path, dpi=170)
	plt.close(fig)
	print(f"[plot] Saved: {out_path}")


def plot_emittance_in_matching_section(
	beamline: Beamline,
	particles0: np.ndarray,
	section_start: str = "D09",
	section_end: str = "DOUT",
	output_dir: str = "plots",
) -> None:
	"""Визуализация эволюции эмиттанса (emit_x/emit_y) в согласующем участке."""
	try:
		import matplotlib.pyplot as plt
	except Exception as ex:  # pragma: no cover
		print(f"[plot] matplotlib недоступен: {ex}")
		return

	os.makedirs(output_dir, exist_ok=True)

	p = particles0.copy()
	s_pos = 0.0
	in_section = False

	s_track: List[float] = []
	ex_track: List[float] = []
	ey_track: List[float] = []
	labels: List[str] = []

	for elem in beamline.elements:
		# точка входа в секцию
		if elem.name == section_start:
			in_section = True

		p = elem.track_particles(p)
		s_pos += float(elem.L)

		if in_section:
			cov = np.cov(p.T, bias=False)
			_, _, ex = sigma_to_twiss(cov[0:2, 0:2])
			_, _, ey = sigma_to_twiss(cov[2:4, 2:4])
			s_track.append(s_pos)
			ex_track.append(ex)
			ey_track.append(ey)
			labels.append(elem.name)

		if elem.name == section_end:
			break

	if not s_track:
		print("[plot] Matching section for emittance plot is empty")
		return

	fig, ax = plt.subplots(1, 1, figsize=(10, 5.6))
	ax.plot(s_track, ex_track, "-o", ms=4, lw=1.8, color="#1f77b4", label="emit_x")
	ax.plot(s_track, ey_track, "-o", ms=4, lw=1.8, color="#2ca02c", label="emit_y")

	# Подпишем ключевые точки (квадруполи/skew/выход)
	for s, ex, ey, name in zip(s_track, ex_track, ey_track, labels):
		if name.startswith("MQ") or name.startswith("SQ") or name == "DOUT":
			ax.annotate(name, (s, ex), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
			ax.annotate(name, (s, ey), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=8)

	ax.set_title("Emittance evolution in matching section (nonlinear tracking)")
	ax.set_xlabel("s [m]")
	ax.set_ylabel("emittance [m·rad]")
	ax.grid(alpha=0.3)
	ax.legend(loc="best")

	out_path = os.path.join(output_dir, "emittance_matching_section.png")
	fig.tight_layout()
	fig.savefig(out_path, dpi=170)
	plt.close(fig)
	print(f"[plot] Saved: {out_path}")


def plot_beamline_presentation_layout(beamline: Beamline, output_dir: str = "plots") -> None:
	"""Картинка для презентации: порядок магнитов + геометрический поворот трубы."""
	try:
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle
	except Exception as ex:  # pragma: no cover
		print(f"[plot] matplotlib недоступен: {ex}")
		return

	os.makedirs(output_dir, exist_ok=True)

	# ---- 1) Порядок элементов по s ----
	seq = []
	s = 0.0
	for elem in beamline.elements:
		start = s
		s += float(elem.L)
		end = s
		seq.append((elem, start, end))

	# ---- 2) Траектория трубы в горизонтальной плоскости ----
	x = 0.0
	z = 0.0
	theta = 0.0
	path_x = [x]
	path_z = [z]
	turn_marks: list[tuple[str, float, float, float]] = []  # name, x, z, theta_deg

	for elem in beamline.elements:
		if isinstance(elem, SectorDipole):
			# Простейшая геометрия: секторный поворот на угол elem.angle
			if abs(elem.angle) < 1e-12:
				dx = elem.L * np.cos(theta)
				dz = elem.L * np.sin(theta)
				x += dx
				z += dz
			else:
				rho = elem.rho
				dth = elem.angle
				x += rho * (np.sin(theta + dth) - np.sin(theta))
				z += -rho * (np.cos(theta + dth) - np.cos(theta))
				theta += dth
				turn_marks.append((elem.name, x, z, np.rad2deg(theta)))
		else:
			dx = elem.L * np.cos(theta)
			dz = elem.L * np.sin(theta)
			x += dx
			z += dz

		path_x.append(x)
		path_z.append(z)

	total_turn_deg = np.rad2deg(theta)

	fig = plt.figure(figsize=(14, 8.5))
	fig.suptitle("Beamline layout for presentation", fontsize=16)

	# ---------- Верх: порядок магнитов ----------
	ax1 = fig.add_axes([0.06, 0.56, 0.90, 0.36])
	ax1.set_title("Order of elements along s", fontsize=12)

	color_map = {
		"Drift": "#cfcfcf",
		"Quadrupole": "#1f77b4",
		"SkewQuadrupole": "#9467bd",
		"Sextupole": "#ff7f0e",
		"SectorDipole": "#2ca02c",
	}

	for elem, s0, s1 in seq:
		cls = elem.__class__.__name__
		c = color_map.get(cls, "#bbbbbb")
		rect = Rectangle((s0, 0.1), s1 - s0, 0.8, facecolor=c, edgecolor="white", lw=1.0)
		ax1.add_patch(rect)
		if (s1 - s0) > 0.10:
			ax1.text(0.5 * (s0 + s1), 0.5, elem.name, ha="center", va="center", fontsize=8, color="black", rotation=90)

	ax1.set_xlim(0.0, max(1e-9, seq[-1][2] if seq else 1.0))
	ax1.set_ylim(0.0, 1.0)
	ax1.set_xlabel("s [m]")
	ax1.set_yticks([])
	ax1.grid(axis="x", alpha=0.25)

	legend_labels = [
		("Drift", color_map["Drift"]),
		("Quadrupole", color_map["Quadrupole"]),
		("Skew", color_map["SkewQuadrupole"]),
		("Sextupole", color_map["Sextupole"]),
		("Dipole", color_map["SectorDipole"]),
	]
	for i, (lbl, c) in enumerate(legend_labels):
		ax1.add_patch(Rectangle((0.01 + i * 0.13, 0.92), 0.025, 0.06, transform=ax1.transAxes, facecolor=c, edgecolor="none"))
		ax1.text(0.04 + i * 0.13, 0.95, lbl, transform=ax1.transAxes, va="center", fontsize=9)

	# ---------- Низ: траектория трубы ----------
	ax2 = fig.add_axes([0.06, 0.08, 0.90, 0.38])
	ax2.set_title("Pipe turning trajectory (horizontal projection)", fontsize=12)
	ax2.plot(path_x, path_z, "-", lw=2.8, color="#006d77", label="pipe centerline")
	ax2.scatter([path_x[0]], [path_z[0]], color="#2b2d42", s=45, zorder=5)
	ax2.text(path_x[0], path_z[0], "  START", va="bottom", fontsize=9)
	ax2.scatter([path_x[-1]], [path_z[-1]], color="#d62828", s=45, zorder=5)
	ax2.text(path_x[-1], path_z[-1], "  END", va="bottom", fontsize=9)

	for name, xx, zz, deg in turn_marks:
		ax2.scatter([xx], [zz], color="#2ca02c", s=28, zorder=4)
		ax2.text(xx, zz, f" {name}\n {deg:+.1f}°", fontsize=8, va="bottom")

	ax2.set_xlabel("X [m]")
	ax2.set_ylabel("Z [m]")
	ax2.grid(alpha=0.25)
	ax2.axis("equal")
	ax2.legend(loc="best", fontsize=9)

	fig.text(0.06, 0.02, f"Total bend angle: {total_turn_deg:+.2f}°", fontsize=12, weight="bold")

	out_path = os.path.join(output_dir, "beamline_layout_presentation.png")
	fig.savefig(out_path, dpi=190)
	plt.close(fig)
	print(f"[plot] Saved: {out_path}")


def _latex_num(x: float) -> str:
	v = float(x)
	if abs(v) < 1e-18:
		return "0"
	if abs(v) >= 1e4 or abs(v) < 1e-3:
		s = f"{v:.6e}"
		m, e = s.split("e")
		return f"{float(m):.6f}\\times 10^{{{int(e)}}}"
	return f"{v:.8g}"


def _latex_matrix(M: np.ndarray) -> str:
	rows = []
	for r in M:
		rows.append(" & ".join(_latex_num(v) for v in r))
	body = " \\\\\n".join(rows)
	return "\\begin{bmatrix}\n" + body + "\n\\end{bmatrix}"


def export_lattice_markdown_report(
	beamline: Beamline,
	sigma0: np.ndarray,
	target_out: dict,
	output_path: str = "plots/lattice_export.md",
) -> str:
	"""
	Экспорт параметров и матриц всех элементов в Markdown с LaTeX:
	- параметры элемента
	- локальная матрица R_i
	- накопленная матрица R_cum
	- параметры пучка после элемента (по Sigma-транспорту)
	"""
	os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

	lines: List[str] = []
	lines.append("# Lattice export report (Markdown + LaTeX)\n")
	lines.append("## Target output parameters\n")
	lines.append(
		"- "
		f"$\\beta_x={_latex_num(target_out['beta_x'])}$, "
		f"$\\alpha_x={_latex_num(target_out['alpha_x'])}$, "
		f"$\\varepsilon_x={_latex_num(target_out['emit_x'])}$\n"
	)
	lines.append(
		"- "
		f"$\\beta_y={_latex_num(target_out['beta_y'])}$, "
		f"$\\alpha_y={_latex_num(target_out['alpha_y'])}$, "
		f"$\\varepsilon_y={_latex_num(target_out['emit_y'])}$\n"
	)
	lines.append(
		"- "
		f"$D_x={_latex_num(target_out['Dx'])}$, $D'_x={_latex_num(target_out['Dpx'])}$, "
		f"$D_y={_latex_num(target_out['Dy'])}$, $D'_y={_latex_num(target_out['Dpy'])}$\n"
	)

	lines.append("## Initial Sigma\n")
	lines.append("$$\\Sigma_0=" + _latex_matrix(sigma0) + "$$\n")

	R_cum = np.eye(6, dtype=float)
	Sigma = sigma0.copy()

	lines.append("## Element-by-element data\n")
	for i, elem in enumerate(beamline.elements, start=1):
		R_i = elem.linear_matrix()
		R_cum = R_i @ R_cum
		Sigma = R_i @ Sigma @ R_i.T

		bx, ax, ex = sigma_to_twiss(Sigma[0:2, 0:2])
		by, ay, ey = sigma_to_twiss(Sigma[2:4, 2:4])
		sdd = Sigma[5, 5] if abs(Sigma[5, 5]) > 1e-24 else 1.0
		Dx = Sigma[0, 5] / sdd
		Dpx = Sigma[1, 5] / sdd
		Dy = Sigma[2, 5] / sdd
		Dpy = Sigma[3, 5] / sdd

		lines.append(f"### {i}. {elem.name} ({elem.__class__.__name__})\n")
		lines.append("**Parameters**\n")
		for k, v in vars(elem).items():
			if k.startswith("_"):
				continue
			if isinstance(v, (int, float, np.integer, np.floating)):
				lines.append(f"- `{k}` = ${_latex_num(float(v))}$")
			else:
				lines.append(f"- `{k}` = `{v}`")
		lines.append("")

		lines.append("**Element matrix** $R_i$\n")
		lines.append("$$R_i=" + _latex_matrix(R_i) + "$$\n")

		lines.append("**Cumulative matrix** $R_{\\mathrm{cum}}$\n")
		lines.append("$$R_{\\mathrm{cum}}=" + _latex_matrix(R_cum) + "$$\n")

		lines.append("**Beam parameters after this element (Sigma transport)**\n")
		lines.append(
			"- "
			f"$\\beta_x={_latex_num(bx)}$, $\\alpha_x={_latex_num(ax)}$, $\\varepsilon_x={_latex_num(ex)}$, "
			f"$D_x={_latex_num(Dx)}$, $D'_x={_latex_num(Dpx)}$"
		)
		lines.append(
			"- "
			f"$\\beta_y={_latex_num(by)}$, $\\alpha_y={_latex_num(ay)}$, $\\varepsilon_y={_latex_num(ey)}$, "
			f"$D_y={_latex_num(Dy)}$, $D'_y={_latex_num(Dpy)}$\n"
		)

	with open(output_path, "w", encoding="utf-8") as f:
		f.write("\n".join(lines))

	print(f"[export] Saved: {output_path}")
	return output_path


def launch_interactive_phase_tuner(
	beamline: Beamline,
	particles0: np.ndarray,
	target_out: dict,
	knobs: List[Knob],
	max_particles: int = 2000,
	plane_element_name: str = "END",
) -> None:
	"""Интерактивная визуализация: выбор ручки + ползунок, эллипсы обновляются в реальном времени."""
	if len(knobs) == 0:
		print("[interactive] Нет ручек для управления")
		return

	try:
		import matplotlib.pyplot as plt
		from matplotlib.widgets import RadioButtons, Slider
	except Exception as ex:  # pragma: no cover
		print(f"[interactive] matplotlib/widgets недоступны: {ex}")
		return

	# Подвыборка для интерактива (ускорение)
	p0 = particles0
	if particles0.shape[0] > max_particles:
		step = max(1, particles0.shape[0] // max_particles)
		p0 = particles0[::step].copy()

	selected_idx = {"i": 0}

	fig = plt.figure(figsize=(12.5, 7.2))
	fig.suptitle("Interactive phase tuner (nonlinear tracking)", fontsize=14)

	ax_x = fig.add_axes([0.07, 0.28, 0.39, 0.63])
	ax_y = fig.add_axes([0.52, 0.28, 0.39, 0.63])
	ax_radio = fig.add_axes([0.07, 0.05, 0.30, 0.18])
	ax_slider = fig.add_axes([0.42, 0.12, 0.50, 0.06])

	radio = RadioButtons(ax_radio, [k.name for k in knobs], active=0)
	slider = Slider(
		ax=ax_slider,
		label=f"{knobs[0].name}",
		valmin=knobs[0].vmin,
		valmax=knobs[0].vmax,
		valinit=knobs[0].get(),
	)

	def redraw() -> None:
		h = _track_history_by_names(beamline, p0, [plane_element_name])
		if plane_element_name in h:
			p = h[plane_element_name]
		else:
			p = beamline.track_nonlinear(p0)

		# X plane
		x2 = p[:, [0, 1]]
		mx = _core_mask_2d(x2, keep_fraction=0.98)
		bx, axx, ex = sigma_to_twiss(np.cov(x2[mx].T, bias=False))
		ex_u, ex_up = _ellipse_curve_from_twiss(bx, axx, ex)
		tx, txp = _ellipse_curve_from_twiss(target_out["beta_x"], target_out["alpha_x"], target_out["emit_x"])

		ax_x.clear()
		ax_x.scatter(x2[:, 0], x2[:, 1], s=4, alpha=0.12, color="#9ecae1", label="particles (all)")
		ax_x.scatter(x2[mx, 0], x2[mx, 1], s=4, alpha=0.22, color="#1f77b4", label="particles (core)")
		ax_x.plot(ex_u, ex_up, color="#0b3d91", lw=2.0, label="fitted ellipse")
		ax_x.plot(tx, txp, "--", color="#d62728", lw=2.0, label="target ellipse")
		ax_x.set_title(f"{plane_element_name} : X-X'")
		ax_x.set_xlabel("x [m]")
		ax_x.set_ylabel("x' [rad]")
		ax_x.grid(alpha=0.25)
		ax_x.legend(loc="best", fontsize=8)

		# Y plane
		y2 = p[:, [2, 3]]
		my = _core_mask_2d(y2, keep_fraction=0.98)
		by, ayy, ey = sigma_to_twiss(np.cov(y2[my].T, bias=False))
		ey_u, ey_up = _ellipse_curve_from_twiss(by, ayy, ey)
		ty, typ = _ellipse_curve_from_twiss(target_out["beta_y"], target_out["alpha_y"], target_out["emit_y"])

		ax_y.clear()
		ax_y.scatter(y2[:, 0], y2[:, 1], s=4, alpha=0.12, color="#a1d99b", label="particles (all)")
		ax_y.scatter(y2[my, 0], y2[my, 1], s=4, alpha=0.22, color="#2ca02c", label="particles (core)")
		ax_y.plot(ey_u, ey_up, color="#146c2e", lw=2.0, label="fitted ellipse")
		ax_y.plot(ty, typ, "--", color="#d62728", lw=2.0, label="target ellipse")
		ax_y.set_title(f"{plane_element_name} : Y-Y'")
		ax_y.set_xlabel("y [m]")
		ax_y.set_ylabel("y' [rad]")
		ax_y.grid(alpha=0.25)
		ax_y.legend(loc="best", fontsize=8)

		fig.canvas.draw_idle()

	def on_radio(label: str) -> None:
		i = next((j for j, k in enumerate(knobs) if k.name == label), 0)
		selected_idx["i"] = i
		k = knobs[i]
		slider.valmin = k.vmin
		slider.valmax = k.vmax
		slider.ax.set_xlim(k.vmin, k.vmax)
		slider.label.set_text(k.name)
		slider.set_val(k.get())

	def on_slider(val: float) -> None:
		k = knobs[selected_idx["i"]]
		k.set(float(val))
		redraw()

	radio.on_clicked(on_radio)
	slider.on_changed(on_slider)

	redraw()
	print("[interactive] Окно открыто: выбирайте элемент слева и крутите ползунок")
	plt.show()


def main(interactive: bool = False) -> None:
	# ===== Входные/выходные параметры из задания =====
	# Таблица 1 (начальные условия):
	#   X: beta=5.0 м, alpha=-0.5, emit=10 нм*рад
	#   Y: beta=2.5 м, alpha=+0.3, emit=2 нм*рад
	#
	# Таблица 2 (цель, параметры на входе в ускоритель):
	#   X: beta=8.0 м, alpha=0.0, emit=10 нм*рад
	#   Y: beta=4.0 м, alpha=0.0, emit=2 нм*рад
	#   D=0, D'=0
	target_out = {
		"beta_x": 8.0,
		"alpha_x": 0.0,
		"emit_x": 10.0e-9,
		"beta_y": 4.0,
		"alpha_y": 0.0,
		"emit_y": 2.0e-9,
		"Dx": 0.0,
		"Dpx": 0.0,
		"Dy": 0.0,
		"Dpy": 0.0,
	}

	# Продольные параметры не заданы в таблице, задаём типичные значения.
	sigma0 = build_sigma6(
		beta_x=5.0,
		alpha_x=-0.5,
		emit_x=10.0e-9,
		beta_y=2.5,
		alpha_y=0.3,
		emit_y=2.0e-9,
		sigma_z=3.0e-3,
		sigma_delta=1.0e-3,
		Dx=0.0,
		Dpx=0.0,
		Dy=0.0,
		Dpy=0.0,
	)

	beamline, quads, match_quads, skews, dipoles, sexts, dout = build_example_beamline()

	# 1) быстрый pre-match по дисперсии (квадруполями)
	Dx_out, Dpx_out = match_zero_dispersion(beamline, quads + match_quads)
	print(f"Initial dispersion match: Dx={Dx_out:+.3e}, Dpx={Dpx_out:+.3e}")

	# 2) реалистичный global match по семействам ручек
	#    (как в реальных ускорителях: response-matrix + bounds)
	knobs: List[Knob] = []
	for q in quads:
		knobs.append(
			Knob(
				name=f"{q.name}.k1",
				get_value=lambda qq=q: qq.k1,
				set_value=lambda v, qq=q: setattr(qq, "k1", v),
				vmin=-8.0,
				vmax=8.0,
			)
		)

	for q in match_quads:
		knobs.append(
			Knob(
				name=f"{q.name}.k1",
				get_value=lambda qq=q: qq.k1,
				set_value=lambda v, qq=q: setattr(qq, "k1", v),
				vmin=-8.0,
				vmax=8.0,
			)
		)

	# Общий полюсный угол края диполей как семейство
	def _get_edge_family() -> float:
		return float(np.mean([b.e1 for b in dipoles]))

	def _set_edge_family(v: float) -> None:
		for b in dipoles:
			b.e1 = float(v)
			b.e2 = float(v)

	knobs.append(
		Knob(
			name="Bedge.family",
			get_value=_get_edge_family,
			set_value=_set_edge_family,
			vmin=np.deg2rad(-12.0),
			vmax=np.deg2rad(12.0),
		)
	)

	# Градиент dipole-семейства (combined-function bending magnets)
	def _get_bend_k1_family() -> float:
		return float(np.mean([b.k1 for b in dipoles]))

	def _set_bend_k1_family(v: float) -> None:
		for b in dipoles:
			b.k1 = float(v)

	knobs.append(
		Knob(
			name="Bend.k1.family",
			get_value=_get_bend_k1_family,
			set_value=_set_bend_k1_family,
			vmin=-1.5,
			vmax=1.5,
		)
	)

	knobs.append(
		Knob(
			name="DOUT.L",
			get_value=lambda: dout.L,
			set_value=lambda v: setattr(dout, "L", float(v)),
			vmin=0.0,
			vmax=3.0,
		)
	)

	match_weights = {
		"beta_x": 2.0,
		"alpha_x": 2.0,
		"beta_y": 2.0,
		"alpha_y": 2.0,
		"Dx": 120.0,
		"Dpx": 120.0,
		"Dy": 120.0,
		"Dpy": 120.0,
	}
	match_keys = ["beta_x", "alpha_x", "beta_y", "alpha_y", "Dx", "Dpx", "Dy", "Dpy"]

	matched = multistart_match_realistic(
		beamline=beamline,
		knobs=knobs,
		sigma_in=sigma0,
		target_out=target_out,
		weights=match_weights,
		keys=match_keys,
		n_starts=400,
		seed=1,
	)

	# Сохраняем рабочую линейную точку после coarse stage
	coarse_state = [kb.get() for kb in knobs]

	# 3) Тонкая подстройка по нелинейному трекингу в узком диапазоне
	#    (после грубого линейного этапа)
	fine_knobs: List[Knob] = []
	for sx in sexts:
		fine_knobs.append(
			Knob(
				name=f"{sx.name}.k2",
				get_value=lambda ss=sx: ss.k2,
				set_value=lambda v, ss=sx: setattr(ss, "k2", v),
				vmin=-80.0,
				vmax=80.0,
			)
		)
	for sq in skews:
		fine_knobs.append(
			Knob(
				name=f"{sq.name}.k1s",
				get_value=lambda ss=sq: ss.k1s,
				set_value=lambda v, ss=sq: setattr(ss, "k1s", v),
				vmin=-2.5,
				vmax=2.5,
			)
		)

	particles_fine = sample_gaussian_beam(n_particles=5000, sigma6=sigma0, seed=777)
	nonlinear_matched = fine_tune_nonlinear(
		beamline=beamline,
		knobs=fine_knobs,
		sigma_in=sigma0,
		particles0=particles_fine,
		target_out=target_out,
		weights={
			"beta_x": 2.0,
			"alpha_x": 2.0,
			"beta_y": 2.0,
			"alpha_y": 2.0,
			"Dx": 80.0,
			"Dpx": 80.0,
			"Dy": 80.0,
			"Dpy": 80.0,
		},
		keys=match_keys,
		max_iter=8,
		local_span_frac=0.02,
		local_span_min=0.002,
		linear_guard=0.8,
	)

	# После тонкой подстройки обновляем линейную оценку на выходе
	matched = compute_linear_exit_params(beamline, sigma0)
	lin_ok_after_fine = (
		abs(matched["beta_x"] - target_out["beta_x"]) <= 0.2
		and abs(matched["alpha_x"] - target_out["alpha_x"]) <= 0.2
		and abs(matched["beta_y"] - target_out["beta_y"]) <= 0.2
		and abs(matched["alpha_y"] - target_out["alpha_y"]) <= 0.2
		and abs(matched["Dx"]) <= 1e-4
		and abs(matched["Dpx"]) <= 1e-4
	)

	# Если fine stage испортил линейный match — откатываемся к coarse решению
	if not lin_ok_after_fine:
		for kb, val in zip(knobs, coarse_state):
			kb.set(val)
		matched = compute_linear_exit_params(beamline, sigma0)
		nonlinear_matched = compute_nonlinear_exit_params(beamline, particles_fine)
		print("[fine] rollback: fine stage degraded linear constraints")
	else:
		print("[fine] accepted")

	matched["objective"] = np.nan

	print("Matched quadrupole strengths (1/m^2):")
	for q in quads:
		print(f"  {q.name}: k1 = {q.k1:+.6f}")
	print("Matched trim quadrupole strengths (1/m^2):")
	for q in match_quads:
		print(f"  {q.name}: k1 = {q.k1:+.6f}")
	print("Matched skew quadrupole strengths (1/m^2):")
	for sq in skews:
		print(f"  {sq.name}: k1s = {sq.k1s:+.6f}")
	print(f"Matched dipole edge family: {np.rad2deg(_get_edge_family()):+.3f} deg")
	print(f"Matched dipole gradient family: {_get_bend_k1_family():+.6f} 1/m^2")
	print(f"Matched output drift length DOUT: {dout.L:+.4f} m")
	print("Matched sextupole strengths (1/m^3):")
	for sx in sexts:
		print(f"  {sx.name}: k2 = {sx.k2:+.6f}")
	print("Nonlinear fine-tune objective:")
	print(f"  objective={nonlinear_matched['objective']:.3e}")

	print("\nMatched linear targets at exit (from Sigma transport):")
	print(
		"  "
		f"beta_x={matched['beta_x']:.5g}, alpha_x={matched['alpha_x']:.5g}, "
		f"Dx={matched['Dx']:+.3e}, Dpx={matched['Dpx']:+.3e}"
	)
	print(
		"  "
		f"beta_y={matched['beta_y']:.5g}, alpha_y={matched['alpha_y']:.5g}, "
		f"Dy={matched['Dy']:+.3e}, Dpy={matched['Dpy']:+.3e}, objective={matched['objective']:.3e}"
	)
	print("  Target errors:")
	print(
		"    "
		f"d(beta_x)={matched['beta_x'] - target_out['beta_x']:+.3e}, "
		f"d(alpha_x)={matched['alpha_x'] - target_out['alpha_x']:+.3e}, "
		f"d(beta_y)={matched['beta_y'] - target_out['beta_y']:+.3e}, "
		f"d(alpha_y)={matched['alpha_y'] - target_out['alpha_y']:+.3e}"
	)
	print("Matched nonlinear targets at exit (particle tracking, fine stage):")
	print(
		"  "
		f"beta_x={nonlinear_matched['beta_x']:.5g}, alpha_x={nonlinear_matched['alpha_x']:.5g}, "
		f"Dx={nonlinear_matched['Dx']:+.3e}, Dpx={nonlinear_matched['Dpx']:+.3e}"
	)
	print(
		"  "
		f"beta_y={nonlinear_matched['beta_y']:.5g}, alpha_y={nonlinear_matched['alpha_y']:.5g}, "
		f"Dy={nonlinear_matched['Dy']:+.3e}, Dpy={nonlinear_matched['Dpy']:+.3e}"
	)
	lin_ok = (
		abs(matched["beta_x"] - target_out["beta_x"]) <= 0.2
		and abs(matched["alpha_x"] - target_out["alpha_x"]) <= 0.2
		and abs(matched["beta_y"] - target_out["beta_y"]) <= 0.2
		and abs(matched["alpha_y"] - target_out["alpha_y"]) <= 0.2
		and abs(matched["Dx"]) <= 1e-4
		and abs(matched["Dpx"]) <= 1e-4
	)
	print(f"  Status (linear Twiss/dispersion): {'OK' if lin_ok else 'NOT OK'}")

	# Линейное распространение матрицей 6x6
	sigma_lin_out = beamline.propagate_sigma_linear(sigma0)

	# Сэмплинг частиц и нелинейный трекинг
	particles0 = sample_gaussian_beam(n_particles=12000, sigma6=sigma0, seed=42)
	p_lin = beamline.track_linear(particles0)
	p_nonlin = beamline.track_nonlinear(particles0)

	in_params = estimate_phase_ellipse(particles0)
	lin_params = estimate_phase_ellipse(p_lin)
	nonlin_params = estimate_phase_ellipse(p_nonlin)

	pretty_print_params("Input phase ellipse:", in_params)
	pretty_print_params("Target phase ellipse at exit:", target_out)
	pretty_print_params("Output phase ellipse (linear particle tracking):", lin_params)
	pretty_print_params("Output phase ellipse (nonlinear tracking):", nonlin_params)

	# Контроль через сигма-матрицу линейной модели
	bx, ax, ex = sigma_to_twiss(sigma_lin_out[0:2, 0:2])
	by, ay, ey = sigma_to_twiss(sigma_lin_out[2:4, 2:4])
	print("\nOutput from Sigma transport (pure 6x6):")
	print(
		"  "
		f"beta_x={bx:.5g}, alpha_x={ax:.5g}, emit_x={ex:.5g}, "
		f"Dx={sigma_lin_out[0,5]/sigma_lin_out[5,5]:+.3e}, "
		f"Dpx={sigma_lin_out[1,5]/sigma_lin_out[5,5]:+.3e}"
	)
	print(
		"  "
		f"beta_y={by:.5g}, alpha_y={ay:.5g}, emit_y={ey:.5g}, "
		f"Dy={sigma_lin_out[2,5]/sigma_lin_out[5,5]:+.3e}, "
		f"Dpy={sigma_lin_out[3,5]/sigma_lin_out[5,5]:+.3e}"
	)

	# Визуализация фазовых эллипсов на разных элементах участка
	plot_phase_ellipses_along_beamline(
		beamline=beamline,
		particles0=particles0,
		target_out=target_out,
		watch_names=["START", "B1", "B2", "B3", "DOUT", "END"],
		output_dir="plots",
	)

	plot_emittance_in_matching_section(
		beamline=beamline,
		particles0=particles0,
		section_start="D09",
		section_end="DOUT",
		output_dir="plots",
	)

	plot_beamline_presentation_layout(
		beamline=beamline,
		output_dir="plots",
	)

	export_lattice_markdown_report(
		beamline=beamline,
		sigma0=sigma0,
		target_out=target_out,
		output_path="plots/lattice_export.md",
	)

	if interactive:
		launch_interactive_phase_tuner(
			beamline=beamline,
			particles0=particles0,
			target_out=target_out,
			knobs=knobs,
			max_particles=2000,
			plane_element_name="END",
		)


if __name__ == "__main__":
	main(interactive=("--interactive" in sys.argv))
