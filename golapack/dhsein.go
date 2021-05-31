package golapack

import (
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dhsein uses inverse iteration to find specified right and/or left
// eigenvectors of a real upper Hessenberg matrix H.
//
// The right eigenvector x and the left eigenvector y of the matrix H
// corresponding to an eigenvalue w are defined by:
//
//              H * x = w * x,     y**h * H = w * y**h
//
// where y**h denotes the conjugate transpose of the vector y.
func Dhsein(side, eigsrc, initv byte, _select *[]bool, n *int, h *mat.Matrix, ldh *int, wr, wi *mat.Vector, vl *mat.Matrix, ldvl *int, vr *mat.Matrix, ldvr, mm, m *int, work *mat.Vector, ifaill, ifailr *[]int, info *int) {
	var bothv, fromqr, leftv, noinit, pair, rightv bool
	var bignum, eps3, hnorm, one, smlnum, ulp, unfl, wki, wkr, zero float64
	var i, iinfo, k, kl, kln, kr, ksi, ksr, ldwork int

	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters.
	bothv = side == 'B'
	rightv = side == 'R' || bothv
	leftv = side == 'L' || bothv

	fromqr = eigsrc == 'Q'

	noinit = initv == 'N'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors, and standardize the array SELECT.
	(*m) = 0
	pair = false
	for k = 1; k <= (*n); k++ {
		if pair {
			pair = false
			(*_select)[k-1] = false
		} else {
			if wi.Get(k-1) == zero {
				if (*_select)[k-1] {
					(*m) = (*m) + 1
				}
			} else {
				pair = true
				if (*_select)[k-1] || (*_select)[k+1-1] {
					(*_select)[k-1] = true
					(*m) = (*m) + 2
				}
			}
		}
	}

	(*info) = 0
	if !rightv && !leftv {
		(*info) = -1
	} else if !fromqr && eigsrc != 'N' {
		(*info) = -2
	} else if !noinit && initv != 'U' {
		(*info) = -3
	} else if (*n) < 0 {
		(*info) = -5
	} else if (*ldh) < maxint(1, *n) {
		(*info) = -7
	} else if (*ldvl) < 1 || (leftv && (*ldvl) < (*n)) {
		(*info) = -11
	} else if (*ldvr) < 1 || (rightv && (*ldvr) < (*n)) {
		(*info) = -13
	} else if (*mm) < (*m) {
		(*info) = -14
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("DHSEIN"), -(*info))
		return
	}

	//     Quick return if possible.
	if (*n) == 0 {
		return
	}

	//     Set machine-dependent constants.
	unfl = Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(*n) / ulp)
	bignum = (one - ulp) / smlnum

	ldwork = (*n) + 1

	kl = 1
	kln = 0
	if fromqr {
		kr = 0
	} else {
		kr = (*n)
	}
	ksr = 1

	for k = 1; k <= (*n); k++ {
		if (*_select)[k-1] {
			//           Compute eigenvector(s) corresponding to W(K).
			if fromqr {
				//              If affiliation of eigenvalues is known, check whether
				//              the matrix splits.
				//
				//              Determine KL and KR such that 1 <= KL <= K <= KR <= N
				//              and H(KL,KL-1) and H(KR+1,KR) are zero (or KL = 1 or
				//              KR = N).
				//
				//              Then inverse iteration can be performed with the
				//              submatrix H(KL:N,KL:N) for a left eigenvector, and with
				//              the submatrix H(1:KR,1:KR) for a right eigenvector.
				for i = k; i >= kl+1; i-- {
					if h.Get(i-1, i-1-1) == zero {
						goto label30
					}
				}
			label30:
				;
				kl = i
				if k > kr {
					for i = k; i <= (*n)-1; i++ {
						if h.Get(i+1-1, i-1) == zero {
							goto label50
						}
					}
				label50:
					;
					kr = i
				}
			}

			if kl != kln {
				kln = kl

				//              Compute infinity-norm of submatrix H(KL:KR,KL:KR) if it
				//              has not ben computed before.
				hnorm = Dlanhs('I', toPtr(kr-kl+1), h.Off(kl-1, kl-1), ldh, work)
				if Disnan(int(hnorm)) {
					(*info) = -6
					return
				} else if hnorm > zero {
					eps3 = hnorm * ulp
				} else {
					eps3 = smlnum
				}
			}

			//           Perturb eigenvalue if it is close to any previous
			//           selected eigenvalues affiliated to the submatrix
			//           H(KL:KR,KL:KR). Close roots are modified by EPS3.
			wkr = wr.Get(k - 1)
			wki = wi.Get(k - 1)
		label60:
			;
			for i = k - 1; i >= kl; i-- {
				if (*_select)[i-1] && math.Abs(wr.Get(i-1)-wkr)+math.Abs(wi.Get(i-1)-wki) < eps3 {
					wkr = wkr + eps3
					goto label60
				}
			}
			wr.Set(k-1, wkr)

			pair = wki != zero
			if pair {
				ksi = ksr + 1
			} else {
				ksi = ksr
			}
			if leftv {
				//              Compute left eigenvector.
				Dlaein(false, noinit, toPtr((*n)-kl+1), h.Off(kl-1, kl-1), ldh, &wkr, &wki, vl.Vector(kl-1, ksr-1), vl.Vector(kl-1, ksi-1), work.Matrix(ldwork, opts), &ldwork, work.Off((*n)*(*n)+(*n)+1-1), &eps3, &smlnum, &bignum, &iinfo)
				if iinfo > 0 {
					if pair {
						(*info) = (*info) + 2
					} else {
						(*info) = (*info) + 1
					}
					(*ifaill)[ksr-1] = k
					(*ifaill)[ksi-1] = k
				} else {
					(*ifaill)[ksr-1] = 0
					(*ifaill)[ksi-1] = 0
				}
				for i = 1; i <= kl-1; i++ {
					vl.Set(i-1, ksr-1, zero)
				}
				if pair {
					for i = 1; i <= kl-1; i++ {
						vl.Set(i-1, ksi-1, zero)
					}
				}
			}
			if rightv {
				//              Compute right eigenvector.
				Dlaein(true, noinit, &kr, h, ldh, &wkr, &wki, vr.Vector(0, ksr-1), vr.Vector(0, ksi-1), work.Matrix(ldwork, opts), &ldwork, work.Off((*n)*(*n)+(*n)+1-1), &eps3, &smlnum, &bignum, &iinfo)
				if iinfo > 0 {
					if pair {
						(*info) = (*info) + 2
					} else {
						(*info) = (*info) + 1
					}
					(*ifailr)[ksr-1] = k
					(*ifailr)[ksi-1] = k
				} else {
					(*ifailr)[ksr-1] = 0
					(*ifailr)[ksi-1] = 0
				}
				for i = kr + 1; i <= (*n); i++ {
					vr.Set(i-1, ksr-1, zero)
				}
				if pair {
					for i = kr + 1; i <= (*n); i++ {
						vr.Set(i-1, ksi-1, zero)
					}
				}
			}

			if pair {
				ksr = ksr + 2
			} else {
				ksr = ksr + 1
			}
		}
	}
}
