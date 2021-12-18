package golapack

import (
	"fmt"
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
func Dhsein(side mat.MatSide, eigsrc, initv byte, _select *[]bool, n int, h *mat.Matrix, wr, wi *mat.Vector, vl, vr *mat.Matrix, mm int, work *mat.Vector, ifaill, ifailr *[]int) (m, info int, err error) {
	var bothv, fromqr, leftv, noinit, pair, rightv bool
	var bignum, eps3, hnorm, one, smlnum, ulp, unfl, wki, wkr, zero float64
	var i, iinfo, k, kl, kln, kr, ksi, ksr, ldwork int

	zero = 0.0
	one = 1.0

	//     Decode and test the input parameters.
	bothv = side == Both
	rightv = side == Right || bothv
	leftv = side == Left || bothv

	fromqr = eigsrc == 'Q'

	noinit = initv == 'N'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors, and standardize the array SELECT.
	m = 0
	pair = false
	for k = 1; k <= n; k++ {
		if pair {
			pair = false
			(*_select)[k-1] = false
		} else {
			if wi.Get(k-1) == zero {
				if (*_select)[k-1] {
					m = m + 1
				}
			} else {
				pair = true
				if (*_select)[k-1] || (*_select)[k] {
					(*_select)[k-1] = true
					m = m + 2
				}
			}
		}
	}

	if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if !fromqr && eigsrc != 'N' {
		err = fmt.Errorf("!fromqr && eigsrc != 'N': eigsrc='%c'", eigsrc)
	} else if !noinit && initv != 'U' {
		err = fmt.Errorf("!noinit && initv != 'U': initv='%c'", initv)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if h.Rows < max(1, n) {
		err = fmt.Errorf("h.Rows < max(1, n): h.Rows=%v, n=%v", h.Rows, n)
	} else if vl.Rows < 1 || (leftv && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=%s, vl.Rows=%v, n=%v", side, vl.Rows, n)
	} else if vr.Rows < 1 || (rightv && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=%s, vr.Rows=%v, n=%v", side, vr.Rows, n)
	} else if mm < m {
		err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
	}
	if err != nil {
		gltest.Xerbla2("Dhsein", err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	//     Set machine-dependent constants.
	unfl = Dlamch(SafeMinimum)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(n) / ulp)
	bignum = (one - ulp) / smlnum

	ldwork = n + 1

	kl = 1
	kln = 0
	if fromqr {
		kr = 0
	} else {
		kr = n
	}
	ksr = 1

	for k = 1; k <= n; k++ {
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
					for i = k; i <= n-1; i++ {
						if h.Get(i, i-1) == zero {
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
				hnorm = Dlanhs('I', kr-kl+1, h.Off(kl-1, kl-1), work)
				if Disnan(int(hnorm)) {
					info = -6
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
				if iinfo = Dlaein(false, noinit, n-kl+1, h.Off(kl-1, kl-1), wkr, wki, vl.Off(kl-1, ksr-1).Vector(), vl.Off(kl-1, ksi-1).Vector(), work.Matrix(ldwork, opts), work.Off(n*n+n), eps3, smlnum, bignum); iinfo > 0 {
					if pair {
						info = info + 2
					} else {
						info = info + 1
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
				if iinfo = Dlaein(true, noinit, kr, h, wkr, wki, vr.Off(0, ksr-1).Vector(), vr.Off(0, ksi-1).Vector(), work.Matrix(ldwork, opts), work.Off(n*n+n), eps3, smlnum, bignum); iinfo > 0 {
					if pair {
						info = info + 2
					} else {
						info = info + 1
					}
					(*ifailr)[ksr-1] = k
					(*ifailr)[ksi-1] = k
				} else {
					(*ifailr)[ksr-1] = 0
					(*ifailr)[ksi-1] = 0
				}
				for i = kr + 1; i <= n; i++ {
					vr.Set(i-1, ksr-1, zero)
				}
				if pair {
					for i = kr + 1; i <= n; i++ {
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

	return
}
