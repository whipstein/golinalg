package golapack

import (
	"fmt"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zhsein uses inverse iteration to find specified right and/or left
// eigenvectors of a complex upper Hessenberg matrix H.
//
// The right eigenvector x and the left eigenvector y of the matrix H
// corresponding to an eigenvalue w are defined by:
//
//              H * x = w * x,     y**h * H = w * y**h
//
// where y**h denotes the conjugate transpose of the vector y.
func Zhsein(side mat.MatSide, eigsrc, initv byte, _select []bool, n int, h *mat.CMatrix, w *mat.CVector, vl, vr *mat.CMatrix, mm int, work *mat.CVector, rwork *mat.Vector, ifaill, ifailr *[]int) (m, info int, err error) {
	var bothv, fromqr, leftv, noinit, rightv bool
	var wk, zero complex128
	var eps3, hnorm, rzero, smlnum, ulp, unfl float64
	var i, iinfo, k, kl, kln, kr, ks, ldwork int

	zero = (0.0 + 0.0*1i)
	rzero = 0.0

	//     Decode and test the input parameters.
	bothv = side == Both
	rightv = side == Right || bothv
	leftv = side == Left || bothv

	fromqr = eigsrc == 'Q'

	noinit = initv == 'N'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors.
	m = 0
	for k = 1; k <= n; k++ {
		if _select[k-1] {
			m = m + 1
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
		gltest.Xerbla2("Zhsein", err)
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

	ldwork = n

	kl = 1
	kln = 0
	if fromqr {
		kr = 0
	} else {
		kr = n
	}
	ks = 1

	for k = 1; k <= n; k++ {
		if _select[k-1] {
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
				for i = k; i >= kl+1; i -= 1 {
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
				hnorm = Zlanhs('I', kr-kl+1, h.Off(kl-1, kl-1), rwork)
				if Disnan(int(hnorm)) {
					err = fmt.Errorf("Disnan(int(hnorm)): hnorm=%v", hnorm)
					return
				} else if hnorm > rzero {
					eps3 = hnorm * ulp
				} else {
					eps3 = smlnum
				}
			}

			//           Perturb eigenvalue if it is close to any previous
			//           selected eigenvalues affiliated to the submatrix
			//           H(KL:KR,KL:KR). Close roots are modified by EPS3.
			wk = w.Get(k - 1)
		label60:
			;
			for i = k - 1; i >= kl; i -= 1 {
				if _select[i-1] && cabs1(w.Get(i-1)-wk) < eps3 {
					wk = wk + complex(eps3, 0)
					goto label60
				}
			}
			w.Set(k-1, wk)

			if leftv {
				//              Compute left eigenvector.
				if iinfo = Zlaein(false, noinit, n-kl+1, h.Off(kl-1, kl-1), wk, vl.CVector(kl-1, ks-1), work.CMatrix(ldwork, opts), rwork, eps3, smlnum); iinfo > 0 {
					info = info + 1
					(*ifaill)[ks-1] = k
				} else {
					(*ifaill)[ks-1] = 0
				}
				for i = 1; i <= kl-1; i++ {
					vl.Set(i-1, ks-1, zero)
				}
			}
			if rightv {
				//              Compute right eigenvector.
				if iinfo = Zlaein(true, noinit, kr, h, wk, vr.CVector(0, ks-1), work.CMatrix(ldwork, opts), rwork, eps3, smlnum); iinfo > 0 {
					info = info + 1
					(*ifailr)[ks-1] = k
				} else {
					(*ifailr)[ks-1] = 0
				}
				for i = kr + 1; i <= n; i++ {
					vr.Set(i-1, ks-1, zero)
				}
			}
			ks = ks + 1
		}
	}

	return
}
