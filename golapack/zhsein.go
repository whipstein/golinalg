package golapack

import (
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
func Zhsein(side, eigsrc, initv byte, _select *[]bool, n *int, h *mat.CMatrix, ldh *int, w *mat.CVector, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr, mm, m *int, work *mat.CVector, rwork *mat.Vector, ifaill, ifailr *[]int, info *int) {
	var bothv, fromqr, leftv, noinit, rightv bool
	var wk, zero complex128
	var eps3, hnorm, rzero, smlnum, ulp, unfl float64
	var i, iinfo, k, kl, kln, kr, ks, ldwork int

	zero = (0.0 + 0.0*1i)
	rzero = 0.0

	//     Decode and test the input parameters.
	bothv = side == 'B'
	rightv = side == 'R' || bothv
	leftv = side == 'L' || bothv

	fromqr = eigsrc == 'Q'

	noinit = initv == 'N'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors.
	(*m) = 0
	for k = 1; k <= (*n); k++ {
		if (*_select)[k-1] {
			(*m) = (*m) + 1
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
		(*info) = -10
	} else if (*ldvr) < 1 || (rightv && (*ldvr) < (*n)) {
		(*info) = -12
	} else if (*mm) < (*m) {
		(*info) = -13
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZHSEIN"), -(*info))
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

	ldwork = (*n)

	kl = 1
	kln = 0
	if fromqr {
		kr = 0
	} else {
		kr = (*n)
	}
	ks = 1

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
				for i = k; i >= kl+1; i -= 1 {
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
				hnorm = Zlanhs('I', toPtr(kr-kl+1), h.Off(kl-1, kl-1), ldh, rwork)
				if Disnan(int(hnorm)) {
					(*info) = -6
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
				if (*_select)[i-1] && cabs1(w.Get(i-1)-wk) < eps3 {
					wk = wk + complex(eps3, 0)
					goto label60
				}
			}
			w.Set(k-1, wk)

			if leftv {
				//              Compute left eigenvector.
				Zlaein(false, noinit, toPtr((*n)-kl+1), h.Off(kl-1, kl-1), ldh, &wk, vl.CVector(kl-1, ks-1), work.CMatrix(ldwork, opts), &ldwork, rwork, &eps3, &smlnum, &iinfo)
				if iinfo > 0 {
					(*info) = (*info) + 1
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
				Zlaein(true, noinit, &kr, h, ldh, &wk, vr.CVector(0, ks-1), work.CMatrix(ldwork, opts), &ldwork, rwork, &eps3, &smlnum, &iinfo)
				if iinfo > 0 {
					(*info) = (*info) + 1
					(*ifailr)[ks-1] = k
				} else {
					(*ifailr)[ks-1] = 0
				}
				for i = kr + 1; i <= (*n); i++ {
					vr.Set(i-1, ks-1, zero)
				}
			}
			ks = ks + 1
		}
	}
}
