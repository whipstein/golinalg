package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrsna estimates reciprocal condition numbers for specified
// eigenvalues and/or right eigenvectors of a complex upper triangular
// matrix T (or of any matrix Q*T*Q**H with Q unitary).
func Ztrsna(job, howmny byte, _select []bool, n *int, t *mat.CMatrix, ldt *int, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr *int, s, sep *mat.Vector, mm, m *int, work *mat.CMatrix, ldwork *int, rwork *mat.Vector, info *int) {
	var somcon, wantbh, wants, wantsp bool
	var normin byte
	var prod complex128
	var bignum, eps, est, lnrm, one, rnrm, scale, smlnum, xnorm, zero float64
	var i, ierr, ix, j, k, kase, ks int
	dummy := cvf(1)
	isave := make([]int, 3)

	zero = 0.0
	one = 1.0 + 0

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantsp = job == 'V' || wantbh

	somcon = howmny == 'S'

	//     Set M to the number of eigenpairs for which condition numbers are
	//     to be computed.
	if somcon {
		(*m) = 0
		for j = 1; j <= (*n); j++ {
			if _select[j-1] {
				(*m) = (*m) + 1
			}
		}
	} else {
		(*m) = (*n)
	}

	(*info) = 0
	if !wants && !wantsp {
		(*info) = -1
	} else if howmny != 'A' && !somcon {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*ldt) < maxint(1, *n) {
		(*info) = -6
	} else if (*ldvl) < 1 || (wants && (*ldvl) < (*n)) {
		(*info) = -8
	} else if (*ldvr) < 1 || (wants && (*ldvr) < (*n)) {
		(*info) = -10
	} else if (*mm) < (*m) {
		(*info) = -13
	} else if (*ldwork) < 1 || (wantsp && (*ldwork) < (*n)) {
		(*info) = -16
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTRSNA"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	if (*n) == 1 {
		if somcon {
			if !_select[0] {
				return
			}
		}
		if wants {
			s.Set(0, one)
		}
		if wantsp {
			sep.Set(0, t.GetMag(0, 0))
		}
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)

	ks = 1
	for k = 1; k <= (*n); k++ {

		if somcon {
			if !_select[k-1] {
				goto label50
			}
		}

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			prod = goblas.Zdotc(*n, vr.CVector(0, ks-1), 1, vl.CVector(0, ks-1), 1)
			rnrm = goblas.Dznrm2(*n, vr.CVector(0, ks-1), 1)
			lnrm = goblas.Dznrm2(*n, vl.CVector(0, ks-1), 1)
			s.Set(ks-1, cmplx.Abs(prod)/(rnrm*lnrm))

		}

		if wantsp {
			//           Estimate the reciprocal condition number of the k-th
			//           eigenvector.
			//
			//           Copy the matrix T to the array WORK and swap the k-th
			//           diagonal element to the (1,1) position.
			Zlacpy('F', n, n, t, ldt, work, ldwork)
			Ztrexc('N', n, work, ldwork, dummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), &k, func() *int { y := 1; return &y }(), &ierr)

			//           Form  C = T22 - lambda*I in WORK(2:N,2:N).
			for i = 2; i <= (*n); i++ {
				work.Set(i-1, i-1, work.Get(i-1, i-1)-work.Get(0, 0))
			}

			//           Estimate a lower bound for the 1-norm of inv(C**H). The 1st
			//           and (N+1)th columns of WORK are used to store work vectors.
			sep.Set(ks-1, zero)
			est = zero
			kase = 0
			normin = 'N'
		label30:
			;
			Zlacn2(toPtr((*n)-1), work.CVector(0, (*n)+1-1), work.CVector(0, 0), &est, &kase, &isave)

			if kase != 0 {
				if kase == 1 {
					//                 Solve C**H*x = scale*b
					Zlatrs('U', 'C', 'N', normin, toPtr((*n)-1), work.Off(1, 1), ldwork, work.CVector(0, 0), &scale, rwork, &ierr)
				} else {
					//                 Solve C*x = scale*b
					Zlatrs('U', 'N', 'N', normin, toPtr((*n)-1), work.Off(1, 1), ldwork, work.CVector(0, 0), &scale, rwork, &ierr)
				}
				normin = 'Y'
				if scale != one {
					//                 Multiply by 1/SCALE if doing so will not cause
					//                 overflow.
					ix = goblas.Izamax((*n)-1, work.CVector(0, 0), 1)
					xnorm = cabs1(work.Get(ix-1, 0))
					if scale < xnorm*smlnum || scale == zero {
						goto label40
					}
					Zdrscl(n, &scale, work.CVector(0, 0), func() *int { y := 1; return &y }())
				}
				goto label30
			}

			sep.Set(ks-1, one/maxf64(est, smlnum))
		}

	label40:
		;
		ks = ks + 1
	label50:
	}
}
