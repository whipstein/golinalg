package golapack

import (
	"fmt"
	"math"
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrsna estimates reciprocal condition numbers for specified
// eigenvalues and/or right eigenvectors of a complex upper triangular
// matrix T (or of any matrix Q*T*Q**H with Q unitary).
func Ztrsna(job, howmny byte, _select []bool, n int, t, vl, vr *mat.CMatrix, s, sep *mat.Vector, mm int, work *mat.CMatrix, rwork *mat.Vector) (m int, err error) {
	var somcon, wantbh, wants, wantsp bool
	var normin byte
	var prod complex128
	var bignum, eps, est, lnrm, one, rnrm, scale, smlnum, xnorm, zero float64
	var i, ix, j, k, kase, ks int

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
		m = 0
		for j = 1; j <= n; j++ {
			if _select[j-1] {
				m = m + 1
			}
		}
	} else {
		m = n
	}

	if !wants && !wantsp {
		err = fmt.Errorf("!wants && !wantsp: job='%c'", job)
	} else if howmny != 'A' && !somcon {
		err = fmt.Errorf("howmny != 'A' && !somcon: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if vl.Rows < 1 || (wants && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (wants && vl.Rows < n): job='%c', vl.Rows=%v, n=%v", job, vl.Rows, n)
	} else if vr.Rows < 1 || (wants && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (wants && vr.Rows < n): job='%c', vr.Rows=%v, n=%v", job, vr.Rows, n)
	} else if mm < m {
		err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
	} else if work.Rows < 1 || (wantsp && work.Rows < n) {
		err = fmt.Errorf("work.Rows < 1 || (wantsp && work.Rows < n): job='%c', work.Rows=%v, n=%v", job, work.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Ztrsna", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	if n == 1 {
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
	smlnum, bignum = Dlabad(smlnum, bignum)

	ks = 1
	for k = 1; k <= n; k++ {

		if somcon {
			if !_select[k-1] {
				goto label50
			}
		}

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			prod = goblas.Zdotc(n, vr.CVector(0, ks-1, 1), vl.CVector(0, ks-1, 1))
			rnrm = goblas.Dznrm2(n, vr.CVector(0, ks-1, 1))
			lnrm = goblas.Dznrm2(n, vl.CVector(0, ks-1, 1))
			s.Set(ks-1, cmplx.Abs(prod)/(rnrm*lnrm))

		}

		if wantsp {
			//           Estimate the reciprocal condition number of the k-th
			//           eigenvector.
			//
			//           Copy the matrix T to the array WORK and swap the k-th
			//           diagonal element to the (1,1) position.
			Zlacpy(Full, n, n, t, work)
			if err = Ztrexc('N', n, work, dummy.CMatrix(1, opts), k, 1); err != nil {
				panic(err)
			}

			//           Form  C = T22 - lambda*I in WORK(2:N,2:N).
			for i = 2; i <= n; i++ {
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
			est, kase = Zlacn2(n-1, work.CVector(0, n), work.CVector(0, 0), est, kase, &isave)

			if kase != 0 {
				if kase == 1 {
					//                 Solve C**H*x = scale*b
					if scale, err = Zlatrs(Upper, ConjTrans, NonUnit, normin, n-1, work.Off(1, 1), work.CVector(0, 0), rwork); err != nil {
						panic(err)
					}
				} else {
					//                 Solve C*x = scale*b
					if scale, err = Zlatrs(Upper, NoTrans, NonUnit, normin, n-1, work.Off(1, 1), work.CVector(0, 0), rwork); err != nil {
						panic(err)
					}
				}
				normin = 'Y'
				if scale != one {
					//                 Multiply by 1/SCALE if doing so will not cause
					//                 overflow.
					ix = goblas.Izamax(n-1, work.CVector(0, 0, 1))
					xnorm = cabs1(work.Get(ix-1, 0))
					if scale < xnorm*smlnum || scale == zero {
						goto label40
					}
					Zdrscl(n, scale, work.CVector(0, 0, 1))
				}
				goto label30
			}

			sep.Set(ks-1, one/math.Max(est, smlnum))
		}

	label40:
		;
		ks = ks + 1
	label50:
	}

	return
}
