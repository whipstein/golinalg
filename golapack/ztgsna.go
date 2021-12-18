package golapack

import (
	"fmt"
	"math/cmplx"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsna estimates reciprocal condition numbers for specified
// eigenvalues and/or eigenvectors of a matrix pair (A, B).
//
// (A, B) must be in generalized Schur canonical form, that is, A and
// B are both upper triangular.
func Ztgsna(job, howmny byte, _select []bool, n int, a, b, vl, vr *mat.CMatrix, s, dif *mat.Vector, mm int, work *mat.CVector, lwork int, iwork *[]int) (m int, err error) {
	var lquery, somcon, wantbh, wantdf, wants bool
	var yhax, yhbx complex128
	var bignum, cond, eps, lnrm, one, rnrm, smlnum, zero float64
	var i, idifjb, ierr, ifst, ilst, k, ks, lwmin, n1, n2 int

	dummy := cvf(1)
	dummy1 := cvf(1)

	zero = 0.0
	one = 1.0
	idifjb = 3

	//     Decode and test the input parameters
	wantbh = job == 'B'
	wants = job == 'E' || wantbh
	wantdf = job == 'V' || wantbh

	somcon = howmny == 'S'

	lquery = (lwork == -1)

	if !wants && !wantdf {
		err = fmt.Errorf("!wants && !wantdf: job='%c'", job)
	} else if howmny != 'A' && !somcon {
		err = fmt.Errorf("howmny != 'A' && !somcon: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	} else if b.Rows < max(1, n) {
		err = fmt.Errorf("b.Rows < max(1, n): b.Rows=%v, n=%v", b.Rows, n)
	} else if wants && vl.Rows < n {
		err = fmt.Errorf("wants && vl.Rows < n: job='%c', vl.Rows=%v, n=%v", job, vl.Rows, n)
	} else if wants && vr.Rows < n {
		err = fmt.Errorf("wants && vr.Rows < n: job='%c', vr.Rows=%v, n=%v", job, vr.Rows, n)
	} else {
		//        Set M to the number of eigenpairs for which condition numbers
		//        are required, and test MM.
		if somcon {
			m = 0
			for k = 1; k <= n; k++ {
				if _select[k-1] {
					m = m + 1
				}
			}
		} else {
			m = n
		}

		if n == 0 {
			lwmin = 1
		} else if job == 'V' || job == 'B' {
			lwmin = 2 * n * n
		} else {
			lwmin = n
		}
		work.SetRe(0, float64(lwmin))

		if mm < m {
			err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
		} else if lwork < lwmin && !lquery {
			err = fmt.Errorf("lwork < lwmin && !lquery: lwork=%v, lwmin=%v, lquery=%v", lwork, lwmin, lquery)
		}
	}

	if err != nil {
		gltest.Xerbla2("Ztgsna", err)
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	smlnum, bignum = Dlabad(smlnum, bignum)
	ks = 0
	for k = 1; k <= n; k++ {
		//        Determine whether condition numbers are required for the k-th
		//        eigenpair.
		if somcon {
			if !_select[k-1] {
				goto label20
			}
		}

		ks = ks + 1

		if wants {
			//           Compute the reciprocal condition number of the k-th
			//           eigenvalue.
			rnrm = vr.Off(0, ks-1).CVector().Nrm2(n, 1)
			lnrm = vl.Off(0, ks-1).CVector().Nrm2(n, 1)
			err = work.Gemv(NoTrans, n, n, complex(one, zero), a, vr.Off(0, ks-1).CVector(), 1, complex(zero, zero), 1)
			yhax = vl.Off(0, ks-1).CVector().Dotc(n, work, 1, 1)
			err = work.Gemv(NoTrans, n, n, complex(one, zero), b, vr.Off(0, ks-1).CVector(), 1, complex(zero, zero), 1)
			yhbx = vl.Off(0, ks-1).CVector().Dotc(n, work, 1, 1)
			cond = Dlapy2(cmplx.Abs(yhax), cmplx.Abs(yhbx))
			if cond == zero {
				s.Set(ks-1, -one)
			} else {
				s.Set(ks-1, cond/(rnrm*lnrm))
			}
		}

		if wantdf {
			if n == 1 {
				dif.Set(ks-1, Dlapy2(a.GetMag(0, 0), b.GetMag(0, 0)))
			} else {
				//              Estimate the reciprocal condition number of the k-th
				//              eigenvectors.
				//
				//              Copy the matrix (A, B) to the array WORK and move the
				//              (k,k)th pair to the (1,1) position.
				Zlacpy(Full, n, n, a, work.CMatrix(n, opts))
				Zlacpy(Full, n, n, b, work.Off(n*n).CMatrix(n, opts))
				ifst = k
				ilst = 1

				if ilst, ierr, err = Ztgexc(false, false, n, work.CMatrix(n, opts), work.Off(n*n).CMatrix(n, opts), dummy.CMatrix(1, opts), dummy1.CMatrix(1, opts), ifst, ilst); err != nil {
					panic(err)
				}

				if ierr > 0 {
					//                 Ill-conditioned problem - swap rejected.
					dif.Set(ks-1, zero)
				} else {
					//                 Reordering successful, solve generalized Sylvester
					//                 equation for R and L,
					//                            A22 * R - L * A11 = A12
					//                            B22 * R - L * B11 = B12,
					//                 and compute estimate of Difl[(A11,B11), (A22, B22)].
					n1 = 1
					n2 = n - n1
					i = n*n + 1
					_, *dif.GetPtr(ks - 1), ierr, err = Ztgsyl(NoTrans, idifjb, n2, n1, work.Off(n*n1+n1).CMatrix(n, opts), work.CMatrix(n, opts), work.Off(n1).CMatrix(n, opts), work.Off(n*n1+n1+i-1).CMatrix(n, opts), work.Off(i-1).CMatrix(n, opts), work.Off(n1+i-1).CMatrix(n, opts), dummy, 1, iwork)
				}
			}
		}

	label20:
	}
	work.SetRe(0, float64(lwmin))

	return
}
