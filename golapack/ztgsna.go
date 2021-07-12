package golapack

import (
	"math/cmplx"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztgsna estimates reciprocal condition numbers for specified
// eigenvalues and/or eigenvectors of a matrix pair (A, B).
//
// (A, B) must be in generalized Schur canonical form, that is, A and
// B are both upper triangular.
func Ztgsna(job, howmny byte, _select []bool, n *int, a *mat.CMatrix, lda *int, b *mat.CMatrix, ldb *int, vl *mat.CMatrix, ldvl *int, vr *mat.CMatrix, ldvr *int, s, dif *mat.Vector, mm *int, m *int, work *mat.CVector, lwork *int, iwork *[]int, info *int) {
	var lquery, somcon, wantbh, wantdf, wants bool
	var yhax, yhbx complex128
	var bignum, cond, eps, lnrm, one, rnrm, scale, smlnum, zero float64
	var i, idifjb, ierr, ifst, ilst, k, ks, lwmin, n1, n2 int
	var err error
	_ = err

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

	(*info) = 0
	lquery = ((*lwork) == -1)

	if !wants && !wantdf {
		(*info) = -1
	} else if howmny != 'A' && !somcon {
		(*info) = -2
	} else if (*n) < 0 {
		(*info) = -4
	} else if (*lda) < max(1, *n) {
		(*info) = -6
	} else if (*ldb) < max(1, *n) {
		(*info) = -8
	} else if wants && (*ldvl) < (*n) {
		(*info) = -10
	} else if wants && (*ldvr) < (*n) {
		(*info) = -12
	} else {
		//        Set M to the number of eigenpairs for which condition numbers
		//        are required, and test MM.
		if somcon {
			(*m) = 0
			for k = 1; k <= (*n); k++ {
				if _select[k-1] {
					(*m) = (*m) + 1
				}
			}
		} else {
			(*m) = (*n)
		}

		if (*n) == 0 {
			lwmin = 1
		} else if job == 'V' || job == 'B' {
			lwmin = 2 * (*n) * (*n)
		} else {
			lwmin = (*n)
		}
		work.SetRe(0, float64(lwmin))

		if (*mm) < (*m) {
			(*info) = -15
		} else if (*lwork) < lwmin && !lquery {
			(*info) = -18
		}
	}

	if (*info) != 0 {
		gltest.Xerbla([]byte("ZTGSNA"), -(*info))
		return
	} else if lquery {
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Get machine constants
	eps = Dlamch(Precision)
	smlnum = Dlamch(SafeMinimum) / eps
	bignum = one / smlnum
	Dlabad(&smlnum, &bignum)
	ks = 0
	for k = 1; k <= (*n); k++ {
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
			rnrm = goblas.Dznrm2(*n, vr.CVector(0, ks-1, 1))
			lnrm = goblas.Dznrm2(*n, vl.CVector(0, ks-1, 1))
			err = goblas.Zgemv(NoTrans, *n, *n, complex(one, zero), a, vr.CVector(0, ks-1, 1), complex(zero, zero), work.Off(0, 1))
			yhax = goblas.Zdotc(*n, work.Off(0, 1), vl.CVector(0, ks-1, 1))
			err = goblas.Zgemv(NoTrans, *n, *n, complex(one, zero), b, vr.CVector(0, ks-1, 1), complex(zero, zero), work.Off(0, 1))
			yhbx = goblas.Zdotc(*n, work.Off(0, 1), vl.CVector(0, ks-1, 1))
			cond = Dlapy2(toPtrf64(cmplx.Abs(yhax)), toPtrf64(cmplx.Abs(yhbx)))
			if cond == zero {
				s.Set(ks-1, -one)
			} else {
				s.Set(ks-1, cond/(rnrm*lnrm))
			}
		}

		if wantdf {
			if (*n) == 1 {
				dif.Set(ks-1, Dlapy2(toPtrf64(a.GetMag(0, 0)), toPtrf64(b.GetMag(0, 0))))
			} else {
				//              Estimate the reciprocal condition number of the k-th
				//              eigenvectors.
				//
				//              Copy the matrix (A, B) to the array WORK and move the
				//              (k,k)th pair to the (1,1) position.
				Zlacpy('F', n, n, a, lda, work.CMatrix(*n, opts), n)
				Zlacpy('F', n, n, b, ldb, work.CMatrixOff((*n)*(*n), *n, opts), n)
				ifst = k
				ilst = 1

				Ztgexc(false, false, n, work.CMatrix(*n, opts), n, work.CMatrixOff((*n)*(*n), *n, opts), n, dummy.CMatrix(1, opts), func() *int { y := 1; return &y }(), dummy1.CMatrix(1, opts), func() *int { y := 1; return &y }(), &ifst, &ilst, &ierr)

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
					n2 = (*n) - n1
					i = (*n)*(*n) + 1
					Ztgsyl('N', &idifjb, &n2, &n1, work.CMatrixOff((*n)*n1+n1, *n, opts), n, work.CMatrix(*n, opts), n, work.CMatrixOff(n1, *n, opts), n, work.CMatrixOff((*n)*n1+n1+i-1, *n, opts), n, work.CMatrixOff(i-1, *n, opts), n, work.CMatrixOff(n1+i-1, *n, opts), n, &scale, dif.GetPtr(ks-1), dummy, func() *int { y := 1; return &y }(), iwork, &ierr)
				}
			}
		}

	label20:
	}
	work.SetRe(0, float64(lwmin))
}
