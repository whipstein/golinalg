package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Ztrevc computes some or all of the right and/or left eigenvectors of
// a complex upper triangular matrix T.
// Matrices of this _type are produced by the Schur factorization of
// a complex general matrix:  A = Q*T*Q**H, as computed by ZHSEQR.
//
// The right eigenvector x and the left eigenvector y of T corresponding
// to an eigenvalue w are defined by:
//
//              T*x = w*x,     (y**H)*T = w*(y**H)
//
// where y**H denotes the conjugate transpose of the vector y.
// The eigenvalues are not input to this routine, but are read directly
// from the diagonal of T.
//
// This routine returns the matrices X and/or Y of right and left
// eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
// input matrix.  If Q is the unitary factor that reduces a matrix A to
// Schur form T, then Q*X and Q*Y are the matrices of right and left
// eigenvectors of A.
func Ztrevc(side mat.MatSide, howmny byte, _select []bool, n int, t, vl, vr *mat.CMatrix, mm int, work *mat.CVector, rwork *mat.Vector) (m int, err error) {
	var allv, bothv, leftv, over, rightv, somev bool
	var cmone, cmzero complex128
	var one, ovfl, remax, scale, smin, smlnum, ulp, unfl, zero float64
	var i, ii, is, j, k, ki int

	zero = 0.0
	one = 1.0
	cmzero = (0.0 + 0.0*1i)
	cmone = (1.0 + 0.0*1i)

	//     Decode and test the input parameters
	bothv = side == Both
	rightv = side == Right || bothv
	leftv = side == Left || bothv

	allv = howmny == 'A'
	over = howmny == 'B'
	somev = howmny == 'S'

	//     Set M to the number of columns required to store the selected
	//     eigenvectors.
	if somev {
		m = 0
		for j = 1; j <= n; j++ {
			if _select[j-1] {
				m = m + 1
			}
		}
	} else {
		m = n
	}

	if !rightv && !leftv {
		err = fmt.Errorf("!rightv && !leftv: side=%s", side)
	} else if !allv && !over && !somev {
		err = fmt.Errorf("!allv && !over && !somev: howmny='%c'", howmny)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if t.Rows < max(1, n) {
		err = fmt.Errorf("t.Rows < max(1, n): t.Rows=%v, n=%v", t.Rows, n)
	} else if vl.Rows < 1 || (leftv && vl.Rows < n) {
		err = fmt.Errorf("vl.Rows < 1 || (leftv && vl.Rows < n): side=%s, vl.Rows=%v, n=%v", side, vl.Rows, n)
	} else if vr.Rows < 1 || (rightv && vr.Rows < n) {
		err = fmt.Errorf("vr.Rows < 1 || (rightv && vr.Rows < n): side=%s, vr.Rows=%v, n=%v", side, vr.Rows, n)
	} else if mm < m {
		err = fmt.Errorf("mm < m: mm=%v, m=%v", mm, m)
	}
	if err != nil {
		gltest.Xerbla2("Ztrevc", err)
		return
	}

	//     Quick return if possible.
	if n == 0 {
		return
	}

	//     Set the constants to control overflow.
	unfl = Dlamch(SafeMinimum)
	ovfl = one / unfl
	unfl, ovfl = Dlabad(unfl, ovfl)
	ulp = Dlamch(Precision)
	smlnum = unfl * (float64(n) / ulp)

	//     Store the diagonal elements of T in working array WORK.
	for i = 1; i <= n; i++ {
		work.Set(i+n-1, t.Get(i-1, i-1))
	}

	//     Compute 1-norm of each column of strictly upper triangular
	//     part of T to control overflow in triangular solver.
	rwork.Set(0, zero)
	for j = 2; j <= n; j++ {
		rwork.Set(j-1, goblas.Dzasum(j-1, t.CVector(0, j-1, 1)))
	}

	if rightv {
		//        Compute right eigenvectors.
		is = m
		for ki = n; ki >= 1; ki -= 1 {

			if somev {
				if !_select[ki-1] {
					goto label80
				}
			}
			smin = math.Max(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			work.Set(0, cmone)

			//           Form right-hand side.
			for k = 1; k <= ki-1; k++ {
				work.Set(k-1, -t.Get(k-1, ki-1))
			}

			//           Solve the triangular system:
			//              (T(1:KI-1,1:KI-1) - T(KI,KI))*X = SCALE*WORK.
			for k = 1; k <= ki-1; k++ {
				t.Set(k-1, k-1, t.Get(k-1, k-1)-t.Get(ki-1, ki-1))
				if cabs1(t.Get(k-1, k-1)) < smin {
					t.SetRe(k-1, k-1, smin)
				}
			}

			if ki > 1 {
				if scale, err = Zlatrs(Upper, NoTrans, NonUnit, 'Y', ki-1, t, work, rwork); err != nil {
					panic(err)
				}
				work.SetRe(ki-1, scale)
			}

			//           Copy the vector x or Q*x to VR and normalize.
			if !over {
				goblas.Zcopy(ki, work.Off(0, 1), vr.CVector(0, is-1, 1))

				ii = goblas.Izamax(ki, vr.CVector(0, is-1, 1))
				remax = one / cabs1(vr.Get(ii-1, is-1))
				goblas.Zdscal(ki, remax, vr.CVector(0, is-1, 1))

				for k = ki + 1; k <= n; k++ {
					vr.Set(k-1, is-1, cmzero)
				}
			} else {
				if ki > 1 {
					if err = goblas.Zgemv(NoTrans, n, ki-1, cmone, vr, work.Off(0, 1), complex(scale, 0), vr.CVector(0, ki-1, 1)); err != nil {
						panic(err)
					}
				}

				ii = goblas.Izamax(n, vr.CVector(0, ki-1, 1))
				remax = one / cabs1(vr.Get(ii-1, ki-1))
				goblas.Zdscal(n, remax, vr.CVector(0, ki-1, 1))
			}

			//           Set back the original diagonal elements of T.
			for k = 1; k <= ki-1; k++ {
				t.Set(k-1, k-1, work.Get(k+n-1))
			}

			is = is - 1
		label80:
		}
	}

	if leftv {
		//        Compute left eigenvectors.
		is = 1
		for ki = 1; ki <= n; ki++ {

			if somev {
				if !_select[ki-1] {
					goto label130
				}
			}
			smin = math.Max(ulp*cabs1(t.Get(ki-1, ki-1)), smlnum)

			work.Set(n-1, cmone)

			//           Form right-hand side.
			for k = ki + 1; k <= n; k++ {
				work.Set(k-1, -t.GetConj(ki-1, k-1))
			}

			//           Solve the triangular system:
			//              (T(KI+1:N,KI+1:N) - T(KI,KI))**H * X = SCALE*WORK.
			for k = ki + 1; k <= n; k++ {
				t.Set(k-1, k-1, t.Get(k-1, k-1)-t.Get(ki-1, ki-1))
				if cabs1(t.Get(k-1, k-1)) < smin {
					t.SetRe(k-1, k-1, smin)
				}
			}

			if ki < n {
				if scale, err = Zlatrs(Upper, ConjTrans, NonUnit, 'Y', n-ki, t.Off(ki, ki), work.Off(ki), rwork); err != nil {
					panic(err)
				}
				work.SetRe(ki-1, scale)
			}

			//           Copy the vector x or Q*x to VL and normalize.
			if !over {
				goblas.Zcopy(n-ki+1, work.Off(ki-1, 1), vl.CVector(ki-1, is-1, 1))

				ii = goblas.Izamax(n-ki+1, vl.CVector(ki-1, is-1, 1)) + ki - 1
				remax = one / cabs1(vl.Get(ii-1, is-1))
				goblas.Zdscal(n-ki+1, remax, vl.CVector(ki-1, is-1, 1))

				for k = 1; k <= ki-1; k++ {
					vl.Set(k-1, is-1, cmzero)
				}
			} else {
				if ki < n {
					if err = goblas.Zgemv(NoTrans, n, n-ki, cmone, vl.Off(0, ki), work.Off(ki, 1), complex(scale, 0), vl.CVector(0, ki-1, 1)); err != nil {
						panic(err)
					}
				}

				ii = goblas.Izamax(n, vl.CVector(0, ki-1, 1))
				remax = one / cabs1(vl.Get(ii-1, ki-1))
				goblas.Zdscal(n, remax, vl.CVector(0, ki-1, 1))
			}

			//           Set back the original diagonal elements of T.
			for k = ki + 1; k <= n; k++ {
				t.Set(k-1, k-1, work.Get(k+n-1))
			}

			is = is + 1
		label130:
		}
	}

	return
}
