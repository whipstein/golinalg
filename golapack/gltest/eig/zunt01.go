package eig

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// zunt01 checks that the matrix U is unitary by computing the ratio
//
//    RESID = norm( I - U*U' ) / ( n * EPS ), if ROWCOL = 'R',
// or
//    RESID = norm( I - U'*U ) / ( m * EPS ), if ROWCOL = 'C'.
//
// Alternatively, if there isn't sufficient workspace to form
// I - U*U' or I - U'*U, the ratio is computed as
//
//    RESID = abs( I - U*U' ) / ( n * EPS ), if ROWCOL = 'R',
// or
//    RESID = abs( I - U'*U ) / ( m * EPS ), if ROWCOL = 'C'.
//
// where EPS is the machine precision.  ROWCOL is used only if m = n;
// if m > n, ROWCOL is assumed to be 'C', and if m < n, ROWCOL is
// assumed to be 'R'.
func zunt01(rowcol byte, m, n int, u *mat.CMatrix, work *mat.CVector, lwork int, rwork *mat.Vector) (resid float64) {
	var transu mat.MatTrans
	var tmp complex128
	var eps, one, zero float64
	var i, j, k, ldwork, mnmin int
	var err error

	zero = 0.0
	one = 1.0

	//     Quick return if possible
	if m <= 0 || n <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)
	if m < n || (m == n && rowcol == 'R') {
		transu = NoTrans
		k = n
	} else {
		transu = ConjTrans
		k = m
	}
	mnmin = min(m, n)

	if (mnmin+1)*mnmin <= lwork {
		ldwork = mnmin
	} else {
		ldwork = 0
	}
	if ldwork > 0 {
		//        Compute I - U*U' or I - U'*U.
		golapack.Zlaset(Upper, mnmin, mnmin, complex(zero, 0), complex(one, 0), work.CMatrix(ldwork, opts))
		if err = work.CMatrix(ldwork, opts).Herk(Upper, transu, mnmin, k, -one, u, one); err != nil {
			panic(err)
		}

		//        Compute norm( I - U*U' ) / ( K * EPS ) .
		resid = golapack.Zlansy('1', Upper, mnmin, work.CMatrix(ldwork, opts), rwork)
		resid = (resid / float64(k)) / eps
	} else if transu == 'C' {
		//        Find the maximum element in abs( I - U'*U ) / ( m * EPS )
		for j = 1; j <= n; j++ {
			for i = 1; i <= j; i++ {
				if i != j {
					tmp = complex(zero, 0)
				} else {
					tmp = complex(one, 0)
				}
				tmp = tmp - u.Off(0, j-1).CVector().Dotc(m, u.Off(0, i-1).CVector(), 1, 1)
				resid = math.Max(resid, cabs1(tmp))
			}
		}
		resid = (resid / float64(m)) / eps
	} else {
		//        Find the maximum element in abs( I - U*U' ) / ( n * EPS )
		for j = 1; j <= m; j++ {
			for i = 1; i <= j; i++ {
				if i != j {
					tmp = complex(zero, 0)
				} else {
					tmp = complex(one, 0)
				}
				tmp = tmp - u.Off(i-1, 0).CVector().Dotc(n, u.Off(j-1, 0).CVector(), u.Rows, u.Rows)
				resid = math.Max(resid, cabs1(tmp))
			}
		}
		resid = (resid / float64(n)) / eps
	}

	return
}
