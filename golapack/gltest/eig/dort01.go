package eig

import (
	"golinalg/goblas"
	"golinalg/golapack"
	"golinalg/mat"
	"math"
)

// Dort01 checks that the matrix U is orthogonal by computing the ratio
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
func Dort01(rowcol byte, m, n *int, u *mat.Matrix, ldu *int, work *mat.Vector, lwork *int, resid *float64) {
	var transu byte
	var eps, one, tmp, zero float64
	var i, j, k, ldwork, mnmin int

	zero = 0.0
	one = 1.0

	(*resid) = zero

	//     Quick return if possible
	if (*m) <= 0 || (*n) <= 0 {
		return
	}

	eps = golapack.Dlamch(Precision)
	if (*m) < (*n) || ((*m) == (*n) && rowcol == 'R') {
		transu = 'N'
		k = (*n)
	} else {
		transu = 'T'
		k = (*m)
	}
	mnmin = minint(*m, *n)

	if (mnmin+1)*mnmin <= (*lwork) {
		ldwork = mnmin
	} else {
		ldwork = 0
	}
	if ldwork > 0 {
		//        Compute I - U*U' or I - U'*U.
		golapack.Dlaset('U', &mnmin, &mnmin, &zero, &one, work.Matrix(ldwork, opts), &ldwork)
		goblas.Dsyrk(Upper, mat.TransByte(transu), &mnmin, &k, toPtrf64(-one), u, ldu, &one, work.Matrix(ldwork, opts), &ldwork)

		//        Compute norm( I - U*U' ) / ( K * EPS ) .
		(*resid) = golapack.Dlansy('1', 'U', &mnmin, work.Matrix(ldwork, opts), &ldwork, work.Off(ldwork*mnmin+1-1))
		(*resid) = ((*resid) / float64(k)) / eps
	} else if transu == 'T' {
		//        Find the maximum element in abs( I - U'*U ) / ( m * EPS )
		for j = 1; j <= (*n); j++ {
			for i = 1; i <= j; i++ {
				if i != j {
					tmp = zero
				} else {
					tmp = one
				}
				tmp = tmp - goblas.Ddot(m, u.Vector(0, i-1), toPtr(1), u.Vector(0, j-1), toPtr(1))
				(*resid) = maxf64(*resid, math.Abs(tmp))
			}
		}
		(*resid) = ((*resid) / float64(*m)) / eps
	} else {
		//        Find the maximum element in abs( I - U*U' ) / ( n * EPS )
		for j = 1; j <= (*m); j++ {
			for i = 1; i <= j; i++ {
				if i != j {
					tmp = zero
				} else {
					tmp = one
				}
				tmp = tmp - goblas.Ddot(n, u.Vector(j-1, 0), ldu, u.Vector(i-1, 0), ldu)
				(*resid) = maxf64(*resid, math.Abs(tmp))
			}
		}
		(*resid) = ((*resid) / float64(*n)) / eps
	}
}
