package lin

import (
	"math"
	"strings"

	"github.com/whipstein/golinalg/golapack"
)

// dlatb4 sets parameters for the matrix generator based on the _type of
// matrix to be generated.
func dlatb4(path string, imat, m, n int) (_type byte, kl, ku int, anorm float64, mode int, cndnum float64, dist byte) {
	var first bool
	var badc1, badc2, eps, large, one, shrink, small, tenth, two float64
	var mat int

	shrink = 0.25
	tenth = 0.1
	one = 1.0
	two = 2.0

	first = true

	//     Set some constants for use in the subroutine.
	if first {
		// first = false
		eps = golapack.Dlamch(Precision)
		badc2 = tenth / eps
		badc1 = math.Sqrt(badc2)
		small = golapack.Dlamch(SafeMinimum)
		large = one / small

		//        If it looks like we're on a Cray, take the square root of
		//        SMALL and LARGE to avoid overflow and underflow problems.
		small, large = golapack.Dlabad(small, large)
		small = shrink * (small / eps)
		large = one / small
	}

	c2 := path[1:3]

	//     Set some parameters we don't plan to change.
	dist = 'S'
	mode = 3

	if string(c2) == "qr" || string(c2) == "lq" || string(c2) == "ql" || string(c2) == "rq" {
		//        xQR, xLQ, xQL, xRQ:  Set parameters to generate a general
		//                             M x N matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the lower and upper bandwidths.
		if imat == 1 {
			kl = 0
			ku = 0
		} else if imat == 2 {
			kl = 0
			ku = max(n-1, 0)
		} else if imat == 3 {
			kl = max(m-1, 0)
			ku = 0
		} else {
			kl = max(m-1, 0)
			ku = max(n-1, 0)
		}

		//        Set the condition number and norm.
		if imat == 5 {
			cndnum = badc1
		} else if imat == 6 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 7 {
			anorm = small
		} else if imat == 8 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "ge" {
		//        xGE:  Set parameters to generate a general M x N matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the lower and upper bandwidths.
		if imat == 1 {
			kl = 0
			ku = 0
		} else if imat == 2 {
			kl = 0
			ku = max(n-1, 0)
		} else if imat == 3 {
			kl = max(m-1, 0)
			ku = 0
		} else {
			kl = max(m-1, 0)
			ku = max(n-1, 0)
		}

		//        Set the condition number and norm.
		if imat == 8 {
			cndnum = badc1
		} else if imat == 9 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 10 {
			anorm = small
		} else if imat == 11 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "gb" {
		//        xGB:  Set parameters to generate a general banded matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the condition number and norm.
		if imat == 5 {
			cndnum = badc1
		} else if imat == 6 {
			cndnum = tenth * badc2
		} else {
			cndnum = two
		}

		if imat == 7 {
			anorm = small
		} else if imat == 8 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "gt" {
		//        xGT:  Set parameters to generate a general tridiagonal matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the lower and upper bandwidths.
		if imat == 1 {
			kl = 0
		} else {
			kl = 1
		}
		ku = kl

		//        Set the condition number and norm.
		if imat == 3 {
			cndnum = badc1
		} else if imat == 4 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 5 || imat == 11 {
			anorm = small
		} else if imat == 6 || imat == 12 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "po" || string(c2) == "pp" {
		//        xPO, xPP: Set parameters to generate a
		//        symmetric positive definite matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = strings.ToUpper(c2)[0]

		//        Set the lower and upper bandwidths.
		if imat == 1 {
			kl = 0
		} else {
			kl = max(n-1, 0)
		}
		ku = kl

		//        Set the condition number and norm.
		if imat == 6 {
			cndnum = badc1
		} else if imat == 7 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 8 {
			anorm = small
		} else if imat == 9 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "sy" || string(c2) == "sp" {
		//        xSY, xSP: Set parameters to generate a
		//        symmetric matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = strings.ToUpper(c2)[0]

		//        Set the lower and upper bandwidths.
		if imat == 1 {
			kl = 0
		} else {
			kl = max(n-1, 0)
		}
		ku = kl

		//        Set the condition number and norm.
		if imat == 7 {
			cndnum = badc1
		} else if imat == 8 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 9 {
			anorm = small
		} else if imat == 10 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "pb" {
		//        xPB:  Set parameters to generate a symmetric band matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'P'

		//        Set the norm and condition number.
		if imat == 5 {
			cndnum = badc1
		} else if imat == 6 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 7 {
			anorm = small
		} else if imat == 8 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "pt" {
		//        xPT:  Set parameters to generate a symmetric positive definite
		//        tridiagonal matrix.
		_type = 'P'
		if imat == 1 {
			kl = 0
		} else {
			kl = 1
		}
		ku = kl

		//        Set the condition number and norm.
		if imat == 3 {
			cndnum = badc1
		} else if imat == 4 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 5 || imat == 11 {
			anorm = small
		} else if imat == 6 || imat == 12 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "tr" || string(c2) == "tp" {
		//        xTR, xTP:  Set parameters to generate a triangular matrix
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the lower and upper bandwidths.
		mat = abs(imat)
		if mat == 1 || mat == 7 {
			kl = 0
			ku = 0
		} else if imat < 0 {
			kl = max(n-1, 0)
			ku = 0
		} else {
			kl = 0
			ku = max(n-1, 0)
		}

		//        Set the condition number and norm.
		if mat == 3 || mat == 9 {
			cndnum = badc1
		} else if mat == 4 {
			cndnum = badc2
		} else if mat == 10 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if mat == 5 {
			anorm = small
		} else if mat == 6 {
			anorm = large
		} else {
			anorm = one
		}

	} else if string(c2) == "tb" {
		//        xTB:  Set parameters to generate a triangular band matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		_type = 'N'

		//        Set the norm and condition number.
		if imat == 2 || imat == 8 {
			cndnum = badc1
		} else if imat == 3 || imat == 9 {
			cndnum = badc2
		} else {
			cndnum = two
		}

		if imat == 4 {
			anorm = small
		} else if imat == 5 {
			anorm = large
		} else {
			anorm = one
		}
	}
	if n <= 1 {
		cndnum = one
	}

	return
}
