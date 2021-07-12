package lin

import (
	"math"

	"github.com/whipstein/golinalg/golapack"
)

// Zlatb4 sets parameters for the matrix generator based on the _type of
// matrix to be generated.
func Zlatb4(path []byte, imat, m, n *int, _type *byte, kl, ku *int, anorm *float64, mode *int, cndnum *float64, dist *byte) {
	var first bool
	var badc1, badc2, eps, large, one, shrink, small, tenth, two float64
	var mat int
	c2 := path[1:3]

	shrink = 0.25
	tenth = 0.1
	one = 1.0
	two = 2.0

	first = true

	//     Set some constants for use in the subroutine.
	if first {
		first = false
		eps = golapack.Dlamch(Precision)
		badc2 = tenth / eps
		badc1 = math.Sqrt(badc2)
		small = golapack.Dlamch(SafeMinimum)
		large = one / small

		//        If it looks like we're on a Cray, take the square root of
		//        SMALL and LARGE to avoid overflow and underflow problems.
		golapack.Dlabad(&small, &large)
		small = shrink * (small / eps)
		large = one / small
	}

	//     Set some parameters we don't plan to change.
	(*dist) = 'S'
	(*mode) = 3

	//     xQR, xLQ, xQL, xRQ:  Set parameters to generate a general
	//                          M x N matrix.
	if string(c2) == "QR" || string(c2) == "LQ" || string(c2) == "QL" || string(c2) == "RQ" {
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the lower and upper bandwidths.
		if (*imat) == 1 {
			(*kl) = 0
			(*ku) = 0
		} else if (*imat) == 2 {
			(*kl) = 0
			(*ku) = max((*n)-1, 0)
		} else if (*imat) == 3 {
			(*kl) = max((*m)-1, 0)
			(*ku) = 0
		} else {
			(*kl) = max((*m)-1, 0)
			(*ku) = max((*n)-1, 0)
		}

		//        Set the condition number and norm.
		if (*imat) == 5 {
			(*cndnum) = badc1
		} else if (*imat) == 6 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 7 {
			(*anorm) = small
		} else if (*imat) == 8 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "GE" {
		//        xGE:  Set parameters to generate a general M x N matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the lower and upper bandwidths.
		if (*imat) == 1 {
			(*kl) = 0
			(*ku) = 0
		} else if (*imat) == 2 {
			(*kl) = 0
			(*ku) = max((*n)-1, 0)
		} else if (*imat) == 3 {
			(*kl) = max((*m)-1, 0)
			(*ku) = 0
		} else {
			(*kl) = max((*m)-1, 0)
			(*ku) = max((*n)-1, 0)
		}

		//        Set the condition number and norm.
		if (*imat) == 8 {
			(*cndnum) = badc1
		} else if (*imat) == 9 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 10 {
			(*anorm) = small
		} else if (*imat) == 11 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "GB" {
		//        xGB:  Set parameters to generate a general banded matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the condition number and norm.
		if (*imat) == 5 {
			(*cndnum) = badc1
		} else if (*imat) == 6 {
			(*cndnum) = tenth * badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 7 {
			(*anorm) = small
		} else if (*imat) == 8 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "GT" {
		//        xGT:  Set parameters to generate a general tridiagonal matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the lower and upper bandwidths.
		if (*imat) == 1 {
			(*kl) = 0
		} else {
			(*kl) = 1
		}
		(*ku) = (*kl)

		//        Set the condition number and norm.
		if (*imat) == 3 {
			(*cndnum) = badc1
		} else if (*imat) == 4 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 5 || (*imat) == 11 {
			(*anorm) = small
		} else if (*imat) == 6 || (*imat) == 12 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "PO" || string(c2) == "PP" {
		//        xPO, xPP: Set parameters to generate a
		//        symmetric or Hermitian positive definite matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = c2[0]

		//        Set the lower and upper bandwidths.
		if (*imat) == 1 {
			(*kl) = 0
		} else {
			(*kl) = max((*n)-1, 0)
		}
		(*ku) = (*kl)

		//        Set the condition number and norm.
		if (*imat) == 6 {
			(*cndnum) = badc1
		} else if (*imat) == 7 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 8 {
			(*anorm) = small
		} else if (*imat) == 9 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "HE" || string(c2) == "HP" || string(c2) == "SY" || string(c2) == "SP" {
		//        xHE, xHP, xSY, xSP: Set parameters to generate a
		//        symmetric or Hermitian matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = c2[0]

		//        Set the lower and upper bandwidths.
		if (*imat) == 1 {
			(*kl) = 0
		} else {
			(*kl) = max((*n)-1, 0)
		}
		(*ku) = (*kl)

		//        Set the condition number and norm.
		if (*imat) == 7 {
			(*cndnum) = badc1
		} else if (*imat) == 8 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 9 {
			(*anorm) = small
		} else if (*imat) == 10 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "PB" {
		//        xPB:  Set parameters to generate a symmetric band matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'P'

		//        Set the norm and condition number.
		if (*imat) == 5 {
			(*cndnum) = badc1
		} else if (*imat) == 6 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 7 {
			(*anorm) = small
		} else if (*imat) == 8 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "PT" {
		//        xPT:  Set parameters to generate a symmetric positive definite
		//        tridiagonal matrix.
		(*_type) = 'P'
		if (*imat) == 1 {
			(*kl) = 0
		} else {
			(*kl) = 1
		}
		(*ku) = (*kl)

		//        Set the condition number and norm.
		if (*imat) == 3 {
			(*cndnum) = badc1
		} else if (*imat) == 4 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 5 || (*imat) == 11 {
			(*anorm) = small
		} else if (*imat) == 6 || (*imat) == 12 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "TR" || string(c2) == "TP" {
		//        xTR, xTP:  Set parameters to generate a triangular matrix
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the lower and upper bandwidths.
		mat = int(math.Abs(float64(*imat)))
		if mat == 1 || mat == 7 {
			(*kl) = 0
			(*ku) = 0
		} else if (*imat) < 0 {
			(*kl) = max((*n)-1, 0)
			(*ku) = 0
		} else {
			(*kl) = 0
			(*ku) = max((*n)-1, 0)
		}

		//        Set the condition number and norm.
		if mat == 3 || mat == 9 {
			(*cndnum) = badc1
		} else if mat == 4 || mat == 10 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if mat == 5 {
			(*anorm) = small
		} else if mat == 6 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}

	} else if string(c2) == "TB" {
		//        xTB:  Set parameters to generate a triangular band matrix.
		//
		//        Set TYPE, the _type of matrix to be generated.
		(*_type) = 'N'

		//        Set the norm and condition number.
		if (*imat) == 2 || (*imat) == 8 {
			(*cndnum) = badc1
		} else if (*imat) == 3 || (*imat) == 9 {
			(*cndnum) = badc2
		} else {
			(*cndnum) = two
		}

		if (*imat) == 4 {
			(*anorm) = small
		} else if (*imat) == 5 {
			(*anorm) = large
		} else {
			(*anorm) = one
		}
	}
	if (*n) <= 1 {
		(*cndnum) = one
	}
}
