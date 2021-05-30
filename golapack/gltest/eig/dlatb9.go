package eig

import (
	"golinalg/golapack"
	"math"
)

// Dlatb9 sets parameters for the matrix generator based on the _type of
// matrix to be generated.
func Dlatb9(path []byte, imat, m, p, n *int, _type *byte, kla, kua, klb, kub *int, anorm, bnorm *float64, modea, modeb *int, cndnma, cndnmb *float64, dista, distb *byte) {
	var first bool
	var badc1, badc2, eps, large, one, shrink, small, ten, tenth float64

	shrink = 0.25
	tenth = 0.1
	one = 1.0
	ten = 1.0e+1

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
	(*_type) = 'N'
	(*dista) = 'S'
	(*distb) = 'S'
	(*modea) = 3
	(*modeb) = 4

	//     Set the lower and upper bandwidths.
	if string(path) == "GRQ" || string(path) == "LSE" || string(path) == "GSV" {
		//        A: M by N, B: P by N
		if (*imat) == 1 {
			//           A: diagonal, B: upper triangular
			(*kla) = 0
			(*kua) = 0
			(*klb) = 0
			(*kub) = maxint((*n)-1, 0)

		} else if (*imat) == 2 {
			//           A: upper triangular, B: upper triangular
			(*kla) = 0
			(*kua) = maxint((*n)-1, 0)
			(*klb) = 0
			(*kub) = maxint((*n)-1, 0)

		} else if (*imat) == 3 {
			//           A: lower triangular, B: upper triangular
			(*kla) = maxint((*m)-1, 0)
			(*kua) = 0
			(*klb) = 0
			(*kub) = maxint((*n)-1, 0)

		} else {
			//           A: general dense, B: general dense
			(*kla) = maxint((*m)-1, 0)
			(*kua) = maxint((*n)-1, 0)
			(*klb) = maxint((*p)-1, 0)
			(*kub) = maxint((*n)-1, 0)

		}

	} else if string(path) == "GQR" || string(path) == "GLM" {
		//        A: N by M, B: N by P
		if (*imat) == 1 {
			//           A: diagonal, B: lower triangular
			(*kla) = 0
			(*kua) = 0
			(*klb) = maxint((*n)-1, 0)
			(*kub) = 0
		} else if (*imat) == 2 {
			//           A: lower triangular, B: diagonal
			(*kla) = maxint((*n)-1, 0)
			(*kua) = 0
			(*klb) = 0
			(*kub) = 0

		} else if (*imat) == 3 {
			//           A: lower triangular, B: upper triangular
			(*kla) = maxint((*n)-1, 0)
			(*kua) = 0
			(*klb) = 0
			(*kub) = maxint((*p)-1, 0)

		} else {
			//           A: general dense, B: general dense
			(*kla) = maxint((*n)-1, 0)
			(*kua) = maxint((*m)-1, 0)
			(*klb) = maxint((*n)-1, 0)
			(*kub) = maxint((*p)-1, 0)
		}

	}

	//     Set the condition number and norm.
	(*cndnma) = ten * ten
	(*cndnmb) = ten
	if string(path) == "GQR" || string(path) == "GRQ" || string(path) == "GSV" {
		if (*imat) == 5 {
			(*cndnma) = badc1
			(*cndnmb) = badc1
		} else if (*imat) == 6 {
			(*cndnma) = badc2
			(*cndnmb) = badc2
		} else if (*imat) == 7 {
			(*cndnma) = badc1
			(*cndnmb) = badc2
		} else if (*imat) == 8 {
			(*cndnma) = badc2
			(*cndnmb) = badc1
		}
	}

	(*anorm) = ten
	(*bnorm) = ten * ten * ten
	if string(path) == "GQR" || string(path) == "GRQ" {
		if (*imat) == 7 {
			(*anorm) = small
			(*bnorm) = large
		} else if (*imat) == 8 {
			(*anorm) = large
			(*bnorm) = small
		}
	}

	if (*n) <= 1 {
		(*cndnma) = one
		(*cndnmb) = one
	}
}
