package golapack

import (
	"golinalg/goblas"
	"golinalg/golapack/gltest"
	"golinalg/mat"
	"math"
)

// Zpstf2 computes the Cholesky factorization with complete
// pivoting of a complex Hermitian positive semidefinite matrix A.
//
// The factorization has the form
//    P**T * A * P = U**H * U ,  if UPLO = 'U',
//    P**T * A * P = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular, and
// P is stored as vector PIV.
//
// This algorithm does not attempt to check that A is positive
// semidefinite. This version of the algorithm calls level 2 BLAS.
func Zpstf2(uplo byte, n *int, a *mat.CMatrix, lda *int, piv *[]int, rank *int, tol *float64, work *mat.Vector, info *int) {
	var upper bool
	var cone, ztemp complex128
	var ajj, dstop, dtemp, one, zero float64
	var i, itemp, j, pvt int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	(*info) = 0
	upper = uplo == 'U'
	if !upper && uplo != 'L' {
		(*info) = -1
	} else if (*n) < 0 {
		(*info) = -2
	} else if (*lda) < maxint(1, *n) {
		(*info) = -4
	}
	if (*info) != 0 {
		gltest.Xerbla([]byte("ZPSTF2"), -(*info))
		return
	}

	//     Quick return if possible
	if (*n) == 0 {
		return
	}

	//     Initialize PIV
	for i = 1; i <= (*n); i++ {
		(*piv)[i-1] = i
	}

	//     Compute stopping value
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, real(a.Get(i-1, i-1)))
	}
	if *n > 1 {
		pvt = maxlocf64(work.Data[:(*n)-1]...)
	} else {
		pvt = 1
	}
	ajj = real(a.Get(pvt-1, pvt-1))
	if ajj <= zero || Disnan(int(ajj)) {
		(*rank) = 0
		(*info) = 1
		return
	}

	//     Compute stopping value if not supplied
	if (*tol) < zero {
		dstop = float64(*n) * Dlamch(Epsilon) * ajj
	} else {
		dstop = (*tol)
	}

	//     Set first half of WORK to zero, holds dot products
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, 0)
	}

	if upper {
		//        Compute the Cholesky factorization P**T * A * P = U**H* U
		for j = 1; j <= (*n); j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= (*n); i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+a.GetConjProd(j-1-1, i-1))
				}
				work.Set((*n)+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

			}

			if j > 1 {
				if (*n)+j < 2*(*n)-1 {
					itemp = maxlocf64(work.Data[((*n) + j) : (2*(*n))-1]...)
				} else {
					itemp = 1
				}
				pvt = itemp + j - 1
				ajj = work.Get((*n) + pvt - 1)
				if ajj <= dstop || Disnan(int(ajj)) {
					a.SetRe(j-1, j-1, ajj)
					goto label190
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Zswap(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }(), a.CVector(0, pvt-1), func() *int { y := 1; return &y }())
				if pvt < (*n) {
					goblas.Zswap(toPtr((*n)-pvt), a.CVector(j-1, pvt+1-1), lda, a.CVector(pvt-1, pvt+1-1), lda)
				}
				for i = j + 1; i <= pvt-1; i++ {
					ztemp = a.GetConj(j-1, i-1)
					a.Set(j-1, i-1, a.GetConj(i-1, pvt-1))
					a.Set(i-1, pvt-1, ztemp)
				}
				a.Set(j-1, pvt-1, a.GetConj(j-1, pvt-1))

				//              Swap dot products and PIV
				dtemp = work.Get(j - 1)
				work.Set(j-1, work.Get(pvt-1))
				work.Set(pvt-1, dtemp)
				itemp = (*piv)[pvt-1]
				(*piv)[pvt-1] = (*piv)[j-1]
				(*piv)[j-1] = itemp
			}

			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J
			if j < (*n) {
				Zlacgv(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zgemv(Trans, toPtr(j-1), toPtr((*n)-j), toPtrc128(-cone), a.Off(0, j+1-1), lda, a.CVector(0, j-1), func() *int { y := 1; return &y }(), &cone, a.CVector(j-1, j+1-1), lda)
				Zlacgv(toPtr(j-1), a.CVector(0, j-1), func() *int { y := 1; return &y }())
				goblas.Zdscal(toPtr((*n)-j), toPtrf64(one/ajj), a.CVector(j-1, j+1-1), lda)
			}

		}

	} else {
		//        Compute the Cholesky factorization P**T * A * P = L * L**H
		for j = 1; j <= (*n); j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= (*n); i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+a.GetConjProd(i-1, j-1-1))
				}
				work.Set((*n)+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

			}

			if j > 1 {
				if (*n)+j < (2*(*n))-1 {
					itemp = maxlocf64(work.Data[((*n) + j) : (2*(*n))-1]...)
				} else {
					itemp = 1
				}
				pvt = itemp + j - 1
				ajj = work.Get((*n) + pvt - 1)
				if ajj <= dstop || Disnan(int(ajj)) {
					a.SetRe(j-1, j-1, ajj)
					goto label190
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Zswap(toPtr(j-1), a.CVector(j-1, 0), lda, a.CVector(pvt-1, 0), lda)
				if pvt < (*n) {
					goblas.Zswap(toPtr((*n)-pvt), a.CVector(pvt+1-1, j-1), func() *int { y := 1; return &y }(), a.CVector(pvt+1-1, pvt-1), func() *int { y := 1; return &y }())
				}
				for i = j + 1; i <= pvt-1; i++ {
					ztemp = a.GetConj(i-1, j-1)
					a.Set(i-1, j-1, a.GetConj(pvt-1, i-1))
					a.Set(pvt-1, i-1, ztemp)
				}
				a.Set(pvt-1, j-1, a.GetConj(pvt-1, j-1))

				//              Swap dot products and PIV
				dtemp = work.Get(j - 1)
				work.Set(j-1, work.Get(pvt-1))
				work.Set(pvt-1, dtemp)
				itemp = (*piv)[pvt-1]
				(*piv)[pvt-1] = (*piv)[j-1]
				(*piv)[j-1] = itemp
			}

			ajj = math.Sqrt(ajj)
			a.SetRe(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J
			if j < (*n) {
				Zlacgv(toPtr(j-1), a.CVector(j-1, 0), lda)
				goblas.Zgemv(NoTrans, toPtr((*n)-j), toPtr(j-1), toPtrc128(-cone), a.Off(j+1-1, 0), lda, a.CVector(j-1, 0), lda, &cone, a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
				Zlacgv(toPtr(j-1), a.CVector(j-1, 0), lda)
				goblas.Zdscal(toPtr((*n)-j), toPtrf64(one/ajj), a.CVector(j+1-1, j-1), func() *int { y := 1; return &y }())
			}

		}

	}

	//     Ran to completion, A has full rank
	(*rank) = (*n)

	return
label190:
	;

	//     Rank is number of steps completed.  Set INFO = 1 to signal
	//     that the factorization cannot be used to solve a system.
	(*rank) = j - 1
	(*info) = 1
}
