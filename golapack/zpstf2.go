package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
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
func Zpstf2(uplo mat.MatUplo, n int, a *mat.CMatrix, piv *[]int, tol float64, work *mat.Vector) (rank, info int, err error) {
	var upper bool
	var cone, ztemp complex128
	var ajj, dstop, dtemp, one, zero float64
	var i, itemp, j, pvt int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpstf2", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Initialize PIV
	for i = 1; i <= n; i++ {
		(*piv)[i-1] = i
	}

	//     Compute stopping value
	for i = 1; i <= n; i++ {
		work.Set(i-1, real(a.Get(i-1, i-1)))
	}
	if n > 1 {
		pvt = maxlocf64(work.Data[:n-1]...)
	} else {
		pvt = 1
	}
	ajj = real(a.Get(pvt-1, pvt-1))
	if ajj <= zero || Disnan(int(ajj)) {
		rank = 0
		info = 1
		return
	}

	//     Compute stopping value if not supplied
	if tol < zero {
		dstop = float64(n) * Dlamch(Epsilon) * ajj
	} else {
		dstop = tol
	}

	//     Set first half of WORK to zero, holds dot products
	for i = 1; i <= n; i++ {
		work.Set(i-1, 0)
	}

	if upper {
		//        Compute the Cholesky factorization P**T * A * P = U**H* U
		for j = 1; j <= n; j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= n; i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+a.GetConjProd(j-1-1, i-1))
				}
				work.Set(n+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

			}

			if j > 1 {
				if n+j < 2*n-1 {
					itemp = maxlocf64(work.Data[(n + j) : (2*n)-1]...)
				} else {
					itemp = 1
				}
				pvt = itemp + j - 1
				ajj = work.Get(n + pvt - 1)
				if ajj <= dstop || Disnan(int(ajj)) {
					a.SetRe(j-1, j-1, ajj)
					goto label190
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Zswap(j-1, a.CVector(0, j-1, 1), a.CVector(0, pvt-1, 1))
				if pvt < n {
					goblas.Zswap(n-pvt, a.CVector(j-1, pvt), a.CVector(pvt-1, pvt))
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
			if j < n {
				Zlacgv(j-1, a.CVector(0, j-1, 1))
				if err = goblas.Zgemv(Trans, j-1, n-j, -cone, a.Off(0, j), a.CVector(0, j-1, 1), cone, a.CVector(j-1, j)); err != nil {
					panic(err)
				}
				Zlacgv(j-1, a.CVector(0, j-1, 1))
				goblas.Zdscal(n-j, one/ajj, a.CVector(j-1, j))
			}

		}

	} else {
		//        Compute the Cholesky factorization P**T * A * P = L * L**H
		for j = 1; j <= n; j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= n; i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+a.GetConjProd(i-1, j-1-1))
				}
				work.Set(n+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

			}

			if j > 1 {
				if n+j < (2*n)-1 {
					itemp = maxlocf64(work.Data[(n + j) : (2*n)-1]...)
				} else {
					itemp = 1
				}
				pvt = itemp + j - 1
				ajj = work.Get(n + pvt - 1)
				if ajj <= dstop || Disnan(int(ajj)) {
					a.SetRe(j-1, j-1, ajj)
					goto label190
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Zswap(j-1, a.CVector(j-1, 0), a.CVector(pvt-1, 0))
				if pvt < n {
					goblas.Zswap(n-pvt, a.CVector(pvt, j-1, 1), a.CVector(pvt, pvt-1, 1))
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
			if j < n {
				Zlacgv(j-1, a.CVector(j-1, 0))
				if err = goblas.Zgemv(NoTrans, n-j, j-1, -cone, a.Off(j, 0), a.CVector(j-1, 0), cone, a.CVector(j, j-1, 1)); err != nil {
					panic(err)
				}
				Zlacgv(j-1, a.CVector(j-1, 0))
				goblas.Zdscal(n-j, one/ajj, a.CVector(j, j-1, 1))
			}

		}

	}

	//     Ran to completion, A has full rank
	rank = n

	return
label190:
	;

	//     Rank is number of steps completed.  Set INFO = 1 to signal
	//     that the factorization cannot be used to solve a system.
	rank = j - 1
	info = 1

	return
}
