package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Dpstf2 computes the Cholesky factorization with complete
// pivoting of a real symmetric positive semidefinite matrix A.
//
// The factorization has the form
//    P**T * A * P = U**T * U ,  if UPLO = 'U',
//    P**T * A * P = L  * L**T,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular, and
// P is stored as vector PIV.
//
// This algorithm does not attempt to check that A is positive
// semidefinite. This version of the algorithm calls level 2 BLAS.
func Dpstf2(uplo mat.MatUplo, n int, a *mat.Matrix, piv *[]int, tol float64, work *mat.Vector) (rank, info int, err error) {
	var upper bool
	var ajj, dstop, dtemp, one, zero float64
	var i, itemp, j, pvt int

	one = 1.0
	zero = 0.0

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
		gltest.Xerbla2("Dpstf2", err)
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
	pvt = 1
	ajj = a.Get(pvt-1, pvt-1)
	for i = 2; i <= n; i++ {
		if a.Get(i-1, i-1) > ajj {
			pvt = i
			ajj = a.Get(pvt-1, pvt-1)
		}
	}
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
		//        Compute the Cholesky factorization P**T * A * P = U**T * U
		for j = 1; j <= n; j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= n; i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+math.Pow(a.Get(j-1-1, i-1), 2))
				}
				work.Set(n+i-1, a.Get(i-1, i-1)-work.Get(i-1))

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
					a.Set(j-1, j-1, ajj)
					goto label160
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Dswap(j-1, a.Vector(0, j-1, 1), a.Vector(0, pvt-1, 1))
				if pvt < n {
					goblas.Dswap(n-pvt, a.Vector(j-1, pvt), a.Vector(pvt-1, pvt))
				}
				goblas.Dswap(pvt-j-1, a.Vector(j-1, j), a.Vector(j, pvt-1, 1))

				//              Swap dot products and PIV
				dtemp = work.Get(j - 1)
				work.Set(j-1, work.Get(pvt-1))
				work.Set(pvt-1, dtemp)
				itemp = (*piv)[pvt-1]
				(*piv)[pvt-1] = (*piv)[j-1]
				(*piv)[j-1] = itemp
			}

			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of row J
			if j < n {
				err = goblas.Dgemv(mat.Trans, j-1, n-j, -one, a.Off(0, j), a.Vector(0, j-1, 1), one, a.Vector(j-1, j))
				goblas.Dscal(n-j, one/ajj, a.Vector(j-1, j))
			}

		}

	} else {
		//        Compute the Cholesky factorization P**T * A * P = L * L**T
		for j = 1; j <= n; j++ {
			//        Find pivot, test for exit, else swap rows and columns
			//        Update dot products, compute possible pivots which are
			//        stored in the second half of WORK
			for i = j; i <= n; i++ {

				if j > 1 {
					work.Set(i-1, work.Get(i-1)+math.Pow(a.Get(i-1, j-1-1), 2))
				}
				work.Set(n+i-1, a.Get(i-1, i-1)-work.Get(i-1))

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
					a.Set(j-1, j-1, ajj)
					goto label160
				}
			}

			if j != pvt {
				//              Pivot OK, so can now swap pivot rows and columns
				a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
				goblas.Dswap(j-1, a.Vector(j-1, 0), a.Vector(pvt-1, 0))
				if pvt < n {
					goblas.Dswap(n-pvt, a.Vector(pvt, j-1, 1), a.Vector(pvt, pvt-1, 1))
				}
				goblas.Dswap(pvt-j-1, a.Vector(j, j-1, 1), a.Vector(pvt-1, j))

				//              Swap dot products and PIV
				dtemp = work.Get(j - 1)
				work.Set(j-1, work.Get(pvt-1))
				work.Set(pvt-1, dtemp)
				itemp = (*piv)[pvt-1]
				(*piv)[pvt-1] = (*piv)[j-1]
				(*piv)[j-1] = itemp
			}

			ajj = math.Sqrt(ajj)
			a.Set(j-1, j-1, ajj)

			//           Compute elements J+1:N of column J
			if j < n {
				err = goblas.Dgemv(NoTrans, n-j, j-1, -one, a.Off(j, 0), a.Vector(j-1, 0), one, a.Vector(j, j-1, 1))
				goblas.Dscal(n-j, one/ajj, a.Vector(j, j-1, 1))
			}

		}

	}
	//     Ran to completion, A has full rank
	rank = n

	return
label160:
	;

	//     Rank is number of steps completed.  Set INFO = 1 to signal
	//     that the factorization cannot be used to solve a system.
	rank = j - 1
	info = 1

	return
}
