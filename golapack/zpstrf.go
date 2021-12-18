package golapack

import (
	"fmt"
	"math"

	"github.com/whipstein/golinalg/golapack/gltest"
	"github.com/whipstein/golinalg/mat"
)

// Zpstrf computes the Cholesky factorization with complete
// pivoting of a complex Hermitian positive semidefinite matrix A.
//
// The factorization has the form
//    P**T * A * P = U**H * U ,  if UPLO = 'U',
//    P**T * A * P = L  * L**H,  if UPLO = 'L',
// where U is an upper triangular matrix and L is lower triangular, and
// P is stored as vector PIV.
//
// This algorithm does not attempt to check that A is positive
// semidefinite. This version of the algorithm calls level 3 BLAS.
func Zpstrf(uplo mat.MatUplo, n int, a *mat.CMatrix, piv *[]int, tol float64, work *mat.Vector) (rank, info int, err error) {
	var upper bool
	var cone, ztemp complex128
	var ajj, dstop, dtemp, one, zero float64
	var i, itemp, j, jb, k, nb, pvt int

	one = 1.0
	zero = 0.0
	cone = (1.0 + 0.0*1i)

	//     Test the input parameters.
	upper = uplo == Upper
	if !upper && uplo != Lower {
		err = fmt.Errorf("!upper && uplo != Lower: uplo=%s", uplo)
	} else if n < 0 {
		err = fmt.Errorf("n < 0: n=%v", n)
	} else if a.Rows < max(1, n) {
		err = fmt.Errorf("a.Rows < max(1, n): a.Rows=%v, n=%v", a.Rows, n)
	}
	if err != nil {
		gltest.Xerbla2("Zpstrf", err)
		return
	}

	//     Quick return if possible
	if n == 0 {
		return
	}

	//     Get block size
	nb = Ilaenv(1, "Zpotrf", []byte{uplo.Byte()}, n, -1, -1, -1)
	if nb <= 1 || nb >= n {
		//        Use unblocked code
		if rank, info, err = Zpstf2(uplo, n, a, piv, tol, work); err != nil {
			panic(err)
		}
		return

	} else {
		//     Initialize PIV
		for i = 1; i <= n; i++ {
			(*piv)[i-1] = i
		}

		//     Compute stopping value
		for i = 1; i <= n; i++ {
			work.Set(i-1, real(a.Get(i-1, i-1)))
		}
		pvt = maxlocf64(work.Get(0), 1)
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

		if upper {
			//           Compute the Cholesky factorization P**T * A * P = U**H * U
			for k = 1; k <= n; k += nb {
				//              Account for last block not being NB wide
				jb = min(nb, n-k+1)

				//              Set relevant part of first half of WORK to zero,
				//              holds dot products
				for i = k; i <= n; i++ {
					work.Set(i-1, 0)
				}

				for j = k; j <= k+jb-1; j++ {
					//              Find pivot, test for exit, else swap rows and columns
					//              Update dot products, compute possible pivots which are
					//              stored in the second half of WORK
					for i = j; i <= n; i++ {

						if j > k {
							work.Set(i-1, work.Get(i-1)+a.GetConjProd(j-1-1, i-1))
						}
						work.Set(n+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

					}

					if j > 1 {
						itemp = maxlocf64(append(work.Data()[(n+j):(2*n)-1], 1)...)
						pvt = itemp + j - 1
						ajj = work.Get(n + pvt - 1)
						if ajj <= dstop || Disnan(int(ajj)) {
							a.SetRe(j-1, j-1, ajj)
							goto label220
						}
					}

					if j != pvt {
						//                    Pivot OK, so can now swap pivot rows and columns
						a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
						a.Off(0, pvt-1).CVector().Swap(j-1, a.Off(0, j-1).CVector(), 1, 1)
						if pvt < n {
							a.Off(pvt-1, pvt).CVector().Swap(n-pvt, a.Off(j-1, pvt).CVector(), a.Rows, a.Rows)
						}
						for i = j + 1; i <= pvt-1; i++ {
							ztemp = a.GetConj(j-1, i-1)
							a.Set(j-1, i-1, a.GetConj(i-1, pvt-1))
							a.Set(i-1, pvt-1, ztemp)
						}
						a.Set(j-1, pvt-1, a.GetConj(j-1, pvt-1))

						//                    Swap dot products and PIV
						dtemp = work.Get(j - 1)
						work.Set(j-1, work.Get(pvt-1))
						work.Set(pvt-1, dtemp)
						itemp = (*piv)[pvt-1]
						(*piv)[pvt-1] = (*piv)[j-1]
						(*piv)[j-1] = itemp
					}

					ajj = math.Sqrt(ajj)
					a.SetRe(j-1, j-1, ajj)

					//                 Compute elements J+1:N of row J.
					if j < n {
						Zlacgv(j-1, a.Off(0, j-1).CVector(), 1)
						if err = a.Off(j-1, j).CVector().Gemv(Trans, j-k, n-j, -cone, a.Off(k-1, j), a.Off(k-1, j-1).CVector(), 1, cone, a.Rows); err != nil {
							panic(err)
						}
						Zlacgv(j-1, a.Off(0, j-1).CVector(), 1)
						a.Off(j-1, j).CVector().Dscal(n-j, one/ajj, a.Rows)
					}

				}

				//              Update trailing matrix, J already incremented
				if k+jb <= n {
					if err = a.Off(j-1, j-1).Herk(Upper, ConjTrans, n-j+1, jb, -one, a.Off(k-1, j-1), one); err != nil {
						panic(err)
					}
				}

			}

		} else {
			//        Compute the Cholesky factorization P**T * A * P = L * L**H
			for k = 1; k <= n; k += nb {
				//              Account for last block not being NB wide
				jb = min(nb, n-k+1)

				//              Set relevant part of first half of WORK to zero,
				//              holds dot products
				for i = k; i <= n; i++ {
					work.Set(i-1, 0)
				}

				for j = k; j <= k+jb-1; j++ {
					//              Find pivot, test for exit, else swap rows and columns
					//              Update dot products, compute possible pivots which are
					//              stored in the second half of WORK
					for i = j; i <= n; i++ {

						if j > k {
							work.Set(i-1, work.Get(i-1)+a.GetConjProd(i-1, j-1-1))
						}
						work.Set(n+i-1, real(a.Get(i-1, i-1))-work.Get(i-1))

					}

					if j > 1 {
						itemp = maxlocf64(append(work.Data()[(n+j):(2*n)-1], 1)...)
						pvt = itemp + j - 1
						ajj = work.Get(n + pvt - 1)
						if ajj <= dstop || Disnan(int(ajj)) {
							a.SetRe(j-1, j-1, ajj)
							goto label220
						}
					}

					if j != pvt {
						//                    Pivot OK, so can now swap pivot rows and columns
						a.Set(pvt-1, pvt-1, a.Get(j-1, j-1))
						a.Off(pvt-1, 0).CVector().Swap(j-1, a.Off(j-1, 0).CVector(), a.Rows, a.Rows)
						if pvt < n {
							a.Off(pvt, pvt-1).CVector().Swap(n-pvt, a.Off(pvt, j-1).CVector(), 1, 1)
						}
						for i = j + 1; i <= pvt-1; i++ {
							ztemp = a.GetConj(i-1, j-1)
							a.Set(i-1, j-1, a.GetConj(pvt-1, i-1))
							a.Set(pvt-1, i-1, ztemp)
						}
						a.Set(pvt-1, j-1, a.GetConj(pvt-1, j-1))

						//                    Swap dot products and PIV
						dtemp = work.Get(j - 1)
						work.Set(j-1, work.Get(pvt-1))
						work.Set(pvt-1, dtemp)
						itemp = (*piv)[pvt-1]
						(*piv)[pvt-1] = (*piv)[j-1]
						(*piv)[j-1] = itemp
					}

					ajj = math.Sqrt(ajj)
					a.SetRe(j-1, j-1, ajj)

					//                 Compute elements J+1:N of column J.
					if j < n {
						Zlacgv(j-1, a.Off(j-1, 0).CVector(), a.Rows)
						if err = a.Off(j, j-1).CVector().Gemv(NoTrans, n-j, j-k, -cone, a.Off(j, k-1), a.Off(j-1, k-1).CVector(), a.Rows, cone, 1); err != nil {
							panic(err)
						}
						Zlacgv(j-1, a.Off(j-1, 0).CVector(), a.Rows)
						a.Off(j, j-1).CVector().Dscal(n-j, one/ajj, 1)
					}

				}

				//              Update trailing matrix, J already incremented
				if k+jb <= n {
					if err = a.Off(j-1, j-1).Herk(Lower, NoTrans, n-j+1, jb, -one, a.Off(j-1, k-1), one); err != nil {
						panic(err)
					}
				}

			}

		}
	}

	//     Ran to completion, A has full rank
	rank = n

	return
label220:
	;

	//     Rank is the number of steps completed.  Set INFO = 1 to signal
	//     that the factorization cannot be used to solve a system.
	rank = j - 1
	info = 1

	return
}
