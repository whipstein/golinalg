package lin

import (
	"github.com/whipstein/golinalg/goblas"
	"github.com/whipstein/golinalg/golapack"
	"github.com/whipstein/golinalg/mat"
)

// Zspt03 computes the residual for a complex symmetric packed matrix
// times its inverse:
//    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
// where EPS is the machine epsilon.
func Zspt03(uplo byte, n *int, a, ainv *mat.CVector, work *mat.CMatrix, ldw *int, rwork *mat.Vector, rcond, resid *float64) {
	var t complex128
	var ainvnm, anorm, eps, one, zero float64
	var i, icol, j, jcol, k, kcol, nall int

	zero = 0.0
	one = 1.0

	//     Quick exit if N = 0.
	if (*n) <= 0 {
		(*rcond) = one
		(*resid) = zero
		return
	}

	//     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
	eps = golapack.Dlamch(Epsilon)
	anorm = golapack.Zlansp('1', uplo, n, a, rwork)
	ainvnm = golapack.Zlansp('1', uplo, n, ainv, rwork)
	if anorm <= zero || ainvnm <= zero {
		(*rcond) = zero
		(*resid) = one / eps
		return
	}
	(*rcond) = (one / anorm) / ainvnm

	//     Case where both A and AINV are upper triangular:
	//     Each element of - A * AINV is computed by taking the dot product
	//     of a row of A with a column of AINV.
	if uplo == 'U' {
		for i = 1; i <= (*n); i++ {
			icol = ((i-1)*i)/2 + 1

			//           Code when J <= I
			for j = 1; j <= i; j++ {
				jcol = ((j-1)*j)/2 + 1
				t = goblas.Zdotu(j, a.Off(icol-1), 1, ainv.Off(jcol-1), 1)
				jcol = jcol + 2*j - 1
				kcol = icol - 1
				for k = j + 1; k <= i; k++ {
					t = t + a.Get(kcol+k-1)*ainv.Get(jcol-1)
					jcol = jcol + k
				}
				kcol = kcol + 2*i
				for k = i + 1; k <= (*n); k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol-1)
					kcol = kcol + k
					jcol = jcol + k
				}
				work.Set(i-1, j-1, -t)
			}

			//           Code when J > I
			for j = i + 1; j <= (*n); j++ {
				jcol = ((j-1)*j)/2 + 1
				t = goblas.Zdotu(i, a.Off(icol-1), 1, ainv.Off(jcol-1), 1)
				jcol = jcol - 1
				kcol = icol + 2*i - 1
				for k = i + 1; k <= j; k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol+k-1)
					kcol = kcol + k
				}
				jcol = jcol + 2*j
				for k = j + 1; k <= (*n); k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol-1)
					kcol = kcol + k
					jcol = jcol + k
				}
				work.Set(i-1, j-1, -t)
			}
		}
	} else {
		//        Case where both A and AINV are lower triangular
		nall = ((*n) * ((*n) + 1)) / 2
		for i = 1; i <= (*n); i++ {
			//           Code when J <= I
			icol = nall - (((*n)-i+1)*((*n)-i+2))/2 + 1
			for j = 1; j <= i; j++ {
				jcol = nall - (((*n)-j)*((*n)-j+1))/2 - ((*n) - i)
				t = goblas.Zdotu((*n)-i+1, a.Off(icol-1), 1, ainv.Off(jcol-1), 1)
				kcol = i
				jcol = j
				for k = 1; k <= j-1; k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol-1)
					jcol = jcol + (*n) - k
					kcol = kcol + (*n) - k
				}
				jcol = jcol - j
				for k = j; k <= i-1; k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol+k-1)
					kcol = kcol + (*n) - k
				}
				work.Set(i-1, j-1, -t)
			}

			//           Code when J > I
			icol = nall - (((*n)-i)*((*n)-i+1))/2
			for j = i + 1; j <= (*n); j++ {
				jcol = nall - (((*n)-j+1)*((*n)-j+2))/2 + 1
				t = goblas.Zdotu((*n)-j+1, a.Off(icol-(*n)+j-1), 1, ainv.Off(jcol-1), 1)
				kcol = i
				jcol = j
				for k = 1; k <= i-1; k++ {
					t = t + a.Get(kcol-1)*ainv.Get(jcol-1)
					jcol = jcol + (*n) - k
					kcol = kcol + (*n) - k
				}
				kcol = kcol - i
				for k = i; k <= j-1; k++ {
					t = t + a.Get(kcol+k-1)*ainv.Get(jcol-1)
					jcol = jcol + (*n) - k
				}
				work.Set(i-1, j-1, -t)
			}
		}
	}

	//     Add the identity matrix to WORK .
	for i = 1; i <= (*n); i++ {
		work.Set(i-1, i-1, work.Get(i-1, i-1)+complex(one, 0))
	}

	//     Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
	(*resid) = golapack.Zlange('1', n, n, work, ldw, rwork)

	(*resid) = (((*resid) * (*rcond)) / eps) / float64(*n)
}
